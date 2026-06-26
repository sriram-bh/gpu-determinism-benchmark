"""
Multi-stream Order DDI experiment: run 2 concurrent GPT-2 inferences
on separate streams, capture the global kernel execution order, and
measure run-to-run variability via Order DDI.

This models the safety-critical scenario where multiple ML workloads
(e.g., perception + planning) run concurrently on the same GPU, and
the GPU scheduler non-deterministically interleaves their kernels.

Run from repo root:
    python3 src/experiments/run_multistream_order_ddi.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.profiler import profile, ProfilerActivity
from src.experiments.gpu_harness import (
    _configure_deterministic_mode,
    _load_model,
    _tensor_bytes_for_event,
    cupti_lf_bridge,
    run_warmup,
    DECODE_STEPS,
    SEED,
    FIXED_INPUT_SENTENCE,
)
from src.metrics.schema import KernelEvent, events_to_dicts
from src.metrics.ddi import align_events, compute_timing_ddi, compute_order_ddi

N_RUNS = 5
N_STREAMS = 4


def run_multistream_pass(model, tokenizer, device):
    """
    Run N_STREAMS concurrent GPT-2 inferences and capture the global
    kernel execution order (all streams interleaved by start time).
    """
    logical_clock_state = {"tag": 0}
    all_events = []
    all_token_ids = [[] for _ in range(N_STREAMS)]

    # All non-default streams — the default stream has implicit
    # synchronization behavior that serializes execution.
    streams = [torch.cuda.Stream(device) for _ in range(N_STREAMS)]

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)

    # Per-stream inference state
    states = []
    for _ in range(N_STREAMS):
        states.append({
            "past": None,
            "input_ids": inputs["input_ids"].clone(),
            "next_token": None,
        })

    for decode_step in range(0, DECODE_STEPS + 1):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as step_prof:
            # Launch the same decode step on all streams concurrently
            for s_idx, (stream, state) in enumerate(zip(streams, states)):
                with torch.cuda.stream(stream):
                    with torch.no_grad():
                        if decode_step == 0:
                            out = model(state["input_ids"], use_cache=True)
                        else:
                            out = model(
                                state["next_token"],
                                past_key_values=state["past"],
                                use_cache=True,
                            )
                        state["past"] = out.past_key_values
                        state["next_token"] = torch.argmax(
                            out.logits[:, -1, :], dim=-1, keepdim=True
                        )
                        all_token_ids[s_idx].append(state["next_token"].item())

            torch.cuda.synchronize()

        # Capture global kernel sequence: all streams interleaved by time
        step_events = sorted(step_prof.events(), key=lambda e: e.time_range.start)
        for evt in step_events:
            if evt.key.startswith("aten::"):
                continue
            cuda_time_us = evt.self_device_time_total
            if cuda_time_us <= 0:
                continue

            duration_ns = int(cuda_time_us * 1000)
            wall_ns = int(evt.time_range.start * 1000)
            tag = cupti_lf_bridge(wall_ns, logical_clock_state)
            tensor_bytes = _tensor_bytes_for_event(evt)

            all_events.append(
                KernelEvent(
                    kernel_name=evt.key,
                    wall_ns=wall_ns,
                    logical_tag=tag,
                    duration_ns=duration_ns,
                    decode_step=decode_step,
                    tensor_bytes=tensor_bytes,
                    stream_id=0,
                )
            )

    return all_events, all_token_ids


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Streams: {N_STREAMS} concurrent GPT-2 inferences")
    print(f"Runs: {N_RUNS}")

    # Non-deterministic mode: allow cuBLAS to run concurrently across
    # streams without workspace serialization. Keep cudnn.benchmark=False
    # to prevent algorithm autotuning from changing kernel names.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.benchmark = False
    print("Mode: NON-DETERMINISTIC (concurrent cuBLAS enabled)")

    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    # Load G* kernel names for tracking
    gstar_path = "results/gstar.json"
    gstar_names = None
    if os.path.exists(gstar_path):
        with open(gstar_path) as f:
            gstar_data = json.load(f)
        gstar_names = set(e["kernel_name"] for e in gstar_data["gstar_events"])
        print(f"G* loaded: {len(gstar_names)} kernel names for comparison")

    print(f"\nRunning {N_RUNS} multi-stream passes...")
    runs = []
    for i in range(N_RUNS):
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        events, token_ids = run_multistream_pass(model, tokenizer, device)
        runs.append({"events": events, "token_ids": token_ids})
        print(f"  Run {i+1}: {len(events)} global events, "
              f"stream 0 tokens: {tokenizer.decode(token_ids[0][:10])}...")

    # Kernel name tracking against G*
    if gstar_names:
        run1_names = set(e.kernel_name for e in runs[0]["events"])
        missing = gstar_names - run1_names
        new = run1_names - gstar_names
        print(f"\n--- Kernel name diff vs G* ---")
        print(f"  G* names: {len(gstar_names)}, Run 1 names: {len(run1_names)}")
        print(f"  In common: {len(gstar_names & run1_names)}")
        if missing:
            print(f"  Missing from run 1: {len(missing)}")
            for n in sorted(missing):
                print(f"    GONE: {n[:90]}")
        if new:
            print(f"  New in run 1: {len(new)}")
            for n in sorted(new):
                print(f"    NEW:  {n[:90]}")
        if not missing and not new:
            print(f"  IDENTICAL kernel name sets.")

    # Diagnostic: check if events from run 1 step 0 show true interleaving
    step0_events = [e for e in runs[0]["events"] if e.decode_step == 0]
    print(f"\n--- Interleaving diagnostic (run 1, step 0) ---")
    print(f"  Events in step 0: {len(step0_events)}")
    print(f"  First 20 kernel names (by start time):")
    for i, e in enumerate(step0_events[:20]):
        print(f"    {i:3d}  wall={e.wall_ns:>12d}  {e.kernel_name[:80]}")

    # Use run 1 as the reference, compare all others against it
    ref_events = runs[0]["events"]

    print(f"\n{'Comparison':<18} {'TimingDDI':>10} {'OrderDDI':>10} {'Matched':>8} {'Tokens':>8}")
    print("-" * 58)

    results = []
    for i in range(1, N_RUNS):
        obs_events = runs[i]["events"]
        matched = align_events(obs_events, ref_events)

        if len(matched) < 2:
            print(f"  Run 1 vs {i+1}: too few matched events ({len(matched)})")
            continue

        timing = compute_timing_ddi(matched)
        order = compute_order_ddi(matched)

        # Check if all streams produced identical tokens to run 1
        tokens_ok = "match" if runs[i]["token_ids"] == runs[0]["token_ids"] else "DIFFER"

        print(f"Run 1 vs {i+1}         {timing:>10.6f} {order:>10.6f} {len(matched):>8} {tokens_ok:>8}")

        results.append({
            "comparison": f"run1_vs_run{i+1}",
            "timing_ddi": timing,
            "order_ddi": order,
            "matched_events": len(matched),
            "tokens_match": runs[i]["token_ids"] == runs[0]["token_ids"],
        })

    # Also compute pairwise stats
    order_ddis = [r["order_ddi"] for r in results]
    timing_ddis = [r["timing_ddi"] for r in results]
    if order_ddis:
        print(f"\nOrder DDI:  mean={sum(order_ddis)/len(order_ddis):.6f}, "
              f"max={max(order_ddis):.6f}")
        print(f"Timing DDI: mean={sum(timing_ddis)/len(timing_ddis):.6f}, "
              f"max={max(timing_ddis):.6f}")

    # Save
    os.makedirs("results", exist_ok=True)
    save_data = {
        "n_streams": N_STREAMS,
        "n_runs": N_RUNS,
        "decode_steps": DECODE_STEPS,
        "events_per_run": len(ref_events),
        "results": results,
        "reference_events": events_to_dicts(ref_events),
        "reference_tokens": runs[0]["token_ids"],
    }
    with open("results/multistream_order_ddi.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to results/multistream_order_ddi.json")


if __name__ == "__main__":
    main()
