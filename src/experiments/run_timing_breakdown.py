"""
Timing DDI breakdown with extended per-kernel metrics.

Runs Config A (G* baseline) and Config B-2streams (heavy contention)
with extended profiler data (FLOPs, CPU time, memory), computes the
6-way Timing DDI breakdown, and checks ncu/nsys availability for
follow-up cache analysis.

Run from repo root:
    python3 src/experiments/run_timing_breakdown.py
"""

import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.profiler import profile, ProfilerActivity
from src.experiments.gpu_harness import (
    _configure_deterministic_mode,
    _load_model,
    _tensor_bytes_for_event,
    run_warmup,
    DECODE_STEPS,
    SEED,
    FIXED_INPUT_SENTENCE,
)
from src.metrics.schema import KernelEvent, events_to_dicts
from src.metrics.ddi import (
    align_events,
    compute_timing_ddi,
    compute_order_ddi,
    full_breakdown_report,
)


def make_compute_contention_fn(device, n_streams, ops_per_step, tensor_size):
    streams = [torch.cuda.Stream() for _ in range(n_streams)]
    data = torch.randn(tensor_size, device=device)
    outs = [torch.empty_like(data) for _ in range(n_streams)]

    def fn(decode_step):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                for _ in range(ops_per_step):
                    torch.cumsum(data, dim=0, out=outs[i])

    return fn


def run_extended_pass(model, tokenizer, device, pre_step_fn=None):
    """Run one pass capturing extended per-kernel metrics."""
    logical_clock_state = {"tag": 0}
    all_events = []
    all_extended = []
    token_ids = []

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    past_key_values = None
    input_ids = inputs["input_ids"]

    for decode_step in range(0, DECODE_STEPS + 1):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            with_flops=True,
            profile_memory=True,
        ) as step_prof:
            if pre_step_fn:
                pre_step_fn(decode_step)
            with torch.no_grad():
                if decode_step == 0:
                    outputs = model(input_ids, use_cache=True)
                else:
                    outputs = model(
                        next_token,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(
                    outputs.logits[:, -1, :], dim=-1, keepdim=True
                )
                token_ids.append(next_token.item())
            torch.cuda.synchronize()

        step_events = sorted(
            step_prof.events(), key=lambda e: e.time_range.start
        )
        for evt in step_events:
            if evt.key.startswith("aten::"):
                continue
            cuda_time_us = evt.self_device_time_total
            if cuda_time_us <= 0:
                continue

            duration_ns = int(cuda_time_us * 1000)
            wall_ns = int(evt.time_range.start * 1000)
            logical_clock_state["tag"] += 1
            tag = logical_clock_state["tag"]
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

            ext = {
                "kernel_name": evt.key,
                "decode_step": decode_step,
                "duration_ns": duration_ns,
                "wall_ns": wall_ns,
                "tensor_bytes": tensor_bytes,
                "cpu_time_ns": int(evt.self_cpu_time_total * 1000),
                "flops": getattr(evt, "flops", 0) or 0,
                "input_shapes": getattr(evt, "input_shapes", []) or [],
            }
            try:
                ext["device_memory_bytes"] = evt.self_device_memory_usage
            except Exception:
                ext["device_memory_bytes"] = 0
            all_extended.append(ext)

    return all_events, all_extended, token_ids


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Decode steps: {DECODE_STEPS}")
    print(f"Seed: {SEED}")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    # ── Config A (G* baseline) ─────────────────────────────────────────
    print("\n=== Config A (baseline) ===")
    _configure_deterministic_mode()
    a_events, a_ext, a_tokens = run_extended_pass(model, tokenizer, device)
    print(f"  Events: {len(a_events)}")
    print(f"  Sample kernel durations (first 5):")
    for e in a_ext[:5]:
        print(f"    {e['kernel_name'][:60]:60s}  {e['duration_ns']:>8d} ns  flops={e['flops']}")

    # ── Config B-2streams (heavy contention) ───────────────────────────
    print("\n=== Config B-2streams (heavy contention) ===")
    _configure_deterministic_mode()
    contention_fn = make_compute_contention_fn(device, 2, 300, 3000000)
    b_events, b_ext, b_tokens = run_extended_pass(
        model, tokenizer, device, pre_step_fn=contention_fn
    )
    print(f"  Events: {len(b_events)}")
    print(f"  Tokens match A: {b_tokens == a_tokens}")

    # ── Timing DDI ─────────────────────────────────────────────────────
    print("\n=== Timing DDI (Config B vs Config A) ===")
    matched = align_events(b_events, a_events)
    print(f"  Matched pairs: {len(matched)}")

    overall_timing = compute_timing_ddi(matched)
    overall_order = compute_order_ddi(matched)
    print(f"  Overall Timing DDI: {overall_timing:.6f}")
    print(f"  Overall Order DDI:  {overall_order:.6f}")

    # ── 6-way breakdown ────────────────────────────────────────────────
    report = full_breakdown_report(matched)

    for metric_name in ("timing", "order"):
        for axis_name in ("decode_step", "kernel_type", "tensor_size"):
            values = report[metric_name][axis_name]
            print(f"\n  {metric_name.upper()} DDI by {axis_name}:")
            if axis_name == "decode_step":
                sorted_items = sorted(values.items(), key=lambda x: int(x[0]))
            else:
                sorted_items = sorted(values.items(), key=lambda x: -x[1])
            for key, val in sorted_items:
                label = "prefill" if axis_name == "decode_step" and key == "0" else key
                print(f"    {str(label):>60s}: {val:.6f}")

    # ── Top 10 highest-jitter kernels ──────────────────────────────────
    print("\n=== Top 10 kernels by Timing DDI ===")
    kernel_timing = report["timing"]["kernel_type"]
    top_kernels = sorted(kernel_timing.items(), key=lambda x: -x[1])[:10]
    for name, ddi in top_kernels:
        count = sum(1 for e in a_events if e.kernel_name == name)
        print(f"  DDI={ddi:.4f}  count={count:>5d}  {name[:80]}")

    # ── ncu / nsys availability ────────────────────────────────────────
    print("\n=== Profiling tools check ===")
    for tool in ("ncu", "nsys", "nv-nsight-cu-cli"):
        r = subprocess.run(["which", tool], capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  {tool}: {r.stdout.strip()}")
        else:
            print(f"  {tool}: NOT FOUND")

    # ── Save ───────────────────────────────────────────────────────────
    os.makedirs("results", exist_ok=True)

    save_data = {
        "device": torch.cuda.get_device_name(device),
        "decode_steps": DECODE_STEPS,
        "seed": SEED,
        "config_a": {
            "events": events_to_dicts(a_events),
            "extended_metrics": a_ext,
            "token_ids": a_tokens,
            "event_count": len(a_events),
        },
        "config_b_2streams": {
            "events": events_to_dicts(b_events),
            "extended_metrics": b_ext,
            "token_ids": b_tokens,
            "event_count": len(b_events),
            "tokens_match_a": b_tokens == a_tokens,
        },
        "ddi": {
            "overall_timing": overall_timing,
            "overall_order": overall_order,
            "matched_count": len(matched),
        },
        "breakdown": report,
        "top_jitter_kernels": [
            {"kernel_name": n, "timing_ddi": d} for n, d in top_kernels
        ],
    }

    outfile = "results/timing_breakdown.json"
    with open(outfile, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {outfile}")
    print("DONE.")


if __name__ == "__main__":
    main()
