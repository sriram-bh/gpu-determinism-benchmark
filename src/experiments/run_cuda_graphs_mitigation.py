"""
Mitigation experiment: CUDA Graphs.

Captures each decode step as a separate CUDA graph (since KV cache
shape changes per step), then replays them. Compares graph-replay
execution against eager execution under contention.

Four conditions:
  1. Eager baseline (Config A)
  2. Eager + contention (Config B-2streams)
  3. Graph-replay baseline (Config A)
  4. Graph-replay + contention (Config B-2streams)

Run from repo root:
    python3 src/experiments/run_cuda_graphs_mitigation.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from torch.profiler import profile, ProfilerActivity
from src.experiments.gpu_harness import (
    _configure_deterministic_mode,
    _load_model,
    _tensor_bytes_for_event,
    _run_prefill_and_decode,
    cupti_lf_bridge,
    run_warmup,
    DECODE_STEPS,
    SEED,
    FIXED_INPUT_SENTENCE,
)
from src.metrics.schema import KernelEvent, events_to_dicts
from src.metrics.ddi import (
    align_events,
    compute_timing_ddi,
    breakdown_by_axis,
)


def make_contention_fn(device):
    streams = [torch.cuda.Stream() for _ in range(2)]
    data = torch.randn(3000000, device=device)
    outs = [torch.empty_like(data) for _ in range(2)]

    def fn(decode_step):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                for _ in range(300):
                    torch.cumsum(data, dim=0, out=outs[i])

    return fn


def run_graph_pass(model, tokenizer, device, pre_step_fn=None):
    """
    Run prefill eagerly, then capture and replay a CUDA graph for each
    decode step. Each step gets its own graph since KV cache shape changes.
    """
    logical_clock_state = {"tag": 0}
    all_events = []
    token_ids = []

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]

    # Step 0 (prefill): run eagerly — different shape from decode steps
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        token_ids.append(next_token.item())
    torch.cuda.synchronize()

    # Profile the prefill step separately (eager, no graph)
    # Re-run it under profiler for measurement
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prefill_prof:
        if pre_step_fn:
            pre_step_fn(0)
        with torch.no_grad():
            outputs = model(input_ids, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
        torch.cuda.synchronize()

    _extract_events(prefill_prof, 0, logical_clock_state, all_events)

    # Decode steps 1..DECODE_STEPS: capture and replay CUDA graphs
    for decode_step in range(1, DECODE_STEPS + 1):
        # Capture phase: record the computation into a graph
        # First, do a warmup run for this step (required before capture)
        with torch.no_grad():
            outputs = model(
                next_token, past_key_values=past_key_values, use_cache=True
            )
        torch.cuda.synchronize()

        # Prepare static inputs for graph capture
        static_next_token = next_token.clone()

        # Handle past_key_values — may be DynamicCache or tuple, may contain Nones
        def clone_past(pkv):
            if hasattr(pkv, 'key_cache'):
                # DynamicCache object — clone the internal lists
                from copy import deepcopy
                return deepcopy(pkv)
            return tuple(
                tuple(t.clone() if t is not None else None for t in layer)
                for layer in pkv
            )

        def copy_past(dst, src):
            if hasattr(dst, 'key_cache'):
                for i in range(len(dst.key_cache)):
                    dst.key_cache[i].copy_(src.key_cache[i])
                    dst.value_cache[i].copy_(src.value_cache[i])
            else:
                for dst_layer, src_layer in zip(dst, src):
                    for d, s in zip(dst_layer, src_layer):
                        if d is not None and s is not None:
                            d.copy_(s)

        static_past = clone_past(past_key_values)

        # Capture the graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                static_outputs = model(
                    static_next_token,
                    past_key_values=static_past,
                    use_cache=True,
                )

        # Replay phase: profile the graph replay under optional contention
        # Copy real data into static buffers
        static_next_token.copy_(next_token)
        copy_past(static_past, past_key_values)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as step_prof:
            if pre_step_fn:
                pre_step_fn(decode_step)
            graph.replay()
            torch.cuda.synchronize()

        _extract_events(step_prof, decode_step, logical_clock_state, all_events)

        # Extract results from static output buffers
        past_key_values = static_outputs.past_key_values
        next_token = torch.argmax(
            static_outputs.logits[:, -1, :], dim=-1, keepdim=True
        )
        token_ids.append(next_token.item())

    return all_events, token_ids


def _extract_events(prof, decode_step, logical_clock_state, all_events):
    step_events = sorted(prof.events(), key=lambda e: e.time_range.start)
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


def print_summary(events, label):
    names = Counter(e.kernel_name for e in events)
    print(f"\n  [{label}]")
    print(f"  Total events: {len(events)}")
    print(f"  Distinct kernel names: {len(names)}")
    print(f"  Events per step (avg): {len(events)/(DECODE_STEPS+1):.0f}")
    print(f"  Top 10 kernels:")
    for name, count in names.most_common(10):
        print(f"    {count:>5d}x  {name[:75]}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Decode steps: {DECODE_STEPS}")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    contention_fn = make_contention_fn(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    results = {}

    # --- Condition 1: Eager baseline ---
    print("\n=== Condition 1: EAGER BASELINE ===")
    _configure_deterministic_mode()
    eager_base, eager_base_tok = _run_prefill_and_decode(model, tokenizer, device)
    print_summary(eager_base, "Eager baseline")

    # --- Condition 2: Eager + contention ---
    print("\n=== Condition 2: EAGER + CONTENTION ===")
    _configure_deterministic_mode()
    eager_cont, eager_cont_tok = _run_prefill_and_decode(
        model, tokenizer, device, pre_step_fn=contention_fn
    )
    print_summary(eager_cont, "Eager contended")
    print(f"  Tokens match: {eager_cont_tok == eager_base_tok}")

    eager_matched = align_events(eager_cont, eager_base)
    eager_ddi = compute_timing_ddi(eager_matched)
    eager_kernel_ddi = breakdown_by_axis(eager_matched, "kernel_type", compute_timing_ddi)
    print(f"  Eager Timing DDI: {eager_ddi:.6f}")

    # --- Condition 3: Graph baseline ---
    print("\n=== Condition 3: CUDA GRAPH BASELINE ===")
    _configure_deterministic_mode()
    try:
        graph_base, graph_base_tok = run_graph_pass(model, tokenizer, device)
        print_summary(graph_base, "Graph baseline")
        print(f"  Tokens match eager: {graph_base_tok == eager_base_tok}")
        graph_available = True
    except Exception as e:
        import traceback
        print(f"  CUDA Graph capture failed: {e}")
        traceback.print_exc()

        # Debug: inspect past_key_values structure
        print("\n  DEBUG: Inspecting past_key_values structure...")
        _configure_deterministic_mode()
        with torch.no_grad():
            dbg_out = model(
                tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)["input_ids"],
                use_cache=True,
            )
        pkv = dbg_out.past_key_values
        print(f"  Type: {type(pkv)}")
        print(f"  Has key_cache: {hasattr(pkv, 'key_cache')}")
        if hasattr(pkv, 'key_cache'):
            print(f"  key_cache length: {len(pkv.key_cache)}")
            print(f"  key_cache[0] type: {type(pkv.key_cache[0])}, shape: {pkv.key_cache[0].shape if pkv.key_cache[0] is not None else None}")
        elif hasattr(pkv, '__len__'):
            print(f"  Length: {len(pkv)}")
            if len(pkv) > 0:
                layer0 = pkv[0]
                print(f"  Layer 0 type: {type(layer0)}, len: {len(layer0) if hasattr(layer0, '__len__') else 'N/A'}")
                if hasattr(layer0, '__len__'):
                    for j, t in enumerate(layer0):
                        print(f"    Element {j}: type={type(t)}, is_none={t is None}, shape={t.shape if hasattr(t, 'shape') else 'N/A'}")

        graph_available = False

    if graph_available:
        # --- Condition 4: Graph + contention ---
        print("\n=== Condition 4: CUDA GRAPH + CONTENTION ===")
        _configure_deterministic_mode()
        graph_cont, graph_cont_tok = run_graph_pass(
            model, tokenizer, device, pre_step_fn=contention_fn
        )
        print_summary(graph_cont, "Graph contended")
        print(f"  Tokens match graph baseline: {graph_cont_tok == graph_base_tok}")

        graph_matched = align_events(graph_cont, graph_base)
        graph_ddi = compute_timing_ddi(graph_matched)
        graph_kernel_ddi = breakdown_by_axis(graph_matched, "kernel_type", compute_timing_ddi)
        print(f"  Graph Timing DDI: {graph_ddi:.6f}")

        # --- Comparison ---
        print("\n" + "=" * 60)
        print("COMPARISON: Eager vs CUDA Graph under contention")
        print("=" * 60)
        print(f"  Kernels/step:  Eager={len(eager_base)/41:.0f}  Graph={len(graph_base)/41:.0f}")
        print(f"  Timing DDI:    Eager={eager_ddi:.4f}  Graph={graph_ddi:.4f}")
        if eager_ddi > 0:
            reduction = 1 - graph_ddi / eager_ddi
            direction = "BETTER" if reduction > 0 else "WORSE"
            print(f"  DDI change:    {reduction:+.0%} ({direction})")

        print(f"\n  Top 5 jitter kernels (EAGER):")
        for name, ddi in sorted(eager_kernel_ddi.items(), key=lambda x: -x[1])[:5]:
            print(f"    DDI={ddi:.4f}  {name[:70]}")
        print(f"\n  Top 5 jitter kernels (GRAPH):")
        for name, ddi in sorted(graph_kernel_ddi.items(), key=lambda x: -x[1])[:5]:
            print(f"    DDI={ddi:.4f}  {name[:70]}")

        results = {
            "eager_baseline_events": len(eager_base),
            "graph_baseline_events": len(graph_base),
            "eager_timing_ddi": eager_ddi,
            "graph_timing_ddi": graph_ddi,
            "eager_tokens": eager_base_tok,
            "graph_tokens": graph_base_tok,
            "tokens_match": graph_base_tok == eager_base_tok,
            "eager_kernel_ddi": eager_kernel_ddi,
            "graph_kernel_ddi": graph_kernel_ddi,
            "eager_events": events_to_dicts(eager_base),
            "graph_events": events_to_dicts(graph_base),
        }

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/cuda_graphs_mitigation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/cuda_graphs_mitigation.json")
    print("DONE.")


if __name__ == "__main__":
    main()
