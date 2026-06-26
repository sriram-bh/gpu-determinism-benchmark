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
    decode step using StaticCache (pre-allocated KV memory).
    """
    from transformers import StaticCache

    logical_clock_state = {"tag": 0}
    all_events = []
    token_ids = []

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    batch_size = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    max_cache_len = seq_len + DECODE_STEPS + 2

    # Pre-allocate static cache
    static_cache = StaticCache(
        config=model.config,
        batch_size=batch_size,
        max_cache_len=max_cache_len,
        device=device,
        dtype=torch.float32,
    )

    # Step 0 (prefill): run eagerly with static cache
    cache_position = torch.arange(seq_len, device=device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prefill_prof:
        if pre_step_fn:
            pre_step_fn(0)
        with torch.no_grad():
            outputs = model(
                input_ids,
                past_key_values=static_cache,
                cache_position=cache_position,
                use_cache=True,
            )
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            token_ids.append(next_token.item())
        torch.cuda.synchronize()

    _extract_events(prefill_prof, 0, logical_clock_state, all_events)

    # Decode steps: capture one graph (all decode steps have the same
    # tensor shapes with StaticCache — only cache_position changes)
    cur_pos = seq_len

    # Static input buffers for graph capture
    static_token = next_token.clone()
    static_pos = torch.tensor([cur_pos], device=device)

    # Warmup run (required before CUDA graph capture)
    with torch.no_grad():
        _ = model(
            static_token,
            past_key_values=static_cache,
            cache_position=static_pos,
            use_cache=True,
        )
    torch.cuda.synchronize()

    # Capture the decode-step graph (one graph reused for all steps)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_outputs = model(
                static_token,
                past_key_values=static_cache,
                cache_position=static_pos,
                use_cache=True,
            )

    # Replay for each decode step
    for decode_step in range(1, DECODE_STEPS + 1):
        cur_pos = seq_len + decode_step - 1
        static_token.copy_(next_token)
        static_pos.fill_(cur_pos)

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as step_prof:
            if pre_step_fn:
                pre_step_fn(decode_step)
            graph.replay()
            torch.cuda.synchronize()

        _extract_events(step_prof, decode_step, logical_clock_state, all_events)

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
