"""
Mitigation experiment: CUDA Graphs with StaticCache.

Uses HuggingFace's StaticCache (pre-allocated KV memory) which is
designed for CUDA Graph compatibility. Captures one graph for the
decode step and replays it for all steps.

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
from src.metrics.ddi import align_events, compute_timing_ddi, breakdown_by_axis


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


def extract_events(prof, decode_step, logical_clock_state, all_events):
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
                kernel_name=evt.key, wall_ns=wall_ns, logical_tag=tag,
                duration_ns=duration_ns, decode_step=decode_step,
                tensor_bytes=tensor_bytes, stream_id=0,
            )
        )


def run_graph_pass(model, tokenizer, device, pre_step_fn=None):
    """
    Run inference using StaticCache + CUDA Graph for decode steps.
    """
    from transformers import StaticCache

    logical_clock_state = {"tag": 0}
    all_events = []
    token_ids = []

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]
    max_cache_len = seq_len + DECODE_STEPS + 2

    # Check StaticCache API
    print(f"    StaticCache class: {StaticCache}")
    import inspect
    sig = inspect.signature(StaticCache.__init__)
    print(f"    StaticCache.__init__ params: {list(sig.parameters.keys())}")

    # Create StaticCache — try both API variants
    try:
        static_cache = StaticCache(
            config=model.config,
            batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.float32,
        )
    except TypeError:
        static_cache = StaticCache(
            config=model.config,
            max_batch_size=1,
            max_cache_len=max_cache_len,
            device=device,
            dtype=torch.float32,
        )
    print(f"    StaticCache created: type={type(static_cache)}")

    # Prefill (eager, no graph — different input shape)
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

    extract_events(prefill_prof, 0, logical_clock_state, all_events)
    print(f"    Prefill done. cache type after: {type(static_cache)}")
    print(f"    outputs.past_key_values type: {type(outputs.past_key_values)}")

    # Decode: capture CUDA graph
    # Static buffers — these must not be reallocated between capture and replay
    static_token = next_token.clone()
    static_pos = torch.tensor([seq_len], device=device, dtype=torch.long)

    # Warmup run (pre-allocate internal buffers)
    print(f"    Decode warmup...")
    with torch.no_grad():
        warmup_out = model(
            static_token,
            past_key_values=static_cache,
            cache_position=static_pos,
            use_cache=True,
        )
    torch.cuda.synchronize()
    print(f"    Warmup done. cache type: {type(static_cache)}")
    print(f"    warmup past_kv type: {type(warmup_out.past_key_values)}")

    # Reset cache position for capture (rewrite the slot we just used)
    static_pos.fill_(seq_len)

    # Reset the cache slot we wrote during warmup by re-running prefill
    static_cache_2 = type(static_cache)(
        config=model.config,
        batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=torch.float32,
    )
    cache_position_2 = torch.arange(seq_len, device=device)
    with torch.no_grad():
        outputs_2 = model(
            input_ids,
            past_key_values=static_cache_2,
            cache_position=cache_position_2,
            use_cache=True,
        )
        next_token = torch.argmax(outputs_2.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()
    static_cache = static_cache_2
    static_token.copy_(next_token)
    static_pos.fill_(seq_len)

    # Capture the graph
    print(f"    Capturing CUDA graph...")
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with torch.no_grad():
            static_out = model(
                static_token,
                past_key_values=static_cache,
                cache_position=static_pos,
                use_cache=True,
            )
    print(f"    Graph captured successfully!")

    # Replay for each decode step
    # Reset cache again for clean measurement
    static_cache_3 = type(static_cache)(
        config=model.config,
        batch_size=1,
        max_cache_len=max_cache_len,
        device=device,
        dtype=torch.float32,
    )
    cache_position_3 = torch.arange(seq_len, device=device)
    with torch.no_grad():
        out_3 = model(
            input_ids,
            past_key_values=static_cache_3,
            cache_position=cache_position_3,
            use_cache=True,
        )
        next_token = torch.argmax(out_3.logits[:, -1, :], dim=-1, keepdim=True)
    torch.cuda.synchronize()

    token_ids = [next_token.item()]

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

        extract_events(step_prof, decode_step, logical_clock_state, all_events)
        next_token = torch.argmax(
            static_out.logits[:, -1, :], dim=-1, keepdim=True
        )
        token_ids.append(next_token.item())

    return all_events, token_ids


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
    eager_matched = align_events(eager_cont, eager_base)
    eager_ddi = compute_timing_ddi(eager_matched)
    eager_kernel_ddi = breakdown_by_axis(eager_matched, "kernel_type", compute_timing_ddi)
    print(f"  Timing DDI: {eager_ddi:.6f}")

    # --- Condition 3: Graph baseline ---
    print("\n=== Condition 3: CUDA GRAPH + STATIC CACHE BASELINE ===")
    _configure_deterministic_mode()
    graph_available = False
    try:
        graph_base, graph_base_tok = run_graph_pass(model, tokenizer, device)
        print_summary(graph_base, "Graph baseline")
        print(f"  Tokens match eager: {graph_base_tok == eager_base_tok}")
        graph_available = True
    except Exception as e:
        import traceback
        print(f"  CUDA Graph + StaticCache failed: {e}")
        traceback.print_exc()

    if graph_available:
        # --- Condition 4: Graph + contention ---
        print("\n=== Condition 4: CUDA GRAPH + CONTENTION ===")
        _configure_deterministic_mode()
        graph_cont, graph_cont_tok = run_graph_pass(
            model, tokenizer, device, pre_step_fn=contention_fn
        )
        print_summary(graph_cont, "Graph contended")
        graph_matched = align_events(graph_cont, graph_base)
        graph_ddi = compute_timing_ddi(graph_matched)
        graph_kernel_ddi = breakdown_by_axis(graph_matched, "kernel_type", compute_timing_ddi)
        print(f"  Timing DDI: {graph_ddi:.6f}")

        print("\n" + "=" * 60)
        print("COMPARISON: Eager vs CUDA Graph under contention")
        print("=" * 60)
        print(f"  Timing DDI:  Eager={eager_ddi:.4f}  Graph={graph_ddi:.4f}")
        reduction = 1 - graph_ddi / eager_ddi if eager_ddi > 0 else 0
        print(f"  Change: {reduction:+.0%} ({'BETTER' if reduction > 0 else 'WORSE'})")

        print(f"\n  Top 5 jitter kernels (EAGER):")
        for n, d in sorted(eager_kernel_ddi.items(), key=lambda x: -x[1])[:5]:
            print(f"    DDI={d:.4f}  {n[:70]}")
        print(f"\n  Top 5 jitter kernels (GRAPH):")
        for n, d in sorted(graph_kernel_ddi.items(), key=lambda x: -x[1])[:5]:
            print(f"    DDI={d:.4f}  {n[:70]}")

        results = {
            "eager_ddi": eager_ddi, "graph_ddi": graph_ddi,
            "eager_events": len(eager_base), "graph_events": len(graph_base),
            "tokens_match": graph_base_tok == eager_base_tok,
            "eager_kernel_ddi": eager_kernel_ddi,
            "graph_kernel_ddi": graph_kernel_ddi,
        }

    os.makedirs("results", exist_ok=True)
    with open("results/cuda_graphs_mitigation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/cuda_graphs_mitigation.json")
    print("DONE.")


if __name__ == "__main__":
    main()
