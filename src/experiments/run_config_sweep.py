"""
Config B/C sweep: run GPT-2 inference under varying levels of compute
and memory contention, compute Timing DDI and Order DDI against G*.

Config B (compute contention): cumsum ops on 1/2/4 competing streams,
    launched inside each step's profiler window.
Config C (memory contention): D2D memory copies on a competing stream,
    varying copy size (50MB / 200MB / 500MB).

Run from repo root:
    python3 src/experiments/run_config_sweep.py
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
    _run_prefill_and_decode,
    run_warmup,
    DECODE_STEPS,
    SEED,
    FIXED_INPUT_SENTENCE,
)
from src.metrics.schema import KernelEvent, dicts_to_events
from src.metrics.ddi import align_events, compute_timing_ddi, compute_order_ddi


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


def make_memory_contention_fn(device, copy_size_mb, copies_per_step):
    stream = torch.cuda.Stream()
    n_elements = (copy_size_mb * 1024 * 1024) // 4
    src = torch.randn(n_elements, device=device)
    dst = torch.empty_like(src)

    def fn(decode_step):
        with torch.cuda.stream(stream):
            for _ in range(copies_per_step):
                dst.copy_(src)

    return fn


def compute_ddi(events, gstar_events, gstar_names):
    filtered = [e for e in events if e.kernel_name in gstar_names]
    matched = align_events(filtered, gstar_events)
    if len(matched) < 2:
        return None, None, len(filtered)
    timing = compute_timing_ddi(matched)
    order = compute_order_ddi(matched)
    return timing, order, len(filtered)


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    gstar_path = "results/gstar.json"
    if not os.path.exists(gstar_path):
        print(f"G* not found at {gstar_path}. Run run_gstar_validation.py first.")
        sys.exit(1)

    with open(gstar_path) as f:
        gstar_data = json.load(f)
    gstar_events = dicts_to_events(gstar_data["gstar_events"])
    gstar_names = set(e.kernel_name for e in gstar_events)
    gstar_tokens = gstar_data["token_ids"]
    print(f"G*: {len(gstar_names)} kernel names, {len(gstar_events)} events")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    configs = [
        ("A (baseline)",      None),
        ("B-1stream-light",   make_compute_contention_fn(device, 1, 100, 1000000)),
        ("B-1stream-heavy",   make_compute_contention_fn(device, 1, 500, 5000000)),
        ("B-2streams",        make_compute_contention_fn(device, 2, 300, 3000000)),
        ("B-4streams",        make_compute_contention_fn(device, 4, 200, 2000000)),
        ("C-50MB",            make_memory_contention_fn(device, 50, 20)),
        ("C-200MB",           make_memory_contention_fn(device, 200, 10)),
        ("C-500MB",           make_memory_contention_fn(device, 500, 5)),
    ]

    print(f"\n{'Config':<22} {'TimingDDI':>10} {'OrderDDI':>10} {'Events':>8} {'Tokens':>8}")
    print("-" * 62)

    results = []
    for name, contention_fn in configs:
        _configure_deterministic_mode()
        events, token_ids = _run_prefill_and_decode(
            model, tokenizer, device, pre_step_fn=contention_fn
        )
        timing, order, n_matched = compute_ddi(events, gstar_events, gstar_names)
        tokens_ok = "match" if token_ids == gstar_tokens else "DIFFER"

        timing_s = f"{timing:.6f}" if timing is not None else "N/A"
        order_s = f"{order:.6f}" if order is not None else "N/A"
        print(f"{name:<22} {timing_s:>10} {order_s:>10} {n_matched:>8} {tokens_ok:>8}")

        results.append({
            "config": name,
            "timing_ddi": timing,
            "order_ddi": order,
            "matched_events": n_matched,
            "tokens_match": token_ids == gstar_tokens,
            "token_ids": token_ids,
        })

    os.makedirs("results", exist_ok=True)
    with open("results/config_sweep.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/config_sweep.json")


if __name__ == "__main__":
    main()
