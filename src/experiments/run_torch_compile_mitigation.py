"""
Mitigation experiment: torch.compile kernel fusion.

Tests whether compiling the model with torch.compile (inductor backend)
reduces timing jitter by fusing small memory-bound kernels into larger
combined kernels. Compares:
  1. Uncompiled baseline (Config A)
  2. Compiled baseline (Config A + torch.compile)
  3. Uncompiled under contention (Config B-2streams)
  4. Compiled under contention (Config B-2streams + torch.compile)

The key metrics: does compilation reduce kernel count, and does the
reduced kernel count translate into lower Timing DDI under contention?

Run from repo root:
    python3 src/experiments/run_torch_compile_mitigation.py
"""

import json
import sys
import os
from collections import Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from src.experiments.gpu_harness import (
    _configure_deterministic_mode,
    _load_model,
    _run_prefill_and_decode,
    run_warmup,
    DECODE_STEPS,
    SEED,
    FIXED_INPUT_SENTENCE,
)
from src.metrics.schema import events_to_dicts
from src.metrics.ddi import (
    align_events,
    compute_timing_ddi,
    compute_order_ddi,
    breakdown_by_axis,
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


def print_kernel_summary(events, label):
    names = Counter(e.kernel_name for e in events)
    step_counts = Counter(e.decode_step for e in events)
    events_per_step = len(events) / (DECODE_STEPS + 1) if events else 0

    print(f"\n  [{label}]")
    print(f"  Total events: {len(events)}")
    print(f"  Distinct kernel names: {len(names)}")
    print(f"  Events per step (avg): {events_per_step:.0f}")
    print(f"  Top 10 kernels:")
    for name, count in names.most_common(10):
        print(f"    {count:>5d}x  {name[:75]}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"Decode steps: {DECODE_STEPS}")

    _configure_deterministic_mode()
    model_eager, tokenizer = _load_model(device)

    # Compile the model
    print("\nCompiling model with torch.compile (inductor backend)...")
    try:
        model_compiled = torch.compile(model_eager, backend="inductor")
        # Trigger compilation with a warmup pass
        print("  Running compilation warmup (first pass triggers JIT)...")
        inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
        with torch.no_grad():
            _ = model_compiled(inputs["input_ids"], use_cache=True)
        torch.cuda.synchronize()
        print("  Compilation successful.")
        compiled_available = True
    except Exception as e:
        print(f"  torch.compile failed: {e}")
        print("  Continuing with eager-only comparison.")
        compiled_available = False

    contention_fn = make_compute_contention_fn(device, 2, 300, 3000000)

    print("\nRunning 3 warmup passes (eager)...")
    run_warmup(model_eager, tokenizer, device, n=3)

    results = {}

    # --- Condition 1: Eager baseline ---
    print("\n=== Condition 1: EAGER BASELINE (Config A) ===")
    _configure_deterministic_mode()
    eager_base_events, eager_base_tokens = _run_prefill_and_decode(
        model_eager, tokenizer, device
    )
    print_kernel_summary(eager_base_events, "Eager baseline")
    results["eager_baseline"] = {
        "events": events_to_dicts(eager_base_events),
        "token_ids": eager_base_tokens,
        "event_count": len(eager_base_events),
        "distinct_kernels": len(set(e.kernel_name for e in eager_base_events)),
    }

    # --- Condition 2: Eager + contention ---
    print("\n=== Condition 2: EAGER + CONTENTION (Config B-2streams) ===")
    _configure_deterministic_mode()
    eager_cont_events, eager_cont_tokens = _run_prefill_and_decode(
        model_eager, tokenizer, device, pre_step_fn=contention_fn
    )
    print_kernel_summary(eager_cont_events, "Eager contended")
    print(f"  Tokens match baseline: {eager_cont_tokens == eager_base_tokens}")

    # Eager DDI
    eager_matched = align_events(eager_cont_events, eager_base_events)
    eager_timing_ddi = compute_timing_ddi(eager_matched)
    eager_kernel_ddi = breakdown_by_axis(eager_matched, "kernel_type", compute_timing_ddi)
    print(f"\n  Eager Timing DDI: {eager_timing_ddi:.6f}")
    results["eager_contended"] = {
        "event_count": len(eager_cont_events),
        "timing_ddi": eager_timing_ddi,
        "matched": len(eager_matched),
        "tokens_match": eager_cont_tokens == eager_base_tokens,
    }

    if compiled_available:
        # --- Condition 3: Compiled baseline ---
        print("\n=== Condition 3: COMPILED BASELINE (Config A + torch.compile) ===")
        _configure_deterministic_mode()
        comp_base_events, comp_base_tokens = _run_prefill_and_decode(
            model_compiled, tokenizer, device
        )
        print_kernel_summary(comp_base_events, "Compiled baseline")
        print(f"  Tokens match eager baseline: {comp_base_tokens == eager_base_tokens}")
        results["compiled_baseline"] = {
            "events": events_to_dicts(comp_base_events),
            "token_ids": comp_base_tokens,
            "event_count": len(comp_base_events),
            "distinct_kernels": len(set(e.kernel_name for e in comp_base_events)),
            "tokens_match_eager": comp_base_tokens == eager_base_tokens,
        }

        # --- Condition 4: Compiled + contention ---
        print("\n=== Condition 4: COMPILED + CONTENTION (Config B + torch.compile) ===")
        _configure_deterministic_mode()
        comp_cont_events, comp_cont_tokens = _run_prefill_and_decode(
            model_compiled, tokenizer, device, pre_step_fn=contention_fn
        )
        print_kernel_summary(comp_cont_events, "Compiled contended")
        print(f"  Tokens match compiled baseline: {comp_cont_tokens == comp_base_tokens}")

        # Compiled DDI (compiled contended vs compiled baseline)
        comp_matched = align_events(comp_cont_events, comp_base_events)
        comp_timing_ddi = compute_timing_ddi(comp_matched)
        comp_kernel_ddi = breakdown_by_axis(comp_matched, "kernel_type", compute_timing_ddi)
        print(f"\n  Compiled Timing DDI: {comp_timing_ddi:.6f}")
        results["compiled_contended"] = {
            "event_count": len(comp_cont_events),
            "timing_ddi": comp_timing_ddi,
            "matched": len(comp_matched),
            "tokens_match": comp_cont_tokens == comp_base_tokens,
        }

        # --- Comparison ---
        print("\n" + "=" * 60)
        print("COMPARISON: Eager vs Compiled under contention")
        print("=" * 60)
        print(f"  Kernels per step:  Eager={len(eager_base_events)/41:.0f}  "
              f"Compiled={len(comp_base_events)/41:.0f}  "
              f"Reduction={1 - len(comp_base_events)/len(eager_base_events):.0%}")
        print(f"  Timing DDI:        Eager={eager_timing_ddi:.4f}  "
              f"Compiled={comp_timing_ddi:.4f}  "
              f"Reduction={1 - comp_timing_ddi/eager_timing_ddi:.0%}")
        print(f"  Distinct kernels:  Eager={len(set(e.kernel_name for e in eager_base_events))}  "
              f"Compiled={len(set(e.kernel_name for e in comp_base_events))}")

        # Top jitter kernels comparison
        print(f"\n  Top 5 jitter kernels (EAGER):")
        for name, ddi in sorted(eager_kernel_ddi.items(), key=lambda x: -x[1])[:5]:
            print(f"    DDI={ddi:.4f}  {name[:70]}")
        print(f"\n  Top 5 jitter kernels (COMPILED):")
        for name, ddi in sorted(comp_kernel_ddi.items(), key=lambda x: -x[1])[:5]:
            print(f"    DDI={ddi:.4f}  {name[:70]}")

        results["comparison"] = {
            "eager_kernels_per_step": len(eager_base_events) / 41,
            "compiled_kernels_per_step": len(comp_base_events) / 41,
            "kernel_reduction_pct": 1 - len(comp_base_events) / len(eager_base_events),
            "eager_timing_ddi": eager_timing_ddi,
            "compiled_timing_ddi": comp_timing_ddi,
            "ddi_reduction_pct": 1 - comp_timing_ddi / eager_timing_ddi if eager_timing_ddi > 0 else 0,
        }

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/torch_compile_mitigation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/torch_compile_mitigation.json")
    print("DONE.")


if __name__ == "__main__":
    main()
