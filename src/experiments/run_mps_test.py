"""
MPS mitigation test: measure Timing DDI with and without MPS active.

When MPS is running, CUDA processes share the GPU through a single
hardware context, enabling true concurrent execution with SM
partitioning. This test compares:
  1. Eager baseline (no contention, for reference)
  2. Eager + contention (without MPS-level partitioning, but MPS daemon active)

MPS SM partitioning is set via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
environment variable. This script tests with the MPS daemon running
(concurrent kernel execution enabled) vs the baseline.

The MPS daemon must be started BEFORE this script runs (see
run_mps_mitigation.sh).

Run from repo root:
    python3 src/experiments/run_mps_test.py
"""

import json
import os
import sys
import subprocess
import multiprocessing
import time
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
from src.metrics.ddi import align_events, compute_timing_ddi, breakdown_by_axis


multiprocessing.set_start_method("spawn", force=True)


def run_contention_process(duration_seconds):
    """Separate process that creates GPU contention via cumsum ops."""
    import torch
    device = torch.device("cuda")
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


def print_summary(events, label):
    names = Counter(e.kernel_name for e in events)
    print(f"\n  [{label}]")
    print(f"  Total events: {len(events)}")
    print(f"  Distinct kernel names: {len(names)}")
    print(f"  Events per step (avg): {len(events)/(DECODE_STEPS+1):.0f}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Decode steps: {DECODE_STEPS}")

    # Check if MPS is active
    mps_pipe = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    print(f"CUDA_MPS_PIPE_DIRECTORY: {mps_pipe}")
    print(f"MPS active: {bool(mps_pipe)}")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    results = {}

    # --- Condition 1: Baseline (no contention) ---
    print("\n=== Condition 1: BASELINE (no contention) ===")
    _configure_deterministic_mode()
    base_events, base_tokens = _run_prefill_and_decode(model, tokenizer, device)
    print_summary(base_events, "Baseline")

    # --- Condition 2: Contention via pre_step_fn (same process) ---
    print("\n=== Condition 2: CONTENTION (same process, pre_step_fn) ===")

    streams = [torch.cuda.Stream() for _ in range(2)]
    c_data = torch.randn(3000000, device=device)
    c_outs = [torch.empty_like(c_data) for _ in range(2)]

    def contention_fn(decode_step):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                for _ in range(300):
                    torch.cumsum(c_data, dim=0, out=c_outs[i])

    _configure_deterministic_mode()
    cont_events, cont_tokens = _run_prefill_and_decode(
        model, tokenizer, device, pre_step_fn=contention_fn
    )
    print_summary(cont_events, "Contended (same process)")
    print(f"  Tokens match baseline: {cont_tokens == base_tokens}")

    cont_matched = align_events(cont_events, base_events)
    cont_ddi = compute_timing_ddi(cont_matched)
    cont_kernel_ddi = breakdown_by_axis(cont_matched, "kernel_type", compute_timing_ddi)
    print(f"  Timing DDI: {cont_ddi:.6f}")

    results["same_process_contention"] = {
        "timing_ddi": cont_ddi,
        "matched": len(cont_matched),
        "tokens_match": cont_tokens == base_tokens,
        "kernel_ddi": cont_kernel_ddi,
    }

    # --- Condition 3: Cross-process contention (MPS may or may not help) ---
    print("\n=== Condition 3: CONTENTION (separate process) ===")
    print("  Starting contention process...")

    contention_proc = multiprocessing.Process(
        target=run_contention_process, args=(120,)
    )
    contention_proc.start()
    time.sleep(2)  # Let contention process initialize

    _configure_deterministic_mode()
    mps_events, mps_tokens = _run_prefill_and_decode(model, tokenizer, device)
    print_summary(mps_events, "Contended (separate process, MPS)")
    print(f"  Tokens match baseline: {mps_tokens == base_tokens}")

    contention_proc.terminate()
    contention_proc.join(timeout=5)

    mps_matched = align_events(mps_events, base_events)
    mps_ddi = compute_timing_ddi(mps_matched)
    mps_kernel_ddi = breakdown_by_axis(mps_matched, "kernel_type", compute_timing_ddi)
    print(f"  Timing DDI: {mps_ddi:.6f}")

    results["cross_process_contention"] = {
        "timing_ddi": mps_ddi,
        "matched": len(mps_matched),
        "tokens_match": mps_tokens == base_tokens,
        "kernel_ddi": mps_kernel_ddi,
    }

    # --- Side-by-side per-kernel comparison (ALL kernels, same order) ---
    print("\n" + "=" * 60)
    print("PER-KERNEL COMPARISON (all kernels, side by side)")
    print("=" * 60)
    all_kernel_names = sorted(set(list(cont_kernel_ddi.keys()) + list(mps_kernel_ddi.keys())))
    print(f"{'Kernel':<55s} {'SameProc':>9s} {'CrossProc':>10s} {'Change':>8s}")
    print("-" * 85)
    for name in all_kernel_names:
        sp = cont_kernel_ddi.get(name, 0)
        cp = mps_kernel_ddi.get(name, 0)
        if sp > 0:
            change = f"{(1 - cp/sp)*100:+.0f}%"
        else:
            change = "N/A"
        short = name[:53]
        print(f"{short:<55s} {sp:>9.4f} {cp:>10.4f} {change:>8s}")

    print(f"\n{'OVERALL':<55s} {cont_ddi:>9.4f} {mps_ddi:>10.4f} {(1-mps_ddi/cont_ddi)*100:+.0f}%")

    results["baseline_events"] = events_to_dicts(base_events)
    results["baseline_tokens"] = base_tokens

    os.makedirs("results", exist_ok=True)
    with open("results/mps_mitigation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to results/mps_mitigation.json")
    print("DONE.")


if __name__ == "__main__":
    main()
