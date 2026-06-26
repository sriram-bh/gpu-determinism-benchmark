"""
Cache pollution experiment: test whether L2 cache state affects kernel
timing by comparing warm-cache vs cold-cache execution with GPU clocks
locked (DVFS eliminated).

Two conditions:
  1. Warm cache: normal execution (no pollution)
  2. Cold cache: flush L2 with a 64 MiB buffer before every step

Prerequisites:
  - Run `sudo jetson_clocks` before starting (pins GPU clocks, disables DVFS)
  - Run inside the Docker container on Thor

Run from repo root:
    python3 src/experiments/run_cache_pollution_test.py
"""

import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from src.experiments.gpu_harness import (
    _configure_deterministic_mode,
    _load_model,
    _run_prefill_and_decode,
    run_warmup,
    DECODE_STEPS,
    SEED,
)
from src.metrics.schema import events_to_dicts
from src.metrics.ddi import align_events, compute_timing_ddi, breakdown_by_axis


L2_CACHE_SIZE_BYTES = 32 * 1024 * 1024  # 32 MiB, queried from cudaDeviceProp
POLLUTION_ELEMENTS = 2**24  # 16,777,216 float32 = exactly 64 MiB, 2x L2


def make_cache_polluter(device):
    pollution_buf = torch.randn(POLLUTION_ELEMENTS, device=device)
    output_buf = torch.empty_like(pollution_buf)
    pollution_bytes = POLLUTION_ELEMENTS * 4

    print(f"  Pollution buffer: {POLLUTION_ELEMENTS} float32 = {pollution_bytes / 1024 / 1024:.1f} MiB")
    print(f"  L2 cache size: {L2_CACHE_SIZE_BYTES / 1024 / 1024:.1f} MiB")
    print(f"  Buffer / L2 ratio: {pollution_bytes / L2_CACHE_SIZE_BYTES:.1f}x")

    durations = []

    def polluter(decode_step):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output_buf.copy_(pollution_buf)
        end.record()
        torch.cuda.synchronize()
        durations.append(start.elapsed_time(end))

    return polluter, durations


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    print(f"Device: {props.name}")
    print(f"L2 Cache: {props.L2_cache_size} bytes = {props.L2_cache_size / 1024 / 1024:.0f} MiB")
    print(f"SMs: {props.multi_processor_count}")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    # --- Condition 1: Warm cache (no pollution) ---
    print("\n=== Condition 1: WARM CACHE ===")
    _configure_deterministic_mode()
    warm_events, warm_tokens = _run_prefill_and_decode(model, tokenizer, device)
    print(f"  Events: {len(warm_events)}")

    # --- Condition 2: Cold cache (pollute before every step) ---
    print("\n=== Condition 2: COLD CACHE (64 MiB L2 flush before every step) ===")
    polluter, pollution_durations = make_cache_polluter(device)
    _configure_deterministic_mode()
    cold_events, cold_tokens = _run_prefill_and_decode(
        model, tokenizer, device, pre_step_fn=polluter
    )
    print(f"  Events: {len(cold_events)}")
    print(f"  Tokens match warm: {cold_tokens == warm_tokens}")

    # Verify pollution actually did work
    if pollution_durations:
        avg_poll = sum(pollution_durations) / len(pollution_durations)
        print(f"  Pollution copy durations: avg={avg_poll:.3f}ms, "
              f"min={min(pollution_durations):.3f}ms, max={max(pollution_durations):.3f}ms")
        if avg_poll < 0.01:
            print("  WARNING: Pollution durations suspiciously low — may not be flushing L2")

    # --- Compare: Timing DDI (cold vs warm) ---
    print("\n=== Timing DDI: Cold Cache vs Warm Cache ===")
    matched = align_events(cold_events, warm_events)
    print(f"  Matched pairs: {len(matched)}")

    overall_timing = compute_timing_ddi(matched)
    print(f"  Overall Timing DDI: {overall_timing:.6f}")

    # By kernel type — the key test: memory-bound should be affected MORE
    kernel_ddi = breakdown_by_axis(matched, "kernel_type", compute_timing_ddi)
    print(f"\n  Timing DDI by kernel type (sorted by impact):")
    for name, ddi in sorted(kernel_ddi.items(), key=lambda x: -x[1]):
        print(f"    {ddi:.4f}  {name[:80]}")

    # By decode step
    step_ddi = breakdown_by_axis(matched, "decode_step", compute_timing_ddi)
    print(f"\n  Timing DDI by decode step:")
    for step in sorted(step_ddi.keys(), key=int):
        label = "prefill" if step == "0" else f"step {step}"
        print(f"    {label:>12s}: {step_ddi[step]:.6f}")

    # --- Per-step total duration comparison ---
    print("\n=== Per-step total CUDA time (µs) ===")
    print(f"{'Step':>6s} {'Warm (µs)':>12s} {'Cold (µs)':>12s} {'Ratio':>8s}")
    print("-" * 42)

    warm_step_time = {}
    cold_step_time = {}
    for e in warm_events:
        warm_step_time[e.decode_step] = warm_step_time.get(e.decode_step, 0) + e.duration_ns
    for e in cold_events:
        cold_step_time[e.decode_step] = cold_step_time.get(e.decode_step, 0) + e.duration_ns

    for step in sorted(warm_step_time.keys()):
        w = warm_step_time[step] / 1000
        c = cold_step_time[step] / 1000
        ratio = c / w if w > 0 else 0
        print(f"{step:>6d} {w:>12.1f} {c:>12.1f} {ratio:>7.2f}x")

    # --- Save ---
    os.makedirs("results", exist_ok=True)
    save_data = {
        "l2_cache_size_bytes": L2_CACHE_SIZE_BYTES,
        "pollution_buffer_bytes": POLLUTION_ELEMENTS * 4,
        "clocks_locked": True,
        "overall_timing_ddi": overall_timing,
        "tokens_match": cold_tokens == warm_tokens,
        "kernel_type_ddi": kernel_ddi,
        "step_ddi": step_ddi,
        "pollution_durations_ms": pollution_durations,
        "warm_events": events_to_dicts(warm_events),
        "cold_events": events_to_dicts(cold_events),
    }
    with open("results/cache_pollution_test.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to results/cache_pollution_test.json")
    print("DONE.")


if __name__ == "__main__":
    main()
