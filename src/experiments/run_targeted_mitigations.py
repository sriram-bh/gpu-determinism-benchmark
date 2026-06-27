"""
Targeted mitigation experiments based on prior findings.

1. Cap contention process SM usage (20/40/60%) while inference is unrestricted
2. L2 warmth under MPS contention (touch weights before each step)
3. Stream priority + L2 warmth combined
4. Stream priority + contention SM cap combined

All experiments use MPS cross-process contention with clocks locked.
Baseline: MPS cross-process, no mitigation, DDI ~0.513.

Run from repo root (MPS daemon must be running):
    python3 src/experiments/run_targeted_mitigations.py
"""

import json
import multiprocessing
import sys
import os
import time
from collections import Counter

multiprocessing.set_start_method("spawn", force=True)

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
    MODEL_NAME,
)
from src.metrics.schema import events_to_dicts
from src.metrics.ddi import align_events, compute_timing_ddi, breakdown_by_axis


# ── Contention process variants ────────────────────────────────────

def run_contention_default(duration_seconds):
    """Contention process: no SM cap, default priority."""
    import torch
    device = torch.device("cuda")
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


def run_contention_capped(duration_seconds, sm_pct):
    """Contention process: SM usage capped via MPS."""
    import os
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(sm_pct)
    import torch
    device = torch.device("cuda")
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


def run_contention_capped_20(duration_seconds):
    run_contention_capped(duration_seconds, 20)

def run_contention_capped_40(duration_seconds):
    run_contention_capped(duration_seconds, 40)

def run_contention_capped_60(duration_seconds):
    run_contention_capped(duration_seconds, 60)


def run_contention_low_prio(duration_seconds):
    """Contention process at LOWEST stream priority (0 = least priority on Thor)."""
    import torch
    device = torch.device("cuda")
    # priority=0 is LOWEST on Thor (range: -3=highest to 0=lowest)
    stream = torch.cuda.Stream(device, priority=0)
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    with torch.cuda.stream(stream):
        while time.time() < end_time:
            torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


def run_contention_prio_capped_20(duration_seconds):
    """Contention at LOWEST priority + 20% SM cap."""
    import os
    os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = "20"
    import torch
    device = torch.device("cuda")
    # priority=0 is LOWEST on Thor
    stream = torch.cuda.Stream(device, priority=0)
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    with torch.cuda.stream(stream):
        while time.time() < end_time:
            torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


# ── L2 warmth hook ────────────────────────────────────────────────

def make_l2_warmth_hook(model, device):
    """Pre-step hook that touches model weights to warm L2."""
    param_list = [p.data for p in model.parameters()]

    def hook(decode_step):
        for p in param_list:
            _ = p.sum()
        torch.cuda.synchronize()

    return hook


# ── Experiment runner ──────────────────────────────────────────────

def run_with_cross_process(label, model, tokenizer, device, base_events,
                           contention_fn=run_contention_default,
                           pre_step_fn=None, duration=120):
    """Run inference with cross-process contention and compute DDI."""
    print(f"\n=== {label} ===")

    proc = multiprocessing.Process(target=contention_fn, args=(duration,))
    proc.start()
    time.sleep(3)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    events, tokens = _run_prefill_and_decode(
        model, tokenizer, device, pre_step_fn=pre_step_fn
    )

    proc.terminate()
    proc.join(timeout=5)

    matched = align_events(events, base_events)
    ddi = compute_timing_ddi(matched)
    kernel_ddi = breakdown_by_axis(matched, "kernel_type", compute_timing_ddi)

    print(f"  Events: {len(events)}, Matched: {len(matched)}")
    print(f"  Timing DDI: {ddi:.6f}")

    return {
        "label": label,
        "timing_ddi": ddi,
        "matched": len(matched),
        "kernel_ddi": kernel_ddi,
    }


def print_table(results):
    """Print side-by-side per-kernel DDI table."""
    all_names = set()
    for r in results:
        all_names.update(r["kernel_ddi"].keys())
    kernel_names = sorted(all_names)

    labels = [r["label"][:14] for r in results]
    header = f"{'Kernel':<50s}" + "".join(f"{l:>15s}" for l in labels)
    print(header)
    print("-" * len(header))

    for name in kernel_names:
        short = name[:48]
        vals = ""
        for r in results:
            v = r["kernel_ddi"].get(name, 0)
            vals += f"{v:>15.4f}"
        print(f"{short:<50s}{vals}")

    overall = ""
    for r in results:
        overall += f"{r['timing_ddi']:>15.4f}"
    print(f"{'OVERALL':<50s}{overall}")


def main():
    if not torch.cuda.is_available():
        print("No CUDA device.")
        sys.exit(1)

    device = torch.device("cuda")
    mps_active = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"MPS active: {bool(mps_active)}")
    print(f"Decode steps: {DECODE_STEPS}")

    if not mps_active:
        print("ERROR: MPS must be active. Run via run_targeted_sweep.sh")
        sys.exit(1)

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    # Baseline (no contention)
    print("\n=== BASELINE (no contention) ===")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base_events, base_tokens = _run_prefill_and_decode(model, tokenizer, device)
    print(f"  Events: {len(base_events)}")

    results = []

    # Reference: MPS cross-process, no mitigation
    r0 = run_with_cross_process(
        "NoMitigation", model, tokenizer, device, base_events,
    )
    results.append(r0)

    # ── Experiment 1: Cap contention process SM usage ──────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: CAP CONTENTION PROCESS SM USAGE")
    print("=" * 60)

    for pct, fn in [(20, run_contention_capped_20),
                     (40, run_contention_capped_40),
                     (60, run_contention_capped_60)]:
        r = run_with_cross_process(
            f"ContCap_{pct}%", model, tokenizer, device, base_events,
            contention_fn=fn,
        )
        results.append(r)

    # ── Experiment 2: L2 warmth under MPS contention ──────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: L2 WARMTH UNDER CONTENTION")
    print("=" * 60)

    l2_hook = make_l2_warmth_hook(model, device)
    r_l2 = run_with_cross_process(
        "L2Warm", model, tokenizer, device, base_events,
        pre_step_fn=l2_hook,
    )
    results.append(r_l2)

    # ── Experiment 3: Stream priority + L2 warmth ─────────────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: STREAM PRIORITY + L2 WARMTH")
    print("=" * 60)

    r_combo1 = run_with_cross_process(
        "Prio+L2Warm", model, tokenizer, device, base_events,
        contention_fn=run_contention_low_prio,
        pre_step_fn=l2_hook,
    )
    results.append(r_combo1)

    # ── Experiment 4: Stream priority + contention SM cap ─────────
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: STREAM PRIORITY + CONTENTION SM CAP")
    print("=" * 60)

    # Find best contention cap from experiment 1
    cap_results = [r for r in results if r["label"].startswith("ContCap")]
    if cap_results:
        best_cap = min(cap_results, key=lambda r: r["timing_ddi"])
        best_pct = int(best_cap["label"].split("_")[1].rstrip("%"))
        print(f"  Best contention cap: {best_pct}% (DDI={best_cap['timing_ddi']:.4f})")

        r_combo2 = run_with_cross_process(
            f"Prio+Cap20%", model, tokenizer, device, base_events,
            contention_fn=run_contention_prio_capped_20,
        )
        results.append(r_combo2)

    # ── Summary ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FULL RESULTS SUMMARY")
    print("=" * 60)

    print(f"\n{'Condition':<20s} {'DDI':>8s} {'vs Baseline':>12s}")
    print("-" * 42)
    baseline_ddi = results[0]["timing_ddi"]
    for r in results:
        change = (1 - r["timing_ddi"] / baseline_ddi) * 100 if baseline_ddi > 0 else 0
        direction = "better" if change > 0 else "worse"
        print(f"{r['label']:<20s} {r['timing_ddi']:>8.4f} {change:>+11.0f}% {direction}")

    print("\n--- Per-kernel breakdown ---")
    print_table(results)

    # Save
    os.makedirs("results", exist_ok=True)
    save_data = {r["label"]: r for r in results}
    with open("results/targeted_mitigations.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to results/targeted_mitigations.json")
    print("DONE.")


if __name__ == "__main__":
    main()
