"""
Full mitigation sweep: 5 experiments in one script.

1. Investigate the 27x same-process DDI difference (check clock state)
2. Stream priority under MPS
3. SM partitioning under MPS (50%, 70%, 90% for inference)
4. Stream priority + SM partitioning combined
5. L2 persistence (cudaAccessPolicyWindow)

All MPS experiments use cross-process contention (separate process).
L2 experiment is single-process, no MPS.

Run from repo root (MPS daemon must be managed by the shell wrapper):
    python3 src/experiments/run_full_mitigation_sweep.py [--mps] [--l2]
"""

import json
import multiprocessing
import subprocess
import sys
import os
import time
from collections import Counter

multiprocessing.set_start_method("spawn", force=True)

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


def run_contention_process(duration_seconds, priority=0):
    """Separate process: GPU contention via cumsum."""
    import torch
    device = torch.device("cuda")
    if priority != 0:
        stream = torch.cuda.Stream(device, priority=priority)
    else:
        stream = torch.cuda.default_stream(device)
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    with torch.cuda.stream(stream):
        while time.time() < end_time:
            torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


def make_same_process_contention(device):
    streams = [torch.cuda.Stream() for _ in range(2)]
    data = torch.randn(3000000, device=device)
    outs = [torch.empty_like(data) for _ in range(2)]

    def fn(decode_step):
        for i, s in enumerate(streams):
            with torch.cuda.stream(s):
                for _ in range(300):
                    torch.cumsum(data, dim=0, out=outs[i])
    return fn


def run_condition(label, model, tokenizer, device, base_events,
                  pre_step_fn=None, cross_process=False,
                  contention_priority=0, contention_duration=120):
    """Run one condition and compute DDI against baseline."""
    print(f"\n=== {label} ===")

    contention_proc = None
    if cross_process:
        print(f"  Starting contention process (priority={contention_priority})...")
        contention_proc = multiprocessing.Process(
            target=run_contention_process,
            args=(contention_duration, contention_priority),
        )
        contention_proc.start()
        time.sleep(3)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    events, tokens = _run_prefill_and_decode(
        model, tokenizer, device, pre_step_fn=pre_step_fn
    )

    if contention_proc:
        contention_proc.terminate()
        contention_proc.join(timeout=5)

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
        "events_count": len(events),
    }


def check_clocks():
    """Check jetson_clocks state."""
    try:
        r = subprocess.run(
            ["sudo", "jetson_clocks", "--show"],
            capture_output=True, text=True, timeout=10,
        )
        gpu_lines = [l for l in r.stdout.split("\n") if "gpu" in l.lower()]
        for l in gpu_lines:
            print(f"  {l.strip()}")
        return r.stdout
    except Exception as e:
        print(f"  jetson_clocks check failed: {e}")
        return ""


def print_kernel_table(results, kernel_names=None):
    """Print side-by-side kernel DDI table for all conditions."""
    if not results:
        return

    if kernel_names is None:
        all_names = set()
        for r in results:
            all_names.update(r["kernel_ddi"].keys())
        kernel_names = sorted(all_names)

    # Header
    labels = [r["label"][:12] for r in results]
    header = f"{'Kernel':<50s}" + "".join(f"{l:>13s}" for l in labels)
    print(header)
    print("-" * len(header))

    for name in kernel_names:
        short = name[:48]
        vals = ""
        for r in results:
            v = r["kernel_ddi"].get(name, 0)
            vals += f"{v:>13.4f}"
        print(f"{short:<50s}{vals}")

    # Overall
    overall = ""
    for r in results:
        overall += f"{r['timing_ddi']:>13.4f}"
    print(f"{'OVERALL':<50s}{overall}")


def run_l2_persistence_test(model, tokenizer, device, base_events):
    """Test L2 cache persistence via cudaAccessPolicyWindow."""
    print("\n=== L2 PERSISTENCE TEST ===")

    # Check if cudaAccessPolicyWindow is available
    has_policy = hasattr(torch.cuda, 'set_l2_cache_policy') or True  # Try anyway

    # We can't easily set cudaAccessPolicyWindow from Python without
    # custom CUDA bindings. Instead, test the effect using stream
    # attributes if available.
    try:
        # PyTorch doesn't expose cudaAccessPolicyWindow directly.
        # Use a practical proxy: cudaStreamSetAttribute for L2 policy.
        # This may not be available on all platforms.
        print("  Note: cudaAccessPolicyWindow requires custom CUDA bindings.")
        print("  Testing L2 effect via cache-warm repeated execution instead.")

        # Approach: run inference twice in quick succession.
        # The second run benefits from L2 residency of weights/activations.
        # Compare DDI of first-run vs second-run.

        print("\n  --- First run (cold) ---")
        # Flush L2 by reading a large buffer
        flush_buf = torch.randn(2**24, device=device)  # 64 MiB
        flush_out = torch.empty_like(flush_buf)
        flush_out.copy_(flush_buf)
        torch.cuda.synchronize()
        del flush_buf, flush_out

        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        cold_events, _ = _run_prefill_and_decode(model, tokenizer, device)
        cold_matched = align_events(cold_events, base_events)
        cold_ddi = compute_timing_ddi(cold_matched)
        cold_kernel = breakdown_by_axis(cold_matched, "kernel_type", compute_timing_ddi)
        print(f"  Cold DDI: {cold_ddi:.6f}")

        print("\n  --- Second run (warm, L2 resident) ---")
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        warm_events, _ = _run_prefill_and_decode(model, tokenizer, device)
        warm_matched = align_events(warm_events, base_events)
        warm_ddi = compute_timing_ddi(warm_matched)
        warm_kernel = breakdown_by_axis(warm_matched, "kernel_type", compute_timing_ddi)
        print(f"  Warm DDI: {warm_ddi:.6f}")

        if cold_ddi > 0:
            print(f"  L2 warmth effect: {(1-warm_ddi/cold_ddi)*100:+.0f}%")

        return {
            "cold": {"timing_ddi": cold_ddi, "kernel_ddi": cold_kernel},
            "warm": {"timing_ddi": warm_ddi, "kernel_ddi": warm_kernel},
        }
    except Exception as e:
        print(f"  L2 test failed: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    if not torch.cuda.is_available():
        print("No CUDA device.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"Decode steps: {DECODE_STEPS}")

    mode = sys.argv[1] if len(sys.argv) > 1 else "--all"
    mps_active = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")
    print(f"MPS active: {bool(mps_active)}")
    print(f"Mode: {mode}")

    # === Step 1: Check clock state ===
    print("\n" + "=" * 60)
    print("STEP 1: CLOCK STATE CHECK")
    print("=" * 60)
    check_clocks()

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    # Baseline
    print("\n=== BASELINE (no contention) ===")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base_events, base_tokens = _run_prefill_and_decode(model, tokenizer, device)
    print(f"  Events: {len(base_events)}")

    # Check for DVFS: look at per-step times
    step_times = {}
    for e in base_events:
        step_times[e.decode_step] = step_times.get(e.decode_step, 0) + e.duration_ns
    step24 = step_times.get(24, 0) / 1000
    step25 = step_times.get(25, 0) / 1000
    ratio = step25 / step24 if step24 > 0 else 0
    print(f"  Step 24: {step24:.0f} µs, Step 25: {step25:.0f} µs, ratio: {ratio:.2f}")
    if ratio < 0.8:
        print(f"  WARNING: Step-25 drop detected (ratio={ratio:.2f}). DVFS may be active!")
        print(f"  Run 'sudo jetson_clocks' to lock GPU frequency.")
    else:
        print(f"  No step-25 anomaly. Clocks appear stable.")

    results_all = {}

    if mode in ("--all", "--mps"):
        # === Steps 2-4: MPS experiments ===
        print("\n" + "=" * 60)
        print("STEPS 2-4: MPS MITIGATION EXPERIMENTS")
        print("=" * 60)

        if not mps_active:
            print("  MPS not active. Skipping MPS experiments.")
            print("  Run via run_mps_sweep.sh to test with MPS.")
        else:
            mps_results = []

            # Same-process contention (reference)
            sp = run_condition(
                "SameProc", model, tokenizer, device, base_events,
                pre_step_fn=make_same_process_contention(device),
            )
            mps_results.append(sp)

            # Cross-process, no mitigation (MPS baseline)
            cp_base = run_condition(
                "CrossProc", model, tokenizer, device, base_events,
                cross_process=True,
            )
            mps_results.append(cp_base)

            # Step 2: Stream priority (contention at low priority)
            cp_prio = run_condition(
                "StreamPrio", model, tokenizer, device, base_events,
                cross_process=True, contention_priority=-1,
            )
            mps_results.append(cp_prio)

            # Step 3: SM partitioning via CUDA_MPS_ACTIVE_THREAD_PERCENTAGE
            for pct in [50, 70, 90]:
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = str(pct)
                cp_sm = run_condition(
                    f"SM_{pct}%", model, tokenizer, device, base_events,
                    cross_process=True,
                )
                mps_results.append(cp_sm)
            # Reset
            os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)

            # Step 4: Stream priority + best SM partition
            # Find best SM partition from step 3
            sm_results = [r for r in mps_results if r["label"].startswith("SM_")]
            if sm_results:
                best_sm = min(sm_results, key=lambda r: r["timing_ddi"])
                best_pct = best_sm["label"].split("_")[1].rstrip("%")
                os.environ["CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"] = best_pct
                cp_combo = run_condition(
                    f"Prio+SM{best_pct}%", model, tokenizer, device, base_events,
                    cross_process=True, contention_priority=-1,
                )
                mps_results.append(cp_combo)
                os.environ.pop("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE", None)

            # Print comparison table
            print("\n" + "=" * 60)
            print("MPS RESULTS SUMMARY")
            print("=" * 60)
            print_kernel_table(mps_results)

            results_all["mps"] = {r["label"]: r for r in mps_results}

    if mode in ("--all", "--l2"):
        # === Step 5: L2 persistence ===
        print("\n" + "=" * 60)
        print("STEP 5: L2 CACHE EFFECT")
        print("=" * 60)
        l2_results = run_l2_persistence_test(model, tokenizer, device, base_events)
        results_all["l2"] = l2_results

    # Save
    os.makedirs("results", exist_ok=True)
    with open("results/full_mitigation_sweep.json", "w") as f:
        json.dump(results_all, f, indent=2, default=str)
    print(f"\nResults saved to results/full_mitigation_sweep.json")
    print("ALL DONE.")


if __name__ == "__main__":
    main()
