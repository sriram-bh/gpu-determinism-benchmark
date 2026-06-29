"""
L2 persistence test: use cudaAccessPolicyWindow to pin model weights
in L2 cache, then measure timing jitter under MPS cross-process
contention.

Tests whether explicit L2 pinning reduces timing jitter when competing
memory traffic would otherwise evict useful data.

Three conditions:
  1. No mitigation (MPS cross-process contention baseline)
  2. L2 persistence: pin attention weights in L2 (16MB persisting region)
  3. L2 persistence + max persisting cache (set to 16MB, half of 32MB L2)

MPS daemon must be running. Run via run_l2_persistence_sweep.sh.

Run from repo root:
    python3 src/experiments/run_l2_persistence_test.py
"""

import ctypes
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
)
from src.metrics.schema import events_to_dicts
from src.metrics.ddi import align_events, compute_timing_ddi, breakdown_by_axis


# ── CUDA L2 persistence via ctypes ────────────────────────────────

def get_cuda_rt():
    return ctypes.CDLL("libcudart.so")


def set_persisting_l2_size(cuda_rt, size_bytes):
    """Set cudaLimitPersistingL2CacheSize."""
    ret = cuda_rt.cudaDeviceSetLimit(0x06, ctypes.c_size_t(size_bytes))
    if ret != 0:
        print(f"  WARNING: cudaDeviceSetLimit failed with ret={ret}")
    return ret


def get_persisting_l2_size(cuda_rt):
    size = ctypes.c_size_t()
    cuda_rt.cudaDeviceGetLimit(ctypes.byref(size), 0x06)
    return size.value


def apply_l2_policy_to_stream(cuda_rt, stream_id, data_ptr, data_size, hit_ratio=0.6):
    """
    Apply cudaAccessPolicyWindow to a CUDA stream, pinning a memory
    region in L2 with the specified hit ratio.
    """

    # cudaAccessPolicyWindow struct layout (from CUDA headers):
    #   void*  base_ptr      (8 bytes)
    #   size_t num_bytes      (8 bytes)
    #   float  hitRatio       (4 bytes)
    #   int    hitProp        (4 bytes) = cudaAccessPropertyPersisting = 2
    #   int    missProp       (4 bytes) = cudaAccessPropertyStreaming = 1
    #   pad                   (4 bytes)

    class CudaAccessPolicyWindow(ctypes.Structure):
        _fields_ = [
            ("base_ptr", ctypes.c_void_p),
            ("num_bytes", ctypes.c_size_t),
            ("hitRatio", ctypes.c_float),
            ("hitProp", ctypes.c_int),
            ("missProp", ctypes.c_int),
            ("pad", ctypes.c_int),
        ]

    # cudaStreamAttrValue union — accessPolicyWindow is the first member
    class CudaStreamAttrValue(ctypes.Union):
        _fields_ = [
            ("accessPolicyWindow", CudaAccessPolicyWindow),
        ]

    policy = CudaAccessPolicyWindow()
    policy.base_ptr = data_ptr
    policy.num_bytes = data_size
    policy.hitRatio = hit_ratio
    policy.hitProp = 2   # cudaAccessPropertyPersisting
    policy.missProp = 1  # cudaAccessPropertyStreaming

    attr_val = CudaStreamAttrValue()
    attr_val.accessPolicyWindow = policy

    # cudaStreamSetAttribute(stream, attr, &value)
    # cudaStreamAttrID::cudaStreamAttributeAccessPolicyWindow = 1
    ret = cuda_rt.cudaStreamSetAttribute(
        ctypes.c_void_p(stream_id),
        ctypes.c_int(1),
        ctypes.byref(attr_val),
    )
    return ret


def setup_l2_persistence(model, device, persist_mb=16, hit_ratio=0.6):
    """
    Set up L2 persistence for the model's largest weight tensors.
    Returns a pre_step_fn hook that applies the policy before each step.
    """
    cuda_rt = get_cuda_rt()

    # Set persisting L2 cache size
    persist_bytes = persist_mb * 1024 * 1024
    set_persisting_l2_size(cuda_rt, persist_bytes)
    actual = get_persisting_l2_size(cuda_rt)
    print(f"  Persisting L2 size: {actual/1024/1024:.0f} MB (requested {persist_mb} MB)")

    # Find the largest weight tensors to pin
    params = [(name, p.data) for name, p in model.named_parameters()]
    params.sort(key=lambda x: -x[1].numel())

    # Pin weights up to the persisting size
    pinned = []
    total_pinned = 0
    for name, p in params:
        p_bytes = p.numel() * p.element_size()
        if total_pinned + p_bytes > persist_bytes:
            break
        pinned.append((name, p))
        total_pinned += p_bytes

    print(f"  Pinning {len(pinned)} tensors ({total_pinned/1024/1024:.1f} MB) in L2")
    if pinned:
        print(f"  First: {pinned[0][0]} ({pinned[0][1].numel()*4/1024/1024:.1f} MB)")
        print(f"  Last:  {pinned[-1][0]} ({pinned[-1][1].numel()*4/1024/1024:.1f} MB)")

    # Get the default stream ID (0 for default stream)
    default_stream = 0

    def pre_step_hook(decode_step):
        # Apply L2 policy for each pinned tensor on the default stream
        for name, p in pinned:
            ptr = p.data_ptr()
            size = p.numel() * p.element_size()
            apply_l2_policy_to_stream(cuda_rt, default_stream, ptr, size, hit_ratio)

    return pre_step_hook


# ── Contention process ────────────────────────────────────────────

def run_contention_default(duration_seconds):
    import torch
    device = torch.device("cuda")
    data = torch.randn(3000000, device=device)
    out = torch.empty_like(data)
    end_time = time.time() + duration_seconds
    while time.time() < end_time:
        torch.cumsum(data, dim=0, out=out)
    torch.cuda.synchronize()


# ── Experiment runner ─────────────────────────────────────────────

def run_condition(label, model, tokenizer, device, base_events,
                  pre_step_fn=None, contention=True, duration=120):
    print(f"\n=== {label} ===")

    proc = None
    if contention:
        print(f"  Starting contention process...")
        proc = multiprocessing.Process(
            target=run_contention_default, args=(duration,)
        )
        proc.start()
        time.sleep(3)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    events, tokens = _run_prefill_and_decode(
        model, tokenizer, device, pre_step_fn=pre_step_fn
    )

    if proc:
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


def main():
    if not torch.cuda.is_available():
        print("No CUDA device.")
        sys.exit(1)

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    mps_active = os.environ.get("CUDA_MPS_PIPE_DIRECTORY", "")

    print(f"Device: {props.name}")
    print(f"L2 cache: {props.L2_cache_size/1024/1024:.0f} MB")
    print(f"MPS active: {bool(mps_active)}")

    # Verify L2 persistence API
    cuda_rt = get_cuda_rt()
    default_persist = get_persisting_l2_size(cuda_rt)
    print(f"Default persisting L2: {default_persist/1024/1024:.1f} MB")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes...")
    run_warmup(model, tokenizer, device, n=3)

    # Baseline (no contention)
    print("\n=== BASELINE (no contention) ===")
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    base_events, _ = _run_prefill_and_decode(model, tokenizer, device)
    print(f"  Events: {len(base_events)}")

    results = []

    # Condition 1: MPS contention, no L2 persistence
    r1 = run_condition(
        "NoL2Pin", model, tokenizer, device, base_events,
    )
    results.append(r1)

    # Condition 2: MPS contention + L2 persistence (8MB, conservative)
    print("\n--- Setting up L2 persistence (8MB) ---")
    hook_8mb = setup_l2_persistence(model, device, persist_mb=8, hit_ratio=0.6)
    r2 = run_condition(
        "L2Pin_8MB", model, tokenizer, device, base_events,
        pre_step_fn=hook_8mb,
    )
    results.append(r2)

    # Condition 3: MPS contention + L2 persistence (16MB, aggressive)
    print("\n--- Setting up L2 persistence (16MB) ---")
    hook_16mb = setup_l2_persistence(model, device, persist_mb=16, hit_ratio=0.8)
    r3 = run_condition(
        "L2Pin_16MB", model, tokenizer, device, base_events,
        pre_step_fn=hook_16mb,
    )
    results.append(r3)

    # Condition 4: L2 persistence only, NO contention (control)
    print("\n--- L2 persistence without contention (control) ---")
    hook_ctrl = setup_l2_persistence(model, device, persist_mb=16, hit_ratio=0.8)
    r4 = run_condition(
        "L2Pin_NoCont", model, tokenizer, device, base_events,
        pre_step_fn=hook_ctrl, contention=False,
    )
    results.append(r4)

    # Reset L2 persistence to default
    set_persisting_l2_size(cuda_rt, default_persist)

    # Summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    baseline_ddi = results[0]["timing_ddi"]
    print(f"\n{'Condition':<20s} {'DDI':>8s} {'vs NoL2Pin':>12s}")
    print("-" * 42)
    for r in results:
        if r["label"] == "NoL2Pin":
            print(f"{r['label']:<20s} {r['timing_ddi']:>8.4f}    (baseline)")
        else:
            change = (1 - r["timing_ddi"] / baseline_ddi) * 100
            direction = "better" if change > 0 else "worse"
            print(f"{r['label']:<20s} {r['timing_ddi']:>8.4f} {change:>+11.0f}% {direction}")

    # Per-kernel table
    print(f"\n{'Kernel':<50s}", end="")
    for r in results:
        print(f"{r['label']:>13s}", end="")
    print()
    print("-" * (50 + 13 * len(results)))

    all_names = set()
    for r in results:
        all_names.update(r["kernel_ddi"].keys())
    for name in sorted(all_names):
        short = name[:48]
        print(f"{short:<50s}", end="")
        for r in results:
            v = r["kernel_ddi"].get(name, 0)
            print(f"{v:>13.4f}", end="")
        print()

    print(f"{'OVERALL':<50s}", end="")
    for r in results:
        print(f"{r['timing_ddi']:>13.4f}", end="")
    print()

    # Save
    os.makedirs("results", exist_ok=True)
    save_data = {r["label"]: r for r in results}
    save_data["l2_cache_size_mb"] = props.L2_cache_size / 1024 / 1024
    save_data["default_persisting_mb"] = default_persist / 1024 / 1024
    with open("results/l2_persistence_test.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to results/l2_persistence_test.json")
    print("DONE.")


if __name__ == "__main__":
    main()
