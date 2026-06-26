"""
Config B kernel name check: run GPT-2 inference on stream 0 with
competing work on stream 1, then diff the kernel name set against G*.

This is the gating check from spec.md Section 3 — must pass before
any cross-config DDI comparison is trusted.

Run from repo root:
    python3 src/experiments/run_config_b_kernel_check.py
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


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")

    # Load G* for comparison
    gstar_path = "results/gstar.json"
    if not os.path.exists(gstar_path):
        print(f"G* not found at {gstar_path}. Run run_gstar_validation.py first.")
        sys.exit(1)

    with open(gstar_path) as f:
        gstar_data = json.load(f)
    gstar_names = set(e["kernel_name"] for e in gstar_data["gstar_events"])
    gstar_count = len(gstar_data["gstar_events"])
    print(f"\nG*: {len(gstar_names)} distinct kernel names, {gstar_count} total events")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print("\nRunning 3 warmup passes (unprofiled, no competing work)...")
    run_warmup(model, tokenizer, device, n=3)

    # Pre-queue competing work on stream 1 (no threading — CUPTI is not
    # thread-safe on Thor). CUDA ops are async, so these execute on the
    # GPU concurrently with the inference on stream 0.
    stream1 = torch.cuda.Stream()
    contention_data = torch.randn(5000000, device=device)
    contention_out = torch.empty_like(contention_data)
    n_contention_ops = 50000

    print(f"\nQueuing {n_contention_ops} cumsum ops on stream 1...")
    with torch.cuda.stream(stream1):
        for _ in range(n_contention_ops):
            torch.cumsum(contention_data, dim=0, out=contention_out)

    print(f"Running Config B: GPT-2 on stream 0 (stream 1 executing concurrently)...")
    events, token_ids = _run_prefill_and_decode(model, tokenizer, device)
    stream1.synchronize()

    # Analyze kernel names
    configb_names = set(e.kernel_name for e in events)

    common = gstar_names & configb_names
    only_in_gstar = gstar_names - configb_names
    only_in_configb = configb_names - gstar_names

    print(f"\n--- Kernel Name Diff: G* vs Config B ---")
    print(f"G* kernel names:       {len(gstar_names)}")
    print(f"Config B kernel names: {len(configb_names)}")
    print(f"In common:             {len(common)}")

    if only_in_gstar:
        print(f"\nKernels in G* but MISSING from Config B ({len(only_in_gstar)}):")
        for n in sorted(only_in_gstar):
            print(f"  MISSING: {n}")
    else:
        print(f"\nAll {len(gstar_names)} G* kernel names present in Config B.")

    if only_in_configb:
        print(f"\nNew kernels in Config B not in G* ({len(only_in_configb)}):")
        for n in sorted(only_in_configb):
            print(f"  NEW: {n}")
    else:
        print(f"\nNo new kernel names in Config B.")

    # Token check
    gstar_tokens = gstar_data["token_ids"]
    tokens_match = token_ids == gstar_tokens
    print(f"\nTokens match G*: {tokens_match}")
    if not tokens_match:
        print(f"  G* tokens:      {gstar_tokens}")
        print(f"  Config B tokens: {token_ids}")

    # Event count comparison
    configb_gpt2_events = [e for e in events if e.kernel_name in gstar_names]
    print(f"\nGPT-2 kernel events (matching G* names): {len(configb_gpt2_events)} (G* has {gstar_count})")

    # Verdict
    print(f"\n--- Verdict ---")
    if not only_in_gstar:
        print("PASS: All G* kernel names present in Config B.")
        print("Cross-config DDI comparison is valid (spec Section 3, outcome 1).")
    elif len(only_in_gstar) <= 3:
        print("PARTIAL: Most G* kernel names present, a few differ.")
        print("May need normalization layer (spec Section 3, outcome 2).")
    else:
        print("FAIL: Substantial kernel name differences.")
        print("Kernel-level DDI comparison may not be meaningful (spec Section 3, outcome 3).")


if __name__ == "__main__":
    main()
