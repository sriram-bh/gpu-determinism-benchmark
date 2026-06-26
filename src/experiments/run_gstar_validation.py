"""
Run G* validation: 5 Config A passes, validate kernel consistency,
check token consistency, save results.

Run from repo root:
    python3 src/experiments/run_gstar_validation.py
"""

import json
import sys
import os
import datetime

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
from src.metrics.ddi import validate_gstar


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    device_name = torch.cuda.get_device_name(device)
    cuda_version = torch.version.cuda
    pytorch_version = torch.__version__

    print(f"Device: {device_name}")
    print(f"CUDA: {cuda_version}, PyTorch: {pytorch_version}")

    print("\n--- Pre-run GPU status ---")
    os.system("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || echo '(no compute processes)'")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print(f"\nInput: \"{FIXED_INPUT_SENTENCE}\"")
    print(f"Decode steps: {DECODE_STEPS}, Seed: {SEED}")

    print("\nRunning 3 warmup passes (unprofiled)...")
    run_warmup(model, tokenizer, device, n=3)

    n_passes = 5
    print(f"\nRunning {n_passes} validation passes...")
    all_events = []
    all_tokens = []
    for i in range(n_passes):
        _configure_deterministic_mode()
        events, token_ids = _run_prefill_and_decode(model, tokenizer, device)
        all_events.append(events)
        all_tokens.append(token_ids)
        decoded = tokenizer.decode(token_ids)
        print(f"  Pass {i+1}: {len(events)} events, {len(token_ids)} tokens: {decoded}")

    # Kernel sequence validation (G*)
    print("\n--- G* Kernel Validation ---")
    gstar_valid = validate_gstar(all_events)

    # Token consistency
    print("\n--- Token Consistency ---")
    tokens_identical = all(t == all_tokens[0] for t in all_tokens[1:])
    print(f"  All {n_passes} runs produced identical tokens: {tokens_identical}")
    if not tokens_identical:
        for i, t in enumerate(all_tokens):
            print(f"    Run {i+1}: {t}")

    print("\n--- Post-run GPU status ---")
    os.system("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader 2>/dev/null || echo '(no compute processes)'")

    # Save results
    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    result_data = {
        "timestamp": timestamp,
        "gstar_valid": gstar_valid,
        "tokens_identical": tokens_identical,
        "config": "A",
        "decode_steps": DECODE_STEPS,
        "seed": SEED,
        "input_sentence": FIXED_INPUT_SENTENCE,
        "device": device_name,
        "pytorch_version": pytorch_version,
        "cuda_version": cuda_version,
        "n_passes": n_passes,
        "events_per_pass": len(all_events[0]),
        "token_ids": all_tokens[0],
        "decoded_text": tokenizer.decode(all_tokens[0]),
    }

    if gstar_valid:
        result_data["gstar_events"] = events_to_dicts(all_events[0])
        outfile = "results/gstar.json"
        with open(outfile, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\nG* CERTIFIED. Saved to {outfile}")
    else:
        result_data["all_runs_events"] = [events_to_dicts(run) for run in all_events]
        result_data["all_runs_tokens"] = all_tokens
        outfile = f"results/gstar_debug_{timestamp}.json"
        with open(outfile, "w") as f:
            json.dump(result_data, f, indent=2)
        print(f"\nG* NOT certified. Debug data saved to {outfile}")


if __name__ == "__main__":
    main()
