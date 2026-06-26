"""
Cache analysis: use nsight compute (ncu) to measure L1/L2 cache hit
rates, DRAM throughput, and SM occupancy for the highest-jitter kernels
identified by the Timing DDI breakdown.

Runs two passes:
  1. Baseline (Config A, no contention)
  2. Contended (Config B-2streams, heavy contention)

Captures metrics for the first N kernel launches in a single decode
step, then compares baseline vs contended cache behavior.

Run from repo root:
    python3 src/experiments/run_cache_analysis.py
"""

import json
import subprocess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import torch
from src.experiments.gpu_harness import (
    _configure_deterministic_mode,
    _load_model,
    run_warmup,
    DECODE_STEPS,
    SEED,
    FIXED_INPUT_SENTENCE,
)

NCU_METRICS = ",".join([
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "l1tex__t_sector_hit_rate.pct",
    "lts__t_sector_hit_rate.pct",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__warps_active.avg.pct_of_peak_sustained_active",
    "sm__inst_executed.avg.pct_of_peak_sustained_elapsed",
])


def write_single_step_script(path, with_contention=False):
    """Write a minimal script that runs one decode step for ncu profiling."""
    contention_setup = ""
    contention_call = ""
    if with_contention:
        contention_setup = """
stream1 = torch.cuda.Stream()
stream2 = torch.cuda.Stream()
c_data = torch.randn(3000000, device=device)
c_out = torch.empty_like(c_data)
"""
        contention_call = """
    with torch.cuda.stream(stream1):
        for _ in range(300):
            torch.cumsum(c_data, dim=0, out=c_out)
    with torch.cuda.stream(stream2):
        for _ in range(300):
            torch.cumsum(c_data, dim=0, out=c_out)
"""

    script = f'''
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
model.eval()

inputs = tokenizer("In 1969, the first human walked on the Moon. The astronaut", return_tensors="pt").to(device)
{contention_setup}
# Warmup
for _ in range(3):
    with torch.no_grad():
        model(inputs["input_ids"], use_cache=True)
    torch.cuda.synchronize()

# Profiled step: single prefill pass
{contention_call}
with torch.no_grad():
    outputs = model(inputs["input_ids"], use_cache=True)
torch.cuda.synchronize()

print("Done.")
'''
    with open(path, "w") as f:
        f.write(script)


def run_ncu(script_path, output_csv, label):
    """Run ncu and capture metrics to CSV."""
    cmd = [
        "ncu",
        "--metrics", NCU_METRICS,
        "--csv",
        "--log-file", output_csv,
        "--target-processes", "all",
        "python3", script_path,
    ]
    print(f"\n  Running ncu ({label})...")
    print(f"  Command: {' '.join(cmd[:6])}...")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    if result.returncode != 0:
        print(f"  ncu failed (exit {result.returncode})")
        print(f"  stderr: {result.stderr[:500]}")
        return False
    print(f"  ncu completed. Output: {output_csv}")
    return True


def parse_ncu_csv(csv_path):
    """Parse ncu CSV output into per-kernel metric dicts."""
    kernels = []
    try:
        with open(csv_path) as f:
            lines = f.readlines()
    except FileNotFoundError:
        return kernels

    # Find header line (starts with "ID")
    header_idx = None
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith('ID'):
            header_idx = i
            break

    if header_idx is None:
        print(f"  Could not find header in {csv_path}")
        print(f"  First 5 lines: {lines[:5]}")
        return kernels

    import csv
    reader = csv.DictReader(lines[header_idx:])
    for row in reader:
        kernels.append(dict(row))
    return kernels


def main():
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    # Check ncu availability
    r = subprocess.run(["which", "ncu"], capture_output=True, text=True)
    if r.returncode != 0:
        print("ncu not found. Cannot run cache analysis.")
        sys.exit(1)
    print(f"ncu found at: {r.stdout.strip()}")

    # Check ncu version
    r = subprocess.run(["ncu", "--version"], capture_output=True, text=True)
    print(f"ncu version: {r.stdout.strip()}")

    os.makedirs("results", exist_ok=True)
    os.makedirs("/tmp/cache_analysis", exist_ok=True)

    # Write minimal scripts for ncu profiling
    baseline_script = "/tmp/cache_analysis/baseline_step.py"
    contended_script = "/tmp/cache_analysis/contended_step.py"
    write_single_step_script(baseline_script, with_contention=False)
    write_single_step_script(contended_script, with_contention=True)

    baseline_csv = "results/ncu_baseline.csv"
    contended_csv = "results/ncu_contended.csv"

    # Run ncu on baseline
    print("\n=== Baseline (Config A, no contention) ===")
    baseline_ok = run_ncu(baseline_script, baseline_csv, "baseline")

    # Run ncu on contended
    print("\n=== Contended (Config B-2streams) ===")
    contended_ok = run_ncu(contended_script, contended_csv, "contended")

    if not baseline_ok or not contended_ok:
        print("\nncu profiling failed. Trying nsys as fallback...")
        for label, script in [("baseline", baseline_script), ("contended", contended_script)]:
            nsys_out = f"results/nsys_{label}"
            cmd = [
                "nsys", "profile",
                "--stats=true",
                "-o", nsys_out,
                "python3", script,
            ]
            print(f"\n  Running nsys ({label})...")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            print(f"  Exit code: {result.returncode}")
            if result.stdout:
                print(f"  stdout (last 2000 chars):\n{result.stdout[-2000:]}")
            if result.stderr:
                print(f"  stderr (last 1000 chars):\n{result.stderr[-1000:]}")
        return

    # Parse and compare
    print("\n=== Parsing results ===")
    baseline_kernels = parse_ncu_csv(baseline_csv)
    contended_kernels = parse_ncu_csv(contended_csv)

    print(f"  Baseline kernels profiled: {len(baseline_kernels)}")
    print(f"  Contended kernels profiled: {len(contended_kernels)}")

    if baseline_kernels:
        print(f"\n  Baseline sample: {json.dumps(baseline_kernels[0], indent=2)[:500]}")
    if contended_kernels:
        print(f"\n  Contended sample: {json.dumps(contended_kernels[0], indent=2)[:500]}")

    # Save parsed results
    save_data = {
        "baseline_kernels": baseline_kernels,
        "contended_kernels": contended_kernels,
        "metrics_requested": NCU_METRICS,
    }
    with open("results/cache_analysis.json", "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to results/cache_analysis.json")

    # Summary comparison
    if baseline_kernels and contended_kernels:
        print("\n=== Cache metric comparison ===")
        print(f"{'Kernel':<50} {'Metric':<20} {'Baseline':>10} {'Contended':>10}")
        print("-" * 95)
        metric_keys = [k for k in baseline_kernels[0].keys()
                       if k not in ("ID", "Kernel Name", "Process ID", "Process Name",
                                    "Host Name", "Kernel Time", "Context", "Stream",
                                    "Section Name", "Block Size", "Grid Size")]
        for bk, ck in zip(baseline_kernels[:10], contended_kernels[:10]):
            kname = bk.get("Kernel Name", "unknown")[:48]
            for mk in metric_keys:
                bv = bk.get(mk, "N/A")
                cv = ck.get(mk, "N/A")
                print(f"{kname:<50} {mk:<20} {str(bv):>10} {str(cv):>10}")
                kname = ""  # only print kernel name on first metric line
            print()

    print("\nDONE.")


if __name__ == "__main__":
    main()
