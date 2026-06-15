# Project Specification — GPU Determinism Benchmark

## Research Question
How much does CUDA kernel scheduling on NVIDIA Thor deviate from a formally certified deterministic reference schedule, and which kernel engineering choices reduce that deviation?

## Workload
GPT-2 (117M parameters) single forward pass inference. Fixed benchmark prompt: "The quick brown fox jumps over the lazy dog." Fixed random seed (42) across all runs. CPU proxy mode available for development pre-Thor; swap `device="cpu"` to `device="cuda"` on hardware.

## Reference Schedule G*
Expressed as a Lingua Franca (LF) reactor pipeline. By LF's execution semantics, for the same input stream the program always produces the same logical-time schedule — this is a language guarantee, not an empirical claim. G* is validated by running the LF harness 5 times under fully controlled conditions (fixed seed, single CUDA stream, cuDNN deterministic mode) and verifying bit-identical kernel dispatch sequences across all 5 runs. A passing validation certifies G* as the ground truth reference. G* is then saved to `results/gstar_validated.json`.

## CUPTI ↔ LF Bridge
Physical GPU kernel events are captured via NVIDIA CUPTI, which provides callbacks at kernel entry and exit with microsecond-level timing. These physical events are lifted into LF logical time via `lf_schedule()` calls inside the CUPTI callbacks, so every captured kernel carries both a wall-clock timestamp and an LF logical tag. Pre-Thor, a CPU proxy patches `torch.nn.functional` operations to record timing in the identical data format CUPTI will produce on hardware.

## DDI Formula
```
DDI(run r) = mean( |t_actual(k,r) - t_ref(k)| / t_ref(k) )  for all kernels k

where:
  t_actual(k,r)  = wall-clock timestamp of kernel k in run r (nanoseconds, from CUPTI)
  t_ref(k)       = logical timestamp of kernel k in G* (nanoseconds, from LF schedule)
  DDI = 0        → perfect determinism (LPU-style)
  DDI = 0.30     → 30% average deviation from reference schedule
```

Two deviation types are tracked separately:
- **Order violations**: kernel fired in the wrong logical order vs. G* (structural non-determinism)
- **Timing jitter**: right order, wrong wall-clock timing (physical non-determinism)

This distinction directly engages Edward Lee's logical/physical time separation in the Reactor model.

## Statistical Reporting
- Minimum 100 runs per configuration
- GPU warmup: 10 runs before measurement (not counted)
- Report: mean DDI, std, p50, p95, p99, min, max, 95% confidence interval
- Significance testing: Welch's t-test vs. baseline for any configuration comparisons

## Experiment: Single Synchronous Pass
One CUDA stream, `cudaDeviceSynchronize` after the forward pass, `cuDNN deterministic=True`, `benchmark=False`, `torch.use_deterministic_algorithms(True)`. This is the baseline configuration and the only configuration in scope until GPU hardware is available and CUDA techniques are better understood. Multi-configuration comparisons (multi-stream, contested, hybrid policy) are deliberately deferred.

## Hardware Target
NVIDIA Thor (Drive OS 6.x). CUPTI 12.x available at `/usr/local/cuda/extras/CUPTI/`.

## Key References
- Olmedo et al., RTAS 2020 — GPU scheduling hierarchy (foundational paper)
- Han et al., ACM TOCS 2025 (REEF) — GPU DNN inference scheduling
- Lohstroh et al., CyPhy 2019 — Reactor model and Lingua Franca formalization
- Prior CPU simulation: github.com/sriram-bh/llm-attention-nondeterminism

## Target Conferences
- RTAS 2027 (primary full paper)
- EMSOFT 2027 (if LF angle is prominent)
- NVIDIA GTC (poster/demo)
