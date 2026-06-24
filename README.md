# GPU Determinism Benchmark

This project measures how far CUDA GPU kernel scheduling deviates from an empirically validated deterministic baseline. A toy GPT-2 inference pipeline — prefill plus a fixed-length token-by-token decode — runs on NVIDIA Thor hardware, and every individual kernel dispatch is captured via CUPTI. The reference schedule, G*, is the captured kernel sequence from 5 bit-identical runs under maximally controlled conditions (single CUDA stream, deterministic algorithm flags, fixed seed) — not generated or defined by Lingua Franca, a deliberate correction from an earlier version of this project that mistakenly treated LF's reactor ordering as a meaningful guarantee at the CUDA kernel level, when in fact LF has no visibility into GPU hardware scheduling at all. The gap between G* and subsequent runs is quantified by two separate metrics: a Timing DDI (physical-time jitter — kernels in the right order, wrong duration) and an Order DDI (logical-time deviation — kernels in the wrong relative sequence, measured via pairwise inversion count rather than naive rank distance). Both metrics are broken down across three independent axes — sequence position, kernel type, and tensor size — to identify specifically where and why non-determinism occurs. Lingua Franca's actual role in this project is narrower than originally proposed: coordinating the hybrid kernel policy's internal mode-switch logic to prevent a real async race condition between CPU-side policy decisions and GPU kernel dispatch — not generating ground truth.

## Repository Structure

```
gpu-determinism-benchmark/
├── README.md
├── spec.md                          # Full project spec — read this for the "why"
├── requirements.txt
├── src/
│   ├── metrics/
│   │   ├── schema.py                # KernelEvent — shared data shape used everywhere
│   │   └── ddi.py                   # Timing DDI, Order DDI, 6-way axis breakdown, G* validation
│   ├── experiments/
│   │   └── gpu_harness.py           # GPU-only harness (no CPU fallback). Runs on Thor.
│   └── lf/
│       └── hybrid_policy_controller.lf   # LF reactor for hybrid mode-switch coordination (race-condition prevention)
├── experiments/
│   ├── baseline/                    # placeholder — Phase 1 output lands here
│   └── configs/                     # placeholder — Config A/B/C results land here
└── results/                         # placeholder — CSVs, G* JSON, plots land here
```

## Status (as of writing)

- **Phase 0 (scope + spec): done, with one significant correction.** The original plan to use an LF reactor program to generate G* was abandoned after working through what LF's guarantee actually covers — it has no mechanism that reaches CUDA kernel scheduling. G* is now purely empirical (captured Config A kernel sequence). LF's role moved to coordinating the hybrid policy's mode-switch logic instead, where it solves a real, identified race condition.
- **Code scaffolding:**
  - `schema.py` — finished, no caveats.
  - `ddi.py` — finished and unit-tested against synthetic data. Known open limitation: Order DDI currently treats all kernel-order differences as equally meaningful; distinguishing "real" dependency violations from "cosmetic" reordering of independent kernels is deferred until real data shows it's necessary (see spec.md Section 4).
  - `gpu_harness.py` — written assuming CUDA/CUPTI are present. Will not run without a GPU. Decode length is fixed deliberately (see spec.md Section 2) — not an arbitrary cap.
  - The old `reference_pipeline.lf` and `gstar_pipeline.lf` files are removed/being replaced — they encoded an ordering guarantee (prefill before decode 1 before decode 2...) that any correct implementation provides for free, and was never a real contribution of using LF.
- **Open empirical question that gates cross-config comparisons:** whether Config A, B, and C dispatch the same set of kernel names or not. Must be checked by diffing raw kernel name lists the moment Thor access starts, before trusting any DDI number across configs. See spec.md Section 3 for the three possible outcomes and what each requires.

## Next Steps (pick up here)

1. Get Thor access, run `gpu_harness.py`, confirm it executes without errors.
2. Run `run_n_validation_passes(n=5)` and confirm `validate_gstar()` passes — this is G*, fully empirical now.
3. **Before computing any cross-config DDI:** run Config B once, diff its raw kernel name set against G*'s. Determine which of the three outcomes in spec.md Section 3 applies.
4. Run the six-axis breakdown on real Config A data; generate the six plots.
5. Write the `hybrid_policy_controller.lf` reactor (mode-switch coordination) once a cache-size threshold is informed by real data from step 4 — not before.
