# GPU Determinism Benchmark

This project measures how far CUDA GPU kernel scheduling deviates from an empirically validated deterministic baseline. A toy GPT-2 inference pipeline — prefill plus a fixed-length token-by-token decode — runs on NVIDIA Thor hardware, and every individual kernel dispatch is captured via CUPTI. The reference schedule, G*, is the captured kernel sequence from 5 bit-identical runs under maximally controlled conditions (single CUDA stream, deterministic algorithm flags, fixed seed). The gap between G* and subsequent runs is quantified by two separate metrics: a Timing DDI (physical-time jitter — kernels in the right order, wrong duration) and an Order DDI (logical-time deviation — kernels in the wrong relative sequence, measured via pairwise inversion count rather than naive rank distance). Both metrics are broken down across three independent axes — sequence position, kernel type, and tensor size — to identify specifically where and why non-determinism occurs. Lingua Franca is used to coordinate the hybrid kernel policy's mode-switch logic, guaranteeing the deterministic/fast mode decision is resolved before each decode step's kernels are dispatched — preventing a race condition that would otherwise exist between the async CPU-side policy decision and GPU kernel launch.

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
│       └── hybrid_policy_controller.lf   # Coordinates hybrid mode-switch decisions
├── experiments/
│   ├── baseline/                    # placeholder — Phase 1 output lands here
│   └── configs/                     # placeholder — Config A/B/C results land here
└── results/                         # placeholder — CSVs, G* JSON, plots land here
```

## Status (as of writing)

- **Phase 0 (scope + spec): done.**
- **Code scaffolding:**
  - `schema.py` — finished, no caveats.
  - `ddi.py` — finished and unit-tested against synthetic data. Known open limitation: Order DDI currently treats all kernel-order differences as equally meaningful, regardless of whether a real dependency exists between the kernels involved — see spec.md Section 4.
  - `gpu_harness.py` — written assuming CUDA/CUPTI are present. Will not run without a GPU.
  - `hybrid_policy_controller.lf` — defines the mode-switch coordination chain. Not yet wired to the real harness; threshold is a placeholder pending real data.
- **Open empirical question that gates cross-config comparisons:** whether Config A, B, and C dispatch the same set of kernel names. Must be checked by diffing raw kernel name lists the moment Thor access starts, before trusting any DDI number across configs. See spec.md Section 3.

## Next Steps (pick up here)

1. Get Thor access, run `gpu_harness.py`, confirm it executes without errors.
2. Run `run_n_validation_passes(n=5)` and confirm `validate_gstar()` passes — this produces G*.
3. **Before computing any cross-config DDI:** run Config B once, diff its raw kernel name set against G*'s. Determine which of the three outcomes in spec.md Section 3 applies.
4. Run the six-axis breakdown on real Config A data; generate the six plots.
5. Wire `hybrid_policy_controller.lf` to the real harness once a cache-size threshold is informed by real data from step 4.
