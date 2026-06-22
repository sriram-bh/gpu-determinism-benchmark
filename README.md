# GPU Determinism Benchmark

This project measures how far CUDA GPU kernel scheduling deviates from a formally certified deterministic reference schedule, expressed using the Lingua Franca reactor model. A toy GPT-2 inference pipeline — prefill plus token-by-token decode — runs on NVIDIA Thor hardware, and every individual kernel dispatch is captured via CUPTI and compared against a mathematically guaranteed ground truth schedule (G*) derived from Lingua Franca's logical time semantics. The gap between G* and actual GPU execution is quantified by two separate metrics: a Timing DDI, capturing physical-time jitter (kernels firing in the right order but at the wrong wall-clock time), and an Order DDI, capturing logical-time deviation (kernels firing in the wrong relative sequence). Both metrics are broken down across three independent axes — sequence position, kernel type, and tensor size — to identify specifically where and why non-determinism occurs, rather than collapsing it into a single number. The project is motivated by the industry tension between GPU probabilistic scheduling and LPU-style determinism, especially in safety-critical systems, and works toward a hybrid kernel policy that targets deterministic execution only where it's empirically shown to matter most.

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
│       └── gstar_pipeline.lf        # LF reactor program defining the prefill/decode ordering
├── experiments/
│   ├── baseline/                    # placeholder — Phase 1 output lands here
│   └── configs/                     # placeholder — Config A/B/C results land here
└── results/                         # placeholder — CSVs, G* JSON, plots land here
```

## Status (as of writing)

- **Phase 0 (scope + spec): done.** Spec finalized with dual-metric, multi-axis design.
- **Code scaffolding: done, untested on hardware.**
  - `schema.py` — finished, no caveats.
  - `ddi.py` — finished AND unit-tested against synthetic data (`python src/metrics/ddi.py` runs 4 self-tests, all passing as of last edit).
  - `gpu_harness.py` — written assuming CUDA/CUPTI are present. **Will not run without a GPU.** One known open question flagged inline: whether `torch.profiler`'s per-step event aggregation needs adjustment once tested on real hardware — check this first when Thor access starts.
  - `gstar_pipeline.lf` — full 20-step decode chain, not a placeholder/truncated version. Not yet compiled against a real `lfc` toolchain.
- **Not started:** anything requiring actual Thor access (G* validation, real DDI numbers, plots, Config B/C).

## Next Steps (pick up here)

1. Get Thor access, run `gpu_harness.py`, confirm it executes without errors.
2. Resolve the `torch.profiler` per-step windowing question flagged in `gpu_harness.py`.
3. Run `run_n_validation_passes(n=5)` and confirm `validate_gstar()` passes before computing any DDI.
4. Run the six-axis breakdown (`full_breakdown_report`) on real data and generate the six plots.
5. Only after the above is solid: attempt compiling `gstar_pipeline.lf` with `lfc` and decide how tightly to integrate it with the harness's output.
