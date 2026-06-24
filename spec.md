# spec.md — GPU Determinism Benchmark

## 1. Problem

GPUs are incredibly powerful SIMD tools in computing, essential to almost any modern AI system. However, this comes with a caveat: GPUs can produce non-deterministic results at three phases — application-level time and processing allocation, thread scheduling, and SM (Streaming Multiprocessor) assignment. Surprisingly, even when controlling for the first two, the third factor introduces non-determinism on its own. SMs are not assigned work in a simple round-robin fashion — depending on the warp counts of competing kernels, the block scheduler can stack multiple blocks onto an already-occupied SM and leave other SMs idle entirely, a behavior empirically documented by Olmedo et al. (RTAS 2020).

Non-determinism is not the only issue — latency is a real cost too. When two different kernels execute on SMs within the same GPC (Graphics Processing Cluster), they share a single bus to L2 memory. If the two kernels are doing unrelated work, that bus carries twice the traffic with zero reuse benefit. Olmedo et al. measured this directly and found that intra-GPC memory contention can increase a kernel's worst-case execution time by up to 10x.

This is a real problem, especially in safety-critical systems. An inconsistent floating-point result or an unpredictable latency spike can cause a critical misjudgment — a missed brake deadline, a delayed surgical robot response. Deterministic chips exist — Groq's LPU, for instance — but they trade away raw throughput to get there. The question I aim to answer is: which kernel engineering techniques reduce non-determinism in multi-stream execution, compared to a deterministic single-stream baseline, without giving up the throughput multi-stream exists to provide? A necessary first step toward that question is knowing *where* non-determinism actually concentrates — which kernels, at which points in generation, at which tensor sizes. If no mitigating technique exists once that picture is built, the empirical latency-vs-determinism tradeoff is itself a valid finding.

**Scope boundary, stated explicitly:** this project measures *scheduling* non-determinism — variation in kernel order and timing under identical inputs and identical computed values. It does not measure *numerical* non-determinism — variation in the actual computed values themselves (e.g. from floating-point non-associativity in atomic operations). That question is the subject of a separate, prior CPU-level simulation (github.com/sriram-bh/llm-attention-nondeterminism). Keeping these separate matters because they have different causal mechanisms and conflating them would make any single metric in this project uninterpretable.

## 2. Workload

GPT-2 Small (117M parameters) on the lab's NVIDIA Thor, running prefill over a fixed 20-token input sentence, followed by a **fixed number** of decode steps. The decode length is fixed, not determined by the model reaching a natural stopping point (e.g. an end-of-sequence token), because allowing length to vary would make length itself a function of numerical non-determinism (small floating-point differences changing which token argmax selects, including possibly selecting an end-of-sequence token at different steps across runs) — which is explicitly out of scope per Section 1. A fixed length is required for G* to be well-defined: G* must have a knowable, constant number of decode steps to serve as a fixed reference.

The specific decode step count is a deliberate measurement choice — long enough to observe a trend across the sequence-position breakdown axis, short enough to keep per-run time manageable — and is recorded explicitly as a parameter (`DECODE_STEPS` in `gpu_harness.py`), not left as an implicit, unjustified default.

## 3. Reference Schedule (G*)

**G* is purely empirical.** It is the captured kernel sequence — kernel names, logical tags, durations, decode-step and tensor-size metadata — from 5 runs of Config A (single CUDA stream, `cudnn.deterministic=True`, `cudnn.benchmark=False`, fixed seed and shape), accepted as valid only if all 5 runs produce an identical kernel sequence with identical logical tags. Lingua Franca plays no role in generating or defining G*.

G* is purely empirical: kernel order and timing are governed entirely by the GPU's hardware scheduler, a closed-source system no software-level language can observe or control. A coordination language can guarantee the order of reactions it defines, but it has no mechanism that reaches CUDA kernel dispatch — so G* is not expressed in Lingua Franca.

**Open empirical question, gating everything downstream:** it is not yet known whether different configurations (A vs. B vs. C, or any future kernel-engineering technique) dispatch the *same set* of CUDA kernels (just potentially in a different order or with different timing), or whether configuration changes (e.g. `cudnn.benchmark=True` enabling algorithm autotuning) cause *different kernels entirely* to be selected for the same mathematical operation. This must be checked empirically — diffing raw kernel name sets across configs — before any cross-config DDI comparison is trusted. Three possible outcomes and their implications:

1. **Identical kernel name sets, order/timing differs only** — current `align_events()` matching approach is sound as-is.
2. **Mostly identical, some kernel names differ** (likely from autotuning) — requires a normalization layer mapping kernel-name variants to a common operation category before alignment.
3. **Substantially different kernel sets** — kernel-level comparison breaks down; DDI would need to shift to comparing at the architectural-operation level instead.

## 4. DDI Formulas

Two separate metrics are tracked, since timing and ordering can deviate independently of each other.

**Timing DDI** — how much longer or shorter each kernel took versus expected (physical-time jitter):
```
Timing DDI = (1/N) × Σ |t_actual(i) - t_ref(i)| / t_ref(i)
```

**Order DDI** — how much the actual kernel execution order deviates from the expected order (logical-time deviation). Computed as a normalized count of pairwise inversions (Kendall's tau distance) rather than per-kernel rank distance, because rank distance over-penalizes a uniform rotation (e.g. the sequence `10,1,2,...,9`, which preserves every pairwise relationship except one) as harshly as a fully scrambled order:
```
Order DDI = (number of pairwise order inversions) / (N choose 2)
```

N is the number of matched kernels in a run. A kernel/run can score well on one metric and poorly on the other — never combined into one number.

**Known limitation, not yet resolved:** Order DDI as currently computed treats every kernel-order difference from G* as equally meaningful, including reordering between two kernels that have no real data dependency on each other and were always free to execute in either order (a "cosmetic" violation) versus reordering that violates the model's actual mathematical dependency graph (a "real" violation — which should be rare/impossible in any correct implementation, but worth verifying). Distinguishing these would require an explicit kernel-category dependency map for GPT-2 (e.g. "softmax kernels depend on the QK-matmul kernel within the same attention block"), maintained separately and by hand, not generated by LF or any other tool. This refinement is deferred until real data shows whether it's empirically necessary.

## 5. Breakdown Axes

A single aggregate value for either DDI says non-determinism occurred, but not where or why — unusable for designing a targeted hybrid policy. Both DDIs are computed broken down across three independent axes:

- **Sequence position** — DDI as a function of decode step. Tests whether GPU scheduling divergence follows a similar shape to the numerical divergence pattern observed in the prior CPU-level simulation, a different shape, or none at all.
- **Kernel type** — DDI grouped by kernel name, independent of position. Likely a bar chart rather than a curve, given the small number of distinct kernel types in GPT-2. The most directly actionable axis for a hybrid policy.
- **Tensor size / memory footprint** — DDI as a function of the primary operand's byte size. Tests whether jitter and order violations are driven by memory bandwidth contention (the GPC-to-L2 bus mechanism in Section 1) rather than position or kernel identity.

Six plots total (Timing × 3 axes, Order × 3 axes), generated from one captured dataset per run.

## 6. Latency Definition

End-to-end wall-clock time for the full prefill + fixed-length decode run, measured via `cudaEventElapsedTime` from first kernel dispatch to last kernel completion. Every config comparison is reported as a (Timing DDI, Order DDI, Latency) triple, never collapsed into a single score.

## 7. Configs to Test

- **Config A — Single stream, synchronous.** Baseline; defines G*. Expected lowest DDI on both metrics, lowest throughput.
- **Config B — Two streams, overlapped.** Expected higher Order DDI due to cross-stream kernel interleaving. Primary target for kernel-engineering mitigation.
- **Config C — Single stream with background memory copy.** Isolates memory contention's effect on Timing DDI without introducing stream-level order non-determinism.

*Config A is the only configuration in active scope until the kernel-matching question in Section 3 is resolved empirically.*

## 8. Role of Lingua Franca

LF coordinates the hybrid kernel policy's mode-switch logic. The hybrid policy (deterministic mode early in decode, throughput-optimized mode later, switching at some cache-size threshold) requires the mode-switch decision to be resolved consistently before each decode step's kernels are dispatched. A naive Python implementation is vulnerable to a real race condition: CUDA kernel launches are asynchronous, so the host thread can dispatch a decode step's kernels using a stale or inconsistently-read mode flag, especially under multi-stream configurations.

`hybrid_policy_controller.lf` eliminates this by construction: a `PolicyController` reactor only fires once cache size has a definite value for the current step, and the kernel-dispatching logic only fires once the policy decision has a value — LF's dependency mechanism guarantees the mode decision can never be applied out of sequence or read as stale. This is the same mechanism Lee's drive-by-wire example uses to guarantee brake-pressed state resolves before torque is set (Lohstroh et al., Section 2.2).
