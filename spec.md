# spec.md — GPU Determinism Benchmark

## 1. Problem

GPUs are incredibly powerful SIMD tools in computing, essential to almost any modern AI system. However, this comes with a caveat: GPUs can produce non-deterministic results at three phases — application-level time and processing allocation, thread scheduling, and SM (Streaming Multiprocessor) assignment. Surprisingly, even when controlling for the first two, the third factor introduces non-determinism on its own. SMs are not assigned work in a simple round-robin fashion — depending on the warp counts of competing kernels, the block scheduler can stack multiple blocks onto an already-occupied SM and leave other SMs idle entirely, a behavior empirically documented by Olmedo et al. (RTAS 2020).

Non-determinism is not the only issue — latency is a real cost too. When two different kernels execute on SMs within the same GPC (Graphics Processing Cluster), they share a single bus to L2 memory. If the two kernels are doing unrelated work, that bus carries twice the traffic with zero reuse benefit. Olmedo et al. measured this directly and found that intra-GPC memory contention can increase a kernel's worst-case execution time by up to 10x.

This is a real problem, especially in safety-critical systems. An inconsistent floating-point result or an unpredictable latency spike can cause a critical misjudgment — a missed brake deadline, a delayed surgical robot response. Deterministic chips exist — Groq's LPU, for instance — but they trade away raw throughput to get there. The question I aim to answer is: which kernel engineering techniques reduce non-determinism in multi-stream execution, compared to a deterministic single-stream baseline, without giving up the throughput multi-stream exists to provide? A necessary first step toward that question is knowing *where* non-determinism actually concentrates — which kernels, at which points in generation, at which tensor sizes — since a hybrid policy can't be designed against a single aggregate number. If no mitigating technique exists once that picture is built, the empirical latency-vs-determinism tradeoff is itself a valid finding.

## 2. Workload

GPT-2 Small (117M parameters) on the lab's NVIDIA Thor, running full autoregressive generation: a prefill pass over a fixed 20-token input sentence, followed by token-by-token decode for a fixed number of steps. Fixed random seed throughout. The decode loop is required (not a single forward pass) because two of the three breakdown axes below — sequence position and tensor size growth — only exist once the KV cache grows over multiple steps.

## 3. Reference Schedule

The reference schedule is expressed as a Lingua Franca (LF) reactor program. LF is natively deterministic by construction — it organizes computation into reactors connected by a dependency graph, and assigns each reaction a logical time tag based on that graph. A reactor that depends on another's output is guaranteed a strictly later logical time, and for a fixed input, the same logical-time schedule is produced on every run. This isn't an empirical observation but a property of the language itself, which is what makes it usable as ground truth rather than just another reference measurement.

Kernel-level data is captured via CUPTI (CUDA Profiling Tools Interface), which timestamps every individual GPU kernel dispatch across the full prefill + decode run — every matmul, softmax, layer norm, and embedding lookup kernel, at every decode step. Each captured event carries: kernel name, wall-clock timestamp, logical tag, the decode step it occurred in, and the size (bytes) of its primary tensor operand. G* is built from 5 runs under fully controlled conditions: single CUDA stream, `cudnn.deterministic=True`, `cudnn.benchmark=False`, fixed seed and shape. G* is only accepted if all 5 runs produce an identical kernel sequence with identical logical tags — checked programmatically before any DDI computation proceeds. If this check fails, the harness is debugged before anything is built on top of it.

**Caveat:** Block-to-SM assignment is performed by an internal hardware scheduler with no CUDA API to control or query it directly. LF's reference schedule can't model that decision directly — it can only define the expected kernel order and timing as a whole. What can be influenced indirectly are the conditions that make non-round-robin SM assignment and inter-kernel contention more or less likely, through choices like stream count and synchronization placement.

## 4. DDI Formulas

Two separate metrics are tracked, since timing and ordering can deviate independently of each other.

**Timing DDI** — how much longer or shorter each kernel took versus expected (physical-time jitter):
```
Timing DDI = (1/N) × Σ |t_actual(i) - t_ref(i)| / t_ref(i)
```

**Order DDI** — how much the actual kernel execution order deviates from the expected order (logical-time deviation). This is computed as a normalized count of pairwise inversions (Kendall's tau distance) rather than per-kernel rank distance, because rank distance over-penalizes a uniform rotation (e.g. the sequence `10,1,2,...,9`, which preserves every pairwise relationship except one) as harshly as a fully scrambled order, which misrepresents how "broken" the schedule actually is:
```
Order DDI = (number of pairwise order inversions) / (N choose 2)
```

N is the number of matched kernels in a run (from CUPTI). A kernel/run can score well on one metric and poorly on the other — they are never combined into one number.

## 5. Breakdown Axes

A single aggregate value for either DDI says that non-determinism occurred, but not where or why — which makes it unusable for designing a targeted hybrid policy. Both DDIs are therefore additionally computed broken down across three independent axes:

- **Sequence position** — DDI as a function of decode step / token index. Mirrors the CPU-level attention divergence simulation's finding that numerical divergence is highest early in generation; tests whether GPU scheduling divergence follows the same shape, a different shape, or none at all.
- **Kernel type** — DDI grouped by kernel name (attention, FFN, layer norm, embedding, etc.), independent of position. Likely to render as a bar chart rather than a curve, given the small number of distinct kernel types in GPT-2. This is the most directly actionable axis for a hybrid policy: a per-kernel-type rule ("always run attention deterministically") is simpler to implement and reason about than a position-dependent threshold.
- **Tensor size / memory footprint** — DDI as a function of the primary operand's byte size. Tests whether jitter and order violations are driven by memory bandwidth contention (per the GPC-to-L2 bus mechanism in Section 1) rather than by position or kernel identity, which would be a meaningfully different causal story.

This produces six plots in total (Timing DDI × 3 axes, Order DDI × 3 axes), all generated from the same underlying captured dataset — no additional measurement infrastructure is needed beyond tagging each kernel event with all three metadata fields at capture time.

## 6. Latency Definition

Latency is defined as end-to-end wall-clock time for the full prefill + decode run, measured via `cudaEventElapsedTime` from first kernel dispatch to last kernel completion. Every config comparison is reported as a (Timing DDI, Order DDI, Latency) triple, never collapsed into a single score.

## 7. Configs to Test

- **Config A — Single stream, synchronous.** Baseline. Expected lowest DDI on both metrics, lowest throughput.
- **Config B — Two streams, overlapped.** Expected higher Order DDI specifically, due to cross-stream kernel interleaving. Primary target for kernel-engineering mitigation.
- **Config C — Single stream with background memory copy.** Isolates memory contention's effect on Timing DDI specifically, without introducing stream-level order non-determinism.

*Config A is the only configuration in active scope until the full capture-and-breakdown pipeline is validated end-to-end on Thor.*
