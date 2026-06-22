"""
src/metrics/ddi.py

DDI computation: Timing DDI and Order DDI, each breakable down across
three axes (sequence position, kernel type, tensor size).

Two metrics, computed independently and never combined:

  Timing DDI = (1/N) * sum( |t_actual(i) - t_ref(i)| / t_ref(i) )
      Physical-time jitter: kernels in the right order, wrong duration.

  Order DDI = (pairwise inversions) / (N choose 2)
      Logical-time deviation: kernels in the wrong relative sequence.
      Uses Kendall's tau distance (pairwise inversion count) rather than
      per-kernel absolute rank distance, because rank distance over-
      penalizes a uniform rotation (e.g. 10,1,2,...,9 -- which preserves
      every pairwise relationship except one) nearly as harshly as a
      fully scrambled order. Counting pairwise inversions instead gives
      a rotation a low score, which honestly reflects how mild a failure
      it actually is.

Both metrics support breakdown by:
  - sequence position (decode_step)
  - kernel type (kernel_name)
  - tensor size (tensor_bytes, bucketed)

All breakdowns operate on the same matched event list -- no separate
capture or measurement is needed per axis.
"""

import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Callable
from collections import defaultdict

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.metrics.schema import KernelEvent


# ─────────────────────────────────────────────────────────────────────────────
# ALIGNMENT: match an observed run's kernels to G*'s kernels
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MatchedPair:
    observed: KernelEvent
    reference: KernelEvent
    ref_rank: int       # this kernel's position in G*'s sequence (0-indexed)
    observed_rank: int  # this kernel's position in the observed sequence


def align_events(observed: List[KernelEvent], reference: List[KernelEvent]) -> List[MatchedPair]:
    """
    Aligns an observed run's kernel events to G*'s reference events.

    Matching strategy: greedy, by kernel_name, walking both sequences in
    their own recorded order. If a kernel name appears multiple times
    (very common -- e.g. many matmuls), matches are consumed in order of
    appearance, so the i-th occurrence of "matmul" in the observed run
    matches the i-th occurrence of "matmul" in G*, not an arbitrary one.

    Kernels in `observed` with no remaining match in `reference` are
    skipped (not included in the matched list) -- these are kernels the
    GPU dispatched that G* didn't expect, and are reported separately
    as a warning rather than silently inflating the DDI.
    """
    # Build a queue of (original_index, event) per kernel name in reference
    ref_queues: Dict[str, List[Tuple[int, KernelEvent]]] = defaultdict(list)
    for idx, evt in enumerate(reference):
        ref_queues[evt.kernel_name].append((idx, evt))

    matched: List[MatchedPair] = []
    unmatched_observed = 0

    for obs_idx, obs_evt in enumerate(observed):
        queue = ref_queues.get(obs_evt.kernel_name)
        if queue:
            ref_idx, ref_evt = queue.pop(0)
            matched.append(
                MatchedPair(
                    observed=obs_evt,
                    reference=ref_evt,
                    ref_rank=ref_idx,
                    observed_rank=obs_idx,
                )
            )
        else:
            unmatched_observed += 1

    if unmatched_observed > 0:
        print(f"  [align_events] WARNING: {unmatched_observed} observed kernels "
              f"had no match in G* (GPU dispatched something unexpected).")

    leftover_ref = sum(len(q) for q in ref_queues.values())
    if leftover_ref > 0:
        print(f"  [align_events] WARNING: {leftover_ref} G* kernels were never "
              f"matched in the observed run (GPU skipped or merged something).")

    return matched


# ─────────────────────────────────────────────────────────────────────────────
# TIMING DDI
# ─────────────────────────────────────────────────────────────────────────────

def compute_timing_ddi(matched: List[MatchedPair]) -> float:
    """
    Timing DDI = (1/N) * sum( |t_actual - t_ref| / t_ref )

    Computed over duration_ns (kernel execution duration), not wall_ns
    (absolute start time) -- duration is the meaningful "did this kernel
    take longer/shorter than expected" quantity. Absolute wall-clock
    start time is not comparable across runs (every run starts at its
    own t=0).
    """
    if not matched:
        raise ValueError("No matched kernels -- cannot compute Timing DDI")

    deviations = []
    for pair in matched:
        t_actual = pair.observed.duration_ns
        t_ref = pair.reference.duration_ns
        if t_ref == 0:
            # Avoid division by zero for instantaneous/zero-cost reference kernels
            deviations.append(abs(t_actual - t_ref) / 1.0)
        else:
            deviations.append(abs(t_actual - t_ref) / t_ref)

    return sum(deviations) / len(deviations)


# ─────────────────────────────────────────────────────────────────────────────
# ORDER DDI (Kendall's tau / pairwise inversions)
# ─────────────────────────────────────────────────────────────────────────────

def compute_order_ddi(matched: List[MatchedPair]) -> float:
    """
    Order DDI = (number of pairwise inversions) / (N choose 2)

    For every pair of matched kernels (i, j), check whether their
    relative order in the observed run matches their relative order in
    G*. If kernel i comes before kernel j in G* (ref_rank_i < ref_rank_j)
    but kernel j came before kernel i in the observed run
    (observed_rank_j < observed_rank_i), that pair is an inversion.

    O(N^2) pairwise comparison. Fine for N in the hundreds (one GPT-2
    forward pass's kernel count); would need a merge-sort-based O(N log N)
    inversion count if N grows into the tens of thousands.
    """
    n = len(matched)
    if n < 2:
        return 0.0  # no pairs to compare -- trivially zero deviation

    inversions = 0
    total_pairs = n * (n - 1) // 2

    for i in range(n):
        for j in range(i + 1, n):
            ref_order_i_before_j = matched[i].ref_rank < matched[j].ref_rank
            obs_order_i_before_j = matched[i].observed_rank < matched[j].observed_rank
            if ref_order_i_before_j != obs_order_i_before_j:
                inversions += 1

    return inversions / total_pairs


# ─────────────────────────────────────────────────────────────────────────────
# MULTI-AXIS BREAKDOWN
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_size_bucket(tensor_bytes: int) -> str:
    """
    Buckets tensor size into human-readable ranges for the breakdown axis.
    Adjust bucket boundaries once real Thor data shows the actual range
    of tensor sizes GPT-2 produces -- these are placeholder boundaries.
    """
    kb = tensor_bytes / 1024
    if kb < 1:
        return "<1KB"
    elif kb < 10:
        return "1-10KB"
    elif kb < 100:
        return "10-100KB"
    elif kb < 1000:
        return "100KB-1MB"
    else:
        return ">1MB"


def breakdown_by_axis(
    matched: List[MatchedPair],
    axis: str,
    metric_fn: Callable[[List[MatchedPair]], float],
) -> Dict[str, float]:
    """
    Groups matched pairs by the given axis, then computes the given
    metric (compute_timing_ddi or compute_order_ddi) independently
    within each group.

    axis: one of "decode_step", "kernel_type", "tensor_size"

    Returns: {group_key: metric_value}

    Note: Order DDI within a group is computed using each kernel's rank
    WITHIN that group, not its global rank in the full run. This is the
    correct interpretation for "how scrambled is the order among kernels
    of this type/position/size" -- comparing a position-3 kernel's global
    rank against a position-50 kernel's would conflate two different
    questions.
    """
    if axis not in ("decode_step", "kernel_type", "tensor_size"):
        raise ValueError(f"Unknown axis: {axis}")

    groups: Dict[str, List[MatchedPair]] = defaultdict(list)

    for pair in matched:
        if axis == "decode_step":
            key = str(pair.reference.decode_step)
        elif axis == "kernel_type":
            key = pair.reference.kernel_name
        else:  # tensor_size
            key = _tensor_size_bucket(pair.reference.tensor_bytes)
        groups[key].append(pair)

    results = {}
    for key, group_pairs in groups.items():
        if metric_fn is compute_order_ddi:
            # Re-rank within this group only, so Order DDI reflects local
            # ordering, not global position.
            re_ranked = _rerank_within_group(group_pairs)
            results[key] = metric_fn(re_ranked)
        else:
            results[key] = metric_fn(group_pairs)

    return results


def _rerank_within_group(group_pairs: List[MatchedPair]) -> List[MatchedPair]:
    """
    Reassigns ref_rank and observed_rank within a group, based on the
    group's own internal ordering (sorted by original global rank),
    so Order DDI computed on this group reflects local scrambling only.
    """
    sorted_by_ref = sorted(group_pairs, key=lambda p: p.ref_rank)
    ref_rank_map = {id(p): i for i, p in enumerate(sorted_by_ref)}

    sorted_by_obs = sorted(group_pairs, key=lambda p: p.observed_rank)
    obs_rank_map = {id(p): i for i, p in enumerate(sorted_by_obs)}

    return [
        MatchedPair(
            observed=p.observed,
            reference=p.reference,
            ref_rank=ref_rank_map[id(p)],
            observed_rank=obs_rank_map[id(p)],
        )
        for p in group_pairs
    ]


def full_breakdown_report(matched: List[MatchedPair]) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Computes all six breakdowns at once: {Timing, Order} x {decode_step,
    kernel_type, tensor_size}. This is the function most callers should
    use -- it produces everything needed for the six-plot report in one
    call over one matched dataset.

    Returns:
    {
        "timing": {
            "decode_step": {...}, "kernel_type": {...}, "tensor_size": {...}
        },
        "order": {
            "decode_step": {...}, "kernel_type": {...}, "tensor_size": {...}
        }
    }
    """
    axes = ["decode_step", "kernel_type", "tensor_size"]
    return {
        "timing": {axis: breakdown_by_axis(matched, axis, compute_timing_ddi) for axis in axes},
        "order": {axis: breakdown_by_axis(matched, axis, compute_order_ddi) for axis in axes},
    }


# ─────────────────────────────────────────────────────────────────────────────
# G* VALIDATION
# ─────────────────────────────────────────────────────────────────────────────

def validate_gstar(runs: List[List[KernelEvent]]) -> bool:
    """
    Validates G* by checking that all runs produce an identical kernel
    sequence (same names, same logical tags, in the same order). At
    least 2 runs required; 5 is the project standard.
    """
    if len(runs) < 2:
        raise ValueError("Need at least 2 runs to validate G*")

    reference = runs[0]
    ref_signature = [(e.kernel_name, e.logical_tag) for e in reference]

    all_identical = True
    for i, run in enumerate(runs[1:], start=2):
        run_signature = [(e.kernel_name, e.logical_tag) for e in run]
        if run_signature != ref_signature:
            print(f"  [VALIDATION FAIL] Run {i} differs from run 1.")
            for j, (a, b) in enumerate(zip(ref_signature, run_signature)):
                if a != b:
                    print(f"    First diff at kernel {j}: expected {a}, got {b}")
                    break
            all_identical = False

    if all_identical:
        print(f"  [VALIDATION PASS] All {len(runs)} runs identical "
              f"({len(reference)} kernels each). G* certified.")
    else:
        print("  [VALIDATION FAIL] G* is NOT certified. Debug the harness "
              "(check deterministic flags, fixed seed, single stream) "
              "before computing any DDI.")

    return all_identical


# ─────────────────────────────────────────────────────────────────────────────
# SELF-TEST (synthetic data, no GPU needed)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("DDI module self-test (synthetic data)...\n")

    # --- Test 1: perfect match -> both DDIs should be 0 ---
    ref = [
        KernelEvent(f"kernel_{i}", wall_ns=i * 1000, logical_tag=i,
                    duration_ns=100, decode_step=0, tensor_bytes=1024)
        for i in range(10)
    ]
    obs_perfect = [
        KernelEvent(e.kernel_name, wall_ns=e.wall_ns, logical_tag=e.logical_tag,
                    duration_ns=e.duration_ns, decode_step=e.decode_step,
                    tensor_bytes=e.tensor_bytes)
        for e in ref
    ]
    matched = align_events(obs_perfect, ref)
    assert compute_timing_ddi(matched) == 0.0
    assert compute_order_ddi(matched) == 0.0
    print("Test 1 (perfect match): Timing DDI=0, Order DDI=0 -- PASS")

    # --- Test 2: rotation should score LOW on Order DDI, not high ---
    # observed order: kernel_9, kernel_0, kernel_1, ..., kernel_8
    rotated_names = ["kernel_9"] + [f"kernel_{i}" for i in range(9)]
    obs_rotated = [
        KernelEvent(name, wall_ns=i * 1000, logical_tag=i, duration_ns=100,
                    decode_step=0, tensor_bytes=1024)
        for i, name in enumerate(rotated_names)
    ]
    matched_rot = align_events(obs_rotated, ref)
    order_ddi_rotation = compute_order_ddi(matched_rot)
    print(f"Test 2 (rotation): Order DDI = {order_ddi_rotation:.3f} "
          f"(expect ~0.20, i.e. 9 inversions / 45 pairs)")
    assert abs(order_ddi_rotation - (9 / 45)) < 1e-6

    # --- Test 3: full reversal should score HIGH (near 1.0) ---
    obs_reversed = list(reversed(obs_perfect))
    matched_rev = align_events(obs_reversed, ref)
    order_ddi_reversal = compute_order_ddi(matched_rev)
    print(f"Test 3 (full reversal): Order DDI = {order_ddi_reversal:.3f} "
          f"(expect 1.0 -- every pair inverted)")
    assert order_ddi_reversal == 1.0

    # --- Test 4: 10% timing jitter ---
    obs_jittered = [
        KernelEvent(e.kernel_name, wall_ns=e.wall_ns, logical_tag=e.logical_tag,
                    duration_ns=int(e.duration_ns * 1.10), decode_step=e.decode_step,
                    tensor_bytes=e.tensor_bytes)
        for e in ref
    ]
    matched_jit = align_events(obs_jittered, ref)
    timing_ddi_jitter = compute_timing_ddi(matched_jit)
    print(f"Test 4 (10% jitter): Timing DDI = {timing_ddi_jitter:.3f} (expect ~0.10)")
    assert abs(timing_ddi_jitter - 0.10) < 1e-6

    print("\nAll self-tests passed.")
