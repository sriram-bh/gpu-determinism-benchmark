"""
src/experiments/gpu_harness.py

GPU-only harness. Assumes CUDA + CUPTI are present (i.e. running on Thor).
This will NOT run successfully without a CUDA device — there is no CPU
fallback path.

What it does:
  1. Loads GPT-2 Small onto the GPU.
  2. Runs prefill on a fixed 20-token input, then decodes a fixed number
     of additional tokens, one at a time.
  3. Captures every individual CUDA kernel dispatch during each step
     via CUPTI (through torch.profiler), tagging each captured kernel
     with: decode_step (which step it occurred in, 0 = prefill) and
     tensor_bytes (size of its primary operand).
  4. Lifts each kernel's wall-clock timestamp into an LF logical tag via
     the CUPTI -> LF bridge (see cupti_lf_bridge in this same file).
  5. Returns a List[KernelEvent] for the run, in the shared schema format
     defined in src/metrics/schema.py.

Usage:
    from gpu_harness import run_single_pass, run_n_validation_passes

    events = run_single_pass(seed=42)
    # events is a List[KernelEvent], ready to feed into the DDI module
    # or to save as a candidate G*.
"""

from typing import List, Tuple

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.metrics.schema import KernelEvent


FIXED_INPUT_SENTENCE = "The quick brown fox jumps over the lazy dog today"  # ~20 tokens
DECODE_STEPS = 40  # number of additional tokens to generate after prefill
MODEL_NAME = "gpt2"
SEED = 42


def _configure_deterministic_mode(seed: int = SEED):
    """
    Forces deterministic CUDA execution. Required for G* validation runs.
    See spec.md Section 3 for why each of these matters.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cupti_lf_bridge(wall_ns: int, logical_clock_state: dict) -> int:
    """
    Lifts a physical CUPTI timestamp into an LF logical tag.

    The logical clock advances monotonically with each kernel dispatch,
    independent of the actual wall-clock gap between dispatches. This is
    what makes the logical tag immune to OS scheduling noise: it only
    cares about ORDER of dispatch, not the physical time between events.

    logical_clock_state is a mutable dict {"tag": int} threaded through
    the capture loop, incremented by one microstep (1 ns, by convention
    here) per kernel dispatched.
    """
    logical_clock_state["tag"] += 1
    return logical_clock_state["tag"]


def _tensor_bytes_for_event(evt) -> int:
    """
    Estimates the size in bytes of a kernel's primary operand from a
    torch.profiler event record. Falls back to 0 if shape info isn't
    available for this event (some kernels, e.g. synchronization
    barriers, have no tensor operand at all).
    """
    try:
        shapes = evt.input_shapes
        if not shapes:
            return 0
        # Take the largest input tensor as the "primary operand"
        max_elems = max(
            (1 if not s else _product(s)) for s in shapes if s is not None
        )
        # Assume float32 (4 bytes/elem) unless dtype info says otherwise.
        # GPT-2 runs in float32 by default per our deterministic config.
        return max_elems * 4
    except Exception:
        return 0


def _product(shape) -> int:
    p = 1
    for dim in shape:
        p *= max(dim, 1)
    return p


def _load_model(device):
    """Load GPT-2 and its tokenizer onto the given device."""
    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()
    return model, tokenizer


def _run_prefill_and_decode(model, tokenizer, device) -> Tuple[List[KernelEvent], List[int]]:
    """
    Runs one full prefill + decode pass with per-step profiling,
    capturing every kernel dispatch and returning them as KernelEvent
    objects tagged with decode_step and tensor_bytes.

    Returns (events, token_ids) where token_ids is the list of generated
    token IDs at each step (length DECODE_STEPS + 1).

    Each decode step gets its own profiler context so events are cleanly
    separated per step — no cumulative aggregation across the loop.
    Uses events() (individual dispatch records) not key_averages()
    (which aggregates by name, destroying per-dispatch identity).
    """
    logical_clock_state = {"tag": 0}
    all_events: List[KernelEvent] = []
    token_ids: List[int] = []

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    past_key_values = None
    input_ids = inputs["input_ids"]

    for decode_step in range(0, DECODE_STEPS + 1):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
        ) as step_prof:
            with torch.no_grad():
                if decode_step == 0:
                    outputs = model(input_ids, use_cache=True)
                else:
                    outputs = model(
                        next_token, past_key_values=past_key_values, use_cache=True
                    )

                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
                token_ids.append(next_token.item())

            torch.cuda.synchronize()

        step_events = sorted(step_prof.events(), key=lambda e: e.time_range.start)
        for evt in step_events:
            cuda_time_us = evt.self_device_time_total
            if cuda_time_us <= 0:
                continue

            duration_ns = int(cuda_time_us * 1000)  # µs -> ns
            wall_ns = int(evt.time_range.start * 1000)  # µs -> ns
            tag = cupti_lf_bridge(wall_ns, logical_clock_state)
            tensor_bytes = _tensor_bytes_for_event(evt)

            all_events.append(
                KernelEvent(
                    kernel_name=evt.key,
                    wall_ns=wall_ns,
                    logical_tag=tag,
                    duration_ns=duration_ns,
                    decode_step=decode_step,
                    tensor_bytes=tensor_bytes,
                    stream_id=0,
                )
            )

    return all_events, token_ids


def run_single_pass(seed: int = SEED) -> List[KernelEvent]:
    """
    Runs one full prefill + decode pass under Config A (single stream,
    synchronous, deterministic) and returns the captured KernelEvents.
    """
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. This harness requires a GPU "
            "(NVIDIA Thor) and will not run on CPU."
        )

    _configure_deterministic_mode(seed)

    model, tokenizer = _load_model(device)
    events, _token_ids = _run_prefill_and_decode(model, tokenizer, device)
    return events


def run_n_validation_passes(n: int = 5, seed: int = SEED) -> Tuple[List[List[KernelEvent]], List[List[int]]]:
    """
    Runs `n` passes under identical controlled conditions, for G*
    validation (see src/metrics/ddi.py:validate_gstar). Returns
    (all_event_lists, all_token_lists) — one per run. The caller is
    responsible for checking bit-identity across runs before accepting
    any of them as G*.

    Loads the model once and reuses it across all passes.
    """
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA device available. This harness requires a GPU "
            "(NVIDIA Thor) and will not run on CPU."
        )

    _configure_deterministic_mode(seed)
    model, tokenizer = _load_model(device)
    all_events = []
    all_tokens = []
    for _ in range(n):
        events, token_ids = _run_prefill_and_decode(model, tokenizer, device)
        all_events.append(events)
        all_tokens.append(token_ids)
    return all_events, all_tokens


def run_warmup(model, tokenizer, device, n: int = 3):
    """
    Runs `n` warmup passes (unprofiled) to stabilize GPU clocks and
    warm caches before any measurement run.
    """
    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    for _ in range(n):
        past_key_values = None
        input_ids = inputs["input_ids"]
        for step in range(DECODE_STEPS + 1):
            with torch.no_grad():
                if step == 0:
                    outputs = model(input_ids, use_cache=True)
                else:
                    outputs = model(
                        next_token, past_key_values=past_key_values, use_cache=True
                    )
                past_key_values = outputs.past_key_values
                next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)
            torch.cuda.synchronize()


if __name__ == "__main__":
    from collections import Counter

    print("GPU Determinism Benchmark Harness")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if not torch.cuda.is_available():
        print("No CUDA device. Exiting.")
        sys.exit(1)

    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(device)}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"PyTorch version: {torch.__version__}")

    _configure_deterministic_mode()
    model, tokenizer = _load_model(device)

    print(f"\nRunning 3 warmup passes (unprofiled)...")
    run_warmup(model, tokenizer, device, n=3)

    print(f"\nRunning single profiled pass ({DECODE_STEPS} decode steps)...")
    events, token_ids = _run_prefill_and_decode(model, tokenizer, device)

    print(f"\nCaptured {len(events)} kernel events total.")

    step_counts = Counter(e.decode_step for e in events)
    print(f"\nEvents per decode step:")
    for step in sorted(step_counts):
        label = "prefill" if step == 0 else f"decode {step}"
        print(f"  {label}: {step_counts[step]} kernels")

    name_counts = Counter(e.kernel_name for e in events)
    print(f"\nDistinct kernel names: {len(name_counts)}")
    print("Top 10 most frequent:")
    for name, count in name_counts.most_common(10):
        print(f"  {count:4d}x  {name}")

    print(f"\nGenerated tokens: {token_ids}")
    print(f"Decoded text: {tokenizer.decode(token_ids)}")

    if events:
        prefill_events = [e for e in events if e.decode_step == 0]
        decode1_events = [e for e in events if e.decode_step == 1]
        if prefill_events:
            e = prefill_events[0]
            print(f"\nSample prefill event:")
            print(f"  name={e.kernel_name}, duration_ns={e.duration_ns}, "
                  f"wall_ns={e.wall_ns}, tensor_bytes={e.tensor_bytes}")
        if decode1_events:
            e = decode1_events[0]
            print(f"Sample decode-1 event:")
            print(f"  name={e.kernel_name}, duration_ns={e.duration_ns}, "
                  f"wall_ns={e.wall_ns}, tensor_bytes={e.tensor_bytes}")
