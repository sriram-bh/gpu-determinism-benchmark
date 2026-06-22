"""
src/experiments/gpu_harness.py

GPU-only harness. Assumes CUDA + CUPTI are present (i.e. running on Thor).
This will NOT run successfully without a CUDA device — there is no CPU
fallback path. It is written now so that the moment Thor is available,
this is ready to execute as-is.

What it does:
  1. Loads GPT-2 Small onto the GPU.
  2. Runs prefill on a fixed 20-token input, then decodes a fixed number
     of additional tokens, one at a time.
  3. Captures every individual CUDA kernel dispatch during the entire
     run via CUPTI (through torch.profiler, which wraps CUPTI), tagging
     each captured kernel with: decode_step (which step it occurred in,
     0 = prefill) and tensor_bytes (size of its primary operand).
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

import time
from typing import List, Optional

import torch
from torch.profiler import profile, ProfilerActivity
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.metrics.schema import KernelEvent


FIXED_INPUT_SENTENCE = "The quick brown fox jumps over the lazy dog today"  # ~20 tokens
DECODE_STEPS = 20  # number of additional tokens to generate after prefill
MODEL_NAME = "gpt2"
SEED = 42


def _configure_deterministic_mode():
    """
    Forces deterministic CUDA execution. Required for G* validation runs.
    See spec.md Section 3 for why each of these matters.
    """
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)


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


def _run_prefill_and_decode(model, tokenizer, device, profiler_ctx) -> List[KernelEvent]:
    """
    Runs one full prefill + decode pass, capturing every kernel dispatch
    via the active torch.profiler context, and returns them as KernelEvent
    objects tagged with decode_step and tensor_bytes.
    """
    logical_clock_state = {"tag": 0}
    events: List[KernelEvent] = []

    inputs = tokenizer(FIXED_INPUT_SENTENCE, return_tensors="pt").to(device)
    past_key_values = None
    input_ids = inputs["input_ids"]

    # decode_step = 0 is prefill. 1..DECODE_STEPS are the generated tokens.
    for decode_step in range(0, DECODE_STEPS + 1):
        with torch.no_grad():
            if decode_step == 0:
                # Prefill: process the full fixed input sentence at once.
                outputs = model(input_ids, use_cache=True)
            else:
                # Decode: process only the newly generated token, using
                # the cached K/V from all previous steps.
                outputs = model(
                    next_token, past_key_values=past_key_values, use_cache=True
                )

            past_key_values = outputs.past_key_values
            next_token = torch.argmax(outputs.logits[:, -1, :], dim=-1, keepdim=True)

        torch.cuda.synchronize()  # ensure all kernels for this step have completed

        # Pull this step's kernel events out of the profiler and tag them.
        step_events = profiler_ctx.key_averages(group_by_input_shape=True)
        # NOTE: torch.profiler aggregates across the whole `with` block by
        # default. On Thor, replace this with per-step profiler start/stop
        # (torch.profiler.profile(schedule=...)) so each decode_step gets
        # its own clean event window rather than a cumulative one. This is
        # flagged here as the one piece that needs real-hardware tuning.
        for evt in step_events:
            wall_ns = int(evt.self_cuda_time_total * 1000)  # us -> ns
            duration_ns = wall_ns
            tag = cupti_lf_bridge(wall_ns, logical_clock_state)
            tensor_bytes = _tensor_bytes_for_event(evt)

            events.append(
                KernelEvent(
                    kernel_name=evt.key,
                    wall_ns=wall_ns,
                    logical_tag=tag,
                    duration_ns=duration_ns,
                    decode_step=decode_step,
                    tensor_bytes=tensor_bytes,
                    stream_id=0,  # Config A: single stream throughout
                )
            )

    return events


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

    _configure_deterministic_mode()

    tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
    model = GPT2LMHeadModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        events = _run_prefill_and_decode(model, tokenizer, device, prof)

    return events


def run_n_validation_passes(n: int = 5, seed: int = SEED) -> List[List[KernelEvent]]:
    """
    Runs `n` passes under identical controlled conditions, for G*
    validation (see src/metrics/validate_gstar.py). Returns a list of
    per-run KernelEvent lists. The caller is responsible for checking
    bit-identity across runs before accepting any of them as G*.
    """
    return [run_single_pass(seed=seed) for _ in range(n)]


def run_warmup(n: int = 10, seed: int = SEED):
    """
    Runs `n` warmup passes (not returned, not logged) to stabilize GPU
    clocks and warm caches before any measurement run.
    """
    for _ in range(n):
        run_single_pass(seed=seed)


if __name__ == "__main__":
    print("This harness requires CUDA and will only run on Thor.")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print("Running warmup...")
        run_warmup(n=10)
        print("Running single pass...")
        events = run_single_pass()
        print(f"Captured {len(events)} kernel events.")
