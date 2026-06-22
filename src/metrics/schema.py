"""
src/metrics/schema.py

Shared data schema for a single captured GPU kernel event.

Every kernel dispatched during a run — whether captured by real CUPTI on
Thor or compared against G* — is represented as one KernelEvent. All three
programs (the GPU harness, the DDI computation code, and the LF reference
schedule comparison) agree on this exact shape, so data flows between them
without translation.

Fields:
    kernel_name   : str   — CUPTI-reported kernel name (e.g. "volta_sgemm_128x64_nn")
    wall_ns       : int   — wall-clock start timestamp in nanoseconds (from CUPTI)
    logical_tag   : int   — LF logical time tag in nanoseconds, assigned via the
                            CUPTI -> LF bridge at the moment of dispatch
    duration_ns   : int   — kernel execution duration in nanoseconds
    decode_step   : int   — which decode step this kernel belongs to.
                            0 = prefill. 1, 2, 3... = decode step index.
                            This is the "sequence position" axis.
    tensor_bytes  : int   — size in bytes of the kernel's primary operand
                            (the largest input/output tensor it touches).
                            This is the "tensor size / memory footprint" axis.
    stream_id     : int   — CUDA stream ID (0 = default stream)

kernel_name doubles as the "kernel type" axis directly — no separate field
needed, since grouping by kernel_name IS grouping by kernel type.
"""

from dataclasses import dataclass, asdict
from typing import List, Dict


@dataclass
class KernelEvent:
    kernel_name: str
    wall_ns: int
    logical_tag: int
    duration_ns: int
    decode_step: int
    tensor_bytes: int
    stream_id: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: Dict) -> "KernelEvent":
        return KernelEvent(**d)


def events_to_dicts(events: List[KernelEvent]) -> List[Dict]:
    """Convert a list of KernelEvent to a list of plain dicts, for JSON/CSV export."""
    return [e.to_dict() for e in events]


def dicts_to_events(dicts: List[Dict]) -> List[KernelEvent]:
    """Convert a list of plain dicts (e.g. loaded from JSON) back to KernelEvent objects."""
    return [KernelEvent.from_dict(d) for d in dicts]
