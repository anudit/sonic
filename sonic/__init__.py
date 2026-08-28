"""Sonic S1 — design-space tooling.

Single source of truth for every number in the S1 plan. If a figure appears in
sonic-s1-plan.html, it is computed here and locked by tests/.
"""

from .modelspec import ModelSpec, load
from .chipspec import ChipSpec, S1, SKUS
from .moe import distinct_experts, spec_decode_gain, batch_gain, occupancy
from .roofline import decode, prefill, power, area

__all__ = [
    "ModelSpec", "load", "ChipSpec", "S1", "SKUS",
    "distinct_experts", "spec_decode_gain", "batch_gain", "occupancy",
    "decode", "prefill", "power", "area",
]
