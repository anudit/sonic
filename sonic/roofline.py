"""Analytical performance, power and area model for Sonic S1.

The plan's headline numbers all come out of here. Deliberately first-order: the
point is that every figure is traceable to an assumption you can edit in
chipspec.py, not that it is signoff-accurate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .chipspec import AREA_FIXED, PHY_AREA, ChipSpec
from .modelspec import ModelSpec


@dataclass(frozen=True)
class Decode:
    tok_s: float
    mb_per_token: float
    watts: float
    mj_per_token: float
    lanes_needed: float


def decode(model: ModelSpec, chip: ChipSpec, ctx: int = 4096) -> Decode:
    """Batch-1 decode. Purely memory-bound -- the array is ~3% utilised."""
    mb = model.bytes_per_token() + model.kv_mb(ctx)
    tok_s = chip.eff_gbps * 1000.0 / mb
    w = power(model, chip, tok_s)
    return Decode(tok_s=tok_s, mb_per_token=mb, watts=w,
                  mj_per_token=w / tok_s * 1000.0,
                  lanes_needed=chip.lanes_to_saturate())


def power(model: ModelSpec, chip: ChipSpec, tok_s: float) -> float:
    """Decode power. DRAM traffic dominates; the array is a rounding error."""
    gbps = tok_s * model.bytes_per_token() / 1000.0
    p_dram = gbps * 1e9 * 8 * chip.e_dram_bit
    p_mac = model.gop_per_token() * 1e9 * tok_s * chip.e_mac_op
    return p_dram + p_mac + chip.p_leak_w + chip.p_stream_w + chip.p_vector_w


@dataclass(frozen=True)
class Prefill:
    ttft_ms: float
    compute_ms: float
    attn_ms: float
    memory_ms: float
    bound: str
    chunks: int
    mb_per_token: float


def prefill(model: ModelSpec, chip: ChipSpec, seq: int,
            chunk: int = 2048, boost: bool = False) -> Prefill:
    """Chunked prefill under the expert-major schedule.

    Expert-major means each expert is read exactly once per chunk per layer, so
    memory cost is `resident / chunk` per token rather than the full sweep per
    token batch. Compute is the linear term plus the quadratic attention term,
    the latter discounted by the flash-attention engine.
    """
    ghz = chip.boost_ghz if boost else chip.clk_ghz
    tops = chip.mac_lanes * 2 * ghz / 1e3

    chunks = math.ceil(seq / chunk)
    # A chunk large enough to touch every expert reads the whole weight set;
    # a very short prompt touches proportionally fewer.
    if model.is_moe:
        from .moe import distinct_experts
        frac = distinct_experts(model.n_experts, model.top_k, min(seq, chunk)) / model.n_experts
        exp_tot = model.expert_total_mb()
        sweep = exp_tot * frac + (model.resident_mb() - exp_tot)
    else:
        sweep = model.resident_mb()

    memory_ms = chunks * sweep / (chip.eff_gbps * 1e3) * 1e3
    compute_ms = model.linear_gop(seq) / (tops * 1e3) * 1e3
    attn_ms = model.attn_gop(seq) / (tops * chip.attn_engine_gain * 1e3) * 1e3

    ttft = max(compute_ms + attn_ms, memory_ms)
    return Prefill(ttft_ms=ttft, compute_ms=compute_ms, attn_ms=attn_ms,
                   memory_ms=memory_ms,
                   bound="memory" if memory_ms > compute_ms + attn_ms else "compute",
                   chunks=chunks, mb_per_token=chunks * sweep / seq)


# Measured from p2/rtl/sonic_tile.sv, synthesized standalone with
# Yosys + `abc -g cmos2` at T = 4, 8, 16:
#
#     cells(T) = 712.0*T^2 + 2158.9*T - 1514.3
#
# Fitted on those three edges and validated at T = 12, where it predicts within
# 1.5%. The T^2 term is the PE array proper. The linear and constant terms are
# the per-tile overhead P1's sweep was getting for free: the accumulator column,
# per-tile control, and partial-sum egress. Because a tile of edge T holds T^2
# lanes, that overhead is amortised over fewer lanes as tiles shrink, so cost
# per lane rises as 1/T -- which is exactly the term that was missing.
TILE_FIT = (712.0, 2158.9, -1514.3)
TILE_REF = 128          # anchor: the ChipSpec default, so published area is unchanged


def tile_area_factor(tile: int, ref: int = TILE_REF) -> float:
    """Array area per lane at `tile`, relative to a monolithic `ref` edge.

    1.0 at the reference edge by construction, so adopting this model moves no
    previously published number; it only prices the sub-tiling that P1-1's sweep
    could previously buy for nothing.
    """
    a, b, c = TILE_FIT
    per_lane = lambda t: a + b / t + c / (t * t)   # noqa: E731
    return per_lane(tile) / per_lane(ref)


def area(chip: ChipSpec) -> dict[str, float]:
    """Die area by block, mm2. Returns blocks plus '_routing' and '_total'."""
    blocks = dict(AREA_FIXED)
    blocks["LPDDR5X x64 PHY"] = PHY_AREA[chip.bus_bits]
    blocks[f"{chip.sram_mb:.0f} MB SRAM, {chip.sram_banks} banks"] = (
        chip.sram_mb * chip.sram_mm2_per_mb)
    blocks[f"{chip.mac_lanes:,}-lane dual-mode array"] = (
        chip.mac_lanes * chip.mac_mm2_per_lane * tile_area_factor(chip.tile))

    core = sum(blocks.values())
    total = core * 1.09  # routing / utilisation overhead at 65% target density
    blocks["_routing"] = total - core
    blocks["_total"] = total
    return blocks


def sram_for_chunk(model: ModelSpec, chunk: int, headroom_mb: float = 4.0) -> float:
    """SRAM needed to hold a prefill chunk's residual stream plus working set."""
    return chunk * model.d / 1e6 + headroom_mb


def min_chunk_for_array(model: ModelSpec, chip: ChipSpec) -> int:
    """Smallest chunk that keeps a tile x tile systolic array fed.

    Tokens per expert is chunk * top_k / n_experts; the array needs at least
    `tile` of them to fill a pass. This coupling between array size and SRAM
    size is the least obvious constraint on the chip.
    """
    if not model.is_moe:
        return chip.tile
    return math.ceil(chip.tile * model.n_experts / model.top_k)
