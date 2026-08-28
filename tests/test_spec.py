"""Lock the numbers the plan publishes.

If a figure in sonic-s1-plan.html changes, a test here must change with it and
the change must be deliberate. These caught three errors during P0 setup:
the 2.6B layer split, the uniform-INT4 traffic idealisation, and the
12-bit accumulator bound.
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic import SKUS, decode, load, prefill
from sonic.moe import (batch_gain, distinct_experts, expected_accepted,
                       occupancy, spec_decode_gain)
from sonic.quant import GATES, UNIFORM_INT4
from sonic.roofline import area, min_chunk_for_array

M8 = load("lfm2.5-8b-a1b")
M2 = load("lfm2.5-2.6b")


def close(a, b, rel=0.01):
    assert abs(a - b) <= rel * abs(b), f"{a} != {b} (within {rel:.0%})"


# --- model topology, straight from config.json --------------------------------

def test_8b_topology():
    assert (M8.n_layers, M8.n_conv, M8.n_attn) == (24, 18, 6)
    assert (M8.n_experts, M8.top_k, M8.moe_inter) == (32, 4, 1792)
    assert (M8.n_dense_layers, M8.dense_inter) == (2, 7168)
    assert M8.rope_theta == 5e6


def test_26b_topology():
    """22 conv / 8 attn. Hand-counting this as 20/10 cost us a published error."""
    assert (M2.n_layers, M2.n_conv, M2.n_attn) == (30, 22, 8)
    assert not M2.is_moe
    assert M2.rope_theta == 1e7


def test_param_counts():
    close(M8.total_params, 8.468e9)
    close(M8.active_params, 1.686e9)
    close(M8.sparsity, 0.199)
    close(M2.total_params, 2.697e9)
    assert M2.sparsity == 1.0


# --- traffic, under the recipe and under the idealisation ---------------------

def test_traffic():
    # Was 978.0 / 1472.1 while the recipe used per-tensor INT8 and a group-64
    # embedding. Measuring the quality gates (p0/gates.py, 65K tokens) moved
    # attention and the dense FFN to per-group INT8 and the embedding to
    # group-32, which costs 12.3 MB/token and buys ppl_delta +6.44 -> +1.87.
    # The traffic rise is the price of a recipe that had never been measured.
    close(M8.bytes_per_token(), 990.3)
    close(M2.bytes_per_token(), 1483.0)
    # The uniform-4.25-bit figure is ~10% optimistic; keep both visible so nobody
    # quietly quotes the idealised one.
    close(M8.bytes_per_token(UNIFORM_INT4), 895.5)
    assert M8.bytes_per_token() > M8.bytes_per_token(UNIFORM_INT4)


def test_avg_bits_meets_gate():
    for m in (M8, M2):
        assert m.avg_bits() <= GATES["avg_bits_max"], f"{m.name}: {m.avg_bits():.2f}"


def test_kv_and_state():
    close(M8.kv_kb_per_token(), 6.144)
    close(M2.kv_kb_per_token(), 8.192)
    close(M8.kv_mb(131072), 805.3)
    close(M8.kv_mb(4096), 25.2)


def test_gop():
    close(M8.gop_per_token(), 3.37)
    close(M2.gop_per_token(), 5.39)


# --- the governing equation ---------------------------------------------------

def test_decode_is_memory_bound():
    """Batch-1 GEMV needs ~119 lanes at SKU B. The array has 16,384."""
    c = SKUS["B"]
    assert c.lanes_to_saturate() < 200
    assert c.mac_lanes / c.lanes_to_saturate() > 50


def test_decode_throughput():
    close(decode(M8, SKUS["B"]).tok_s, 68.2, rel=0.02)
    close(decode(M2, SKUS["B"]).tok_s, 45.4, rel=0.02)
    close(decode(M8, SKUS["B"]).watts, 3.73, rel=0.05)


def test_moe_beats_dense_on_the_same_bus():
    for c in SKUS.values():
        assert decode(M8, c).tok_s > decode(M2, c).tok_s


# --- MoE routing economics ----------------------------------------------------

def test_distinct_experts():
    close(distinct_experts(32, 4, 8), 21.0)
    assert distinct_experts(32, 4, 1) == 4.0


def test_expected_accepted_is_not_the_batch_size():
    """The term whose omission overstates speculative gain by ~2x."""
    assert expected_accepted(4, 0.80) < 5
    close(expected_accepted(4, 0.80), 3.36)


def test_speculation_barely_helps_moe_but_helps_dense():
    assert 1.10 < spec_decode_gain(M8, 2)["gain"] < 1.25
    assert spec_decode_gain(M8, 8)["gain"] < spec_decode_gain(M8, 2)["gain"]
    assert spec_decode_gain(M2, 4, drafter_mb=174.0)["gain"] > 2.0


def test_batching_helps_moe_a_lot():
    assert batch_gain(M8, 32)["gain"] > 6
    assert batch_gain(M8, 128)["gain"] > 20


# --- prefill ------------------------------------------------------------------

def test_array_chunk_coupling():
    """A 128-edge tile needs chunk >= 1024 -- the constraint that sizes SRAM."""
    assert min_chunk_for_array(M8, SKUS["B"]) == 1024


def test_occupancy_gate_is_marginal_at_tile_128():
    import numpy as np
    from sonic.moe import route_counts
    occ = occupancy(route_counts(M8, 2048, 0.5, np.random.default_rng(0)), 128)
    assert 0.75 < occ < 0.85, f"occupancy {occ:.3f}"
    # Sub-tiling the same lanes is the fix.
    sub = occupancy(route_counts(M8, 2048, 0.5, np.random.default_rng(0)), 64)
    assert sub > occ + 0.05


# --- measured routing, captured from the full 8.47 B model -------------------
# These run only when p0/out/real_routing.npz exists (make p3 regenerates it).

TRACE = Path(__file__).resolve().parent.parent / "p0" / "out" / "real_routing.npz"


def _trace():
    import numpy as np
    if not TRACE.exists():
        return None
    return np.load(TRACE)["routing"].astype(int)


def test_measured_routing_beats_the_uniform_bound():
    """Real sequences reuse experts. The uniform model assumes they do not."""
    tr = _trace()
    if tr is None:
        return
    from sonic.moe import measured_distinct
    for b in (2, 4, 8, 16):
        meas = measured_distinct(tr, b)
        bound = distinct_experts(32, 4, b)
        assert meas < bound, f"batch {b}: measured {meas:.2f} >= bound {bound:.2f}"
    # The gap is large enough to matter: ~1.6x at the DSpark verify batch.
    assert distinct_experts(32, 4, 8) / measured_distinct(tr, 8) > 1.4


def test_measured_occupancy_meets_the_gate_with_subtiling():
    """4 x 64^2 clears 0.80 with margin; a monolithic 128 edge does not."""
    import numpy as np
    tr = _trace()
    if tr is None:
        return
    counts = np.bincount(tr[:2048].ravel(), minlength=32)
    assert occupancy(counts, 64) > 0.85, "sub-tiled array fails the gate"
    assert occupancy(counts, 128) < occupancy(counts, 64)


def test_dspark_gain_on_measured_routing():
    tr = _trace()
    if tr is None:
        return
    from sonic.moe import dspark_gain, dspark_gain_measured
    for p in (0.80, 0.90):
        m = dspark_gain_measured(M8, tr, p=p)["gain"]
        b = dspark_gain(M8, p=p)["gain"]
        assert m > b, f"p={p}: measured {m:.2f} should beat bound {b:.2f}"
    assert 1.5 < dspark_gain_measured(M8, tr, p=0.80)["gain"] < 1.8


def test_ttft():
    b = SKUS["B"]
    close(prefill(M8, b, 2048).ttft_ms, 213, rel=0.03)
    assert prefill(M8, b, 2048).bound == "compute"
    assert prefill(M8, b, 128).bound == "memory"
    # Crossover: dense is faster on short prompts, MoE on long ones.
    assert prefill(M2, b, 128).ttft_ms < prefill(M8, b, 128).ttft_ms
    assert prefill(M2, b, 4096).ttft_ms > prefill(M8, b, 4096).ttft_ms


def test_attention_share_grows_quadratically():
    shares = [M8.attn_gop(p) / (M8.attn_gop(p) + M8.linear_gop(p))
              for p in (512, 8192, 131072)]
    assert shares[0] < 0.02 and 0.10 < shares[1] < 0.13 and shares[2] > 0.6
    assert shares == sorted(shares)


# --- area ---------------------------------------------------------------------

def test_die_area():
    close(area(SKUS["B"])["_total"], 20.56)
    assert area(SKUS["A"])["_total"] < area(SKUS["C"])["_total"]


def test_not_bump_limited():
    """614 bumps at 110 um fine-pitch C4 need >= 9.9 mm2; the die is 20.6."""
    bumps, pitch_mm, usable = 614, 0.110, 0.75
    assert bumps * pitch_mm ** 2 / usable < area(SKUS["B"])["_total"]
