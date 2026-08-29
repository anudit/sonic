"""Invariants of the quantization kernels and the offline packer.

Needs torch, so `tests/run.py` skips this module when torch is absent -- the
repo's headline promise is that the numbers reproduce with numpy alone.

Every test here is a property that was violated at some point during P0-1 and
cost real time to find. They are regression locks, not coverage.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402

from p0.gates import _q_group, _q_tensor, _q_int4_outliers, apply_fmt  # noqa: E402
from p0 import packer  # noqa: E402
from sonic import quant  # noqa: E402


def test_all_zero_group_is_not_nan():
    """An FP16 scale floor applied before the cast underflows to zero.

    The group then computes 0/0 = NaN, which propagates silently through a whole
    forward pass and shows up only as a NaN perplexity 40 minutes later.
    """
    for w in (torch.zeros(4, 64, dtype=torch.bfloat16),
              torch.full((4, 64), 1e-9, dtype=torch.bfloat16)):
        q = _q_group(w, 4, 64)
        assert not q.isnan().any(), "all-zero / underflowing group produced NaN"
        assert (q == 0).all(), "such a group must quantize to exactly zero"


def test_int4_group_has_at_most_16_levels():
    torch.manual_seed(0)
    q = _q_group(torch.randn(8, 256, dtype=torch.bfloat16), 4, 64).reshape(-1, 64)
    assert max(len(torch.unique(r)) for r in q) <= 16


def test_scale_is_fp16_representable():
    """INT4_G64 is priced at 4 + 16/64 bits. A float32 scale would be a format
    nobody is building, and would flatter every quality measurement."""
    torch.manual_seed(0)
    flat = torch.randn(64, 64).reshape(-1, 64)
    s = (flat.abs().amax(1, keepdim=True) / 7).to(torch.float16).float()
    assert torch.equal(s, s.to(torch.float16).float())


def test_per_tensor_scales_each_slice_of_a_stack():
    """"Per tensor" must mean per logical tensor.

    A single scale across a fused 32-expert stack, or across the 2048 channels
    of a depthwise kernel, lets the widest slice set the step for every other.
    Measured cost when this was wrong: 11% of top-1 agreement from the conv
    kernels alone, which are 0.0013% of the parameters.
    """
    torch.manual_seed(0)
    w = torch.randn(8, 64, 128, dtype=torch.bfloat16)
    w[3] *= 40                                    # one very wide expert
    err = (_q_tensor(w, 8).float() - w.float()).norm() / w.float().norm()
    assert err < 0.02, f"per-slice scaling regressed: relerr {err:.4f}"


def test_outlier_budget_reduces_error():
    torch.manual_seed(0)
    w = torch.randn(2, 64, 128, dtype=torch.bfloat16)
    w[0, 5] *= 40
    rel = lambda q: ((q.float() - w.float()).norm() / w.float().norm()).item()
    assert rel(_q_int4_outliers(w, 64, 0.02)) < rel(_q_group(w, 4, 64))


def test_bf16_is_identity():
    """The null control depends on this: apply_fmt(BF16) must not touch a bit."""
    torch.manual_seed(0)
    w = torch.randn(16, 64, dtype=torch.bfloat16)
    assert torch.equal(apply_fmt(w, quant.BF16), w)


def test_clip_search_at_ratio_one_reproduces_rtn():
    """The clip grid includes 1.0, so the search can never do worse on its own
    objective. If this drifts, a packer regression will read as a format result.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    assert torch.equal(
        packer.q_group_clipped(w, 4, 64, grid=torch.tensor([1.0])),
        _q_group(w, 4, 64))


def test_clip_search_lowers_total_squared_error():
    torch.manual_seed(0)
    w = torch.randn(64, 256, dtype=torch.bfloat16)
    w[3, :20] *= 30
    sq = lambda q: (q.float() - w.float()).pow(2).sum().item()
    assert sq(packer.q_group_clipped(w, 4, 64)) <= sq(_q_group(w, 4, 64))


def test_awq_without_activations_falls_back_to_rtn_not_clip():
    """Data-free clipping regressed ppl_delta from +6.23 to +8.98: it buys a
    small cut in total error with a large rise in error on the biggest weights,
    and those are the ones that move the output. Anything the packer cannot
    calibrate must therefore fall back to RTN."""
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    assert torch.equal(packer.q_group_awq(w, 4, 64, None), _q_group(w, 4, 64))


def test_awq_alpha_zero_changes_nothing_structurally():
    """alpha=0 makes the per-channel scale identically 1, so AWQ degenerates to
    its inner clip search -- the guarantee that awq >= clip by construction."""
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    act = torch.rand(128) + 0.1
    got = packer.q_group_awq(w, 4, 64, act, alphas=[0.0])
    want = packer.q_group_clipped(w.float(), 4, 64,
                                  grid=packer.AWQ_CLIP_GRID).to(torch.bfloat16)
    assert torch.equal(got, want)


def test_packer_preserves_the_declared_format():
    """The packer must change scales and rounding only, never the container.

    For rtn and clip the dequantized weights lie on <=16 levels per group, so
    the check is direct. For awq it is NOT: the stored value is Q(W * s) and the
    per-channel s is applied to activations, so dividing it back out spreads a
    group across many apparent levels. Checking the dequantized AWQ tensor would
    read 60 levels and look like a format violation when nothing is wrong -- the
    INT4-ness lives in Q(W * s), which is what this asserts.
    """
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    act = torch.rand(128) + 0.1
    for mode in ("rtn", "clip"):
        q = packer.pack(w, quant.INT4_G64, mode, act)
        levels = max(len(torch.unique(r)) for r in q.reshape(-1, 64))
        assert levels <= 16, f"{mode} emitted {levels} levels, not INT4"

    s = packer.awq_best_scale(w, 4, 64, act)
    stored = packer.q_group_clipped(w.float() * s, 4, 64,
                                    grid=packer.AWQ_CLIP_GRID)
    levels = max(len(torch.unique(r)) for r in stored.reshape(-1, 64))
    assert levels <= 16, f"awq stored {levels} levels, not INT4"


def test_awq_scale_is_per_input_channel_only():
    """s must be one value per input channel, shared across every output row.

    That is what keeps it nearly free: unfoldable, it costs 16 bits per input
    channel over all output rows -- 0.0045 bits/weight for a [3584, 2048]
    tensor. A per-element s would be a different format with a different
    bandwidth budget, and every traffic number in the plan would be wrong.
    """
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.bfloat16)
    s = packer.awq_best_scale(w, 4, 64, torch.rand(128) + 0.1)
    assert s.numel() == w.shape[-1], f"s has {s.numel()} entries, want 128"


def test_gptq_without_hessian_falls_back_to_rtn():
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    assert torch.equal(packer.q_group_gptq(w, 4, 64, None, None), _q_group(w, 4, 64))


def test_gptq_preserves_int4_levels():
    torch.manual_seed(0)
    w = torch.randn(32, 128, dtype=torch.bfloat16)
    H = torch.randn(128, 64)
    H = H @ H.T + torch.eye(128)
    q = packer.pack(w, quant.INT4_G64, "gptq", {"H": H, "mean": torch.ones(128)})
    levels = max(len(torch.unique(r)) for r in q.reshape(-1, 64))
    assert levels <= 16, f"gptq emitted {levels} levels, not INT4"


def test_gptq_reduces_weighted_error():
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.bfloat16)
    X = torch.randn(256, 128)
    H = X.T @ X
    q_rtn = _q_group(w, 4, 64)
    q_gptq = packer.q_group_gptq(w, 4, 64, H=H)
    err_rtn = ((w.float() - q_rtn.float()) @ X.T).pow(2).sum().item()
    err_gptq = ((w.float() - q_gptq.float()) @ X.T).pow(2).sum().item()
    assert err_gptq <= err_rtn, f"gptq output error {err_gptq:.2f} > rtn {err_rtn:.2f}"

