#!/usr/bin/env python3
"""The offline packer: calibrated quantization to the frozen formats.

`p0/gates.py` measured the recipe under naive round-to-nearest and it missed
`ppl_delta` by 40x (+6.23 against a 0.15 gate). RTN is the weakest 4-bit scheme
there is, so that result indicts the *method*, not the formats. This module is
the method.

Nothing here changes what the silicon reads. The output is still INT4 group-64
with FP16 scales, plus the INT8 outlier-row budget -- exactly what
`sonic/quant.py` declares and what the RTL implements. All three strategies
below only choose *better scales and rounding* for the same container, which is
work the packer does once, offline.

    rtn    round-to-nearest against the group max. The baseline; what gates.py
           measured first.
    clip   search the clipping ratio that minimises squared error instead of
           assuming the group max is the right full-scale point. One outlier in
           a group of 64 otherwise sets the step size for all 64. Data-free.
    awq    activation-aware scaling. Weight columns that multiply large
           activations are scaled up before quantization and back down after, so
           the fixed number of levels is spent where it changes the output.
           Needs calibration data. Subsumes `clip`.

The scaling in `awq` is applied as W_q = Q(W * s) / s. In deployment the 1/s is
folded into the preceding op so it costs nothing at runtime; here it is applied
directly, which is numerically identical and keeps the packer independent of the
surrounding graph.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic import quant  # noqa: E402

# Clipping ratios searched. 1.0 is plain RTN, so the search can never be worse
# than the baseline -- it is a strict improvement or a tie.
CLIP_GRID = torch.linspace(0.55, 1.0, 16)
# AWQ scaling exponents. 0.0 disables scaling, so awq >= clip by construction.
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8]
# Inside the AWQ search the clip grid is coarse: the alpha search dominates the
# result and the product of two fine grids is what makes packing slow.
AWQ_CLIP_GRID = torch.tensor([0.8, 0.9, 1.0])


def _quant_dequant(flat: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    q = (flat / scale).round().clamp(-qmax - 1, qmax)
    return q * scale


def _scales_for(flat: torch.Tensor, qmax: int, ratio: float) -> torch.Tensor:
    """Per-group FP16 scales at a given fraction of the group max."""
    s = (flat.abs().amax(1, keepdim=True) * ratio / qmax).to(torch.float16).float()
    return torch.where(s == 0, torch.ones_like(s), s)


def q_group_clipped(w: torch.Tensor, bits: int, group: int,
                    weight: torch.Tensor | None = None,
                    grid: torch.Tensor = CLIP_GRID) -> torch.Tensor:
    """Group quantize, choosing the clip ratio that minimises error.

    `weight` optionally weights the error per input channel, so the search
    minimises what the layer's output actually cares about rather than raw
    weight MSE. One ratio is chosen per tensor, not per group: a per-group
    search costs 64x the memory for a fraction of the benefit, because the
    ratio that suits a tensor's weight distribution suits most of its groups.
    """
    qmax = 2 ** (bits - 1) - 1
    n = w.shape[-1]
    if n % group:
        return w
    flat = w.reshape(-1, group).float()
    ew = None
    if weight is not None:
        ew = weight.reshape(1, -1).expand(w.shape[0] if w.ndim == 2 else -1, -1)
        ew = ew.reshape(-1, group) if ew.numel() == w.numel() else None

    best_err, best = None, None
    for r in grid.tolist():
        dq = _quant_dequant(flat, _scales_for(flat, qmax, r), qmax)
        e = (dq - flat).pow(2)
        if ew is not None:
            e = e * ew
        err = e.sum()
        if best_err is None or err < best_err:
            best_err, best = err, dq
    return best.reshape(w.shape).to(w.dtype)


def q_group_awq(w: torch.Tensor, bits: int, group: int, act: torch.Tensor,
                alphas=ALPHA_GRID) -> torch.Tensor:
    """Activation-aware scaling, then clipped group quantization.

    `act` is the mean |x| per input channel, measured on calibration data. The
    per-channel scale s_j = act_j^alpha (normalised to unit geometric mean)
    expands the columns that matter before rounding and contracts them after,
    which spends the 16 available levels where they change the output most.

    The objective is the activation-weighted weight error
    ||(W - W_q) diag(act)||_F, which is the standard cheap proxy for output
    error and needs only the per-channel magnitudes rather than the cached
    activations themselves.
    """
    s = awq_best_scale(w, bits, group, act, alphas)
    if s is None:
        # Fall back to plain RTN, NOT to clip search. Measured: data-free MSE
        # clipping regresses ppl_delta from +6.23 to +8.98, because it trades a
        # 6% cut in total squared error for a 52% rise in error on the largest
        # 0.1% of weights -- and those are the ones that move the output. Only
        # the activation-weighted objective gets this trade right.
        from p0.gates import _q_group
        return _q_group(w, bits, group)
    return (q_group_clipped(w.float() * s, bits, group,
                            grid=AWQ_CLIP_GRID) / s).to(w.dtype)


def awq_best_scale(w, bits, group, act, alphas=ALPHA_GRID):
    """The per-input-channel scale s minimising activation-weighted error.

    Returned separately because it is what determines the *storage* claim. What
    the packer writes is Q(W * s) -- genuine INT4 group-64 -- and s is applied
    to activations, not stored per weight. In deployment s folds into the
    preceding RMSNorm and costs nothing. Where it cannot fold, s is one FP16 per
    input channel shared across every output row: for a [3584, 2048] tensor that
    is 2048 x 16 bits over 7.3M weights, 0.0045 bits/weight. Either way the 4.25
    bits/weight budget survives -- but the RTL must then apply a per-channel
    activation scale, which is a real requirement, not free.
    """
    if act is None or w.shape[-1] != act.numel():
        return None
    a = act.float().clamp(min=1e-6).to(w.device)
    best_err, best_s = None, None
    for alpha in alphas:
        s = a.pow(alpha)
        s = s / s.log().mean().exp()               # unit geometric mean
        dq = q_group_clipped(w.float() * s, bits, group, grid=AWQ_CLIP_GRID) / s
        err = ((dq - w.float()) * a).pow(2).sum()
        if best_err is None or err < best_err:
            best_err, best_s = err, s
    return best_s


# ------------------------------------------------------- calibration capture

# Which parameter each module's input activation belongs to. The MoE expert
# stack is the awkward one: gate_up_proj consumes the block input, but
# down_proj consumes the intermediate produced inside the fused kernel, which
# no forward hook can see. down_proj therefore falls back to RTN -- recorded
# here rather than hidden, because it is a real limitation and down_proj is
# where quant.py says the outliers live.
def collect_act_scales(model, ids: torch.Tensor, window: int, device: str,
                       max_windows: int = 4) -> dict[str, torch.Tensor]:
    """Mean |x| per input channel for every module we can hook."""
    import torch.nn as nn

    acc: dict[str, list] = {}

    def hook(name):
        def f(mod, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0].detach()
            x = x.reshape(-1, x.shape[-1]).abs().float().mean(0).cpu()
            if name in acc:
                acc[name][0] += x
                acc[name][1] += 1
            else:
                acc[name] = [x, 1]
        return f

    handles = []
    for n, m in model.named_modules():
        t = type(m).__name__
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(hook(n + ".weight")))
        elif t == "Lfm2MoeExperts":
            handles.append(m.register_forward_pre_hook(hook(n + ".gate_up_proj")))
        elif t == "Lfm2MoeTopKRouter":
            handles.append(m.register_forward_pre_hook(hook(n + ".weight")))

    try:
        with torch.no_grad():
            for i in range(min(max_windows, ids.numel() // window)):
                w = ids[i * window:(i + 1) * window].unsqueeze(0).to(device)
                model(w)
    finally:
        for h in handles:
            h.remove()

    return {k: v[0] / v[1] for k, v in acc.items()}


def pack(w: torch.Tensor, f: quant.Fmt, mode: str,
         act: torch.Tensor | None = None) -> torch.Tensor:
    """Quantize one tensor to format `f` using strategy `mode`."""
    from p0.gates import _q_group, _q_tensor, _q_int4_outliers

    if f.kind == "bf16":
        return w
    if mode == "rtn":
        from p0.gates import apply_fmt
        return apply_fmt(w, f)

    bits = {"int4": 4, "int8g": 8, "int12": 12}.get(f.kind)
    if bits is None:                                  # per-tensor INT8
        return _q_tensor(w, 8)

    group = f.group or 64
    if f.outlier_rows:
        # Same outlier budget as RTN, but both tiers are calibrated.
        lo = (q_group_awq(w, 4, group, act) if mode == "awq"
              else q_group_clipped(w, 4, group))
        row_max = w.abs().amax(-1)
        k = max(1, int(round(f.outlier_rows * row_max.shape[-1])))
        idx = row_max.topk(k, dim=-1).indices
        mask = torch.zeros_like(row_max, dtype=torch.bool).scatter_(-1, idx, True)
        hi = (q_group_awq(w, 8, group, act) if mode == "awq"
              else q_group_clipped(w, 8, group))
        return torch.where(mask.unsqueeze(-1), hi, lo)

    if mode == "awq":
        return q_group_awq(w, bits, group, act)
    return q_group_clipped(w, bits, group)
