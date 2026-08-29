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


def _cholesky_inv_upper(H: torch.Tensor, damp: float) -> torch.Tensor | None:
    """Upper-triangular Cholesky factor of H^-1, with escalating damping.

    GPTQ's update needs *this* matrix, not H^-1 itself: the compensation
    `err * Hinv[j, j+1:]` is derived from the Cholesky factorisation, and
    applying it to a raw inverse is a different (and wrong) linear algebra.
    Returns None if no amount of damping makes H positive definite, so the
    caller can fall back rather than propagate garbage.
    """
    K = H.shape[-1]
    eye = torch.eye(K, device=H.device, dtype=torch.float32)
    d = torch.diagonal(H)
    mean_d = d.mean().item()
    if not (mean_d > 0):
        return None
    # Rank-deficient Hessians are the norm here, not the exception: an expert
    # that saw five tokens during calibration contributes a rank-5 Gram matrix
    # of full width. Damping is what makes it invertible, so escalate until it
    # is rather than failing on the first try.
    for mult in (1.0, 10.0, 100.0, 1000.0):
        Hd = H + (damp * mult * mean_d) * eye
        try:
            L = torch.linalg.cholesky(Hd)
        except Exception:
            continue
        H_inv = torch.cholesky_inverse(L)
        try:
            return torch.linalg.cholesky(H_inv, upper=True)
        except Exception:
            continue
    return None


def q_group_gptq(w: torch.Tensor, bits: int, group: int,
                 H: torch.Tensor | None = None,
                 act: torch.Tensor | None = None,
                 damp: float = 0.01) -> torch.Tensor:
    """GPTQ error compensation with per-group FP16 scales.

    Quantizes column by column in blocks of `group`, and after each column
    pushes its rounding error onto the columns not yet quantized, weighted by
    the inverse Hessian of the layer input. What RTN throws away, this spends
    on the remaining weights.

    The Hessian comes from calibration (`collect_act_scales` / `expert_hook`)
    as the Gram matrix X^T X. Where it is missing or unusable the result is
    plain group RTN -- never NaN, which is what the previous implementation
    produced on the 8B and is why this gate read NaN rather than a number.
    """
    if w.ndim == 3:
        # Fused expert stack [E, out, in]: each expert has its own Hessian,
        # and experts that never fired during calibration have none at all.
        E = w.shape[0]
        out = torch.empty_like(w)
        for e in range(E):
            He = H[e] if (H is not None and H.ndim == 3) else H
            ae = act[e] if (act is not None and act.ndim == 2) else act
            out[e] = q_group_gptq(w[e], bits, group, H=He, act=ae, damp=damp)
        return out

    from p0.gates import _q_group

    if H is None and act is not None and act.ndim == 1 and act.numel() == w.shape[-1]:
        H = torch.diag(act.detach().float().cpu().pow(2))
    if H is None:
        return _q_group(w, bits, group)

    K = w.shape[-1]
    if K % group:                      # cannot group-quantize; nor can RTN
        return _q_group(w, bits, group)

    # Everything below runs on CPU in float32. The Hessians are accumulated on
    # CPU by the calibration hooks, the model may be on MPS or CUDA, and
    # torch.linalg.cholesky is only dependably available on CPU anyway. One
    # explicit hop beats a device mismatch mid-factorisation.
    H_mat = H.detach().float().cpu()
    if H_mat.ndim == 1:
        H_mat = torch.diag(H_mat)
    if H_mat.shape[-1] != K or not torch.isfinite(H_mat).all():
        return _q_group(w, bits, group)

    qmax = 2 ** (bits - 1) - 1
    W = w.detach().float().cpu().clone()
    H_mat = H_mat.clone()

    # Dead input channels -- a hidden dimension that was identically zero across
    # the whole calibration set -- carry no signal and no curvature. Zeroing the
    # weight and unit-loading the diagonal keeps the factorisation defined
    # without inventing a compensation direction out of nothing.
    dead = torch.diagonal(H_mat) == 0
    if dead.any():
        H_mat[dead, dead] = 1.0
        W[:, dead] = 0.0

    Hinv = _cholesky_inv_upper(H_mat, damp)
    if Hinv is None:
        return _q_group(w, bits, group)

    Q = torch.zeros_like(W)
    for b in range(0, K, group):
        b_end = b + group
        W1 = W[:, b:b_end].clone()
        Q1 = torch.zeros_like(W1)
        E1 = torch.zeros_like(W1)
        Hinv1 = Hinv[b:b_end, b:b_end]

        # One FP16 scale per group per output row, taken before compensation
        # starts so the scale describes the group the packer will actually see.
        s = (W1.abs().amax(dim=1, keepdim=True) / qmax).to(torch.float16).float()
        s = torch.where(s == 0, torch.ones_like(s), s)

        for j in range(group):
            col = W1[:, j]
            q_col = (col / s[:, 0]).round().clamp(-qmax - 1, qmax) * s[:, 0]
            Q1[:, j] = q_col
            # The division by the Cholesky diagonal is the step the previous
            # implementation was missing; without it the compensation is scaled
            # by the wrong factor at every column.
            err = (col - q_col) / Hinv1[j, j]
            if j + 1 < group:
                W1[:, j + 1:] -= err.unsqueeze(1) * Hinv1[j, j + 1:].unsqueeze(0)
            E1[:, j] = err

        Q[:, b:b_end] = Q1
        if b_end < K:
            W[:, b_end:] -= E1 @ Hinv[b:b_end, b_end:]

    # Last line of defence. An ill-conditioned block can still diverge, and one
    # non-finite weight in a fused expert stack takes the whole model's
    # perplexity to NaN -- which is exactly what happened. Degrade to RTN for
    # this tensor and say so, rather than poisoning the run.
    if not torch.isfinite(Q).all():
        print(f"    WARNING: GPTQ diverged on a {tuple(w.shape)} tensor "
              f"({int((~torch.isfinite(Q)).sum())} non-finite); using RTN here.")
        return _q_group(w, bits, group)
    return Q.to(device=w.device, dtype=w.dtype)


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
    if act is None:
        return None
    # act is either [in] (one vector for the tensor) or [E, in] (one per expert
    # in a fused stack). The per-expert form matters: each expert sees a
    # different subset of tokens, so a single averaged vector would scale the
    # quiet experts by the busy ones' statistics.
    if act.ndim == 2 and w.ndim == 3:
        if act.shape != (w.shape[0], w.shape[-1]):
            return None
        a = act.float().clamp(min=1e-6).to(w.device).unsqueeze(1)   # [E,1,in]
    elif act.ndim == 1 and w.shape[-1] == act.numel():
        a = act.float().clamp(min=1e-6).to(w.device)
    else:
        return None
    best_err, best_s = None, None
    for alpha in alphas:
        s = a.pow(alpha)
        s = s / s.log().mean(-1, keepdim=True).exp()   # unit geometric mean
        dq = q_group_clipped(w.float() * s, bits, group, grid=AWQ_CLIP_GRID) / s
        err = ((dq - w.float()) * a).pow(2).sum()
        if best_err is None or err < best_err:
            best_err, best_s = err, s
    return best_s


# ------------------------------------------------------- calibration capture

class HessianStore:
    """Per-module Hessians, accumulated in a file-backed array.

    These dominate calibration memory and nothing else comes close. One fused
    expert stack contributes a [32, 2048, 2048] float32 Gram matrix -- 536 MB --
    and LFM2.5-8B-A1B has 22 MoE layers, each with two of them. Held in RAM
    together that is roughly 21 GB on top of a 17 GB BF16 model, which on a
    68 GB box does not fit: measured, it drove 14 GB of swap and turned the
    calibration pass into the longest phase of the gate by a wide margin.

    GPTQ reads these one tensor at a time, so they do not need to be resident
    together. Backing them with numpy memmaps puts them in the page cache
    instead of anonymous memory, which matters more than the disk round trip:
    clean file-backed pages are dropped under pressure, anonymous pages have to
    be written to swap first. Peak resident becomes one tensor's Hessian.
    """

    def __init__(self, root: Path | None = None):
        import tempfile
        self.root = Path(root or tempfile.mkdtemp(prefix="sonic-hessian-"))
        self.root.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, tuple[tuple[int, ...], int]] = {}

    def _path(self, key: str) -> Path:
        return self.root / (key.replace("/", "_").replace(".", "_") + ".f32")

    def add(self, key: str, h: torch.Tensor) -> None:
        import numpy as np
        a = h.detach().float().cpu().numpy()
        if key not in self._meta:
            mm = np.memmap(self._path(key), dtype=np.float32, mode="w+", shape=a.shape)
            mm[:] = a
            self._meta[key] = (a.shape, 1)
        else:
            shape, n = self._meta[key]
            mm = np.memmap(self._path(key), dtype=np.float32, mode="r+", shape=shape)
            mm += a
            self._meta[key] = (shape, n + 1)
        mm.flush()
        del mm

    def get(self, key: str) -> torch.Tensor | None:
        import numpy as np
        if key not in self._meta:
            return None
        shape, n = self._meta[key]
        mm = np.memmap(self._path(key), dtype=np.float32, mode="r", shape=shape)
        return torch.from_numpy(np.array(mm)) / n

    def close(self) -> None:
        import shutil
        shutil.rmtree(self.root, ignore_errors=True)


class ModuleStats:
    """One module's calibration stats: `mean` resident, `H` fetched on demand."""

    def __init__(self, mean: torch.Tensor, store: HessianStore, key: str):
        self._mean, self._store, self._key = mean, store, key

    def get(self, k: str, default=None):
        if k == "mean":
            return self._mean
        if k == "H":
            return self._store.get(self._key)
        return default



# Every quantized tensor gets an activation vector, including both projections
# of the fused expert stack -- see expert_hook, which replays the routing to
# reconstruct what down_proj consumes.
def expert_hook(name: str, acc: dict, store: HessianStore):
    """Per-expert input magnitudes and Hessians for BOTH projections of a fused expert stack."""
    import torch.nn.functional as F

    def f(mod, args):
        if len(args) < 2:
            return
        hs, topk = args[0], args[1]
        hs = hs.reshape(-1, hs.shape[-1])
        with torch.no_grad():
            mask = F.one_hot(topk.reshape(hs.shape[0], -1),
                             num_classes=mod.num_experts).permute(2, 1, 0)
            gu = torch.zeros(mod.num_experts, mod.hidden_dim)
            dn = torch.zeros(mod.num_experts, mod.intermediate_dim)
            gu_H = torch.zeros(mod.num_experts, mod.hidden_dim, mod.hidden_dim)
            dn_H = torch.zeros(mod.num_experts, mod.intermediate_dim, mod.intermediate_dim)
            for ei in (mask.sum(dim=(-1, -2)) > 0).nonzero():
                ei = ei[0]
                _, tok = torch.where(mask[ei])
                cs = hs[tok]
                # Cast before the Gram matmul, not after. A BF16 X^T X over
                # thousands of tokens loses the small eigenvalues that damping
                # is supposed to be the only thing setting, and those are what
                # the inverse is most sensitive to. collect_act_scales already
                # casts first; this path did not.
                csf = cs.float()
                gu[ei] = csf.abs().mean(0).cpu()
                gu_H[ei] = (csf.T @ csf).cpu()
                gate, up = F.linear(cs, mod.gate_up_proj[ei]).chunk(2, dim=-1)
                d_in = (mod.act_fn(gate) * up).float()
                dn[ei] = d_in.abs().mean(0).cpu()
                dn_H[ei] = (d_in.T @ d_in).cpu()
        for k, (v_mean, v_H) in ((name + ".gate_up_proj", (gu, gu_H)),
                                 (name + ".down_proj", (dn, dn_H))):
            store.add(k, v_H)
            if k in acc:
                acc[k]["mean"] += v_mean
                acc[k]["count"] += 1
            else:
                acc[k] = {"mean": v_mean, "count": 1}
    return f


def collect_act_scales(model, ids: torch.Tensor, window: int, device: str,
                       max_windows: int = 4,
                       store: HessianStore | None = None) -> dict[str, ModuleStats]:
    """Mean |x| and covariance H per input channel for every module we can hook.

    The Hessians go to `store` (a HessianStore, created here if not supplied)
    rather than staying resident -- see that class for why. The caller owns the
    store and should `.close()` it once packing is done.
    """
    import torch.nn as nn

    store = store if store is not None else HessianStore()
    acc: dict[str, dict] = {}

    def hook(name):
        def f(mod, args):
            if not args or not isinstance(args[0], torch.Tensor):
                return
            x = args[0].detach()
            xf = x.reshape(-1, x.shape[-1]).float()
            x_mean = xf.abs().mean(0).cpu()
            store.add(name, (xf.T @ xf).cpu())
            if name in acc:
                acc[name]["mean"] += x_mean
                acc[name]["count"] += 1
            else:
                acc[name] = {"mean": x_mean, "count": 1}
        return f

    handles = []
    for n, m in model.named_modules():
        t = type(m).__name__
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(hook(n + ".weight")))
        elif t == "Lfm2MoeExperts":
            handles.append(m.register_forward_pre_hook(expert_hook(n, acc, store)))
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

    return {k: ModuleStats(v["mean"] / v["count"], store, k) for k, v in acc.items()}


def pack(w: torch.Tensor, f: quant.Fmt, mode: str,
         act: dict | torch.Tensor | None = None) -> torch.Tensor:
    """Quantize one tensor to format `f` using strategy `mode`."""
    from p0.gates import _q_group, _q_tensor

    if f.kind == "bf16":
        return w
    if mode == "rtn":
        from p0.gates import apply_fmt
        return apply_fmt(w, f)

    # Extract mean and Hessian. `act` is either a bare tensor of per-channel
    # magnitudes, or anything with a .get -- a plain dict, or the lazy
    # ModuleStats below, whose "H" is read back from disk only when asked for.
    H, act_mean = None, None
    if isinstance(act, torch.Tensor):
        act_mean = act
    elif act is not None:
        H = act.get("H")
        act_mean = act.get("mean")

    bits = {"int4": 4, "int8g": 8, "int12": 12}.get(f.kind)
    if bits is None:                                  # per-tensor INT8
        return _q_tensor(w, 8)

    group = f.group or 64
    if f.outlier_rows:
        if mode == "gptq":
            lo = q_group_gptq(w, 4, group, H=H, act=act_mean)
        elif mode == "awq":
            lo = q_group_awq(w, 4, group, act_mean)
        else:
            lo = q_group_clipped(w, 4, group)

        row_max = w.abs().amax(-1)
        k = max(1, int(round(f.outlier_rows * row_max.shape[-1])))
        idx = row_max.topk(k, dim=-1).indices
        mask = torch.zeros_like(row_max, dtype=torch.bool).scatter_(-1, idx, True)

        if mode == "gptq":
            hi = q_group_gptq(w, 8, group, H=H, act=act_mean)
        elif mode == "awq":
            hi = q_group_awq(w, 8, group, act_mean)
        else:
            hi = q_group_clipped(w, 8, group)

        return torch.where(mask.unsqueeze(-1), hi, lo)

    if mode == "gptq":
        return q_group_gptq(w, bits, group, H=H, act=act_mean)
    if mode == "awq":
        return q_group_awq(w, bits, group, act_mean)
    return q_group_clipped(w, bits, group)
