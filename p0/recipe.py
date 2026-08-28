#!/usr/bin/env python3
"""P0-1: the frozen quantization recipe.

This file IS the format spec. RTL, the C golden model and the offline packer all
implement what is declared here; if they disagree, this file wins.

Assignment rationale is in sonic-s1-plan.html section 07. The short version:
spend bits where traffic is small (attention, routers, conv kernels) and save
them where traffic is large (experts, embedding).
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sonic import load  # noqa: E402


@dataclass(frozen=True)
class Fmt:
    bits: float          # effective bits per weight including scale overhead
    kind: str            # int4 | int8 | bf16
    group: int = 0       # 0 = per-tensor
    outlier_rows: float = 0.0
    why: str = ""


INT4_G64 = Fmt(4.25, "int4", 64, why="4 + 16/64 bits with an FP16 scale per group")
INT4_OUT = Fmt(4.33, "int4", 64, 0.02, "as INT4_G64 with a 2% INT8 outlier-row budget")
INT8 = Fmt(8.0, "int8", 0, why="per-tensor scale")
BF16 = Fmt(16.0, "bf16", 0, why="tiny cost, removes a class of debugging")

# Matched in order; first hit wins. Patterns are on HF parameter names.
RECIPE: list[tuple[str, Fmt]] = [
    (r"\.feed_forward\.experts\..*\.w2\.", INT4_OUT),
    (r"\.feed_forward\.experts\.", INT4_G64),
    (r"\.feed_forward\.gate\.|router|expert_bias", BF16),
    (r"layers\.[01]\.feed_forward\.w[123]\.", INT8),   # MoE leading dense layers
    (r"\.conv\.conv\.weight", INT8),              # 6,144 values -- not worth quantizing
    (r"\.conv\.(in|out)_proj\.", INT4_G64),
    (r"\.self_attn\.|\.attn\.", INT8),
    (r"embed_tokens|lm_head", INT4_G64),          # first 4K rows promoted to INT8
    (r"norm|\.bias$", BF16),
]

# Numeric behaviour the RTL and the C model must match bit for bit.
ACCUM = dict(local_bits=16, fold=16, mid_bits=24, out_bits=32,
             why="16b fast path folded every 16 additions into 24b, 32b epilogue; "
                 "12b is NOT sufficient for INT4 x INT8 -- see p0/golden")
ACTIVATION = dict(kind="int8", scale="per-token dynamic", clamp="calibrated percentile")
KV = dict(default="int8", long_ctx="int4", threshold_ctx=32768, scale="per-head")
SILU = dict(kind="pwl", segments=16, coeffs="firmware-loadable")
SOFTMAX = dict(kind="online", accum="bf16", why="running max/sum, flash ordering")

GATES = dict(
    ppl_delta_max=0.15,
    bench_drop_max=1.5,
    top1_agreement_min=0.99,
    routing_agreement_min=0.995,   # MoE only; no dense-model equivalent
    avg_bits_max=4.75,
)


def match(name: str) -> Fmt:
    for pat, fmt in RECIPE:
        if re.search(pat, name):
            return fmt
    return INT4_G64


def budget(model_name: str) -> None:
    """Effective bits/weight and resulting traffic under this recipe."""
    m = load(model_name)
    # Map ModelSpec blocks onto representative parameter names.
    rep = {
        "MoE experts": "layers.5.feed_forward.experts.3.w1.weight",
        "Dense SwiGLU FFN": "layers.5.feed_forward.w1.weight",   # dense model -> INT4
        "Dense FFN": "layers.0.feed_forward.w1.weight",
        "Short-conv blocks": "layers.1.conv.in_proj.weight",
        "GQA attention": "layers.2.self_attn.q_proj.weight",
        "Tied embedding / LM head": "model.embed_tokens.weight",
        "Routers": "layers.5.feed_forward.gate.weight",
    }
    tot_bits = act_bits = 0.0
    print(f"\n{m.name}  ({m.n_conv} conv / {m.n_attn} attn, "
          f"{'MoE' if m.is_moe else 'dense'})")
    print(f"  {'block':28s}{'format':>10}{'bits':>7}{'MB/token':>11}")
    for b in m.blocks:
        f = match(rep.get(b.name, ""))
        mb = b.active * f.bits / 8 / 1e6
        tot_bits += b.total * f.bits
        act_bits += b.active * f.bits
        print(f"  {b.name:28s}{f.kind:>10}{f.bits:>7.2f}{mb:>11.1f}")
    print(f"  {'':28s}{'':>10}{act_bits / m.active_params:>7.2f}"
          f"{act_bits / 8 / 1e6:>11.1f}   <- avg bits, MB/token")
    print(f"  resident {tot_bits / 8 / 1e6:,.0f} MB   "
          f"gate: avg bits <= {GATES['avg_bits_max']}  "
          f"{'PASS' if act_bits / m.active_params <= GATES['avg_bits_max'] else 'FAIL'}")


if __name__ == "__main__":
    names = sys.argv[1:] or ["lfm2.5-8b-a1b", "lfm2.5-2.6b"]
    print(__doc__.strip().split("\n")[0])
    print(f"\naccumulator: {ACCUM['local_bits']}b -> fold {ACCUM['fold']} -> "
          f"{ACCUM['mid_bits']}b -> {ACCUM['out_bits']}b")
    for n in names:
        budget(n)
    print("\nacceptance gates:")
    for k, v in GATES.items():
        print(f"  {k:26s} {v}")
