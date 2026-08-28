"""The frozen quantization recipe -- the single source of format truth.

RTL, the C golden model and the offline packer all implement what is declared
here. Rationale per tensor is in sonic-s1-plan.html section 07.

Effective bits include scale overhead: INT4 with one FP16 scale per group of 64
costs 4 + 16/64 = 4.25 bits per weight, not 4.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fmt:
    bits: float
    kind: str
    group: int = 0
    outlier_rows: float = 0.0
    why: str = ""


INT4_G64 = Fmt(4.25, "int4", 64, why="4 + 16/64 bits, FP16 scale per group of 64")
INT4_OUT = Fmt(4.33, "int4", 64, 0.02, "INT4_G64 plus a 2% INT8 outlier-row budget")
INT8 = Fmt(8.0, "int8", 0, why="per-tensor scale")
BF16 = Fmt(16.0, "bf16", 0, why="tiny cost, removes a class of debugging")
# Router datapath, measured on 512 real layer-5 hidden states from the 8.47 B
# checkpoint (p3/export_vectors.py, p2/tb/tb_router.cpp). INT8 weights cap
# routing agreement at 0.986 whatever the activation width; INT4 collapses it
# to 0.76. INT12 x INT12 is the cheapest point clearing the 0.995 gate, and it
# is far cheaper than the BF16 this recipe originally specified.
INT12 = Fmt(12.25, "int12", 64, why="measured: 0.998 routing agreement; INT8 caps at 0.986")

# Format per ModelSpec block name. Blocks are named in sonic/modelspec.py.
BLOCK_FMT: dict[str, Fmt] = {
    "MoE experts": INT4_OUT,            # w2 outliers dominate the average
    "Dense SwiGLU FFN": INT4_G64,
    "Dense FFN": INT8,                  # only the 2 leading layers of the MoE
    "Short-conv blocks": INT4_G64,      # conv kernels themselves are INT8, 0.04% of the block
    "Tied embedding / LM head": INT4_G64,
    "GQA attention": INT8,
    "Routers": INT12,
}

# Numeric behaviour the RTL and the C model must match bit for bit.
ACCUM = dict(local_bits=16, fold=16, mid_bits=24, out_bits=32)
ACTIVATION = dict(kind="int8", scale="per-token dynamic", clamp="calibrated percentile")
KV = dict(default="int8", long_ctx="int4", threshold_ctx=32768, scale="per-head")
SILU = dict(kind="pwl", segments=16, coeffs="firmware-loadable", fit="minimax")
# The router's sigmoid needs 4x the resolution of the FFN's SiLU: 16 segments
# give 0.971 routing agreement, 32 give 0.990, 64 give 0.996.
ROUTER_SIGMOID = dict(kind="pwl", segments=64, coeffs="firmware-loadable",
                      fit="minimax", act_bits=12, weight_bits=12)
SOFTMAX = dict(kind="online", accum="bf16")

GATES = dict(ppl_delta_max=0.15, bench_drop_max=1.5, top1_agreement_min=0.99,
             routing_agreement_min=0.995, avg_bits_max=4.75)

UNIFORM_INT4 = {k: INT4_G64 for k in BLOCK_FMT}


def fmt(block_name: str) -> Fmt:
    return BLOCK_FMT.get(block_name, INT4_G64)
