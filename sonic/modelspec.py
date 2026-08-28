"""Parse an LFM2.5 config.json into a parameter and traffic budget.

Everything downstream keys off this. The parameter counts are derived from the
config alone -- no checkpoint download required -- and are validated against the
published totals in tests/test_modelspec.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

CONFIGS = Path(__file__).resolve().parent.parent / "configs"

# INT4 weights carry one FP16 scale per group of 64 -> 4 + 16/64 = 4.25 bits.
BITS_INT4_G64 = 4.25


@dataclass(frozen=True)
class Block:
    name: str
    total: int          # parameters resident in DRAM
    active: int         # parameters read per decoded token


@dataclass(frozen=True)
class ModelSpec:
    name: str
    d: int
    n_layers: int
    n_conv: int
    n_attn: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    conv_k: int
    vocab: int
    rope_theta: float
    max_pos: int
    blocks: tuple[Block, ...] = field(default_factory=tuple)
    # MoE only
    n_experts: int = 0
    top_k: int = 0
    moe_inter: int = 0
    n_dense_layers: int = 0
    dense_inter: int = 0

    @property
    def is_moe(self) -> bool:
        return self.n_experts > 0

    @property
    def total_params(self) -> int:
        return sum(b.total for b in self.blocks)

    @property
    def active_params(self) -> int:
        return sum(b.active for b in self.blocks)

    @property
    def sparsity(self) -> float:
        """Fraction of parameters read per token."""
        return self.active_params / self.total_params

    def bytes_per_token(self, quant: dict | None = None) -> float:
        """Weight bytes streamed per decoded token, in MB, under the recipe.

        Pass quant=UNIFORM_INT4 to get the idealised uniform-4.25-bit figure.
        The recipe is ~8% heavier because attention, the leading dense layers
        and the routers are deliberately promoted.
        """
        from .quant import fmt
        q = (lambda n: quant[n]) if quant else fmt
        return sum(b.active * q(b.name).bits for b in self.blocks) / 8 / 1e6

    def resident_mb(self, quant: dict | None = None) -> float:
        from .quant import fmt
        q = (lambda n: quant[n]) if quant else fmt
        return sum(b.total * q(b.name).bits for b in self.blocks) / 8 / 1e6

    def avg_bits(self, quant: dict | None = None) -> float:
        return self.bytes_per_token(quant) * 8e6 / self.active_params

    def expert_mb(self) -> float:
        """One expert, one layer, in MB. Zero for dense models."""
        if not self.is_moe:
            return 0.0
        from .quant import fmt
        return 3 * self.d * self.moe_inter * fmt("MoE experts").bits / 8 / 1e6

    def expert_total_mb(self) -> float:
        """All experts across all MoE layers -- what a prefill chunk sweeps."""
        if not self.is_moe:
            return 0.0
        return self.expert_mb() * self.n_experts * self.n_moe_layers

    @property
    def n_moe_layers(self) -> int:
        return self.n_layers - self.n_dense_layers if self.is_moe else 0

    def gop_per_token(self) -> float:
        """Multiply-accumulate work per decoded token, in GOP (2 ops per MAC)."""
        return 2 * self.active_params / 1e9

    def kv_kb_per_token(self, bits: int = 8) -> float:
        """Decimal KB, to match how every other capacity here is quoted."""
        return self.n_attn * 2 * self.n_kv_heads * self.head_dim * bits / 8 / 1e3

    def kv_mb(self, ctx: int, bits: int = 8) -> float:
        return self.kv_kb_per_token(bits) * ctx / 1e3

    def attn_gop(self, seq: int) -> float:
        """Quadratic QK^T + PV work for a full prefill of `seq` tokens, in GOP."""
        return 4 * seq * seq * self.d * self.n_attn / 1e9

    def linear_gop(self, seq: int) -> float:
        return self.gop_per_token() * seq


def _conv_params(m_d: int, conv_k: int, n: int) -> int:
    # in_proj (d -> 3d for B/C/x gates) + depthwise kernel + out_proj
    return n * (m_d * 3 * m_d + m_d * conv_k + m_d * m_d)


def _attn_params(m_d: int, kv: int, hd: int, n: int) -> int:
    # q, o are d x d; k, v are d x (kv_heads * head_dim)
    return n * (2 * m_d * m_d + 2 * m_d * kv * hd)


def load(name_or_path: str | Path) -> ModelSpec:
    """Load a config.json by path, or by short name from configs/."""
    p = Path(name_or_path)
    if not p.exists():
        p = CONFIGS / f"{name_or_path}.json"
    cfg = json.loads(p.read_text())

    d = cfg["hidden_size"]
    layers = cfg["layer_types"]
    n_conv = layers.count("conv")
    n_attn = layers.count("full_attention")
    kv = cfg["num_key_value_heads"]
    heads = cfg["num_attention_heads"]
    hd = d // heads
    conv_k = cfg["conv_L_cache"]
    vocab = cfg["vocab_size"]
    n_layers = cfg["num_hidden_layers"]

    conv = _conv_params(d, conv_k, n_conv)
    attn = _attn_params(d, kv, hd, n_attn)
    emb = vocab * d  # tied with the LM head, so counted once but read every token

    blocks: list[Block] = []
    moe: dict = {}

    if cfg.get("model_type") == "lfm2_moe":
        E = cfg["num_experts"]
        K = cfg["num_experts_per_tok"]
        mi = cfg["moe_intermediate_size"]
        nd = cfg["num_dense_layers"]
        di = cfg["intermediate_size"]
        ml = n_layers - nd

        dense_ffn = 3 * d * di * nd
        expert_layer = 3 * d * mi           # one expert, one layer
        moe_total = expert_layer * E * ml
        moe_active = expert_layer * K * ml
        router = ml * (d * E + E)

        blocks = [
            Block("MoE experts", moe_total, moe_active),
            Block("Short-conv blocks", conv, conv),
            Block("Tied embedding / LM head", emb, emb),
            Block("Dense FFN", dense_ffn, dense_ffn),
            Block("GQA attention", attn, attn),
            Block("Routers", router, router),
        ]
        moe = dict(n_experts=E, top_k=K, moe_inter=mi, n_dense_layers=nd, dense_inter=di)
    else:
        di = cfg["intermediate_size"]
        ffn = 3 * d * di * n_layers
        blocks = [
            Block("Dense SwiGLU FFN", ffn, ffn),
            Block("Short-conv blocks", conv, conv),
            Block("Tied embedding / LM head", emb, emb),
            Block("GQA attention", attn, attn),
        ]

    return ModelSpec(
        name=p.stem, d=d, n_layers=n_layers, n_conv=n_conv, n_attn=n_attn,
        n_heads=heads, n_kv_heads=kv, head_dim=hd, conv_k=conv_k, vocab=vocab,
        rope_theta=float(cfg["rope_parameters"]["rope_theta"]),
        max_pos=cfg["max_position_embeddings"],
        blocks=tuple(sorted(blocks, key=lambda b: -b.active)), **moe,
    )
