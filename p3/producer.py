#!/usr/bin/env python3
"""P3-3a: Descriptor-Ring Producer for Sonic S1 Sequencer.

Generates the static instruction / descriptor ring program for sonic_seq
from a target model specification (e.g. LFM2.5-8B-A1B or Dense 2.6B).

Descriptor format (64 bits):
    [63:60] Opcode:
        0x0 = NOP
        0x1 = GEMV (Dense linear projection)
        0x2 = CONV (1D causal convolution)
        0x3 = EXPERT (MoE expert fetch + GEMV, patched at runtime)
        0x4 = ATTN (Flash / Online softmax attention)
        0x5 = NORM (RMSNorm)
        0x6 = LMHEAD (LM Head + top-1 sampler)
    [59:56] Mode / Flags:
        0 = Decode (weight-streaming, batch-1)
        1 = Prefill (weight-stationary)
    [55:40] Dim M / Outputs (16 bits)
    [39:24] Dim K / Inputs (16 bits)
    [23:0]  DRAM Base Offset / Target Address in 64B lines (24 bits)
"""

from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path
from typing import NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sonic import load


class Opcode:
    NOP = 0x0
    GEMV = 0x1
    CONV = 0x2
    EXPERT = 0x3
    ATTN = 0x4
    NORM = 0x5
    LMHEAD = 0x6


class Descriptor(NamedTuple):
    opcode: int
    mode: int
    dim_m: int
    dim_k: int
    addr_offset: int

    def encode(self) -> int:
        """Pack fields into a 64-bit unsigned integer."""
        u64 = ((self.opcode & 0xF) << 60) | \
              ((self.mode & 0xF) << 56) | \
              ((self.dim_m & 0xFFFF) << 40) | \
              ((self.dim_k & 0xFFFF) << 24) | \
              (self.addr_offset & 0xFFFFFF)
        return u64


def build_descriptor_ring(model_name: str = "lfm2.5-8b-a1b", mode: int = 0) -> list[Descriptor]:
    """Build the sequential descriptor stream for one full token pass."""
    m = load(model_name)
    ring: list[Descriptor] = []
    dram_offset = 0

    d_kv = m.n_kv_heads * m.head_dim

    # 1. Embedding / Input Norm
    ring.append(Descriptor(Opcode.NORM, mode, m.d, 1, dram_offset))
    dram_offset += (m.d * 2) // 64

    # 2. Iterate through layers
    # In LFM architecture, every 4th layer (3, 7, 11, 15, 19, 23) is full attention, others are short-conv.
    for layer in range(m.n_layers):
        is_attn = (layer % 4 == 3) if m.n_attn > 0 else False
        is_moe = m.is_moe and (layer >= m.n_dense_layers)
        d_ffn = m.moe_inter if is_moe else m.dense_inter

        # Attention or Short-Conv
        if not is_attn:
            # Conv input projection (B, C, x')
            ring.append(Descriptor(Opcode.GEMV, mode, 3 * m.d, m.d, dram_offset))
            dram_offset += (3 * m.d * m.d // 2) // 64
            # 1D causal convolution
            ring.append(Descriptor(Opcode.CONV, mode, m.d, m.conv_k, dram_offset))
            dram_offset += (m.d * m.conv_k) // 64
            # Conv output projection
            ring.append(Descriptor(Opcode.GEMV, mode, m.d, m.d, dram_offset))
            dram_offset += (m.d * m.d // 2) // 64
        else:
            # GQA Attention: Q, K, V projections
            ring.append(Descriptor(Opcode.GEMV, mode, m.d + 2 * d_kv, m.d, dram_offset))
            dram_offset += ((m.d + 2 * d_kv) * m.d) // 64
            # Softmax Attention reduction
            ring.append(Descriptor(Opcode.ATTN, mode, m.d, d_kv, dram_offset))
            # O projection
            ring.append(Descriptor(Opcode.GEMV, mode, m.d, m.d, dram_offset))
            dram_offset += (m.d * m.d // 2) // 64

        # Pre-FFN Norm
        ring.append(Descriptor(Opcode.NORM, mode, m.d, 1, dram_offset))

        # FFN / MoE
        if is_moe:
            # Router projection (hidden_dim -> n_experts)
            ring.append(Descriptor(Opcode.GEMV, mode, m.n_experts, m.d, dram_offset))
            dram_offset += (m.n_experts * m.d) // 64

            # Top-k expert fetches (OP_EXPERT dynamically patched by router)
            for _ in range(m.top_k):
                ring.append(Descriptor(Opcode.EXPERT, mode, 2 * d_ffn, m.d, dram_offset))
                ring.append(Descriptor(Opcode.EXPERT, mode, m.d, d_ffn, dram_offset))
            dram_offset += (m.n_experts * 3 * m.d * d_ffn // 2) // 64
        else:
            # Dense FFN
            ring.append(Descriptor(Opcode.GEMV, mode, 2 * d_ffn, m.d, dram_offset))
            dram_offset += (2 * d_ffn * m.d // 2) // 64
            ring.append(Descriptor(Opcode.GEMV, mode, m.d, d_ffn, dram_offset))
            dram_offset += (m.d * d_ffn // 2) // 64

    # 3. Final Norm and LM Head
    ring.append(Descriptor(Opcode.NORM, mode, m.d, 1, dram_offset))
    ring.append(Descriptor(Opcode.LMHEAD, mode, m.vocab, m.d, dram_offset))

    return ring


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="lfm2.5-8b-a1b", choices=["lfm2.5-8b-a1b", "lfm2.5-2.6b"])
    ap.add_argument("--mode", type=int, default=0, help="0=Decode, 1=Prefill")
    ap.add_argument("--out", type=Path, default=Path("p3/out/ring.bin"))
    a = ap.parse_args()

    ring = build_descriptor_ring(a.model, a.mode)
    print(f"Generated {len(ring)} descriptors for {a.model} (mode={a.mode})")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as f:
        for desc in ring:
            val = desc.encode()
            f.write(struct.pack("<Q", val))

    print(f"Wrote binary ring to {a.out} ({len(ring)*8} bytes)")
    print("\nFirst 5 descriptors:")
    for i, desc in enumerate(ring[:5]):
        print(f"  [{i:02d}] op={desc.opcode:#x} mode={desc.mode} M={desc.dim_m:<5} K={desc.dim_k:<5} raw={desc.encode():#018x}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
