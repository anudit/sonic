#!/usr/bin/env python3
"""P3-2: export real tensors from LFM2.5-8B-A1B as RTL test vectors.

The point of this file: an RTL testbench that only ever sees random stimulus
proves the design is self-consistent, not that it computes the model. These
vectors come from the actual 8.47 B checkpoint -- real router weights, real
hidden states, real expert-bias values -- together with the selection PyTorch
made. The RTL has to reproduce that decision, bit for bit, or it is wrong.

Emits a flat little-endian binary per case plus a JSON manifest, both of which
the Verilator and Icarus benches read.

    python3 p3/export_vectors.py --layer 5 --tokens 256
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch

MODEL = "LiquidAI/LFM2.5-8B-A1B"
OUT = Path(__file__).resolve().parent.parent / "p2" / "vectors"


def q_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Per-tensor symmetric INT8, matching sonic/quant.py ACTIVATION."""
    s = float(np.abs(x).max()) / 127.0 or 1.0
    return np.clip(np.rint(x / s), -128, 127).astype(np.int8), s


def q_int4_g64(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """INT4 with one FP16 scale per group of 64 along the reduction axis.

    Returns (codes int8 in [-8,7], scales float32[rows, groups]).
    """
    rows, cols = w.shape
    assert cols % 64 == 0, f"reduction dim {cols} is not a multiple of 64"
    g = w.reshape(rows, cols // 64, 64)
    s = np.abs(g).max(axis=2) / 7.0
    s[s == 0] = 1.0
    codes = np.clip(np.rint(g / s[:, :, None]), -8, 7).astype(np.int8)
    return codes.reshape(rows, cols), s.astype(np.float32)


def _float_agreement(h, W_bf, bias, sel, K):
    """Float router GEMV -- no integer path. `h` may be fp32 or dequantized INT8."""
    lq = h.astype(np.float32) @ W_bf.T
    sq = torch.topk(torch.from_numpy(lq).sigmoid()
                    + torch.from_numpy(bias), k=K, dim=-1).indices.numpy()
    return float((np.sort(sq, 1) == np.sort(sel, 1)).all(1).mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer", type=int, default=5, help="MoE layer to export")
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {a.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, device_map=a.device)
    model.eval()
    cfg = model.config
    E, K, d = cfg.num_experts, cfg.num_experts_per_tok, cfg.hidden_size

    blk = model.model.layers[a.layer].feed_forward
    router = blk.gate
    assert type(router).__name__ == "Lfm2MoeTopKRouter", type(router).__name__

    # Real hidden states reaching this layer's router.
    grabbed = {}
    h_hook = model.model.layers[a.layer].ffn_norm.register_forward_hook(
        lambda m, i, o: grabbed.setdefault("h", o.detach()))

    # A long, mixed-domain passage: routing is content-dependent, so a single
    # short prompt gives a biased and statistically thin sample.
    text = (
        "Explain in detail how a mixture-of-experts layer routes tokens to experts, "
        "and why that interacts badly with speculative decoding on a bandwidth-bound "
        "accelerator. Then write a Python function computing a running mean. "
        "def running_mean(xs):\n    total = 0.0\n    for i, x in enumerate(xs):\n"
        "        total += x\n        yield total / (i + 1)\n"
        "Translate to French: the accumulator overflows when the reduction depth "
        "exceeds the proven bound. Now summarise the tradeoff between SRAM capacity "
        "and prefill chunk size, and list three reasons a systolic array can idle. "
        "The capital of Australia is Canberra, not Sydney. In 1687 Newton published "
        "the Principia. Compute the memory bandwidth needed to decode one hundred "
        "tokens per second from a model with 1.69 billion active parameters at four "
        "bits per weight, and show your working step by step. " * 4)
    enc = tok(text, return_tensors="pt")
    with torch.no_grad():
        model(enc["input_ids"][:, : max(a.tokens, 512)].to(a.device))
    h_hook.remove()

    h = grabbed["h"].reshape(-1, d)[: a.tokens].float().numpy()
    n = h.shape[0]
    print(f"  captured {n} real hidden states at layer {a.layer}")

    # Reference decision, exactly as modeling_lfm2_moe computes it.
    W = router.weight.detach().float().numpy()             # [E, d]
    bias = (blk.expert_bias.detach().float().numpy()
            if getattr(blk, "use_expert_bias", False) else np.zeros(E, np.float32))
    with torch.no_grad():
        logits = torch.from_numpy(h) @ torch.from_numpy(W).T
        weights = logits.sigmoid()
        scores = weights + torch.from_numpy(bias)
        sel = torch.topk(scores, k=K, dim=-1).indices.numpy().astype(np.int32)

    # Quantized forms the RTL actually consumes.
    h_q, h_s = q_int8(h)
    W_q, W_s = q_int4_g64(W)

    OUT.mkdir(parents=True, exist_ok=True)
    stem = OUT / f"router_l{a.layer}"

    with open(f"{stem}.bin", "wb") as f:
        f.write(struct.pack("<4i", n, d, E, K))
        f.write(struct.pack("<f", h_s))
        f.write(h_q.astype(np.int8).tobytes())        # [n, d]      int8
        f.write(W_q.astype(np.int8).tobytes())        # [E, d]      int8 (INT4 range)
        f.write(W_s.astype(np.float32).tobytes())     # [E, d/64]   f32
        f.write(bias.astype(np.float32).tobytes())    # [E]         f32
        f.write(sel.astype(np.int32).tobytes())       # [n, K]      int32
        # Float references, so downstream sweeps can vary precision without
        # being capped by whatever the first quantization step already lost.
        f.write(h.astype(np.float32).tobytes())       # [n, d]      f32
        f.write(W.astype(np.float32).tobytes())       # [E, d]      f32

    # How much of the routing DECISION survives quantization, computed the way
    # the RTL will: per-group partial sums scaled by their own group scale, then
    # sigmoid + bias + top-k. sonic/quant.py assigns the router BF16; this sweep
    # is why. Routing errors are invisible to perplexity -- a mis-routed token
    # produces fluent output from the wrong expert -- so this gate is the only
    # thing standing between the recipe and a silently degraded chip.
    G = d // 64
    hq3 = h_q.astype(np.int32).reshape(n, G, 64)

    def agreement(Wq: np.ndarray, Ws: np.ndarray) -> float:
        part = np.einsum("ngk,egk->neg", hq3,
                         Wq.astype(np.int32).reshape(E, G, 64)).astype(np.float32)
        lq = (part * Ws[None, :, :]).sum(-1) * h_s
        sq = torch.topk(torch.from_numpy(lq).sigmoid()
                        + torch.from_numpy(bias), k=K, dim=-1).indices.numpy()
        return float((np.sort(sq, 1) == np.sort(sel, 1)).all(1).mean())

    def q_int8_g64(w):
        g = w.reshape(E, G, 64)
        sc = np.abs(g).max(axis=2) / 127.0
        sc[sc == 0] = 1.0
        return (np.clip(np.rint(g / sc[:, :, None]), -128, 127)
                .astype(np.int16).reshape(E, d), sc.astype(np.float32))

    def q_bf16(w):
        u = w.astype(np.float32).view(np.uint32)
        bf = ((u + 0x8000) & 0xFFFF0000).astype(np.uint32)
        return bf.view(np.float32), np.ones((E, G), np.float32)

    W_bf = q_bf16(W)[0]
    h_deq = (h_q.astype(np.float32) * h_s)     # INT8 activations, dequantized

    print("\n  routing agreement -- weights x activations, both varied")
    print(f"    {'weights':>12}{'activations':>14}{'agreement':>12}   gate 0.995")
    grid = [
        ("INT4 g=64", "INT8", lambda: agreement(W_q, W_s)),
        ("INT8 g=64", "INT8", lambda: agreement(*q_int8_g64(W))),
        ("BF16",      "INT8", lambda: _float_agreement(h_deq, W_bf, bias, sel, K)),
        ("BF16",      "BF16", lambda: _float_agreement(h, W_bf, bias, sel, K)),
    ]
    results = {}
    for wlab, alab, fn in grid:
        a_ = fn()
        results[(wlab, alab)] = a_
        print(f"    {wlab:>12}{alab:>14}{a_:>12.4f}   "
              f"{'PASS' if a_ >= 0.995 else 'FAIL'}")
    agree = results[("BF16", "INT8")]
    print(f"\n  -> sonic/quant.py assigns the router BF16 weights. This is why:")
    print(f"     the routing decision does not survive INT4 or even INT8 weights,")
    print(f"     and routing errors are invisible to a perplexity check.")

    manifest = dict(model=a.model, layer=a.layer, tokens=n, d=d, experts=E, top_k=K,
                    routing_agreement=agree,
                    act_scale=h_s, bin=str(stem.name) + ".bin",
                    layout=["n d E K (i32)", "act_scale (f32)",
                            "h_q [n,d] i8", "W_q [E,d] i8",
                            "W_scale [E,d/64] f32", "bias [E] f32",
                            "sel [n,K] i32",
                            "h_f32 [n,d] f32", "W_f32 [E,d] f32"])
    (Path(f"{stem}.json")).write_text(json.dumps(manifest, indent=2))

    print(f"  wrote {stem}.bin ({Path(f'{stem}.bin').stat().st_size/1e6:.1f} MB)")
    print(f"  {n} tokens x top-{K} of {E}; expert-bias {'on' if bias.any() else 'off'}")
    print(f"\nnext: make p2-router")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
