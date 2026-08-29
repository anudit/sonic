#!/usr/bin/env python3
"""P3-5: export one complete MoE layer so the RTL can compute it end to end.

`export_vectors.py` exports the router's inputs, which proved the router picks
the experts the model picks. This exports everything needed to compute the
LAYER: after routing, the selected experts' weights, and the hidden state
PyTorch produces from them. That is the chip's core datapath -- route, gather,
two GEMMs, gate, combine -- run on real weights and checked against the model.

Only the experts a token actually selects are exported. All 32 would be 370 MB
per layer; the top-4 for one token is ~46 MB.

    python3 p3/export_layer.py --layer 5 --tokens 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "LiquidAI/LFM2.5-8B-A1B"
PROMPT = ("A systolic array computes a matrix multiplication by streaming "
          "operands through a grid of multiply-accumulate cells.")


def q_int8(x: np.ndarray) -> tuple[np.ndarray, float]:
    s = float(np.abs(x).max()) / 127.0 or 1.0
    return np.clip(np.rint(x / s), -127, 127).astype(np.int8), s


def q_int4_g64(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """INT4 codes plus one FP16-rounded scale per group of 64 along the last axis."""
    g = 64
    flat = w.reshape(-1, g).astype(np.float32)
    s = (np.abs(flat).max(1, keepdims=True) / 7.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.rint(flat / s), -8, 7).astype(np.int8)
    return q.reshape(w.shape), s.reshape(w.shape[0], -1)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("p2/vectors/layer_l5.bin"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, device_map=a.device).eval()
    cfg = model.config
    d, E, K = cfg.hidden_size, cfg.num_experts, cfg.num_experts_per_tok

    blk = model.model.layers[a.layer].feed_forward
    is_dense = not (type(blk).__name__ == "Lfm2MoeSparseMoeBlock" or hasattr(blk, "experts"))

    # Capture the block's input and output: the reference this must reproduce.
    io = {}
    h_pre = blk.register_forward_pre_hook(
        lambda m, args: io.__setitem__("x", args[0].detach().reshape(-1, d).clone()))
    h_post = blk.register_forward_hook(
        lambda m, args, out: io.__setitem__(
            "y", (out[0] if isinstance(out, tuple) else out).detach().reshape(-1, d).clone()))
    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :a.tokens]
    with torch.no_grad():
        model(ids.to(a.device))
    h_pre.remove(); h_post.remove()

    x = io["x"][:a.tokens].numpy().astype(np.float32)
    y = io["y"][:a.tokens].numpy().astype(np.float32)
    n = x.shape[0]

    if is_dense:
        # Dense MLP layer (e.g. 2.6B or leading layers of MoE)
        E, K = 1, 1
        sel = np.zeros((n, 1), dtype=np.int32)
        rwt = np.ones((n, 1), dtype=np.float32)
        Wr = np.zeros((1, d), dtype=np.float32)
        bias = np.zeros(1, dtype=np.float32)
        uniq = [0]

        # Handle gate_up_proj and down_proj
        if hasattr(blk, "gate_up_proj"):
            gu = blk.gate_up_proj.weight.detach().numpy().astype(np.float32) # [2*I, d]
            dn = blk.down_proj.weight.detach().numpy().astype(np.float32)    # [d, I]
        elif hasattr(blk, "w1") and hasattr(blk, "w2") and hasattr(blk, "w3"):
            w1 = blk.w1.weight.detach().numpy().astype(np.float32)
            w3 = blk.w3.weight.detach().numpy().astype(np.float32)
            gu = np.concatenate([w1, w3], axis=0)
            dn = blk.w2.weight.detach().numpy().astype(np.float32)
        else:
            raise SystemExit(f"Unknown dense block architecture: {type(blk).__name__}")
        I = dn.shape[-1]
        print(f"  {n} token(s), dense FFN; intermediate={I}")
    else:
        gate = blk.gate                       # Lfm2MoeTopKRouter
        experts = blk.experts                 # Lfm2MoeExperts
        Wr = gate.weight.detach().numpy().astype(np.float32)          # [E, d]
        eb = getattr(blk, "expert_bias", None)
        bias = (eb.detach().numpy().astype(np.float32) if eb is not None
                else np.zeros(E, np.float32))

        with torch.no_grad():
            rl, rw, rsel = gate(io["x"][:a.tokens], eb)
        sel = rsel.reshape(n, K).numpy().astype(np.int32)
        rwt = rw.reshape(n, K).numpy().astype(np.float32)
        uniq = sorted(set(sel.ravel().tolist()))
        print(f"  {n} token(s), top-{K} of {E}; unique experts {uniq}")

        gu_all = experts.gate_up_proj.detach().numpy().astype(np.float32)   # [E, 2*I, d]
        dn_all = experts.down_proj.detach().numpy().astype(np.float32)      # [E, d, I]
        I = dn_all.shape[-1]

    h_q, act_scale = q_int8(x)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("wb") as f:
        np.array([n, d, E, K, I, len(uniq)], np.int32).tofile(f)
        np.array([act_scale], np.float32).tofile(f)
        h_q.tofile(f)
        x.tofile(f)
        y.tofile(f)
        Wr.tofile(f)
        bias.tofile(f)
        sel.tofile(f)
        rwt.tofile(f)
        np.array(uniq, np.int32).tofile(f)
        for e in uniq:
            gu_mat = gu if is_dense else gu_all[e]
            dn_mat = dn if is_dense else dn_all[e]
            q, s = q_int4_g64(gu_mat); q.tofile(f); s.tofile(f)
            q, s = q_int4_g64(dn_mat); q.tofile(f); s.tofile(f)

    meta = dict(model=a.model, layer=a.layer, tokens=n, d=d, experts=E, top_k=K,
                inter=I, unique_experts=uniq, act_scale=float(act_scale),
                bin=a.out.name, size_mb=round(a.out.stat().st_size / 1e6, 1),
                layout=["n d E K I U (i32)", "act_scale f32", "h_q[n,d] i8",
                        "x[n,d] f32", "y_ref[n,d] f32", "Wr[E,d] f32",
                        "bias[E] f32", "sel[n,K] i32", "rwt[n,K] f32",
                        "uniq[U] i32",
                        "per expert: gu_q[2I,d] i8, gu_s[2I,d/64] f32, "
                        "dn_q[d,I] i8, dn_s[d,I/64] f32"])
    a.out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"  wrote {a.out} ({meta['size_mb']} MB) and {a.out.with_suffix('.json')}")
    print(f"  reference |y| mean {np.abs(y).mean():.5f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
