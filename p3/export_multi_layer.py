#!/usr/bin/env python3
"""T3.2: export N *consecutive real* transformer layers in a single forward pass.

`export_layer.py` captures one layer's FFN input/output. This captures several
layers at once, all from the *same* forward call on the *same* prompt, so each
layer's `x` is the real residual-stream value the model actually produced at
that depth -- not independently re-derived. That is what makes a multi-layer
RTL bring-up real: layer 6's reference input is layer 5's reference output,
genuinely, not a repeated single-layer number.

This does not implement attention/conv/residual-add in RTL -- sonic_top's
verified datapath is the FFN (route, gather, two GEMMs, gate, combine), per
HANDOFF.md 1.2/3.1. So "layer boundary" here means the FFN sub-block boundary
at each of several real depths, each checked independently against what the
real model produced there. It does not claim KV state is carried in RTL.

    python3 p3/export_multi_layer.py --layers 5 6 7 8 --tokens 1
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
    g = 64
    flat = w.reshape(-1, g).astype(np.float32)
    s = (np.abs(flat).max(1, keepdims=True) / 7.0).astype(np.float16).astype(np.float32)
    s[s == 0] = 1.0
    q = np.clip(np.rint(flat / s), -8, 7).astype(np.int8)
    return q.reshape(w.shape), s.reshape(w.shape[0], -1)


def export_one(blk, io, a, n, d, out_path: Path) -> dict:
    is_dense = not (type(blk).__name__ == "Lfm2MoeSparseMoeBlock" or hasattr(blk, "experts"))
    x = io["x"][:n].numpy().astype(np.float32)
    y = io["y"][:n].numpy().astype(np.float32)

    if is_dense:
        E, K = 1, 1
        sel = np.zeros((n, 1), dtype=np.int32)
        rwt = np.ones((n, 1), dtype=np.float32)
        Wr = np.zeros((1, d), dtype=np.float32)
        bias = np.zeros(1, dtype=np.float32)
        uniq = [0]
        if hasattr(blk, "gate_up_proj"):
            gu = blk.gate_up_proj.weight.detach().numpy().astype(np.float32)
            dn = blk.down_proj.weight.detach().numpy().astype(np.float32)
        elif hasattr(blk, "w1") and hasattr(blk, "w2") and hasattr(blk, "w3"):
            w1 = blk.w1.weight.detach().numpy().astype(np.float32)
            w3 = blk.w3.weight.detach().numpy().astype(np.float32)
            gu = np.concatenate([w1, w3], axis=0)
            dn = blk.w2.weight.detach().numpy().astype(np.float32)
        else:
            raise SystemExit(f"Unknown dense block architecture: {type(blk).__name__}")
        I = dn.shape[-1]
    else:
        gate = blk.gate
        experts = blk.experts
        Wr = gate.weight.detach().numpy().astype(np.float32)
        eb = getattr(blk, "expert_bias", None)
        bias = (eb.detach().numpy().astype(np.float32) if eb is not None
                else np.zeros(Wr.shape[0], np.float32))
        with torch.no_grad():
            rl, rw, rsel = gate(io["x"][:n], eb)
        K = rsel.reshape(n, -1).shape[1]
        E = Wr.shape[0]
        sel = rsel.reshape(n, K).numpy().astype(np.int32)
        rwt = rw.reshape(n, K).numpy().astype(np.float32)
        uniq = sorted(set(sel.ravel().tolist()))
        gu_all = experts.gate_up_proj.detach().numpy().astype(np.float32)
        dn_all = experts.down_proj.detach().numpy().astype(np.float32)
        I = dn_all.shape[-1]

    h_q, act_scale = q_int8(x)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
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

    return dict(is_dense=is_dense, unique_experts=uniq, tokens=n, d=d,
                experts=E, top_k=K, inter=I,
                bin=out_path.name, size_mb=round(out_path.stat().st_size / 1e6, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--layers", type=int, nargs="+", default=[5, 6, 7, 8])
    ap.add_argument("--tokens", type=int, default=1)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", type=Path, default=Path("p2/vectors/multi"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"loading {a.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.float32, device_map=a.device).eval()
    cfg = model.config
    d = cfg.hidden_size

    io = {l: {} for l in a.layers}
    handles = []
    for l in a.layers:
        blk = model.model.layers[l].feed_forward
        handles.append(blk.register_forward_pre_hook(
            (lambda l: lambda m, args: io[l].__setitem__(
                "x", args[0].detach().reshape(-1, d).clone()))(l)))
        handles.append(blk.register_forward_hook(
            (lambda l: lambda m, args, out: io[l].__setitem__(
                "y", (out[0] if isinstance(out, tuple) else out).detach().reshape(-1, d).clone()))(l)))

    ids = tok(PROMPT, return_tensors="pt").input_ids[:, :a.tokens]
    with torch.no_grad():
        model(ids.to(a.device))
    for h in handles:
        h.remove()

    manifest = {"model": a.model, "layers": [], "note":
                "each layer's x is the model's real residual-stream input at "
                "that depth, from one shared forward pass -- so layer[i+1].x "
                "is genuinely downstream of layer[i].y, not re-synthesized."}
    for l in a.layers:
        blk = model.model.layers[l].feed_forward
        n = io[l]["x"].shape[0]
        out_path = a.out_dir / f"layer_l{l}.bin"
        meta = export_one(blk, io[l], a, n, d, out_path)
        meta["layer"] = l
        (a.out_dir / f"layer_l{l}.json").write_text(json.dumps(meta, indent=2))
        print(f"  layer {l}: {'dense' if meta['is_dense'] else 'moe top-' + str(meta['top_k'])}, "
              f"{meta['size_mb']} MB -> {out_path}")
        manifest["layers"].append(meta)

    (a.out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote manifest to {a.out_dir / 'manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
