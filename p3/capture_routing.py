#!/usr/bin/env python3
"""P3-1: capture real MoE routing decisions from LFM2.5-8B-A1B.

This closes the highest-value open item in p0/README.md. Every number in p1/
currently rests on a synthetic lognormal-plus-stickiness routing model, and the
systolic occupancy gate is marginal under it (0.790 against a 0.80 threshold).
The real distribution decides whether the prefill engine works.

What it does: registers a forward hook on every MoE block, runs prompts through
the model, and records the top-k expert indices per token per layer. Output is
an .npz consumable by p0/routing_trace.py and p1/occupancy.py.

    python3 p3/capture_routing.py --tokens 4096 --out p0/out/real_routing.npz
    python3 p0/routing_trace.py trace --trace p0/out/real_routing.npz
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL = "LiquidAI/LFM2.5-8B-A1B"

PROMPTS = [
    "Explain why mixture-of-experts models are bandwidth-friendly at large batch "
    "sizes but not at batch one.",
    "Write a Python function that computes the expected number of distinct "
    "experts touched by a batch of tokens under top-k routing.",
    "Summarise the tradeoff between SRAM capacity and prefill chunk size in an "
    "MoE inference accelerator.",
    "What is the difference between a weight-stationary and an output-stationary "
    "systolic dataflow?",
    "Translate to French: The accumulator overflows when the reduction depth "
    "exceeds the proven bound.",
    "def fibonacci(n):\n    # return the nth Fibonacci number\n",
    "The capital of Australia is",
    "Derive the memory bandwidth required to decode 100 tokens per second from a "
    "model with 1.69 billion active parameters at 4 bits per weight.",
]


def find_routers(model):
    """Every top-k router.

    transformers' Lfm2MoeTopKRouter.forward returns
        (router_logits, routing_weights, selected_experts)
    so the routing decision is output[2] -- no need to recompute it, and no
    guessing at tensor shapes.
    """
    out = [(n, m) for n, m in model.named_modules()
           if type(m).__name__ == "Lfm2MoeTopKRouter"]
    if not out:
        raise RuntimeError(
            "No Lfm2MoeTopKRouter found. The implementation changed -- inspect "
            "transformers/models/lfm2_moe/modeling_lfm2_moe.py and update this.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokens", type=int, default=4096, help="target tokens to capture")
    ap.add_argument("--max-new", type=int, default=96)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--out", type=Path, default=Path("p0/out/real_routing.npz"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {a.model} on {a.device} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    cfg = model.config
    E = cfg.num_experts
    K = cfg.num_experts_per_tok
    print(f"  {cfg.num_hidden_layers} layers, {E} experts, top-{K}")

    blocks = find_routers(model)
    print(f"  hooking {len(blocks)} routers: {blocks[0][0]} ... {blocks[-1][0]}")

    captured: dict[str, list[np.ndarray]] = {n: [] for n, _ in blocks}

    def make_hook(name):
        def hook(mod, args, output):
            # output = (router_logits, routing_weights, selected_experts)
            sel = output[2]
            captured[name].append(
                sel.reshape(-1, K).to("cpu", torch.int16).numpy())
        return hook

    handles = [m.register_forward_hook(make_hook(n)) for n, m in blocks]

    total = 0
    try:
        for i, p in enumerate(PROMPTS):
            if total >= a.tokens:
                break
            # transformers 5.x returns a BatchEncoding here, not a bare tensor.
            enc = tok.apply_chat_template([{"role": "user", "content": p}],
                                          add_generation_prompt=True,
                                          return_tensors="pt", return_dict=True)
            ids = enc["input_ids"].to(a.device)
            t1 = time.time()
            with torch.no_grad():
                out = model.generate(ids, max_new_tokens=a.max_new, do_sample=False)
            n_new = out.shape[-1] - ids.shape[-1]
            total += out.shape[-1]
            print(f"  [{i+1}/{len(PROMPTS)}] prompt {ids.shape[-1]:4d} tok "
                  f"+ {n_new:3d} generated in {time.time()-t1:5.1f}s "
                  f"({n_new/(time.time()-t1):5.1f} tok/s)  total={total}", flush=True)
    finally:
        for h in handles:
            h.remove()

    per_layer = {n: np.concatenate(v, 0) for n, v in captured.items() if v}
    if not per_layer:
        print("\nNo routing captured. The hook did not recognise this "
              "implementation's output shape -- inspect the MoE block's forward "
              "signature and extend make_hook().")
        return 1

    allrt = np.concatenate(list(per_layer.values()), 0)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, routing=allrt,
                        **{f"layer_{i}": v for i, v in enumerate(per_layer.values())})

    print(f"\ncaptured {allrt.shape[0]:,} routing decisions across "
          f"{len(per_layer)} layers -> {a.out}")
    counts = np.bincount(allrt.ravel(), minlength=E)
    print(f"  load CV {counts.std()/counts.mean():.3f}   "
          f"min/max expert share {counts.min()/counts.sum():.4f} / "
          f"{counts.max()/counts.sum():.4f}")
    print(f"\nnext: python3 p0/routing_trace.py trace --trace {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
