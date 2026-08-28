#!/usr/bin/env python3
"""Text in, text out, under the chip's weight formats.

The obvious question after `make p3-layer` is whether the thing actually talks.
Running generation through the RTL is not on: one MoE layer takes minutes in
Verilator, and a sentence needs 24 layers times tens of tokens. So this answers
the question the RTL result leaves open -- does the model still produce sensible
text once every weight is in the format the silicon reads -- by generating twice
from the same prompts, once in BF16 and once with `sonic/quant.py` applied.

Scope, stated plainly. This covers the WEIGHT FORMATS: INT4 group-64 with an
outlier budget on the experts, INT8 group-64 on attention and the dense FFN,
INT4 group-32 on the tied embedding, INT12 on the routers. It does NOT cover the
accumulator widths or the PWL activation tables -- those are `p0/accbound.py`
and `p2/tb/tb_layer.cpp`, which measure them directly on the RTL.

    python3 p3/generate.py --tokens 48
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from p0.gates import quantize_  # noqa: E402
from sonic import quant  # noqa: E402

MODEL = "LiquidAI/LFM2.5-8B-A1B"

PROMPTS = [
    "Explain in two sentences why a mixture-of-experts model is bandwidth-bound "
    "at batch size one.",
    "Write a Python function that returns the nth triangular number.",
    "What is the capital of Australia, and why is it not Sydney?",
]


def generate(model, tok, prompt: str, n: int, device: str) -> str:
    enc = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                  add_generation_prompt=True,
                                  return_tensors="pt", return_dict=True)
    ids = enc["input_ids"].to(device)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=n, do_sample=False)
    return tok.decode(out[0, ids.shape[-1]:], skip_special_tokens=True).strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"loading {a.model} on {a.device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device).eval()

    print("\n" + "=" * 74)
    print("BF16 reference")
    print("=" * 74)
    base = []
    for p in PROMPTS:
        t0 = time.time()
        r = generate(model, tok, p, a.tokens, a.device)
        base.append(r)
        print(f"\n> {p}\n{r}\n  [{time.time()-t0:.1f}s]")

    print("\n" + "=" * 74)
    print("applying sonic/quant.py to every weight")
    print("=" * 74)
    cov = quantize_(model, quant.BLOCK_FMT)

    print("\n" + "=" * 74)
    print("Sonic S1 weight formats")
    print("=" * 74)
    same = 0
    for p, b in zip(PROMPTS, base):
        t0 = time.time()
        r = generate(model, tok, p, a.tokens, a.device)
        same += (r == b)
        print(f"\n> {p}\n{r}\n  [{time.time()-t0:.1f}s]")

    per = cov["per_block"]
    bits = sum(v[1] for v in per.values()) / sum(v[0] for v in per.values())
    print("\n" + "=" * 74)
    print(f"{len(PROMPTS)} prompts, {same} identical to BF16 token for token")
    print(f"applied {bits:.3f} bits per resident weight")
    print("Read the text, not the count: greedy decode diverges permanently after")
    print("one differing token, so 'identical' is a strict bar and a paraphrase")
    print("is not a failure. What would be a failure is incoherence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
