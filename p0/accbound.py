#!/usr/bin/env python3
"""P0-5: per-layer accumulator bounds from real activations.

`sonic_golden.h` proves a worst-case bound and `quant.ACCUM` ships 16 local
bits because of it:

    fold 16 x |w|max 8 x |a|max 128 = 16,384  ->  16 bits

That bound assumes all 16 products are simultaneously maximal and identically
signed. Real activations do not do that, and p2/README.md records what the
conservative choice cost: the 12 -> 16 bit fix added 4 logic levels, 66 -> 70.
Given P4 has just shown the router is depth-limited, buying levels back is worth
measuring rather than assuming.

What is measured: the LOCAL accumulator, which holds a fold-16 partial sum of
raw INT4 x INT8 *codes*. `sonic_streamer.sv` applies the FP16 group scale to the
accumulated sum, not to individual weights, so the accumulator never sees
dequantized values -- codes are the right domain and dequantizing here would
measure a different machine.

    python3 p0/accbound.py --tokens 512

This reports a measured maximum, which is evidence and not a proof. The gap
between it and the worst case is the safety margin being proposed, and the
report prints that gap explicitly so the decision is made with it in view.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic import quant  # noqa: E402
from p0.gates import classify  # noqa: E402

MODEL = "LiquidAI/LFM2.5-8B-A1B"

PROMPTS = [
    "The accumulator overflows when the reduction depth exceeds the proven bound.",
    "Explain how a systolic array computes a matrix multiplication.",
    "def merge_sort(a):\n    if len(a) <= 1:\n        return a\n",
    "Summarise the causes of the First World War in one paragraph.",
    "Translate to Spanish: The memory controller reorders requests to exploit "
    "row-buffer locality.",
    "What is the difference between static and dynamic random access memory?",
]


def q_act_int8(x: torch.Tensor) -> torch.Tensor:
    """Per-token dynamic INT8, as `quant.ACTIVATION` declares."""
    s = x.abs().amax(-1, keepdim=True).clamp(min=1e-8) / 127.0
    return (x / s).round().clamp(-127, 127)


def q_wt_codes(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Integer codes the array actually multiplies, per group of `group`."""
    qmax = 2 ** (bits - 1) - 1
    flat = w.reshape(-1, group).float()
    s = (flat.abs().amax(1, keepdim=True) / qmax).to(torch.float16).float()
    s = torch.where(s == 0, torch.ones_like(s), s)
    return (flat / s).round().clamp(-qmax - 1, qmax).reshape(w.shape)


@torch.no_grad()
def local_max(qx: torch.Tensor, qw: torch.Tensor, fold: int) -> tuple[float, np.ndarray]:
    """Max |fold-deep partial sum| over every token, output row and group.

    Done group by group as a (T x fold) @ (fold x O) matmul so nothing of size
    T*O*K is ever materialised. Returns the max and a sample of magnitudes for
    the tail report.
    """
    T, K = qx.shape
    O = qw.shape[0]
    K -= K % fold
    g = K // fold
    xs = qx[:, :K].reshape(T, g, fold)
    ws = qw[:, :K].reshape(O, g, fold)
    peak, samples = 0.0, []
    for i in range(g):
        acc = xs[:, i, :] @ ws[:, i, :].T          # [T, O]
        peak = max(peak, float(acc.abs().amax()))
        if i % max(1, g // 8) == 0:
            samples.append(acc.abs().flatten()[::97].to("cpu", torch.float32).numpy())
    return peak, np.concatenate(samples) if samples else np.zeros(1)


def bits_for(v: float) -> int:
    """Signed bits needed to hold +/- v."""
    return int(np.ceil(np.log2(max(v, 1.0) + 1))) + 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--tokens", type=int, default=512)
    ap.add_argument("--fold", type=int, default=quant.ACCUM["fold"])
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--corpus-offset", type=int, default=0,
                    help="token offset into WikiText-2; -1 uses the built-in prompts")
    ap.add_argument("--out", type=Path, default=Path("p0/out/accbound.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch.nn as nn

    print(f"loading {a.model} on {a.device} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device).eval()

    # Capture the input to every quantized matmul. The fused expert stack needs
    # the routing replayed to see what each expert actually consumes; that is
    # already solved in the packer, so reuse it rather than re-deriving it.
    caught: dict[str, torch.Tensor] = {}

    def grab(name):
        def f(mod, args):
            if args and isinstance(args[0], torch.Tensor) and name not in caught:
                x = args[0].detach()
                caught[name] = x.reshape(-1, x.shape[-1])[:a.tokens].float()
        return f

    handles = []
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            handles.append(m.register_forward_pre_hook(grab(n + ".weight")))
        elif type(m).__name__ == "Lfm2MoeExperts":
            handles.append(m.register_forward_pre_hook(grab(n + ".gate_up_proj")))

    # WikiText-2 rather than the short built-in prompts: with only ~100 tokens
    # of text, --tokens was silently a no-op and every sweep measured the same
    # activations, which makes a stability check meaningless.
    if a.corpus_offset >= 0:
        from p0.gates import wikitext2_text
        full = tok(wikitext2_text()[:400_000], return_tensors="pt").input_ids[0]
        ids = full[a.corpus_offset:a.corpus_offset + a.tokens].unsqueeze(0)
    else:
        ids = tok("\n\n".join(PROMPTS), return_tensors="pt").input_ids[:, :a.tokens]
    with torch.no_grad():
        model(ids.to(a.device))
    for h in handles:
        h.remove()
    print(f"  captured {len(caught)} activation tensors")

    params = dict(model.named_parameters())
    rows, t0 = [], time.time()
    for pname, x in sorted(caught.items()):
        if pname not in params:
            continue
        w = params[pname].data
        block = classify(pname, w)
        if block is None:
            continue
        f = quant.BLOCK_FMT[block]
        bits = {"int4": 4, "int8g": 8, "int12": 12}.get(f.kind)
        if bits is None:                      # per-tensor INT8: conv kernels
            continue
        w2 = w[0] if w.ndim == 3 else w       # one expert of a fused stack
        if w2.shape[-1] != x.shape[-1]:
            continue
        qx = q_act_int8(x.to(w.device))
        qw = q_wt_codes(w2, bits, f.group or 64)
        peak, samp = local_max(qx, qw, a.fold)
        rows.append(dict(param=pname, block=block, wbits=bits,
                         peak=peak, bits=bits_for(peak),
                         p99_9=float(np.percentile(samp, 99.9)),
                         worst_case=a.fold * (2 ** (bits - 1)) * 128))
    print(f"  swept {len(rows)} tensors in {time.time()-t0:.1f}s\n")

    by_block: dict[str, list] = {}
    for r in rows:
        by_block.setdefault(r["block"], []).append(r)

    print(f"{'block':26s} {'w':>3s} {'worst case':>11s} {'measured max':>13s} "
          f"{'bits':>5s} {'margin':>8s}")
    print("-" * 74)
    for b, rs in sorted(by_block.items()):
        peak = max(r["peak"] for r in rs)
        wc = rs[0]["worst_case"]
        print(f"{b:26s} {rs[0]['wbits']:3d} {wc:11,d} {peak:13,.0f} "
              f"{bits_for(peak):5d} {wc/max(peak,1):7.1f}x")
    print("-" * 74)
    overall = max(r["peak"] for r in rows) if rows else 0
    need = bits_for(overall)
    print(f"{'ALL BLOCKS':26s} {'':3s} {'':11s} {overall:13,.0f} {need:5d}")
    print(f"\nshipping local_bits = {quant.ACCUM['local_bits']}  "
          f"(worst-case proof); measured needs {need}")
    twelve = 2 ** 11 - 1
    print(f"12-bit signed holds +/-{twelve:,}: "
          f"{'FITS' if overall <= twelve else 'DOES NOT FIT'} "
          f"(measured max {overall:,.0f}, {'%.1fx' % (twelve/max(overall,1))} headroom)")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(dict(
        tokens=a.tokens, fold=a.fold, measured_max=overall, bits_needed=need,
        shipping_local_bits=quant.ACCUM["local_bits"], rows=rows), indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
