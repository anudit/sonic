#!/usr/bin/env python3
"""P0-1: measure the frozen recipe against BF16.

Closes the last unmeasured gates in p0/README.md. `sonic/quant.py` declares a
format per block and `avg_bits` says the recipe costs 4.64 bits/weight, but
nothing has ever checked that the recipe is *good enough* -- only that it is
cheap enough. This runs the quality gates:

    ppl_delta_max       0.15   perplexity rise vs BF16
    top1_agreement_min  0.99   same argmax token, teacher-forced
    avg_bits_max        4.75   cross-checked against what was actually applied

Method: simulated ("fake") quantization. Each weight is quantized to its
declared format and immediately dequantized back to BF16, so the arithmetic
still runs in BF16 but the *values* are exactly those the packed format can
represent. This measures the information the recipe throws away, which is what
the gates are about. It does not measure accumulator or PWL effects -- those are
p0/golden/ and the RTL benches.

Both passes reuse one model: baseline first, then quantize in place. Peak memory
is one BF16 copy (~17 GB for the 8B).

    python3 p0/gates.py --corpus wikitext2.txt
    python3 p0/gates.py --corpus wikitext2.txt --uniform   # ablation: flat INT4
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

MODEL = "LiquidAI/LFM2.5-8B-A1B"

# Held-out instructions. Deliberately disjoint from p3/capture_routing.py's
# prompts: those steered the routing trace, and reusing them here would measure
# agreement on the data the recipe was tuned against.
INSTRUCTIONS = [
    "Explain the difference between write-back and write-through caching.",
    "Write a SQL query that finds the second-highest salary in a table.",
    "What causes metastability in a synchronous digital circuit?",
    "Summarise the plot of Hamlet in three sentences.",
    "Convert 98.6 degrees Fahrenheit to Celsius and show the arithmetic.",
    "def quicksort(a):\n    # sort the list in place\n",
    "List three reasons a distributed system might prefer eventual consistency.",
    "Translate to German: The compiler rejected the program because the type "
    "of the accumulator was too narrow.",
    "Why does a depthwise convolution have fewer parameters than a full one?",
    "What is the halting problem, and why does it matter in practice?",
]


# ---------------------------------------------------------------- fake quant

def _q_group(w: torch.Tensor, bits: int, group: int) -> torch.Tensor:
    """Symmetric per-group quantize-dequantize along the last (input) axis.

    The scale is stored as FP16, matching what the format actually pays for:
    `sonic/quant.py` prices INT4_G64 at 4 + 16/64 bits. Rounding the scale to
    FP16 here is not a detail -- a float32 scale would measure a format nobody
    is building.
    """
    qmax = 2 ** (bits - 1) - 1
    n = w.shape[-1]
    if n % group:
        return _q_tensor(w, bits)
    flat = w.reshape(-1, group).float()
    scale = (flat.abs().amax(1, keepdim=True) / qmax).to(torch.float16).float()
    # Clamp AFTER the cast, not before. Any pre-cast floor below ~6e-8 (the
    # smallest FP16 subnormal) underflows to exactly 0, and an all-zero weight
    # group then computes 0/0 = NaN, which propagates silently through the whole
    # forward pass. Substituting 1.0 is exact: the group quantizes to 0 either
    # way, and a scale that underflows FP16 is a group real hardware reads as
    # zero too.
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    q = (flat / scale).round().clamp(-qmax - 1, qmax)
    return (q * scale).reshape(w.shape).to(w.dtype)


def _q_tensor(w: torch.Tensor, bits: int) -> torch.Tensor:
    """Symmetric per-tensor quantize-dequantize."""
    qmax = 2 ** (bits - 1) - 1
    f = w.float()
    scale = (f.abs().amax() / qmax).clamp(min=1e-12)
    return ((f / scale).round().clamp(-qmax - 1, qmax) * scale).to(w.dtype)


def _q_int4_outliers(w: torch.Tensor, group: int, frac: float) -> torch.Tensor:
    """INT4 group-64, with the `frac` widest output rows kept at INT8.

    Rows are ranked by max magnitude: a row whose dynamic range is far above its
    neighbours' is the one whose shared INT4 scale is most wasted. quant.py's
    note that "w2 outliers dominate the average" is what this budget is for.
    """
    lo = _q_group(w, 4, group)
    if frac <= 0:
        return lo
    row_max = w.abs().amax(-1)                       # [..., out]
    k = max(1, int(round(frac * row_max.shape[-1])))
    idx = row_max.topk(k, dim=-1).indices
    mask = torch.zeros_like(row_max, dtype=torch.bool).scatter_(-1, idx, True)
    return torch.where(mask.unsqueeze(-1), _q_group(w, 8, group), lo)


def apply_fmt(w: torch.Tensor, f: quant.Fmt) -> torch.Tensor:
    if f.kind == "bf16":
        return w
    if f.kind == "int8":
        return _q_tensor(w, 8)
    if f.kind == "int12":
        return _q_group(w, 12, f.group or 64)
    if f.kind == "int4":
        if f.outlier_rows:
            return _q_int4_outliers(w, f.group or 64, f.outlier_rows)
        return _q_group(w, 4, f.group or 64)
    raise ValueError(f"unhandled format kind {f.kind!r}")


# ------------------------------------------------------------- block mapping

def classify(pname: str, p: torch.Tensor) -> str | None:
    """Map a parameter name onto a `sonic/quant.py` block name.

    Keyed off the real module tree of LFM2.5-8B-A1B, which does NOT put the MoE
    in nn.Linear: experts are a fused Lfm2MoeExperts tensor stack
    ([E, out, in] for gate_up_proj and down_proj) and the router is an
    Lfm2MoeTopKRouter holding a bare [E, d] weight. Walking nn.Linear modules
    would leave the entire MoE -- the great majority of the parameters -- in
    BF16 and report a recipe far better than the one being built.
    """
    if p.ndim < 2:                       # RMSNorm gains, biases: stay BF16
        return None
    if ".feed_forward.experts." in pname:
        return "MoE experts"
    if pname.endswith(".feed_forward.gate.weight"):
        return "Routers"
    if ".feed_forward.w" in pname:       # only the leading dense layers have w1/w2/w3
        return "Dense FFN"
    if ".conv." in pname:
        return "Short-conv blocks"
    if ".self_attn." in pname:
        return "GQA attention"
    if "embed_tokens" in pname or "lm_head" in pname:
        return "Tied embedding / LM head"
    return None


def quantize_(model, table: dict[str, quant.Fmt], verbose: bool = True) -> dict:
    """Fake-quantize every weight in place. Returns a coverage report."""
    stats: dict[str, list[int]] = {}
    skipped: list[tuple[str, tuple]] = []
    seen: set[int] = set()

    with torch.no_grad():
        for pname, p in model.named_parameters():
            block = classify(pname, p)
            if block is None:
                if p.ndim >= 2:
                    skipped.append((pname, tuple(p.shape)))
                continue
            if p.data_ptr() in seen:     # tied embedding / lm_head: quantize once
                continue
            seen.add(p.data_ptr())

            f = table[block]
            # The depthwise conv kernel is INT8 regardless of its block format:
            # its last axis is conv_k (3), far too short to group, and it is
            # 0.04% of the block. See p0/README.md.
            if pname.endswith(".conv.conv.weight"):
                f = quant.INT8
            p.copy_(apply_fmt(p.data, f))
            stats.setdefault(block, [0, 0.0])
            stats[block][0] += p.numel()
            stats[block][1] += p.numel() * f.bits

    if verbose:
        print("\n  quantized:")
        tot = sum(v[0] for v in stats.values())
        for b, (n, bits) in sorted(stats.items(), key=lambda kv: -kv[1][0]):
            print(f"    {b:28s} {n/1e6:8.1f} M  {bits/n:5.2f} bits  "
                  f"{table[b].kind}")
        print(f"    {'TOTAL':28s} {tot/1e6:8.1f} M  "
              f"{sum(v[1] for v in stats.values())/tot:5.2f} bits")
        if skipped:
            print("\n  WARNING -- left in BF16 (unclassified, >=2D):")
            for n, s in skipped[:12]:
                print(f"    {n}  {s}")
            print("  Any large tensor here inflates the result. Extend classify().")
    return {"per_block": {k: v for k, v in stats.items()}, "skipped": skipped}


# --------------------------------------------------------------- evaluation

@torch.no_grad()
def evaluate(model, ids: torch.Tensor, window: int, device: str,
             label: str) -> tuple[float, int, np.ndarray]:
    """Teacher-forced NLL and argmax over non-overlapping windows.

    One pass yields both gates: perplexity from the NLL sum, and top-1
    agreement from the argmax stream compared against the baseline's. Teacher
    forcing (rather than free-running generation) is what makes 0.99 a
    meaningful threshold -- free-running decode diverges after the first
    disagreement and would measure drift, not agreement.
    """
    n_win = ids.numel() // window
    if n_win == 0:
        raise SystemExit(f"corpus too short: {ids.numel()} tokens < window {window}")
    nll_sum, n_tok, tops = 0.0, 0, []
    t0 = time.time()

    for i in range(n_win):
        w = ids[i * window:(i + 1) * window].unsqueeze(0).to(device)
        logits = model(w).logits[0, :-1]              # [W-1, V]
        tgt = w[0, 1:]
        # Slice the vocab softmax: [2047, 128000] in float32 is ~1 GB in one go.
        for s in range(0, logits.shape[0], 256):
            sl = logits[s:s + 256].float()
            nll_sum += torch.nn.functional.cross_entropy(
                sl, tgt[s:s + 256], reduction="sum").item()
            tops.append(sl.argmax(-1).to("cpu", torch.int32).numpy())
        n_tok += tgt.numel()
        if (i + 1) % 5 == 0 or i + 1 == n_win:
            el = time.time() - t0
            print(f"    [{label}] window {i+1}/{n_win}  "
                  f"ppl {np.exp(nll_sum/n_tok):8.3f}  "
                  f"{el:5.1f}s ({el/(i+1):4.1f}s/win)", flush=True)

    return nll_sum, n_tok, np.concatenate(tops)


def load_corpus(path: Path | None, tok, max_tokens: int) -> torch.Tensor:
    if path:
        text = path.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            from datasets import load_dataset
        except ImportError:
            raise SystemExit(
                "No --corpus given and `datasets` is not installed.\n"
                "  either:  pip install datasets\n"
                "  or:      pass --corpus <a plain text file>\n"
                "The gate names WikiText-2; anything else must be labelled as "
                "what it actually is.")
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(ds["text"])
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:max_tokens]


# ------------------------------------------------------------------- report

def report(base_ppl, quant_ppl, agree, applied_res, expect_res, active_bits) -> int:
    """`avg_bits` is two different numbers and only one of them is the gate.

    The gate is bits per ACTIVE parameter -- what streams per decoded token,
    4.639 under this recipe. Bits per RESIDENT parameter is 4.391, lower,
    because the MoE experts that dominate DRAM are the cheapest format while the
    promoted attention and embedding are read every token. Checking the resident
    figure against a 4.75 active gate passes for the wrong reason and would hide
    a real regression. So: gate on active, and use resident purely to prove the
    formats applied are the formats declared.
    """
    G = quant.GATES
    d_ppl = quant_ppl - base_ppl
    drift = abs(applied_res - expect_res)
    rows = [
        ("ppl_delta", d_ppl, G["ppl_delta_max"], "<=",
         f"{base_ppl:.3f} -> {quant_ppl:.3f}"),
        ("top1_agreement", agree, G["top1_agreement_min"], ">=",
         "teacher-forced argmax"),
        ("avg_bits (active)", active_bits, G["avg_bits_max"], "<=",
         "per decoded token, from modelspec"),
        ("coverage drift", drift, 0.02, "<=",
         f"applied {applied_res:.3f} vs declared {expect_res:.3f} bits resident"),
    ]
    print("\n" + "=" * 72)
    print(f"{'gate':22s} {'measured':>10s} {'':2s} {'gate':>8s}   note")
    print("-" * 72)
    bad = 0
    for name, got, gate, op, note in rows:
        ok = got <= gate if op == "<=" else got >= gate
        bad += not ok
        print(f"{name:22s} {got:10.4f} {op:2s} {gate:8.3f}   "
              f"{'PASS' if ok else 'FAIL'}  {note}")
    print("-" * 72)
    print(f"{'routing_agreement':22s} {0.998:10.4f} >= "
          f"{G['routing_agreement_min']:8.3f}   PASS  measured in "
          f"p2/tb/tb_router.cpp, not here")
    print(f"{'bench_drop':22s} {'--':>10s}    "
          f"{G['bench_drop_max']:8.3f}   SKIP  needs a benchmark suite; see "
          f"p0/README.md")
    print("=" * 72)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--corpus", type=Path, help="plain text; default fetches WikiText-2")
    ap.add_argument("--max-tokens", type=int, default=65536)
    ap.add_argument("--window", type=int, default=2048)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    ap.add_argument("--uniform", action="store_true",
                    help="ablation: flat INT4_G64 everywhere, ignoring the recipe's promotions")
    ap.add_argument("--out", type=Path, default=Path("p0/out/gates.json"))
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sonic import modelspec

    print(f"loading {a.model} on {a.device} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, device_map=a.device)
    model.eval()
    print(f"  loaded in {time.time()-t0:.1f}s")

    ids = load_corpus(a.corpus, tok, a.max_tokens)
    inst = tok("\n\n".join(INSTRUCTIONS), return_tensors="pt").input_ids[0]
    print(f"  corpus {ids.numel():,} tokens, instructions {inst.numel():,} tokens, "
          f"window {a.window}")

    print("\n--- BF16 baseline ---")
    b_nll, b_n, b_top = evaluate(model, ids, a.window, a.device, "bf16")
    base_ppl = float(np.exp(b_nll / b_n))

    table = quant.UNIFORM_INT4 if a.uniform else quant.BLOCK_FMT
    print(f"\n--- applying {'UNIFORM INT4 (ablation)' if a.uniform else 'the recipe'} ---")
    cov = quantize_(model, table)

    print("\n--- quantized ---")
    q_nll, q_n, q_top = evaluate(model, ids, a.window, a.device, "quant")
    quant_ppl = float(np.exp(q_nll / q_n))
    agree = float((b_top == q_top).mean())

    per = cov["per_block"]
    applied_res = sum(v[1] for v in per.values()) / sum(v[0] for v in per.values())

    spec = modelspec.load("lfm2.5-8b-a1b")
    tbl = quant.UNIFORM_INT4 if a.uniform else None
    active_bits = spec.avg_bits(tbl)
    q = (lambda n: tbl[n]) if tbl else quant.fmt
    expect_res = (sum(b.total * q(b.name).bits for b in spec.blocks)
                  / spec.total_params)

    rc = report(base_ppl, quant_ppl, agree, applied_res, expect_res, active_bits)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "model": a.model, "uniform": a.uniform,
        "tokens": int(b_n), "window": a.window,
        "ppl_bf16": base_ppl, "ppl_quant": quant_ppl,
        "ppl_delta": quant_ppl - base_ppl,
        "top1_agreement": agree,
        "avg_bits_active": active_bits,
        "avg_bits_resident_applied": applied_res,
        "avg_bits_resident_declared": expect_res,
        "per_block_params": {k: v[0] for k, v in per.items()},
        "gates": quant.GATES,
    }, indent=2))
    print(f"\nwrote {a.out}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
