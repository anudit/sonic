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
    """Symmetric per-tensor quantize-dequantize.

    "Per tensor" means per *logical* tensor. A 3-D weight here is a stack of
    matrices that are separate in hardware -- 32 experts in a fused
    Lfm2MoeExperts [E, out, in], or the 2048 independent channels of a depthwise
    conv kernel [C, 1, K] -- so each leading slice gets its own scale. One scale
    spanning the whole stack lets the single widest expert or channel set the
    step size for every other, quantizing the quiet ones to zero. Measured: that
    mistake alone cost 11% of top-1 agreement from the conv kernels, which are
    0.0013% of the parameters.
    """
    qmax = 2 ** (bits - 1) - 1
    f = w.float()
    dims = (-2, -1) if f.ndim == 3 else tuple(range(f.ndim))
    scale = f.abs().amax(dim=dims, keepdim=True) / qmax
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
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
    if f.kind == "int8g":
        return _q_group(w, 8, f.group or 64)
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


def quantize_(model, table: dict[str, quant.Fmt], verbose: bool = True,
              mode: str = "rtn", acts: dict | None = None) -> dict:
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
            if pname.endswith(".conv.conv.weight") and f.kind != "bf16":
                f = quant.INT8
            if mode == "rtn":
                p.copy_(apply_fmt(p.data, f))
            else:
                from p0 import packer
                p.copy_(packer.pack(p.data, f, mode,
                                    (acts or {}).get(pname)))
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

def bootstrap_delta(nll_b: np.ndarray, nll_q: np.ndarray, win: int,
                    n_boot: int = 4000, seed: int = 0) -> tuple[float, float]:
    """95% CI on ppl_delta, resampling WINDOWS rather than tokens.

    Tokens inside a window are strongly correlated -- the model is conditioning
    on the same prefix -- so a token-level bootstrap would report a confidence
    interval several times too narrow. Windows are the independent unit.

    The comparison is paired: both passes see identical token sequences in
    identical order, so the same resampled windows are used for both. That
    cancels most of the corpus-difficulty variance and is why a few tens of
    thousands of tokens resolve a 0.15 gate that would need far more if the two
    runs were independent samples.
    """
    n = (nll_b.size // win) * win
    b = nll_b[:n].reshape(-1, win)
    q = nll_q[:n].reshape(-1, win)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, b.shape[0], size=(n_boot, b.shape[0]))
    d = np.exp(q[idx].mean((1, 2))) - np.exp(b[idx].mean((1, 2)))
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


@torch.no_grad()
def evaluate(model, ids: torch.Tensor, window: int, device: str,
             label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    nlls, tops, confs = [], [], []
    t0 = time.time()

    for i in range(n_win):
        w = ids[i * window:(i + 1) * window].unsqueeze(0).to(device)
        logits = model(w).logits[0, :-1]              # [W-1, V]
        tgt = w[0, 1:]
        # Slice the vocab softmax: [2047, 128000] in float32 is ~1 GB in one go.
        for s in range(0, logits.shape[0], 256):
            sl = logits[s:s + 256].float()
            nlls.append(torch.nn.functional.cross_entropy(
                sl, tgt[s:s + 256], reduction="none").to("cpu").numpy())
            tops.append(sl.argmax(-1).to("cpu", torch.int32).numpy())
            # Baseline top-1 probability, to stratify agreement by confidence.
            # Where the model is near-tied, argmax flips under any perturbation
            # and says nothing about the format.
            confs.append(sl.softmax(-1).amax(-1).to("cpu").numpy())
        if (i + 1) % 5 == 0 or i + 1 == n_win:
            el, run = time.time() - t0, np.concatenate(nlls)
            print(f"    [{label}] window {i+1}/{n_win}  "
                  f"ppl {np.exp(run.mean()):8.3f}  "
                  f"{el:5.1f}s ({el/(i+1):4.1f}s/win)", flush=True)

    # Per-token NLL, not a running sum: the paired bootstrap needs the sequence.
    return np.concatenate(nlls), np.concatenate(tops), np.concatenate(confs)


WIKITEXT2_TEST = ("Salesforce/wikitext",
                  "wikitext-2-raw-v1/test-00000-of-00001.parquet")


def wikitext2_text() -> str:
    """WikiText-2 test split as plain text.

    The split is 0.73 MB on disk and ~280 K tokens -- it is the standard
    perplexity corpus precisely because it is small, so there is nothing
    meaningful to gain by finding a smaller one. What costs time is
    --max-tokens, not the download.

    Prefers `datasets`, but falls back to reading the parquet directly with
    pyarrow + huggingface_hub. pyarrow alone is a far lighter dependency than
    `datasets`, and it has a cp314 wheel -- which is not a given on this Python
    (see installed.md on cocotb).
    """
    try:
        from datasets import load_dataset
        return "\n\n".join(load_dataset(
            "wikitext", "wikitext-2-raw-v1", split="test")["text"])
    except ImportError:
        pass
    try:
        import pyarrow.parquet as pq
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise SystemExit(
            "No --corpus given, and neither `datasets` nor `pyarrow` is "
            "installed.\n"
            "  lightest:  pip install pyarrow\n"
            "  or:        pip install datasets\n"
            "  or:        pass --corpus <a plain text file>\n"
            "The gate names WikiText-2; anything else must be reported as what "
            "it actually is.")
    p = hf_hub_download(WIKITEXT2_TEST[0], WIKITEXT2_TEST[1], repo_type="dataset")
    rows = pq.read_table(p).column("text").to_pylist()
    return "\n\n".join(r for r in rows if r.strip())


def load_corpus(path: Path | None, tok, max_tokens: int) -> tuple[torch.Tensor, str]:
    if path:
        text, name = path.read_text(encoding="utf-8", errors="replace"), path.name
    else:
        text, name = wikitext2_text(), "wikitext-2 test"
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids[:max_tokens], name


# ------------------------------------------------------------------- report

def report(base_ppl, quant_ppl, agree, applied_res, expect_res, active_bits,
           ci=None, n_tok=0, corpus="", strat=None) -> int:
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
    print(f"{corpus}, {n_tok:,} tokens")
    print(f"{'gate':22s} {'measured':>10s} {'':2s} {'gate':>8s}   note")
    print("-" * 72)
    bad = 0
    for name, got, gate, op, note in rows:
        ok = got <= gate if op == "<=" else got >= gate
        bad += not ok
        print(f"{name:22s} {got:10.4f} {op:2s} {gate:8.3f}   "
              f"{'PASS' if ok else 'FAIL'}  {note}")
    print("-" * 72)
    if strat is not None:
        print("agreement by baseline top-1 confidence "
              "(where BF16 is near-tied, argmax flips under any perturbation "
              "and says nothing about the format):")
        for lo_p, hi_p, n, ag in strat:
            print(f"  p_top1 {lo_p:.2f}-{hi_p:.2f}  {n:7,d} tok  agreement {ag:.4f}")
        print("-" * 72)
    if ci:
        lo, hi = ci
        gate = G["ppl_delta_max"]
        # A verdict that is inside the gate but whose interval straddles it has
        # not actually decided anything -- report that rather than let a lucky
        # point estimate read as a pass.
        if lo <= gate <= hi:
            v = "UNRESOLVED -- interval straddles the gate, raise --max-tokens"
        elif hi < gate:
            v = "resolved: below the gate"
        else:
            v = "resolved: above the gate"
        print(f"{'ppl_delta 95% CI':22s} [{lo:+.4f}, {hi:+.4f}]   {v}")
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
    ap.add_argument("--pack", choices=["rtn", "clip", "awq"], default="rtn",
                    help="packer strategy. rtn is naive round-to-nearest; clip searches the\n"
                         "clipping ratio; awq adds activation-aware scaling. All three\n"
                         "emit the SAME format -- only the scales and rounding differ")
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--int8-group", action="store_true",
                    help="swap quant.py's PER-TENSOR INT8 for per-group INT8 (8.25 "
                         "bits). Measured: attention at per-tensor INT8 does 21x more "
                         "damage per parameter than the experts at INT4, because the "
                         "promotion to 8 bits is wasted by a single tensor-wide scale")
    ap.add_argument("--uniform", action="store_true",
                    help="ablation: flat INT4_G64 everywhere, ignoring the recipe's promotions")
    ap.add_argument("--force", choices=["int12", "int8g", "int8", "int4", "bf16"],
                    help="override every block's format. Controls: bf16 is the null "
                         "(must be exactly 0.0000 / 1.0000); int12 group-64 is the "
                         "positive control (relerr 5e-5, must be near-lossless). "
                         "A failure in either means the harness is wrong rather than "
                         "the recipe. Note int8 here is quant.py's PER-TENSOR INT8, "
                         "which is a genuinely lossy scheme -- int8g is its per-group "
                         "counterpart")
    ap.add_argument("--only", help="comma-separated block names to quantize; every "
                                   "other block stays BF16. Isolates which format "
                                   "is responsible for a failure")
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

    ids, corpus_name = load_corpus(a.corpus, tok, a.max_tokens)
    inst = tok("\n\n".join(INSTRUCTIONS), return_tensors="pt").input_ids[0]
    n_win = ids.numel() // a.window
    print(f"  {corpus_name}: {ids.numel():,} tokens, {n_win} windows of {a.window}"
          f"  (+{inst.numel():,} instruction tokens)")
    if n_win < 8:
        print("  NOTE: fewer than 8 windows -- the bootstrap CI will be wide. "
              "Raise --max-tokens for a decisive result.")

    print("\n--- BF16 baseline ---")
    b_nll, b_top, b_conf = evaluate(model, ids, a.window, a.device, "bf16")
    base_ppl = float(np.exp(b_nll.mean()))

    acts = None
    if a.pack == "awq":
        from p0 import packer
        print("\n--- calibrating (activation magnitudes, BF16 model) ---")
        t1 = time.time()
        acts = packer.collect_act_scales(model, ids, a.window, a.device,
                                         a.calib_windows)
        # lm_head is tied to embed_tokens, and the parameter we quantize is the
        # embedding one; carry its activation across or the LM head silently
        # falls back to clip search.
        if "lm_head.weight" in acts:
            acts.setdefault("model.embed_tokens.weight", acts["lm_head.weight"])
        print(f"  captured {len(acts)} activation vectors in {time.time()-t1:.1f}s")

    table = dict(quant.UNIFORM_INT4 if a.uniform else quant.BLOCK_FMT)
    label = "UNIFORM INT4 (ablation)" if a.uniform else "the recipe"
    if a.int8_group:
        g8 = quant.Fmt(8.25, "int8g", 64, why="per-group INT8; 8 + 16/64 bits")
        table = {k: (g8 if v.kind == "int8" else v) for k, v in table.items()}
        label += " with per-group INT8"
    if a.force:
        INT8_G64 = quant.Fmt(8.25, "int8g", 64, why="control: per-group INT8")
        table = {k: {"int12": quant.INT12, "int8g": INT8_G64, "int8": quant.INT8,
                     "int4": quant.INT4_G64, "bf16": quant.BF16}[a.force]
                 for k in table}
        label = f"FORCED {a.force.upper()} (control)"
    if a.only:
        keep = {s.strip() for s in a.only.split(",")}
        unknown = keep - set(table)
        if unknown:
            raise SystemExit(f"unknown block(s) {sorted(unknown)}; "
                             f"choose from {sorted(table)}")
        table = {k: (v if k in keep else quant.BF16) for k, v in table.items()}
        label = f"ONLY {sorted(keep)}"
    print(f"\n--- applying {label} via packer={a.pack} ---")
    cov = quantize_(model, table, mode=a.pack, acts=acts)

    print("\n--- quantized ---")
    q_nll, q_top, _ = evaluate(model, ids, a.window, a.device, "quant")
    quant_ppl = float(np.exp(q_nll.mean()))
    agree = float((b_top == q_top).mean())
    ci = bootstrap_delta(b_nll, q_nll, a.window - 1)

    per = cov["per_block"]
    applied_res = sum(v[1] for v in per.values()) / sum(v[0] for v in per.values())

    # Compare against whatever table was actually applied, so --only and --force
    # runs still get a meaningful coverage check rather than false drift.
    spec = modelspec.load("lfm2.5-8b-a1b")
    active_bits = spec.avg_bits(table)
    expect_res = (sum(b.total * table[b.name].bits for b in spec.blocks)
                  / spec.total_params)

    # Stratify agreement by how confident BF16 was. A 0.99 gate applied to
    # every corpus position is unreachable by construction: with a 128K vocab
    # and a model at this perplexity, a large share of positions are near-tied
    # and flip under a 5e-5 weight perturbation. Confident positions are where
    # a disagreement actually indicts the format.
    strat = []
    for lo_p, hi_p in [(0.0, 0.5), (0.5, 0.9), (0.9, 1.01)]:
        m = (b_conf >= lo_p) & (b_conf < hi_p)
        if m.sum():
            strat.append((lo_p, min(hi_p, 1.0), int(m.sum()),
                          float((b_top[m] == q_top[m]).mean())))

    rc = report(base_ppl, quant_ppl, agree, applied_res, expect_res, active_bits,
                ci=ci, n_tok=b_nll.size, corpus=corpus_name, strat=strat)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "model": a.model, "uniform": a.uniform, "corpus": corpus_name,
        "pack": a.pack,
        "tokens": int(b_nll.size), "window": a.window,
        "ppl_bf16": base_ppl, "ppl_quant": quant_ppl,
        "ppl_delta": quant_ppl - base_ppl,
        "ppl_delta_ci95": list(ci),
        "top1_agreement": agree,
        "top1_agreement_by_confidence": strat,
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
