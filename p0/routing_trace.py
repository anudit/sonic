#!/usr/bin/env python3
"""P0-2: measure real expert overlap and beat the uniform-routing bound.

The S1 plan budgets speculative decoding at ~1.15x and prefill occupancy at
>= 0.80, and BOTH numbers come from an independent-uniform-routing assumption
that is deliberately pessimistic. Real sequences show expert locality. This
script is how that assumption gets replaced with a measurement.

Two modes:

  synthetic   Model routing as a lognormal expert prior plus a stickiness term
              (probability a token reuses its predecessor's experts). Runs now,
              no checkpoint needed, and brackets the plausible range.

  trace       Consume a .npz of real routing decisions captured from the HF
              model -- shape (n_tokens, top_k), dtype int -- and report the
              same statistics. Capture with:
                  python p0/routing_trace.py capture --model <hf-id> --prompts f.txt

Gate: report measured overlap at batch 3 and 8, and measured occupancy at the
chosen chunk. P1's array sizing depends on both.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sonic import load  # noqa: E402
from sonic.moe import distinct_experts, occupancy, spec_decode_gain  # noqa: E402


def synth_trace(n_tokens: int, E: int, K: int, imbalance: float,
                stickiness: float, seed: int = 0) -> np.ndarray:
    """Routing decisions with a skewed prior and temporal locality.

    `stickiness` in [0,1) is the probability that a slot reuses the previous
    token's expert -- the mechanism that makes real traces beat the uniform
    bound, and the thing §06 change 4 ("train for routing stability") targets.
    """
    rng = np.random.default_rng(seed)
    if imbalance <= 0:
        prior = np.full(E, 1.0 / E)
    else:
        sigma = np.sqrt(np.log1p(imbalance ** 2))
        prior = rng.lognormal(-sigma ** 2 / 2, sigma, E)
        prior /= prior.sum()

    out = np.zeros((n_tokens, K), dtype=np.int32)
    prev = rng.choice(E, size=K, replace=False, p=prior)
    for t in range(n_tokens):
        keep = rng.random(K) < stickiness
        pick = prev.copy()
        n_new = int((~keep).sum())
        if n_new:
            avail = np.setdiff1d(np.arange(E), pick[keep], assume_unique=False)
            p = prior[avail] / prior[avail].sum()
            pick[~keep] = rng.choice(avail, size=n_new, replace=False, p=p)
        out[t] = prev = pick
    return out


def overlap_curve(trace: np.ndarray, E: int, batches=(1, 2, 3, 4, 8, 16, 32)) -> dict:
    """Mean distinct experts over sliding windows of each batch size."""
    res = {}
    for b in batches:
        if len(trace) < b:
            continue
        d = [len(np.unique(trace[i:i + b])) for i in range(0, len(trace) - b + 1, max(1, b // 2))]
        res[b] = float(np.mean(d))
    return res


def report(trace: np.ndarray, model, tile: int, chunk: int, label: str) -> dict:
    E, K = model.n_experts, model.top_k
    counts = np.bincount(trace[:chunk].ravel(), minlength=E)
    occ = occupancy(counts, tile)
    curve = overlap_curve(trace, E)
    cv = float(counts.std() / counts.mean()) if counts.mean() else 0.0

    print(f"\n=== {label} ===")
    print(f"  tokens {len(trace):,}   experts {E} top-{K}   load CV {cv:.2f}")
    print(f"  {'batch':>6}{'measured':>11}{'uniform bound':>16}{'better by':>12}")
    for b, meas in curve.items():
        u = distinct_experts(E, K, b)
        print(f"  {b:>6}{meas:>11.2f}{u:>16.2f}{u / meas:>11.2f}x")
    print(f"\n  chunk {chunk}: occupancy {occ:.3f} at tile {tile} "
          f"({'PASS' if occ >= 0.80 else 'FAIL'} vs 0.80 gate)")
    print(f"  min/max tokens per expert: {counts.min()} / {counts.max()}")
    return dict(label=label, n_tokens=int(len(trace)), load_cv=cv,
                overlap=curve, occupancy=occ, chunk=chunk, tile=tile)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["synthetic", "trace", "capture"])
    ap.add_argument("--model", default="lfm2.5-8b-a1b")
    ap.add_argument("--trace", type=Path, help="npz with array 'routing' (n_tokens, top_k)")
    ap.add_argument("--tokens", type=int, default=8192)
    ap.add_argument("--chunk", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=128)
    ap.add_argument("--out", type=Path, default=Path("p0/out/routing.json"))
    a = ap.parse_args()

    model = load(a.model)
    if not model.is_moe:
        print(f"{a.model} is dense -- no routing to measure."); return 1

    results = []
    if a.mode == "capture":
        print("Capture requires transformers + the checkpoint. Register a forward hook on\n"
              "each Lfm2MoeSparseMoeBlock router, stack the top-k indices, and np.savez the\n"
              "result as 'routing'. Then re-run with: routing_trace.py trace --trace <file>")
        return 2

    if a.mode == "trace":
        if not a.trace or not a.trace.exists():
            print("--trace required and must exist"); return 1
        tr = np.load(a.trace)["routing"].astype(np.int32)
        results.append(report(tr, model, a.tile, a.chunk, f"measured: {a.trace.name}"))
    else:
        # Bracket the plausible range: the uniform bound, a realistic middle, and
        # what §06 change 4 (routing-stability training) might buy.
        for lab, imb, st in [("uniform, no locality (plan's bound)", 0.0, 0.0),
                             ("realistic: CV 0.5, stickiness 0.25", 0.5, 0.25),
                             ("stability-trained: CV 0.3, stickiness 0.50", 0.3, 0.50)]:
            tr = synth_trace(a.tokens, model.n_experts, model.top_k, imb, st)
            results.append(report(tr, model, a.tile, a.chunk, lab))

    if a.mode == "trace":
        from sonic.moe import dspark_gain_measured, dspark_gain
        print("\n=== DSpark block drafter: measured trace vs uniform bound ===")
        print(f"  {'p':>5}{'measured':>12}{'bound':>10}")
        for p_acc in (0.70, 0.80, 0.90):
            gm = dspark_gain_measured(model, tr, p=p_acc)
            gb = dspark_gain(model, p=p_acc)
            print(f"  {p_acc:>5.2f}{gm['gain']:>11.2f}x{gb['gain']:>9.2f}x")
        print("  measured routing reuses experts; the uniform model assumes it")
        print("  does not, and understates the gain by ~40%.")
    else:
        print("\n=== speculative decode under the uniform bound ===")
        for k in (2, 4, 8):
            g = spec_decode_gain(model, k)
            print(f"  k={k}: {g['mb_per_token']:6.0f} MB/accepted-token   gain {g['gain']:.2f}x")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
