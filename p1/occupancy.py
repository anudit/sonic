#!/usr/bin/env python3
"""P1-2: does routing imbalance starve the systolic array?

This is the gate that decides whether the prefill engine works at all. A 128x128
array fed by ragged per-expert GEMMs wastes every partially-filled pass, so the
question is not "how many lanes" but "how many lanes stay busy".

Run it across the plausible imbalance range until P0 supplies the real number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sonic import load  # noqa: E402
from sonic.moe import occupancy, route_counts  # noqa: E402

GATE = 0.80


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lfm2.5-8b-a1b")
    ap.add_argument("--tiles", type=int, nargs="+", default=[64, 96, 128, 181])
    ap.add_argument("--chunks", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    ap.add_argument("--trials", type=int, default=16)
    ap.add_argument("--experts", type=int, default=None, help="override num experts (e.g. 8 for P1-7)")
    ap.add_argument("--top-k", type=int, default=None, help="override top-k (e.g. 1 for P1-7)")
    a = ap.parse_args()

    m = load(a.model)
    if not m.is_moe:
        print("dense model -- occupancy is 1.0 by construction"); return 0

    import dataclasses
    n_experts = a.experts or m.n_experts
    top_k = a.top_k or m.top_k
    m = dataclasses.replace(m, n_experts=n_experts, top_k=top_k)
    print(f"{m.name}: {n_experts} experts top-{top_k}, gate >= {GATE:.2f}")
    print("mean occupancy over", a.trials, "trials; ! marks a gate failure\n")

    for cv in (0.0, 0.3, 0.5, 0.8):
        print(f"  routing load CV {cv:.1f}" + ("   <- uniform bound" if cv == 0 else ""))
        hdr = "".join(f"{t:>10}" for t in a.tiles)
        print(f"    {'chunk':>7}{hdr}   tokens/expert")
        for chunk in a.chunks:
            cells = []
            for tile in a.tiles:
                vals = [occupancy(route_counts(m, chunk, cv, np.random.default_rng(s)), tile)
                        for s in range(a.trials)]
                mu = float(np.mean(vals))
                cells.append(f"{mu:>9.3f}{'!' if mu < GATE else ' '}")
            print(f"    {chunk:>7}{''.join(cells)}   {chunk * top_k // n_experts:>6}")
        print()

    print("Reading: occupancy is set by tokens-per-expert relative to the tile edge.")
    print("A 128-edge tile needs chunk >= 1024 even under uniform routing, and more")
    print("once the load is skewed. That coupling is why SRAM sizing is a prefill")
    print("decision, not a decode one.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
