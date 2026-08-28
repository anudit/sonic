#!/usr/bin/env python3
"""P1-3: DRAM traces for the access patterns Sonic S1 actually issues.

`chipspec.dram_eff` has been a flat 1.00 against a 0.85 gate, and p1/README.md
flagged the expert gather as "exactly the case where a flat derate is least
trustworthy". This generates DRAMsim3 traces so that stops being a guess.

The result turns on one number the plan already contains but never applied here:
**an expert is 5.96 MB of contiguous weights**, 93,112 cache lines. The MoE
"gather" is therefore not a scatter of small reads -- it is four jumps per layer
between megabyte-scale sequential runs. Whether that costs anything is an
empirical question about row-buffer locality across those jumps, which is what
these traces measure.

Patterns:

    stream   one contiguous sweep. The upper bound: what a perfectly linear
             weight read achieves on this device, including refresh and bus
             turnaround, which is already below peak.
    expert   the real decode pattern: per layer, top_k expert blocks chosen by
             measured routing, each read contiguously.
    scatter  a deliberately pessimal control -- the same byte count as `expert`
             but in randomly placed cache lines. This is the pattern the flat
             derate was implicitly feared to be, and it bounds how bad a gather
             could get if experts were ever stored non-contiguously.

    python3 p1/dram_trace.py --pattern expert --out /tmp/expert.trace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sonic import modelspec  # noqa: E402

LINE = 64                      # bytes per DRAM transaction
TRACE_NPZ = Path("p0/out/real_routing.npz")


def expert_base(layer: int, expert: int, expert_bytes: int, n_experts: int) -> int:
    """Byte address of one expert's weights. Experts are laid out contiguously
    within a layer, layers contiguously within the weight region."""
    return (layer * n_experts + expert) * expert_bytes


def gen(pattern: str, model, n_lines: int, rng, routing=None) -> list[int]:
    """Return a list of byte addresses, in issue order."""
    ebytes = int(model.expert_mb() * 1e6)
    ebytes -= ebytes % LINE
    E, K = model.n_experts, model.top_k
    span = model.n_moe_layers * E * ebytes

    addrs: list[int] = []
    if pattern == "stream":
        a = 0
        while len(addrs) < n_lines:
            addrs.append(a % span)
            a += LINE
        return addrs

    if pattern == "scatter":
        # Same volume, no locality at all.
        return [int(rng.integers(0, span // LINE)) * LINE for _ in range(n_lines)]

    # expert: per layer, read K whole expert blocks back to back.
    layer = 0
    while len(addrs) < n_lines:
        if routing is not None:
            sel = routing[rng.integers(0, routing.shape[0])]
        else:
            sel = rng.choice(E, size=K, replace=False)
        for e in sel:
            base = expert_base(layer % model.n_moe_layers, int(e), ebytes, E)
            for off in range(0, ebytes, LINE):
                addrs.append(base + off)
                if len(addrs) >= n_lines:
                    return addrs
        layer += 1
    return addrs


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pattern", choices=["stream", "expert", "scatter"],
                    default="expert")
    ap.add_argument("--lines", type=int, default=200_000,
                    help="cache lines to emit (64 B each)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    model = modelspec.load("lfm2.5-8b-a1b")
    rng = np.random.default_rng(a.seed)

    # Use the captured routing when it exists, so `expert` reflects measured
    # locality rather than a uniform draw -- the whole point of P0-2.
    routing = None
    if a.pattern == "expert" and TRACE_NPZ.exists():
        routing = np.load(TRACE_NPZ)["routing"].astype(int)
        print(f"using measured routing: {routing.shape[0]:,} decisions", flush=True)

    addrs = gen(a.pattern, model, a.lines, rng, routing)

    # Issue every request at cycle 0: we want the controller's steady-state
    # throughput, not a latency study throttled by a synthetic arrival rate.
    with a.out.open("w") as f:
        for ad in addrs:
            f.write(f"{ad:x} READ 0\n")
    print(f"{a.pattern}: {len(addrs):,} lines "
          f"({len(addrs)*LINE/1e6:.1f} MB) -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
