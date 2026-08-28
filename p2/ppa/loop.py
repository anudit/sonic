#!/usr/bin/env python3
"""P2: the automated PPA loop.

The two-axis method from the Kimi K3 write-up, with one correction that matters
for this chip.

  Y-axis: iterate RTL inside a fixed harness.
  X-axis: when Y plateaus, rewrite the harness -- a plateau usually means the
          framing is wrong, not the code.

The correction: that loop optimizes what it can measure, and what it can measure
is f_max of a multiplier array. On Sonic S1 the array is 0.3% of decode power
and 3% utilised, so an unattended loop will happily hand you a faster array
bolted to a memory system that can feed 119 lanes. Every run here therefore
reports cell area and logic depth ALONGSIDE a reminder of what the roofline says
the unit is actually worth.

Usage:
    python3 p2/ppa/loop.py                 # all units
    python3 p2/ppa/loop.py --unit sonic_acc --sweep ACC_LOCAL=12,16,20,24
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
RTL = ROOT / "p2" / "rtl"
OUT = ROOT / "p2" / "ppa" / "out"

UNITS = {
    "sonic_acc":      ["sonic_acc.sv"],
    "sonic_pe":       ["sonic_acc.sv", "sonic_pe.sv"],
    "sonic_conv":     ["sonic_conv.sv"],
    "sonic_softmax":  ["sonic_softmax.sv"],
    "sonic_lmhead":   ["sonic_lmhead.sv"],
    "sonic_streamer": ["sonic_streamer.sv"],
    "sonic_seq":      ["sonic_seq.sv"],
    # The router was missing from this table until P4 found why it mattered:
    # block-level P&R reported WNS -1086.6 ns at a 10 ns clock, and an
    # independent depth measurement put its combinational path at 2,518 levels
    # against sonic_acc's 59. The one unit with a pathological critical path was
    # the one unit excluded from the loop built to measure critical paths.
    "sonic_router":   ["sonic_router.sv"],
}

# What the roofline says each unit is worth, so the loop cannot mislead.
CONTEXT = {
    "sonic_acc":      "0.3% of decode power; correctness >> f_max",
    "sonic_pe":       "array is 3% utilised in decode, sized for TTFT",
    "sonic_conv":     "O(P) and nearly free; 72 KB of state, context-independent",
    "sonic_softmax":  "only matters above 8K context, where attention is 32% of prefill",
    "sonic_lmhead":   "0.14 mm2 that saves ~8% of all decode traffic",
    "sonic_streamer": "THE critical path -- if this stalls, the chip stops",
    "sonic_seq":      "control, not datapath; f_max here is irrelevant",
    "sonic_router":   "2,518 levels deep -- CANNOT be clocked; needs pipelining",
}


@dataclass
class Result:
    unit: str
    cells: int
    area: float
    depth: int
    params: dict

    def __str__(self) -> str:
        p = " ".join(f"{k}={v}" for k, v in self.params.items()) or "default"
        return (f"{self.unit:12s} {p:20s} cells={self.cells:7,d} "
                f"dffs={self.area:5,.0f} depth={self.depth:4d}")


# Units that need a define before they will elaborate standalone.
# SEGS=8, not the shipping 64: ABC's technology mapping on the segment-select
# mux tree runs for hours at 64 and never converges -- reproduced locally and on
# the P4 box. 8 is the tractable point for a relative signal, and the shipping
# config is DEEPER than whatever this reports, never shallower.
REQUIRED_DEFINES = {"sonic_router": {"ROUTER_PWL_SEGS": 8}}


def synth(unit: str, params: dict | None = None) -> Result | None:
    params = {**REQUIRED_DEFINES.get(unit, {}), **(params or {})}
    srcs = " ".join(str(RTL / f) for f in UNITS[unit])
    defines = " ".join(f"-D{k}={v}" for k, v in params.items())
    OUT.mkdir(parents=True, exist_ok=True)

    script = f"""
read_verilog -sv -I{RTL} {defines} {srcs}
hierarchy -top {unit}
proc; opt; fsm; opt; memory; opt
flatten; opt
techmap; opt
abc -g cmos2 -dff
opt_clean
stat
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as f:
        f.write(script)
        path = f.name

    r = subprocess.run(["yosys", "-s", path], capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  yosys failed for {unit}: {r.stderr.strip().splitlines()[-1:]}")
        return None

    txt = r.stdout
    cells = area = 0
    # yosys `stat` prints bare "<n> cells"; with no liberty there is no area,
    # so use the DFF count as the sequential-cost proxy instead.
    m = re.search(r"^\s+(\d+)\s+cells\s*$", txt, re.M)
    if m:
        cells = int(m.group(1))
    area = float(sum(int(x) for x in re.findall(r"^\s+(\d+)\s+\$_DFF", txt, re.M)))

    # Logic depth: a cheap proxy for the critical path without a liberty file.
    depth = longest_path(unit, defines, srcs)
    return Result(unit, cells, area, depth, params)


def longest_path(unit: str, defines: str, srcs: str) -> int:
    """Combinational logic levels, via yosys `ltp` on the mapped netlist."""
    script = f"""
read_verilog -sv -I{RTL} {defines} {srcs}
hierarchy -top {unit}
proc; opt; flatten; opt; techmap; opt
abc -g cmos2 -dff
opt_clean
ltp -noff
"""
    with tempfile.NamedTemporaryFile("w", suffix=".ys", delete=False) as f:
        f.write(script)
        path = f.name
    r = subprocess.run(["yosys", "-s", path], capture_output=True, text=True)
    m = re.search(r"Longest topological path in \S+ \(length=(\d+)\)", r.stdout)
    return int(m.group(1)) if m else -1


# Measurable on request, but not in the default sweep: even at SEGS=8 the
# router takes many minutes under this flow's `abc -g cmos2 -dff` sequential
# mapping, and `make p2` has to stay fast. Run it deliberately:
#     python3 p2/ppa/loop.py --unit sonic_router
SLOW = {"sonic_router"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--unit", choices=list(UNITS))
    ap.add_argument("--sweep", help="PARAM=v1,v2,v3")
    a = ap.parse_args()

    units = [a.unit] if a.unit else [u for u in UNITS if u not in SLOW]
    if not a.unit and SLOW:
        print(f"skipping {', '.join(sorted(SLOW))} (slow; pass --unit to include)",
              flush=True)
    results: list[Result] = []

    for u in units:
        print(f"\n{u}  --  {CONTEXT.get(u, '')}", flush=True)
        if a.sweep:
            key, vals = a.sweep.split("=")
            for v in vals.split(","):
                r = synth(u, {key: v})
                if r:
                    print(f"  {r}", flush=True)
                    results.append(r)
        else:
            r = synth(u)
            if r:
                print(f"  {r}", flush=True)
                results.append(r)

    if results:
        (OUT / "ppa.json").write_text(json.dumps(
            [dict(unit=r.unit, cells=r.cells, area=r.area, depth=r.depth,
                  params=r.params) for r in results], indent=2))
        print(f"\nwrote {OUT / 'ppa.json'}")
        print("\nReminder: depth and area are relative signals from a generic")
        print("cell library. Absolute f_max needs the foundry liberty at P4, and")
        print("the roofline says this array is not what limits the chip.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
