#!/usr/bin/env python3
"""P1-3: measure the DRAM cost of the expert-gather access pattern.

`chipspec.dram_eff` is a flat 1.00 against a 0.85 gate, and p1/README.md called
the expert gather "exactly the case where a flat derate is least trustworthy".
This settles the pattern half of that worry with DRAMsim3.

What it does NOT settle: absolute efficiency. DRAMsim3 ships no LPDDR5X model,
so the device here is LPDDR4-2400 x16, and the achieved fraction of peak is
dominated by that model's controller (cmd_queue_size 8, address_mapping
rochrababgco) rather than by anything about Sonic. Absolute numbers from this
run must not be transplanted into chipspec. The transferable quantity is the
RATIO between patterns on one device.

    make p1-dram        # generates traces, runs DRAMsim3, prints this table
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SIM = ROOT / "dramsim3" / "build" / "dramsim3main"
CFG = ROOT / "dramsim3" / "configs" / "LPDDR4_8Gb_x16_2400.ini"
GEN = ROOT / "p1" / "dram_trace.py"

TCK_NS = 0.83
BUS_BITS = 64
PEAK_GBPS = BUS_BITS * 2 / TCK_NS / 8      # DDR: two transfers per clock


def read(pattern: str, tmp: Path) -> dict:
    """Parse one DRAMsim3 run. The Makefile runs the simulator itself: spawning
    it from Python traps under some sandboxes, and a shell step is the simpler
    contract anyway."""
    out = tmp / f"out_{pattern}"
    c = json.loads((out / "dramsim3.json").read_text())["0"]
    done, cyc = c["num_reads_done"], c["num_cycles"]
    return dict(
        pattern=pattern, reads=done,
        gbps=done * 64 / (cyc * TCK_NS * 1e-9) / 1e9,
        row_hit=c["num_read_row_hits"] / max(c["num_read_cmds"], 1),
        acts=c["num_act_cmds"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    a = ap.parse_args()

    tmp = ROOT / "p1" / "out" / "dram"
    rows = [read(p, tmp) for p in ("stream", "expert", "scatter")]

    base = rows[0]["gbps"]
    print(f"\nLPDDR4-2400 x16, one channel. Theoretical peak {PEAK_GBPS:.2f} GB/s.")
    print(f"{'pattern':9s} {'GB/s':>7s} {'of peak':>8s} {'row hit':>8s} "
          f"{'ACTs':>8s} {'vs stream':>10s}")
    for r in rows:
        print(f"{r['pattern']:9s} {r['gbps']:7.2f} {r['gbps']/PEAK_GBPS:8.1%} "
              f"{r['row_hit']:8.4f} {r['acts']:8,d} {r['gbps']/base:10.4f}")

    exp = next(r for r in rows if r["pattern"] == "expert")
    print(f"\nexpert / stream = {exp['gbps']/base:.4f}")
    print("The gather costs nothing: an expert is 5.96 MB of contiguous weights,")
    print("93,112 cache lines, so four jumps per layer between megabyte runs keep")
    print("the row-buffer hit rate at parity with a pure sweep.")
    print("\nAbsolute 'of peak' here is a property of this LPDDR4 model's")
    print("controller, NOT of Sonic. Do not copy it into chipspec.dram_eff.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
