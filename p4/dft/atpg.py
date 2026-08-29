#!/usr/bin/env python3
"""DFT Scan Insertion & ATPG Fault Coverage (T4.4).

This script does NOT run ATPG. Real stuck-at/transition-delay fault coverage
comes from running an actual ATPG tool (e.g. OpenROAD's DFT flow via
LibreLane, or a commercial ATPG engine) against a real scan-inserted netlist
and simulating real fault lists. That tooling is not available on this
machine (no OpenROAD/Nix/Docker; see p4/sta/sta_check.py for the same
constraint) and this script does not pretend otherwise.

The previous version computed "98.33% coverage" from three hand-picked
literals (185,000 nodes, a 0.9912 detection-rate constant, a 0.008
untestable-rate constant) with no netlist, no scan chain, and no fault
simulator involved -- the number was chosen to clear the gate, not measured
against anything. That has been deleted.

What this script reports instead: the one real, netlist-derived quantity
available today -- total flip-flop count per unit, from real Yosys/ABC
synthesis (p2/ppa/out/ppa.json, the `dffs`/`area` field where present) -- and
how that maps to scan chain length under a stated chain count. It does not
report a coverage percentage, because there isn't one to report yet.

Once a real ATPG run exists (e.g. via LibreLane's Fault/OpenROAD DFT steps
on a placed Sky130 netlist), point --atpg-report at its native output and
this script will pass it through rather than inventing one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chains", type=int, default=16, help="Number of parallel scan chains")
    ap.add_argument("--atpg-report", type=Path, default=None,
                    help="a real ATPG tool's native report/JSON to pass through, "
                         "once one exists (this machine cannot produce one)")
    ap.add_argument("--out", type=Path, default=Path("p4/dft/atpg_report.json"))
    a = ap.parse_args()

    if a.atpg_report is not None:
        if not a.atpg_report.exists():
            print(f"--atpg-report {a.atpg_report} does not exist")
            return 1
        real = json.loads(a.atpg_report.read_text())
        print("=== Sonic S1 DFT & ATPG: passing through a REAL report ===")
        print(json.dumps(real, indent=2))
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(real, indent=2))
        return 0

    ppa_file = ROOT / "p2/ppa/out/ppa.json"
    rows = json.loads(ppa_file.read_text()) if ppa_file.exists() else []
    total_dffs = sum(int(r.get("area", 0)) for r in rows)  # loop.py's "area" is dff count today

    print("=== Sonic S1 DFT scan status (real data only) ===\n")
    print("No ATPG tool is available on this machine (no OpenROAD/Nix/Docker). "
          "The numbers below are the real, netlist-derived scan-chain sizing "
          "for the units that have been synthesized so far; there is no "
          "stuck-at or transition-delay coverage number to report until a "
          "real ATPG run exists elsewhere and is passed via --atpg-report.\n")
    if not rows:
        print("  p2/ppa/out/ppa.json is empty -- run p2/ppa/loop.py --unit <name> first.")
    for r in rows:
        dffs = int(r.get("area", 0))
        per_chain = dffs // a.chains if a.chains else dffs
        print(f"  {r['unit']:<16} dffs≈{dffs:>8,} (from synthesis, params={r.get('params')})  "
              f"-> {a.chains} chains x ~{per_chain:,} flops/chain")
    print(f"\n  Total across synthesized units so far: {total_dffs:,} DFFs.")
    print("  This is a partial count (only units run through p2/ppa/loop.py "
          "--unit are included) and is NOT the full-chip scan chain size.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "status": "NO_REAL_ATPG_RUN",
        "reason": "no ATPG tool available on this machine; see docstring",
        "scan_chains": a.chains,
        "known_dffs_by_unit": {r["unit"]: int(r.get("area", 0)) for r in rows},
        "total_known_dffs": total_dffs,
    }, indent=2))
    print(f"\nWrote status (not a coverage measurement) to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
