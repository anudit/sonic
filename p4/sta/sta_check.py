#!/usr/bin/env python3
"""STA timing summary (T2.2/T2.3/T4.3).

This script does NOT run static timing analysis. Real STA needs a real
Liberty-backed delay model on a real, placed-and-routed netlist -- that is
what OpenROAD's `sta` does inside the LibreLane flow, and it is the only
thing this program treats as "measured" timing (see p4/openlane/router/
results/segs8/metrics.json and HANDOFF.md 1.3).

The previous version of this file computed a fake per-unit STA by inventing
"14nm FinFET" gate-delay constants (t_clk_q, avg_gate_delay, ...) and, worse,
silently capped logic depth to 68 for exactly the four units whose real depth
would otherwise miss 1 GHz -- manufacturing a PASS instead of measuring one.
That has been deleted, not patched: there is no honest way to turn
`depth * constant` into a timing signoff, at 14nm or any node, without a
real cell library and a real router.

What this script does instead: it reports the one real, Liberty-backed data
point this program has (the Sky130 router run, see below) and prints, for
every other unit, its real Yosys/ABC logic depth from p2/ppa/out/ppa.json
with NO derived frequency claim -- because a depth number times an invented
constant is not a frequency, it's a guess wearing a frequency's units.

Once real Sky130 P&R lands for sonic_tile / sonic_sram_bank (see
p4/openlane/{tile,sram}/config.json and p4/README.md for the run commands --
this machine has no OpenROAD binary available, so those runs happen
elsewhere), rerun this script with --metrics pointing at each block's real
`runs/*/final/metrics.json` and it will report their real worst-corner slack
the same way it already does for the router. A 14nm-equivalent projection
from Sky130 numbers is a separate, explicitly-labeled scaling step -- see
`project_14nm()` below -- not something this script does silently.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
ROUTER_METRICS = ROOT / "p4/openlane/router/results/segs8/metrics.json"

# Sky130 HD is a real, fabricable 130nm node -- not 14nm. Any "14nm-equivalent"
# number derived from it is a MODEL, not a measurement. This is a standard,
# citable delay-scaling exponent (delay ~ node^1 to node^1.3 across published
# process comparisons; we use the conservative linear bound), applied only
# when --project-14nm is passed, and always labeled as such in the output.
SKY130_NODE_NM = 130.0
NODE_14NM_NM = 14.0


def load_router_real_slack() -> dict | None:
    if not ROUTER_METRICS.exists():
        return None
    m = json.loads(ROUTER_METRICS.read_text())
    # metrics.json keys vary by LibreLane version; report whatever slack/period
    # keys are present rather than guessing a schema.
    keys = {k: v for k, v in m.items()
            if any(s in k.lower() for s in ("slack", "period", "wns", "tns"))}
    return keys


def project_14nm(sky130_ps: float) -> float:
    """Linear node-scaling projection ONLY -- not a measurement. Delay scales
    roughly with feature size for a fixed design; this is the simplest
    defensible bound, not a foundry-calibrated model."""
    return sky130_ps * (NODE_14NM_NM / SKY130_NODE_NM)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics", type=Path, nargs="*", default=[],
                    help="additional real LibreLane metrics.json files "
                         "(e.g. p4/openlane/tile/runs/*/final/metrics.json) "
                         "to report alongside the router's")
    ap.add_argument("--project-14nm", action="store_true",
                    help="also print the labeled, non-measured 14nm linear "
                         "scaling projection of any Sky130 slack found")
    a = ap.parse_args()

    ppa_file = ROOT / "p2/ppa/out/ppa.json"
    depths = {}
    if ppa_file.exists():
        for row in json.loads(ppa_file.read_text()):
            depths[row["unit"]] = row

    print("=== Sonic S1 timing status (real data only) ===\n")

    print("-- Real, Liberty-backed Sky130 data --")
    router_slack = load_router_real_slack()
    if router_slack:
        print(f"  sonic_router (LibreLane/OpenROAD, {ROUTER_METRICS}):")
        for k, v in router_slack.items():
            print(f"    {k}: {v}")
        if a.project_14nm:
            print("    [MODELED, NOT MEASURED] linear 14nm-equivalent scaling "
                  "of the numbers above by (14/130) -- see project_14nm() "
                  "docstring for what this is and is not.")
    else:
        print(f"  no router metrics found at {ROUTER_METRICS}")

    for extra in a.metrics:
        if not extra.exists():
            print(f"  MISSING: {extra}")
            continue
        m = json.loads(extra.read_text())
        keys = {k: v for k, v in m.items()
                if any(s in k.lower() for s in ("slack", "period", "wns", "tns"))}
        print(f"\n  {extra}:")
        for k, v in keys.items():
            print(f"    {k}: {v}")

    print("\n-- Real Yosys/ABC logic depth, generic library, NO frequency claim --")
    print(f"  {'unit':<16} {'depth':>6} {'cells':>10} {'params'}")
    for unit, row in sorted(depths.items(), key=lambda kv: kv[1]["depth"]):
        print(f"  {unit:<16} {row['depth']:>6} {row['cells']:>10,} {row.get('params')}")
    print("\n  These depths are real (measured by p2/ppa/loop.py against a "
          "generic CMOS2 library). They are NOT converted to picoseconds or "
          "GHz here, because no real cell library backs that conversion for "
          "these units yet. Run each through LibreLane on Sky130 (configs "
          "in p4/openlane/{tile,sram}/) to get a real number, the same way "
          "the router already has one.")

    print(f"\nNote: no OpenROAD/OpenSTA binary is available on this machine "
          f"(macOS ARM64, no Nix/Docker) -- real P&R for tile/sram/top must "
          f"be run elsewhere and its metrics.json passed via --metrics.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
