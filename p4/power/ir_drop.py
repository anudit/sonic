#!/usr/bin/env python3
"""Power Delivery Network (PDN) & Dynamic IR Drop Analysis (T4.6).

This script does NOT extract a real PDN. Real static/dynamic IR-drop numbers
come from a parasitic-extracted power grid (real R/L/C from a real routed
layout's power straps and vias) fed to a vectored or vectorless IR-drop tool
(e.g. OpenROAD's `psm`/`pdn` analysis inside LibreLane). That tooling and a
real routed full-chip layout are both unavailable on this machine (see
p4/sta/sta_check.py for the same constraint) -- there is no full-chip GDS to
extract a PDN from yet, only the one Sky130 sonic_router block.

The previous version of this file used the same static-IR (I*R) and dynamic
L*di/dt formulas below, but with R_pdn=1.8 mOhm, L_pkg=8.5 pH, and a 0.82
"decap damping factor" invented outright and tuned until the output cleared
the <5% gate. The formulas are legitimate physics; the inputs were fiction
presented as a "verified" 2.82% droop. That framing has been deleted.

This version keeps the same formulas (they are the right first-order model)
but requires every R/L/C/current input to be passed explicitly and labeled
as an ASSUMPTION in the output -- nothing has a hardcoded default anymore,
and the report says so instead of implying a measurement. Once a real PDN
extraction exists for a routed block, replace these assumed inputs with the
extracted numbers and this becomes a real (if still single-block, not
full-chip) IR-drop estimate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class IRDropReport:
    vdd_nominal_v: float
    peak_current_a: float
    r_pdn_effective_mohm: float
    c_decap_total_nf: float
    l_pkg_loop_ph: float
    decap_damping_assumed: float
    static_ir_drop_mv: float
    static_ir_drop_pct: float
    dynamic_ir_droop_mv: float
    dynamic_ir_droop_pct: float
    effective_core_voltage_v: float
    status: str
    all_inputs_are_assumptions: bool = True


def analyze_ir_drop(vdd_nominal_v: float, peak_current_a: float,
                     r_pdn_effective_mohm: float, c_decap_total_nf: float,
                     l_pkg_loop_ph: float, decap_damping: float) -> IRDropReport:
    """Static I*R and dynamic L*di/dt(1-damping) droop. Every argument here is
    an ASSUMPTION unless it was extracted from a real routed PDN -- see
    module docstring. No argument has a default; the caller must state what
    it is assuming and why."""
    static_ir_drop_v = peak_current_a * (r_pdn_effective_mohm * 1e-3)
    static_pct = (static_ir_drop_v / vdd_nominal_v) * 100.0

    di_dt = peak_current_a / 500e-12  # assumed 500ps current step
    inductive_droop_v = (l_pkg_loop_ph * 1e-12) * di_dt
    dynamic_droop_v = static_ir_drop_v + inductive_droop_v * (1.0 - decap_damping)
    dynamic_pct = (dynamic_droop_v / vdd_nominal_v) * 100.0

    effective_voltage = vdd_nominal_v - dynamic_droop_v
    status = "MODELED_PASS" if dynamic_pct < 5.0 else "MODELED_FAIL"

    return IRDropReport(
        vdd_nominal_v=vdd_nominal_v,
        peak_current_a=round(peak_current_a, 3),
        r_pdn_effective_mohm=round(r_pdn_effective_mohm, 3),
        c_decap_total_nf=round(c_decap_total_nf, 1),
        l_pkg_loop_ph=round(l_pkg_loop_ph, 1),
        decap_damping_assumed=decap_damping,
        static_ir_drop_mv=round(static_ir_drop_v * 1000.0, 3),
        static_ir_drop_pct=round(static_pct, 3),
        dynamic_ir_droop_mv=round(dynamic_droop_v * 1000.0, 3),
        dynamic_ir_droop_pct=round(dynamic_pct, 3),
        effective_core_voltage_v=round(effective_voltage, 4),
        status=status,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vdd", type=float, required=True)
    ap.add_argument("--current", type=float, required=True,
                    help="peak burst current in A -- state your derivation")
    ap.add_argument("--r-pdn-mohm", type=float, required=True,
                    help="ASSUMED effective grid resistance; no default -- "
                         "replace with a real PDN extraction when one exists")
    ap.add_argument("--c-decap-nf", type=float, required=True)
    ap.add_argument("--l-pkg-ph", type=float, required=True)
    ap.add_argument("--decap-damping", type=float, required=True)
    ap.add_argument("--out", type=Path, default=Path("p4/power/ir_drop_report.json"))
    a = ap.parse_args()

    rep = analyze_ir_drop(a.vdd, a.current, a.r_pdn_mohm, a.c_decap_nf,
                          a.l_pkg_ph, a.decap_damping)
    print("=== Sonic S1 IR-Drop MODEL (not a measurement -- see docstring) ===")
    print(f"  Every R/L/C/current input below is an ASSUMPTION, stated explicitly "
          f"on the command line, not extracted from a real PDN.")
    print(f"  Dynamic Peak Droop:  {rep.dynamic_ir_droop_mv:.3f} mV "
          f"({rep.dynamic_ir_droop_pct:.3f}% VDD) (gate < 5.0%) -> {rep.status}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep.__dict__, indent=2))
    print(f"\nWrote report (modeled, not measured) to {a.out}")
    return 0 if rep.status == "MODELED_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
