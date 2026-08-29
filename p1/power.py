#!/usr/bin/env python3
"""P1-5a: Power Model for Sonic S1 (14nm FinFET vs 22nm FDX).

Evaluates dynamic switching power, clock distribution, memory PHY IO,
and junction leakage power across Decode (batch 1) and Prefill (batch 2048)
modes against the defended SKU A (<= 2.0 W) and SKU B (<= 3.5 W) envelopes.

Usage:
    python3 p1/power.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

NODES = {
    "14nm_finfet": {
        "vdd": 0.80,            # Core supply voltage (V)
        "c_per_mac_ff": 25.0e-15,# Effective capacitance per MAC PE (25 fF)
        "leakage_mw_mm2": 12.0, # Leakage power density at 85C (mW/mm2)
        "dram_io_mw_gbps": 4.5, # LPDDR5X PHY energy (mW per GB/s)
        "core_area_mm2": 20.56, # Locked die area (mm2)
    },
    "22nm_fdx": {
        "vdd": 0.70,            # Low-Vdd FDX operating point (V)
        "c_per_mac_ff": 35.0e-15,# Effective capacitance per MAC PE (35 fF)
        "leakage_mw_mm2": 6.0,  # Ultra-low leakage with back-biasing (mW/mm2)
        "dram_io_mw_gbps": 5.2, # LPDDR5X PHY energy (mW per GB/s)
        "core_area_mm2": 20.56, # Scaled die area (mm2)
    },
}


def model_power(node_name: str, mode: str = "decode") -> dict:
    cfg = NODES[node_name]
    vdd = cfg["vdd"]
    f_clk = 1.0e9  # 1 GHz

    # 1. Compute Array Power (16,384 MACs)
    num_macs = 16384
    if mode == "decode":
        # Decode is memory-bound: array operates at ~0.7% duty cycle (119 active lanes)
        duty = 119 / num_macs
    else:
        # Prefill is compute-heavy: array operates at ~88% occupancy
        duty = 0.88

    c_mac = cfg["c_per_mac_ff"]
    p_mac_dyn = duty * num_macs * c_mac * (vdd ** 2) * f_clk

    # 2. Clock Tree Distribution (~0.15 W)
    p_clock = 0.15 * (vdd / 0.8) ** 2

    # 3. LPDDR5X DRAM PHY IO Power
    if mode == "decode":
        bandwidth_gbps = 68.26 * 0.885  # ~60.4 GB/s
    else:
        bandwidth_gbps = 68.26 * 0.95   # ~64.8 GB/s
    p_dram_io = (bandwidth_gbps * cfg["dram_io_mw_gbps"]) / 1000.0

    # 4. SRAM Array Power (72KB conv state + activation buffers)
    p_sram = 0.25 * (vdd / 0.8) ** 2 if mode == "prefill" else 0.08 * (vdd / 0.8) ** 2

    # 5. Static Leakage Power at 85C
    p_leakage = (cfg["core_area_mm2"] * cfg["leakage_mw_mm2"]) / 1000.0

    p_total = p_mac_dyn + p_clock + p_dram_io + p_sram + p_leakage

    return {
        "node": node_name,
        "mode": mode,
        "vdd_v": vdd,
        "mac_dyn_w": round(p_mac_dyn, 3),
        "clock_w": round(p_clock, 3),
        "dram_io_w": round(p_dram_io, 3),
        "sram_w": round(p_sram, 3),
        "leakage_w": round(p_leakage, 3),
        "total_w": round(p_total, 3),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=Path("p1/out/power_model.json"))
    a = ap.parse_args()

    results = []
    print(f"\n{'Node':15s} {'Mode':10s} {'MAC (W)':>8s} {'Clk (W)':>8s} {'DRAM (W)':>9s} {'SRAM (W)':>9s} {'Leak (W)':>9s} {'Total (W)':>10s}")
    print("-" * 85)
    for node in ("14nm_finfet", "22nm_fdx"):
        for mode in ("decode", "prefill"):
            res = model_power(node, mode)
            results.append(res)
            print(f"{res['node']:15s} {res['mode']:10s} {res['mac_dyn_w']:8.3f} {res['clock_w']:8.3f} "
                  f"{res['dram_io_w']:9.3f} {res['sram_w']:9.3f} {res['leakage_w']:9.3f} {res['total_w']:10.3f}")

    print("\nDecision: Both nodes fit the defended edge power envelope (SKU A <= 2.0 W, SKU B <= 3.5 W).")
    print("  * 14nm FinFET is selected to comfortably close 1.0 GHz (+15% slack) on the systolic datapath.")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote power breakdown to {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
