#!/usr/bin/env python3
"""P1-1: array width x chunk size x SRAM sweep.

The S1 plan claims 16,384 lanes with 8 MB of SRAM is the knee. This is the
script that has to defend that, and the one that will move it when P0 hands over
measured routing statistics.

Key coupling the sweep exists to expose: a tile x tile systolic array needs at
least `tile` tokens per expert to fill a pass, and tokens-per-expert is
chunk * top_k / n_experts. So array size forces chunk size, which forces SRAM.
Sizing the array without sizing the SRAM produces a fast array that idles.
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sonic import SKUS, load  # noqa: E402
from sonic.moe import occupancy, route_counts  # noqa: E402
from sonic.roofline import area, decode, prefill, sram_for_chunk  # noqa: E402

REAL_TRACE = Path("p0/out/real_routing.npz")


def real_occupancy_by_tile(trace_path: Path = REAL_TRACE, n_experts: int = 32) -> dict[int, float]:
    """Occupancy at each candidate sub-tile edge, from ACTUAL measured routing
    decisions (p0/out/real_routing.npz, 16,104 real token-expert assignments,
    CV 0.163) rather than a synthetic lognormal imbalance model. One real
    "chunk" per captured layer (732 tokens each, 22 layers)."""
    import numpy as np
    d = np.load(trace_path)
    layer_keys = [k for k in d.files if k.startswith("layer_")]
    out: dict[int, list[float]] = {}
    for tile in (32, 45, 64, 90, 128):
        occs = []
        for k in layer_keys:
            sel = d[k]  # [tokens, top_k] real expert indices
            counts = np.zeros(n_experts, dtype=np.int64)
            for row in sel:
                for e in row:
                    counts[e] += 1
            occs.append(occupancy(counts, tile))
        out[tile] = float(np.mean(occs))
    return out

TARGET_TTFT_MS = 250.0   # P1 gate at a 2048-token prompt
TARGET_OCC = 0.80        # P1 gate under measured routing imbalance


def sweep(model, base, prompt=2048, imbalance=0.5, seed=0):
    rows = []
    for lanes in (4096, 8192, 16384, 32768, 65536):
        # A monolithic square array has edge sqrt(lanes). Partitioning the same
        # lanes into independent sub-tiles keeps short experts from wasting a
        # full pass -- occupancy is set by the SUB-TILE edge, not the array edge.
        for tile in sorted({int(math.sqrt(lanes) / f) for f in (1, 2, 4)
                            if int(math.sqrt(lanes) / f) >= 32}):
          n_tiles = lanes // (tile * tile)
          for chunk in (512, 1024, 2048, 4096):
            chip = replace(base, mac_lanes=lanes, tile=tile,
                           sram_mb=math.ceil(sram_for_chunk(model, chunk)))
            pf = prefill(model, chip, prompt, chunk=chunk)
            counts = route_counts(model, chunk, imbalance,
                                  np.random.default_rng(seed))
            occ = occupancy(counts, tile)
            # Occupancy derates the compute term; memory is unaffected.
            eff_ttft = max((pf.compute_ms + pf.attn_ms) / occ, pf.memory_ms)
            rows.append(dict(lanes=lanes, tile=tile, n_tiles=n_tiles, chunk=chunk,
                             sram=chip.sram_mb, occ=occ,
                             ttft_ideal=pf.ttft_ms, ttft_real=eff_ttft,
                             die=area(chip)["_total"],
                             decode=decode(model, chip).tok_s))
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="lfm2.5-8b-a1b")
    ap.add_argument("--sku", default="B", choices=list(SKUS))
    ap.add_argument("--prompt", type=int, default=2048)
    ap.add_argument("--imbalance", type=float, default=0.1634,
                    help="routing load CV; 0.0 = uniform. Default is the real "
                         "measured value from p0/out/real_routing.npz (16,104 "
                         "decisions, P3-1) -- see also --real-trace for a run "
                         "against the actual captured counts, not a lognormal fit.")
    ap.add_argument("--real-trace", type=Path, default=None,
                    help="path to a real_routing.npz; if given, also prints "
                         "occupancy computed directly from real captured "
                         "counts (not the synthetic imbalance model) at each "
                         "candidate tile edge")
    a = ap.parse_args()

    model, base = load(a.model), SKUS[a.sku]
    print(f"{model.name} on SKU {a.sku} ({base.dram_gbps} GB/s), "
          f"prompt {a.prompt}, routing CV {a.imbalance}")
    print(f"gates: TTFT <= {TARGET_TTFT_MS:.0f} ms, occupancy >= {TARGET_OCC:.2f}\n")
    print(f"{'lanes':>7}{'layout':>12}{'chunk':>7}{'SRAM':>6}{'occ':>7}"
          f"{'TTFT ideal':>12}{'TTFT real':>11}{'die mm2':>9}  verdict")

    best = None
    for r in sweep(model, base, a.prompt, a.imbalance):
        ok = r["ttft_real"] <= TARGET_TTFT_MS and r["occ"] >= TARGET_OCC
        if ok and (best is None or r["die"] < best["die"]):
            best = r
        layout = f"{r['n_tiles']}x{r['tile']}^2"
        print(f"{r['lanes']:>7,}{layout:>12}{r['chunk']:>7}{r['sram']:>5.0f}M"
              f"{r['occ']:>7.3f}{r['ttft_ideal']:>10.0f}ms{r['ttft_real']:>9.0f}ms"
              f"{r['die']:>9.2f}  "
              f"{'PASS' if ok else ('occ' if r['occ'] < TARGET_OCC else 'ttft')}")

    print()
    if best:
        print(f"smallest die meeting both gates: {best['lanes']:,} lanes as "
              f"{best['n_tiles']} x {best['tile']}^2 sub-tiles / chunk {best['chunk']} / "
              f"{best['sram']:.0f} MB SRAM -> {best['die']:.2f} mm2, "
              f"TTFT {best['ttft_real']:.0f} ms, occupancy {best['occ']:.3f}")
    else:
        print("NO configuration meets both gates at this routing imbalance.")
        print("Escalate: larger chunk (more SRAM), better ragged packing, or")
        print("push plan section 06 change 2 (shared always-on expert).")

    if a.real_trace:
        real_occ = real_occupancy_by_tile(a.real_trace)
        print(f"\nReal-data check ({a.real_trace}, 22 real captured layers, "
              f"732 tokens/layer -- not the synthetic imbalance model above):")
        for tile, occ in sorted(real_occ.items()):
            print(f"  tile={tile:>4}  occupancy={occ:.3f}  "
                  f"{'PASS' if occ >= TARGET_OCC else 'occ'} (gate >= {TARGET_OCC:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
