#!/usr/bin/env python3
"""Regenerate the floorplan's data and re-inject it into floorplan.html.

The diagram is only worth anything if its rectangles are the model's real
areas, so the numbers are never hand-edited: this reads them out of
sonic/roofline.py and rewrites the `const DATA = {...}` line in place.

    make demo-data
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sonic import load                                    # noqa: E402
from sonic.chipspec import SKUS                           # noqa: E402
from sonic.quant import BLOCK_FMT, GATES                  # noqa: E402
from sonic.roofline import area, decode, prefill          # noqa: E402


def build() -> dict:
    m8, m2 = load("lfm2.5-8b-a1b"), load("lfm2.5-2.6b")
    out: dict = {"skus": {}}
    for n, c in SKUS.items():
        A, d8, d2 = area(c), decode(m8, c), decode(m2, c)
        pf = prefill(m8, c, 2048)
        out["skus"][n] = dict(
            bus=c.bus_bits, gbps=c.dram_gbps, gb=c.dram_gb,
            blocks={k: round(v, 4) for k, v in A.items()},
            tok_s_8b=round(d8.tok_s, 1), tok_s_26b=round(d2.tok_s, 1),
            watts=round(d8.watts, 2), mj=round(d8.mj_per_token, 1),
            ttft=round(pf.ttft_ms, 1), bound=pf.bound,
            lanes=c.mac_lanes, tile=c.tile, sram=c.sram_mb,
            lanes_needed=round(c.lanes_to_saturate(), 1))
    out["model"] = dict(
        total=m8.total_params, active=m8.active_params,
        sparsity=round(m8.sparsity, 4), mb_tok=round(m8.bytes_per_token(), 1),
        avg_bits=round(m8.avg_bits(), 3), experts=m8.n_experts, top_k=m8.top_k,
        layers=m8.n_layers, conv=m8.n_conv, attn=m8.n_attn, d=m8.d,
        kv_kb=round(m8.kv_kb_per_token(), 3), gop=round(m8.gop_per_token(), 2))
    out["quant"] = {k: dict(bits=v.bits, kind=v.kind, group=v.group, why=v.why)
                    for k, v in BLOCK_FMT.items()}
    out["gates"] = GATES

    # Incorporate real physical design and hardware signoff measurements only
    # -- no invented STA/ATPG/IR-drop numbers. See HANDOFF.md and JOURNEY.md
    # ("a report that described work that hadn't happened") for why this
    # section was rewritten: it used to read p4/sta/sta_report.json,
    # p4/dft/atpg_report.json, p4/power/ir_drop_report.json, all of which
    # were computed from hardcoded literals with no real tool behind them.
    # Those files and the scripts that wrote them have been deleted/rewritten;
    # this only reports what real LibreLane/OpenROAD runs actually measured.
    verif: dict = {
        "top_single_layer_cosine": 0.99119,
        "top_single_layer_status": "PASSED (>=0.99 gate)",
        # Real per-layer result from `make p3-top` Test 3 (4 distinct real
        # MoE layers, p3/export_multi_layer.py) -- NOT all passing. This is
        # the honest number after the previous fake test (which printed the
        # single-layer cosine 4 times) was fixed and found a real gate miss.
        "top_multilayer_layers": [5, 6, 7, 8],
        "top_multilayer_cosine": [0.99119, 0.98993, 0.99086, 0.98689],
        "top_multilayer_status": "2/4 layers PASS >=0.99 gate (l6, l8 FAIL) -- open, not yet root-caused",
    }

    blocks_real = {}
    for name, path in (
        ("sonic_router", ROOT / "p4/openlane/router/results/segs8/metrics.json"),
        ("sonic_tile", ROOT / "p4/openlane/tile/results/tile8/metrics.json"),
        ("sonic_sram_bank", ROOT / "p4/openlane/sram/results/sram8/metrics.json"),
    ):
        if not path.exists():
            continue
        try:
            m = json.loads(path.read_text())
        except Exception:
            continue
        blocks_real[name] = dict(
            wns_ps=m.get("timing__setup__wns"),
            tns_ps=m.get("timing__setup__tns"),
            slew_violations=m.get("design__max_slew_violation__count"),
            fanout_violations=m.get("design__max_fanout_violation__count"),
            cap_violations=m.get("design__max_cap_violation__count"),
            utilization=m.get("design__instance__utilization"),
            std_cells=m.get("design__instance__count__stdcell"),
            power_w=m.get("power__total"),
            # Real OpenROAD PDN static-drop check, where present (sram run
            # has it; router/tile runs predate/omit it). This is a per-net
            # static check on THIS reduced-scale block, not the full-chip
            # dynamic prefill-burst scenario p4/power/ir_drop.py models --
            # label it as such wherever quoted.
            pdn_drop_worst_v=m.get("design_powergrid__drop__worst"),
        )
    if blocks_real:
        verif["sky130_real"] = blocks_real
        verif["sky130_note"] = ("Real LibreLane/OpenROAD Sky130 results at reduced scale "
                                 "(LANES=4 router, TILE=8 tile, ADDR_WIDTH=8 sram) -- Sky130 "
                                 "is a real 130nm node, not 14nm, and these are flow-"
                                 "qualification runs, not shipping-scale signoff. See "
                                 "p4/sta/sta_check.py --metrics for the full per-corner data.")

    pf_file = ROOT / "p3/out/prefill_schedule.json"
    if pf_file.exists():
        try:
            verif["prefill_schedule"] = json.loads(pf_file.read_text())
        except Exception:
            pass

    # Real rendered GDS images (KLayout, real Sky130 layer colors), embedded
    # as data URIs so the page stays a single self-contained file.
    renders: dict = {}
    render_dir = Path(__file__).parent / "render"
    for name, path in (
        ("tile_overview", render_dir / "sonic_tile.png"),
        ("tile_wires", render_dir / "sonic_tile_crop.png"),
        ("router_overview", ROOT / "p4/openlane/router/results/segs8/sonic_router.jpg"),
        ("sram_overview", render_dir / "sonic_sram.png"),
        ("sram_wires", render_dir / "sonic_sram_crop.png"),
    ):
        if path.exists():
            import base64
            mime = "image/png" if path.suffix == ".png" else "image/jpeg"
            renders[name] = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"
    out["renders"] = renders

    out["verification"] = verif
    return out


def main() -> int:
    data = build()
    (Path(__file__).parent / "data.json").write_text(json.dumps(data, indent=1))

    page = Path(__file__).parent / "floorplan.html"
    src = page.read_text(encoding="utf-8")
    blob = json.dumps(data, separators=(",", ":"))
    new, n = re.subn(r"const DATA = .*?;\n", f"const DATA = {blob};\n", src,
                     count=1, flags=re.S)
    if n != 1:
        raise SystemExit("could not find the `const DATA = ...;` line to replace")
    # The page is written ASCII-only so it renders correctly however it is
    # served -- a stray U+00B2 shows up as mojibake without a charset header.
    bad = [c for c in new if ord(c) > 127]
    if bad:
        raise SystemExit(f"non-ASCII crept in: {sorted(set(bad))}")
    page.write_text(new, encoding="utf-8")

    b = data["skus"]["B"]
    print(f"  SKU B: {b['blocks']['_total']:.2f} mm2, {b['tok_s_8b']} tok/s, "
          f"{len(b['blocks']) - 1} blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
