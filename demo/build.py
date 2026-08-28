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
