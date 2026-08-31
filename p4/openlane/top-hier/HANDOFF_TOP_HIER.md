# Handoff: `sonic_top` hierarchical P&R (T4.2 proper)

This is the real next step after the flat `sonic_top` smoke run
(`p4/openlane/top/`, `N_TILES=1`, `N_SRAM_BANKS=1` — see its own
`HANDOFF_TOP_RUN.md`). That run proved the full 18-file, 12-block hierarchy
makes it through synthesis-to-GDS *as one flattened netlist*; this run
proves the shipping approach instead — harden `sonic_tile`, `sonic_router`
and `sonic_sram_bank` once each, then instance them as macros. Same x86-64
Linux + Nix prerequisites as that doc; not repeated here.

## Why this is a separate step, not just editing `top/config.json`

Macro-based assembly needs GDS + LEF + multi-corner `.lib` + a blackbox
netlist *for each macro*, produced by that macro's own standalone run. The
flat run never produced those (it re-synthesizes tile/router/sram_bank's
internals every time as part of the full netlist). `p4/openlane/top-hier/`
is a new config, `DESIGN_NAME` still `sonic_top`, but `VERILOG_FILES` drops
`sonic_tile.sv`/`sonic_router.sv`/`sonic_acc.sv`/`sonic_pe.sv` (folded into
the `sonic_tile` macro) and `sonic_sram_bank.sv` stays out too — `sonic_sram.sv`
(the bank-instancing wrapper) is still in the file list, but each
`sonic_sram_bank` instance inside it resolves to the macro.

## Prerequisite artifacts — what exists, what's missing

| Macro | Config that hardened it | GDS/LEF | `.lib` (9 corners) | Blackbox netlist |
|---|---|---|---|---|
| `sonic_tile` (T=8) | `p4/openlane/tile/config.json` | **not vendored** — only `results/tile8/metrics.{json,csv}` and `lib/` are committed | present, `results/tile8/lib/*/` | **not vendored** |
| `sonic_router` (LANES=4/SEGS=8) | `p4/openlane/router/config.json` | not vendored (only `results/segs8/metrics.*`, `.rpt`, `.jpg`) | **missing entirely** — no `lib/` dir under `results/segs8/` at all | not vendored |
| `sonic_sram_bank` (ADDR_WIDTH=8) | `p4/openlane/sram/config.json` | not vendored | present, `results/sram8/lib/*/` | not vendored |

None of the three is ready to consume yet. Concretely, before
`p4/openlane/top-hier/config.json` will run:

1. **Re-run `p4-tile`, `p4-router`, `p4-sram`** with their existing
   `config.json` unchanged (same `T=8`/`LANES=4`+`SEGS=8`/`ADDR_WIDTH=8` this
   config's `MACROS` block assumes — do not "improve" these while re-running,
   or the hardened macro won't match what `sonic_top`'s parameterized
   instances expect).
2. **This time, keep the GDS and LEF**, not just `metrics.json`. Copy
   `runs/<tag>/final/gds/*.gds` → `results/<tag8>/gds/`,
   `runs/<tag>/final/lef/*.lef` → `results/<tag8>/lef/`, and the blackbox
   netlist (`runs/<tag>/final/nl/*.nl.v` or LibreLane's equivalent structural
   view for macro use) → `results/<tag8>/nl/`, mirroring the existing
   `lib/` layout so `config.json`'s `dir::../<block>/results/<tag8>/...`
   paths resolve.
3. **`sonic_router` specifically is missing its `.lib` outright** — tile and
   sram both have all 9 corners under `results/*/lib/`, router has none.
   Without it, OpenSTA has no timing model for the router macro and top-level
   STA cannot report a meaningful WNS/TNS once the macro is dropped in. This
   has to be fixed by re-running `p4-router` and actually copying
   `runs/<tag>/final/lib/` this time — it's not a flow bug, the artifact was
   just never carried over from that run.

If your LibreLane version's macro-consumption step names these files
differently (check `openlane.config.Macro` / the `MACROS` schema in whatever
LibreLane revision `nix run github:librelane/librelane` resolves to), adjust
the `gds`/`lef`/`nl`/`lib` keys in `config.json` to match — the instance
names and `location`/`orientation` values are the part that must not drift
from `macro_placement.cfg`, the file paths are just plumbing.

## What's already prepared here

- `config.json`: `MACROS` block with all three macro types, real instance
  paths verified against `p2/rtl/sonic_top.sv` and `p2/rtl/sonic_sram.sv`
  (`g_tiles[i].u_tile`, `u_router`, `u_sram.g_sram_banks[b].u_bank` — **not**
  the placeholder `u_tile_0..3` names the old sketch used), a real
  Sky130-scale floorplan (5150 x 4300 um, not the old sketch's shipping-scale
  4534x4534 "14nm" die), and `N_TILES=4`/`N_SRAM_BANKS=4` — a genuine step up
  from the flat run's `N_TILES=1`/`N_SRAM_BANKS=1`, affordable specifically
  because macro instancing doesn't re-run ABC on tile/router/sram_bank.
- `macro_placement.cfg`: human-readable version of the same layout, for
  cross-checking by eye — not consumed by the flow.

## The run, once prerequisites are met

```
cd p4/openlane/top-hier
nix run github:librelane/librelane -- config.json
```

(`make p4-top-hier` from the repo root does the same and opens the result in
KLayout on macOS, mirroring `p4-top`/`p4-router`.)

## What to watch for, beyond the flat run's landmines

`HANDOFF_TOP_RUN.md`'s landmine list (ABC non-convergence, `GPL-0301`,
`DPL-0036`, generous `CLOCK_PERIOD`, no real SRAM compiler) all still apply
to the glue logic. New risks specific to macro instancing:

- **Macro halo / channel spacing.** `macro_placement.cfg` uses 250um gaps.
  If global routing can't get PDN straps or signal routing between macros at
  that spacing, widen the gaps (and `DIE_AREA`) rather than shrinking
  macros — the macros' internal layout is fixed, only the floorplan around
  them can move.
- **Pin access at macro boundaries.** Unlike flat synthesis, the macro's
  pins are fixed at the LEF-defined locations on its boundary; if routing
  congestion concentrates at one macro's pin side, that's a floorplan
  orientation/rotation question (`macro_placement.cfg`'s `N` column), not a
  resizer problem.
- **First hierarchical run in this repo — nothing here has verified LEF
  pin-side conventions or a working `MACROS.instances` schema against
  whatever LibreLane version actually resolves.** Treat the first attempt as
  a schema-shakedown, same posture the flat run took toward the four block
  runs before it.

## When it succeeds

Same reporting convention as `HANDOFF_TOP_RUN.md`: copy `metrics.json` etc.
to `p4/openlane/top-hier/results/<tag>/`, check the same
DRC/LVS/antenna/hold fields the flat run's writeup calls out (that run had 2
hold violations and 7 open antenna nets — worth specifically checking
whether the hierarchical floorplan does better or worse), render the GDS,
and diff cell count / area / electrical-limit violations against the flat
run's numbers in `HANDOFF.md` T4.2. That diff is the actual point of this
run — it's what tells you whether hierarchical assembly bought anything over
the flat smoke test besides `N_TILES`/`N_SRAM_BANKS`.
