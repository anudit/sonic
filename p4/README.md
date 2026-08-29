# P4 — Physical design

Block-level place-and-route on an open PDK. This is a **flow-validation and
area-feedback** exercise, not the chip's real physical design: Sky130 is 130 nm
and Sonic S1 is specced at 14 nm, so absolute numbers here do not transfer.
What does transfer is whether the RTL is synthesizable, routable, and free of
structures that blow up a P&R tool.

## How this runs: CI, not this laptop

There is no prebuilt OpenROAD for Darwin ARM, no brew formula, and the Docker
path is x86 emulation. The remaining local option is Nix, which on macOS needs
its own encrypted APFS volume at `/nix` and therefore sudo — a large, permanent
change to the machine in exchange for one block-level run.

So P&R runs on an x86-64 Linux runner instead, where OpenROAD is actually
supported and where Nix is free (no volume, no sudo, nothing to uninstall):

```
make -C .. p4-router-ci     # dispatch the workflow and watch it
make -C .. p4-pull          # download the GDS + PNG, open the render
```

Override the width without a commit:

```
make -C .. p4-router-ci LANES=64
```

The tool is **LibreLane**, the FOSSi Foundation's continuation of OpenLane 2
after Efabless shut down. `github:efabless/openlane2` no longer resolves.

`make p4-router` still exists for a machine that already has Nix.

## Status

**One block is through the flow end to end.** `sonic_router` reached GDS on
2026-08-28 with clean DRC, LVS, antenna and timing signoff — see
[RESULTS.md](RESULTS.md) for the metrics, the caveats, and what the run
demanded of the RTL.

| Step | State |
|---|---|
| KLayout viewer | installed; 0.26.2 segfaults on `.py` macros, use >= 0.30 |
| Yosys synthesis | works (198,703 cells at LANES=64) |
| Sky130 PDK | fetched by LibreLane on first run (`volare`) |
| LibreLane on CI | `.github/workflows/router-gds.yml` (disabled) |
| `sonic_router` config | `p4/openlane/router/config.json`, matching the run that finished |
| `sonic_router` GDS | **done** — 4 h 40 min, 80/80 stages, DRC/LVS/antenna clean |
| results | `p4/openlane/router/results/segs8/` |

## Sizing note

`sonic_router` at the shipping LANES=64 is ~199k cells. On Sky130 that is a
multi-hour run. `config.json` builds **LANES=4**, which is what the completed
run used: same structure, same routing character, 1/16 the multiplier array.

`LANES` was never the tractability knob, though — `ROUTER_PWL_SEGS` is. At
SEGS=64 the sigmoid table is 4,096 flip-flops behind a 64-way 32-bit mux, and
ABC does not converge on it: two attempts, one killed at 2 h and one at ~11 h of
CPU. SEGS=8 clears synthesis in 2 h. That is a finding about the RTL, not about
the tool — see RESULTS.md.

## What a full-chip layout would need, and why it is not this

1. **The RTL does not exist yet.** P2-5 covers the systolic array, streamer,
   conv, attention, LM head, NoC and sequencer — around 60% of the 20.6 mm².
2. **Wrong node.** Sky130 is 130 nm. The same design is 50-100x the area there;
   a 16,384-MAC array would not route in tolerable time and the picture would
   not describe the chip being planned.
3. **No hard macros.** There is no open 8 MB SRAM compiler and no LPDDR5X PHY.
   Those are 44% of the die and would appear as empty blockages.

Full-chip P&R is the commercial-tool, real-PDK deliverable in the program plan.
