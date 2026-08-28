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

| Step | State |
|---|---|
| KLayout viewer | installed (local render also done in CI) |
| Yosys synthesis | works (198,703 cells at LANES=64) |
| Sky130 PDK | fetched by LibreLane on first run (`volare`) |
| LibreLane on CI | `.github/workflows/router-gds.yml` |
| `sonic_router` config | written, `p4/openlane/router/config.json` |

## Sizing note

`sonic_router` at the shipping LANES=64 is ~199k cells. On Sky130 that is a
multi-hour run. `config.json` therefore builds **LANES=16** by default: same
structure, same routing character, roughly a quarter the multiplier array, and a
run that finishes in well under an hour. Raise it once the flow is proven.

## What a full-chip layout would need, and why it is not this

1. **The RTL does not exist yet.** P2-5 covers the systolic array, streamer,
   conv, attention, LM head, NoC and sequencer — around 60% of the 20.6 mm².
2. **Wrong node.** Sky130 is 130 nm. The same design is 50-100x the area there;
   a 16,384-MAC array would not route in tolerable time and the picture would
   not describe the chip being planned.
3. **No hard macros.** There is no open 8 MB SRAM compiler and no LPDDR5X PHY.
   Those are 44% of the die and would appear as empty blockages.

Full-chip P&R is the commercial-tool, real-PDK deliverable in the program plan.
