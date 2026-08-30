# Handoff: `sonic_top` flat GDS smoke run

You're picking this up because it needs an x86-64 Linux box — there is no
OpenROAD build for macOS/ARM, and the Docker path there is x86 emulation
(unusably slow for a multi-hour P&R run). This doc is self-contained; you
should not need to read the rest of the repo to execute the run, though the
"why" section at the end links to where the numbers came from.

## What this run is, and is not

`sonic_top` is the Sonic S1 chip's real, fully-wired RTL top level —
`p2/rtl/sonic_top.sv` instantiates all 12 digital blocks (sequencer, weight
streamer, MoE router, 4x systolic sub-tiles, short-conv, online-softmax,
LM head, vector unit, 8 MB SRAM + gating + MBIST, RV32 controller, NoC,
I/O ring) and is already verified end-to-end in simulation — `make p3-top`
drives a real transformer layer through it against a golden C model at
cosine similarity 0.99119. That part is done; you don't need to touch RTL.

**This run is a flat LibreLane (OpenLane 2 successor) synthesis-to-GDS pass
on the open Sky130 PDK, at every size knob shrunk to its smallest tractable
value.** It is a *flow-qualification smoke test* — proving the entire
18-file hierarchy makes it through synthesis, floorplan, place, CTS, route
and DRC/LVS signoff as one netlist — not a shipping-scale or 14 nm result.
This repo's own `HANDOFF.md` (search "T4.2") is explicit that the real
16,384-MAC, 32-expert, 8 MB-SRAM chip will never go through a *flat* run;
the real path is hierarchical (harden each block once, instance as macros).
That hierarchical config does not exist yet — this flat smoke run is the
next rung, matching the same "reduced scale, prove the flow" strategy that
already worked for the four block-level runs in this repo (below).

**Do not report any timing, power, or area number from this run as a
shipping-scale or 14 nm claim.** Sky130 is a 130 nm PDK; Sonic S1 targets
14 nm. Report what actually happened: did the flow complete, how many
stages, what DRC/LVS/antenna signoff said, wall-clock time, cell count.

## Precedent — what already worked, so you know what "normal" looks like

Four blocks from this same RTL have already been through this exact flow on
a fresh x86-64 Linux + Nix box:

| Block | Config | Result |
|---|---|---|
| `sonic_router` (MoE top-k router alone) | `p4/openlane/router/config.json` | **Full GDS, clean DRC/LVS/antenna**, 224,588 cells, 4h40m, WNS/TNS=0 at `CLOCK_PERIOD=1500`. Real electrical-limit violations remain (42,310 max-slew, 3,105 max-fanout) — see `p4/RESULTS.md`. |
| `sonic_tile` at `T=8` | `p4/openlane/tile/config.json` | Completed, 51,199 cells, WNS/TNS=0 at 1000 ns. Real slew/fanout violations too. |
| `sonic_sram_bank` at `ADDR_WIDTH=8` | `p4/openlane/sram/config.json` | Completed, 49,514 cells, WNS/TNS=0 at 1000 ns, 0.86 mW. |
| `sonic_seq` at `DEPTH=64` | `p4/openlane/seq/config.json` | Config exists, not yet run. |

So: this flow works on this RTL, on this PDK, on an x86 box with Nix. The
`sonic_top` run below is the union of all of the above plus 8 more small
blocks in one netlist — expect it to be at least as slow as the router
alone (the single largest contributor), likely several hours longer.

## Prerequisites on the target box

- x86-64 Linux (the CI workflow this repo already has, `router-gds.yml`,
  runs on `ubuntu-latest` — that's a known-good target).
- **Nix**, with flakes enabled. Install:
  ```
  sh <(curl -L https://nixos.org/nix/install) --daemon
  ```
  then either add `experimental-features = nix-command flakes` to
  `/etc/nix/nix.conf` / `~/.config/nix/nix.conf`, or pass
  `--extra-experimental-features 'nix-command flakes'` on every `nix` call.
- **Disk**: reserve at least 20-30 GB free. The Sky130 PDK (`volare`) plus
  the LibreLane Nix closure plus one run directory (the router's `runs/ci/`
  alone holds a 113 MB GDS and many intermediate views) add up. The CI
  workflow reclaims disk first (`rm -rf /usr/share/dotnet /usr/local/lib/android
  /opt/ghc /usr/local/share/boost`) — do the same if space is tight.
- **Memory**: no hard number recorded from the block runs, but ABC technology
  mapping is single-threaded and memory-hungry on wide combinational cones
  (see Known Landmines below) — 16 GB+ recommended, more is safer.
- **Time budget**: budget most of a day. The router alone was 4h40m; this
  run includes it plus everything else. If it's still running after ~10-12
  hours with no error, that alone is not evidence of a hang — check
  `flow.log` for forward progress (which stage is active) before killing it.
- Optional but recommended for the FOSSi binary cache (avoids compiling
  LibreLane's dependencies from source):
  ```
  extra-substituters = https://nix-cache.fossi-foundation.org
  extra-trusted-public-keys = nix-cache.fossi-foundation.org:3+K59iFwXqKsL7BNu6Guy0v+uTlwsxYQxjspXzqLYQs=
  ```

## Get the repo

```
git clone <this repo's remote> sonic
cd sonic
git log -1   # sanity check you're on the commit that has p4/openlane/top/config.json
             # with SYNTH_PARAMETERS (not the old flat-attempt config)
```

## The run

```
cd p4/openlane/top
nix run github:librelane/librelane -- config.json
```

(Equivalent to `make p4-top` from the repo root, which also auto-opens the
resulting GDS in KLayout afterward — skip that step on a headless box, see
Rendering below.)

This is the exact invocation pattern `p4-router`/`p4-tile`/`p4-sram`/`p4-seq`
already use in this repo's `Makefile` — nothing new to learn if you've run
any of those.

`config.json` already carries every parameter override needed
(`SYNTH_PARAMETERS`): `T=8` (tile edge, not the shipping 64), `N_TILES=1`
(not 4), `ROUTER_LANES=4`/`ROUTER_PWL_SEGS=8`/`ROUTER_PWL_RANGE=4` (the
exact recipe that finished the standalone router run), `CONV_CH=4`,
`LMHEAD_K=4`, `SEQ_DEPTH=32`, `N_SRAM_BANKS=1`, `SRAM_AW=6`. You should not
need to change these for a first run — read the `"// scope"` and `"// util"`
comments in the config for the reasoning if you want it, but don't tighten
utilization or raise these numbers before a first successful run establishes
a baseline.

## Watching it run

```
tail -f runs/*/flow.log
```

LibreLane numbers each stage (`01-`, `02-`, ...); `flow.log` names the
current one. `p4/openlane/router/results/segs8/step-runtimes.txt` (already
in this repo) shows where the router run's time actually went — synthesis
(ABC) and the post-CTS resizer were 59% of the total. Expect the same shape
here, likely worse (more blocks to map).

## Known landmines (from the four runs that already hit them)

1. **ABC does not converge on wide tables/muxes, no matter how much time you
   give it.** The router's PWL table blew up ABC for 2+ hours at
   `ROUTER_PWL_SEGS=64` (4,096 flip-flops behind a 64-way mux) and a later
   experiment burned ~11 hours and never converged at that setting even with
   better RTL. `config.json` already uses `ROUTER_PWL_SEGS=8` — the value
   that actually cleared this. **Do not raise it** without expecting a
   multi-hour-to-never blowup. If synthesis (stage `06-yosys-synthesis` or
   similar) is stuck for 2+ hours with no forward progress in the log, this
   is almost certainly what's happening — check which module ABC is mapping
   in the log tail, not just wall-clock time.

2. **A closed WNS/TNS=0 headline does not mean the design meets timing.**
   Every run so far uses a deliberately generous `CLOCK_PERIOD` (1500 ns
   here) specifically to let the flow *reach routing* — it is not a
   frequency claim. Report the real electrical-limit violations
   (`design__max_slew_violation__count`, `design__max_fanout_violation__count`,
   `design__max_cap_violation__count` in `metrics.json`) alongside WNS/TNS,
   the way `p4/RESULTS.md` does — a WNS=0 headline without them is
   misleading.

3. **Initial placement density needs headroom for what CTS adds later, not
   just room for the synthesized cell count.** Both `sonic_tile` and
   `sonic_sram_bank` failed on their first *two* attempts each: global
   placement (`GPL-0301`, over-100%-utilization) on attempt 1, then
   *detailed* placement post-CTS (`DPL-0036`) on attempt 2 because ~70-80%
   initial utilization left no room for the hold/fanout resizer's inserted
   buffers to legalize into. Both were fixed by dropping to the router's
   own margin (40-45% target utilization). `config.json` here starts even
   lower (35%/40%) specifically because this is the first-ever run
   combining all 12 blocks and the resulting congestion profile is
   unmeasured — if it fails placement anyway, the fix is almost certainly
   "go lower still," not "go higher."

4. **If global placement fails with over-100% utilization (`GPL-0301`)**
   instead, that means the auto-sized die (from `FP_CORE_UTIL`) is too small
   for the actual synthesized cell count — counter-intuitively the *opposite*
   direction from #3. This has also happened in this repo (both `tile` and
   `sram` hit it before hitting #3). If it recurs here, the fix that worked
   both times was switching to an absolute, deliberately generous
   `DIE_AREA`/`CORE_AREA` (see `p4/openlane/tile/config.json` and
   `p4/openlane/sram/config.json` for the exact keys) rather than trusting
   the percentage-based auto-sizing a second time.

5. **Everything is flip-flops. Nothing here is a real SRAM macro.** The 8 MB
   cache, the router's PWL table, the descriptor ring — all synthesize to
   flip-flop arrays because no `MACROS` key points this flow at a hardened
   SRAM cell. That's expected and consistent with the block-level runs. Do
   not report cell/power/area numbers as if a real memory compiler was used.

6. **macOS `ar` is irrelevant here** (that's a note for Verilator sims on
   this repo's original macOS dev machine, not for LibreLane on Linux) —
   mentioned only so you don't go looking for it as a red herring if you see
   it referenced elsewhere in this repo.

## If it fails

Read `runs/<tag>/error.log` and the tail of the specific stage's own
`.log` file first — LibreLane numbers stages, so `ls runs/*/` shows which
one it died on. Cross-reference against the "Known landmines" list above;
all four block runs in this repo failed at least once before succeeding, and
every failure so far has been one of: ABC non-convergence (landmine 1),
`GPL-0301` (landmine 4), or `DPL-0036` (landmine 3). If you hit something
not on this list, that's new information worth recording, not necessarily a
tooling bug — this exact combination of blocks has never been run before.

## When it succeeds

1. **Metrics**: `runs/<tag>/final/metrics.json` (or wherever LibreLane's
   `final/` view lands) — copy it to `p4/openlane/top/results/<tag>/`,
   mirroring how `router/results/segs8/`, `tile/results/tile8/`, and
   `sram/results/sram8/` are laid out in this repo, so it's easy to diff
   against the precedent table above.
2. **DRC/LVS/antenna signoff**: check `design__lvs_error__count`,
   `magic__drc_error__count`, `klayout__drc_error__count`,
   `route__antenna_violation__count`, `design__xor_difference__count` in
   `metrics.json` — all should be 0 for a clean signoff, exactly as they are
   in the router's `results/segs8/metrics.json` already in this repo (use it
   as the template for which fields matter).
3. **Render the full-chip image** (this is what the user actually wants a
   picture of):
   ```
   sudo apt-get install -y klayout xvfb   # klayout >= 0.30; 0.26.2 segfaults
                                            # on .py macros, see p4/RESULTS.md
   GDS=$(ls -t runs/*/final/gds/*.gds | head -1)
   xvfb-run -a klayout -z -nc -r ../../../p4/render_gds.py \
     -rd gds="$GDS" -rd out=sonic_top.png -rd size=4000
   ```
   `p4/render_gds.py` is already in this repo (loads the GDS, zooms to fit,
   dumps a PNG) — it's exactly what produced
   `p4/openlane/router/results/segs8/sonic_router.jpg`, the image already
   sitting in this repo as a reference for what the output should look like.
4. **Commit back**: the GDS itself is not meant to be vendored (113 MB for
   the router alone; this will be bigger) — per `p4/RESULTS.md`'s own
   convention, commit `metrics.json`, `metrics.csv`, any DRC/LVS/antenna
   reports, `step-runtimes.txt`, and the rendered PNG/JPG. Leave the GDS as
   a downloadable artifact if there's a CI/storage path for it, otherwise
   note where it's parked.
5. **Write it up** the way `p4/RESULTS.md` documents the router run: what
   config actually ran (if you changed anything to get it through), headline
   metrics, the real electrical-limit caveat, wall-clock breakdown by stage.

## Where the numbers in this doc came from, if you want to verify them

- `p4/README.md` — overall P4 status and why this doesn't run on macOS/ARM.
- `p4/RESULTS.md` — the router's full run writeup (ABC blowup, WNS/TNS
  caveat, electrical violations, wall-clock breakdown, tooling notes).
- `HANDOFF.md`, search `T4.1`/`T4.2` — the tile/sram runs' failure-then-fix
  history (the placement density lessons above), and the explicit statement
  that shipping-scale P&R must be hierarchical, not flat.
- `p2/README.md`, "Finding 16" and after — why the router's original
  critical path was pathological and how it was fixed in RTL (already
  fixed, nothing to do here, just context for why `CLOCK_PERIOD=1500`
  looks large).
- `.github/workflows/router-gds.yml` — this repo's working CI recipe for
  the same flow on the same kind of box (disabled by default, but the Nix
  setup / PDK caching / rendering steps in it are all directly reusable
  patterns if you want to script this rather than run it by hand).
