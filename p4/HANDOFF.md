# P4 handoff: run sonic_router RTL-to-GDS on an x86-64 Linux box

## The task

Take `p2/rtl/sonic_router.sv` through a full RTL-to-GDS flow on the Sky130 open
PDK using **LibreLane**, and produce two things:

1. `runs/<tag>/final/gds/sonic_router.gds` — the routed layout
2. a PNG render of it (the picture)

This is **flow validation and area feedback**, not the chip's real physical
design. Sky130 is 130 nm; Sonic S1 is specced at 14 nm. Absolute numbers do not
transfer. What transfers is whether the RTL is synthesisable, routable, and free
of structures that blow up a P&R tool — which is exactly what we are still
trying to establish.

## Why not the Mac, and why not CI

- No prebuilt OpenROAD for Darwin ARM, no brew formula, Docker path is x86
  emulation. Local Nix on macOS needs a dedicated encrypted APFS volume at
  `/nix` plus sudo.
- GitHub Actions works but the runner is slow where it hurts (below), and a
  private repo draws on a 2000 min/month quota that a multi-hour P&R run eats.

## Why this box should do better than the CI runner

The bottleneck we actually hit is **ABC technology mapping inside Yosys**, which
is single-threaded. Two GitHub runs reached 41 minutes still inside
`06-yosys-synthesis`, with all of place-and-route still ahead.

`ubuntu-latest` is 4 shared vCPUs at roughly 2.4-3.0 GHz. This box is a Xeon
E3-1275 v6, 4 real cores at 3.8 GHz boosting to 4.2. On the single-threaded step
that is the actual wall, expect meaningfully better throughput — and no job
timeout hanging over it.

**Check RAM before starting.** ABC on this design is the memory risk, and the
failure mode is the OOM killer, not a useful error:

```
free -g && nproc
```

Below ~16 GB, start at a lower `LANES` (see Knobs).

## Setup

```bash
# 1. Nix (Linux root: no volume, no sudo dance -- unlike macOS)
sh <(curl -L https://nixos.org/nix/install) --daemon

# 2. Flakes + the FOSSi binary cache. WITHOUT the cache lines LibreLane
#    compiles its entire toolchain from source; with them it downloads.
cat >> /etc/nix/nix.conf <<'EOF'
experimental-features = nix-command flakes
extra-substituters = https://nix-cache.fossi-foundation.org
extra-trusted-public-keys = nix-cache.fossi-foundation.org:3+K59iFwXqKsL7BNu6Guy0v+uTlwsxYQxjspXzqLYQs=
EOF
systemctl restart nix-daemon

# 3. The repo is PRIVATE -- needs credentials
gh auth login          # or a deploy key / PAT
git clone https://github.com/anudit/sonic.git && cd sonic
```

## Run

```bash
cd p4/openlane/router
nix run github:librelane/librelane -- \
  --run-tag local \
  --pdk-root ~/.volare \
  config.json
```

The Sky130 PDK is fetched automatically by `volare` on first run and cached in
`~/.volare` — later runs skip it.

## Render the picture

```bash
apt-get install -y klayout xvfb
GDS=$(ls -t runs/local/final/gds/*.gds | head -1)
xvfb-run -a klayout -z -nc \
  -r ../../render_gds.py \
  -rd gds="$GDS" \
  -rd out="$PWD/sonic_router.png" \
  -rd size=4000
```

`p4/render_gds.py` is already in the repo. `-z` gives a hidden main window,
which still owns the `LayoutView` that `save_image` needs; `xvfb-run` covers the
headless server.

## Knobs, in the order worth trying

`config.json` currently ships `LANES=16` and `ROUTER_PWL_SEGS=64`.

**Start at `LANES=4`.** Get one green end-to-end run and a real PNG first, then
climb. A flow you have never completed is not the place to also be chasing a
three-hour synthesis.

```bash
jq '.SYNTH_PARAMETERS = ["LANES=4"]' config.json > c.tmp && mv c.tmp config.json
```

Three things to know before you tune:

1. **`LANES` does not shrink the hard part.** It scales the multiplier array.
   The congestion and ABC hot spot is the top-k network — a wide combinational
   cone over a **1024-bit score register** — and that is unchanged at every
   `LANES` value. If ABC still grinds at `LANES=4`, the cone is the cause, and
   `ROUTER_PWL_SEGS` is the knob that actually touches it.

2. **`GRT_ALLOW_CONGESTION: true` defers, it does not fix.** It stops global
   route failing on congestion and hands the problem to detailed routing, which
   can then iterate a long time and still finish with violations. If detailed
   route is where it stalls, lower `FP_CORE_UTIL` from 40 rather than waiting.

3. **This design has a multiplier-sizing history.** Findings 12-15 in
   JOURNEY.md: the router epilogue at 64x64 made Yosys never finish (three
   killed 500 s runs) and was fixed by multiplying at 32x32; `sonic_conv` went
   688,016 -> 330,640 cells, bit-identical, by not widening tap operands before
   multiplying. If synthesis explodes again, that is the family of cause to
   suspect first.

## What success looks like

```
runs/local/final/gds/sonic_router.gds     the layout
runs/local/final/metrics.json             area, cell count, violations, WNS/TNS
runs/local/*/                             per-step logs and reports
```

`metrics.json` is the actual deliverable of P4 — the GDS is what you look at,
the metrics are what you learn from. Copy both back.

## State as of handoff

- Two GitHub runs cancelled mid-synthesis; no GDS was ever produced.
- The `router RTL-to-GDS` workflow is **disabled** so pushes cannot start
  another multi-hour job. Re-enable with `gh workflow enable router-gds.yml`.
- `config.json` comments use `//`-prefixed keys: LibreLane **errors** on
  unrecognised config keys and exempts only those. Do not reintroduce
  `__comment_*` keys.
- `github:efabless/openlane2` no longer resolves. Efabless shut down; LibreLane
  under the FOSSi Foundation is the continuation.
