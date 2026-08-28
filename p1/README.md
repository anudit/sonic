# P1 — Architecture model and package decision

**Weeks 5–14 · 2 architects, overlapping P0.** Turns the plan's asserted
configuration into a defended one, and settles the two decisions that cannot be
revisited later: package technology and process node.

## Deliverables

| # | Deliverable | Artifact | Status |
|---|---|---|---|
| P1-1 | Array × chunk × SRAM sweep | `p1/sweep.py` | analytical |
| P1-2 | Systolic occupancy under routing imbalance | `p1/occupancy.py` | **measured** |
| P1-3 | DRAM efficiency against a real timing model | — | **not started** |
| P1-4 | Package/bump-pitch decision | — | not started |
| P1-5 | 14 nm vs 22 nm against a measured power model | — | not started |

## Gates

```
decode   >= 70 tok/s at 4K context, SKU B
TTFT     <= 250 ms at a 2048-token prompt
occupancy >= 0.80 under MEASURED routing imbalance   [MET: 0.888 at 4 x 64^2]
dram_eff >= 0.85 of peak
power     within a defended envelope; SKU A <= 2 W
```

## Run

```
make p1
python3 p1/sweep.py --imbalance 0.5      # until P0 supplies the real CV
python3 p1/occupancy.py
```

## The finding that matters

**Settled with measured data.** A monolithic 128×128 array sits on the gate with
no margin; sub-tiling the same 16,384 lanes clears it comfortably at identical
die area, because a short expert wastes a quarter-sized pass instead of a full
one. Occupancy on real traces (`p0/out/real_routing.npz`), chunk 2048:

| Sub-tile edge | 32 | 64 | 96 | 128 |
|---|---|---|---|---|
| Occupancy | 0.942 | **0.888** | 0.843 | 0.805 |

**Decision: 4 × 64² sub-tiles, chunk 2048, 8 MB SRAM.** 64 is chosen over 32
because the sweep's preference for ever-smaller tiles is an artefact of the cost
model (below), and 0.888 is margin enough.

Chunk size still matters independently — at chunk 512 even a 32-edge tile only
reaches 0.794:

| Chunk | tile 32 | tile 64 | tile 128 |
|---|---|---|---|
| 512 | 0.794 ✗ | 0.670 ✗ | 0.461 ✗ |
| 1024 | 0.895 | 0.809 | 0.687 ✗ |
| **2048** | 0.942 | **0.888** | 0.805 |
| 4096 | 0.968 | 0.939 | 0.887 |

**Caveat that must be closed before acting on this:** the area model charges
nothing for sub-tiling. More, smaller tiles mean more accumulator sets, more
control, less weight reuse per tile, and more partial-sum traffic. Treat 4 × 64²
as the recommendation and 16 × 32² as a direction to model properly — the sweep
currently rewards arbitrarily small tiles, which is a sign the cost model is
incomplete, not that 8 × 8 tiles are optimal.

## Open items

1. **Charge for sub-tiling** in `sonic/roofline.area` before trusting P1-1.
2. **Hook up a real DRAM timing model** (DRAMsim3). `dram_eff` is currently a
   flat 1.0 in `chipspec.py` with 0.85 as the gate; the expert-gather access
   pattern is exactly the case where a flat derate is least trustworthy.
3. ~~Re-run every sweep against measured routing~~ — **done**; P0-2 landed and
   the numbers above are measured, not modelled.
4. **Model the fewer-larger-experts variant** (8 experts top-1) — it makes every
   occupancy number better and is a co-design ask that must reach the model team
   before their next architecture freeze.
