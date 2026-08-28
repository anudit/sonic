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

## Sub-tiling is now priced, and it does not change the answer

`roofline.area` charged a flat cost per lane, so `tile` did not appear in it at
all — the sweep saw sub-tiling's occupancy benefit in full and none of its cost.
That is a one-sided incentive, not a missing coefficient, which is why the sweep
rewarded arbitrarily small tiles.

The cost is now measured rather than asserted. `p2/rtl/sonic_tile.sv`
synthesized standalone at T = 4, 8, 16 with Yosys + `abc -g cmos2`:

```
cells(T) = 712.0*T^2 + 2158.9*T - 1514.3        validated at T=12, within 1.5%
```

The `T^2` term is the PE array; the linear and constant terms are the per-tile
overhead — accumulator column, control, partial-sum egress — amortised over
fewer lanes as tiles shrink. Array area per lane, against a monolithic 128 edge:

| sub-tile edge | 128 | 64 | 32 | 16 |
|---|---:|---:|---:|---:|
| array area factor | 1.000 | 1.023 | 1.068 | 1.154 |
| die, 16,384 lanes | 20.56 | 20.70 | 20.98 | 21.52 mm² |

**The charge is real and it is small.** Going 128 -> 32 costs 6.8% of array area
and about 2% of the die, against an occupancy gain of 0.805 -> 0.942. The sweep
still selects **16 x 32²**, unchanged.

So the open item is closed with a negative result: **area was not the missing
cost.** Either small tiles genuinely are better and 4 x 64² was conservative, or
the real penalty is bandwidth — weight reuse per tile and cross-tile partial-sum
traffic — which is not an area term and will not be found by refining one. That
is the next thing to model, and it is now the only remaining reason to doubt
16 x 32².

The reference edge is anchored at 128 so adopting this moves no published
figure: the S1 default still reports 20.56 mm².

## Open items

1. ~~Charge for sub-tiling in `sonic/roofline.area`~~ — **done, above.** It costs
   ~2% of die and does not change the recommendation. Model the *bandwidth* cost
   of sub-tiling instead: that is where the penalty must be, if there is one.
2. **Hook up a real DRAM timing model** (DRAMsim3). `dram_eff` is currently a
   flat 1.0 in `chipspec.py` with 0.85 as the gate; the expert-gather access
   pattern is exactly the case where a flat derate is least trustworthy.
3. ~~Re-run every sweep against measured routing~~ — **done**; P0-2 landed and
   the numbers above are measured, not modelled.
4. **Model the fewer-larger-experts variant** (8 experts top-1) — it makes every
   occupancy number better and is a co-design ask that must reach the model team
   before their next architecture freeze.
