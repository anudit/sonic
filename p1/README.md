# P1 — Architecture model and package decision

**Weeks 5–14 · 2 architects, overlapping P0.** Turns the plan's asserted
configuration into a defended one, and settles the two decisions that cannot be
revisited later: package technology and process node.

## Deliverables

| # | Deliverable | Artifact | Status |
|---|---|---|---|
| P1-1 | Array × chunk × SRAM sweep | `p1/sweep.py` | analytical |
| P1-2 | Systolic occupancy under routing imbalance | `p1/occupancy.py` | **measured (0.888 on 4x64²; 0.94 on 8-expert top-1)** |
| P1-3 | DRAM efficiency against a real timing model | `dramsim3/configs/LPDDR5X_16Gb_x16_8533.ini`, `p1/dram.py` | **DONE — measured (0.885 vs 0.85 gate)** |
| P1-4 | Package/bump-pitch decision | `p1/PACKAGE.md` | **DONE — 130 µm pitch FO-WLP, 20.56 mm² die, 9x9 mm² 256-BGA** |
| P1-5 | 14 nm vs 22 nm against a measured power model | `p1/power.py`, `p1/out/power_model.json` | **DONE — 14nm FinFET (0.75W decode / 1.17W prefill <= 2.0W envelope)** |

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

## The expert gather costs nothing (P1-3, measured)

`make p1-dram`. Three patterns through DRAMsim3, every request injected at
cycle 0 so the controller runs saturated, throughput read as reads completed in
a fixed 100 K-cycle window:

| pattern | GB/s | of peak | row-hit rate | ACTs | vs stream |
|---|---:|---:|---:|---:|---:|
| stream | 9.14 | 47.4% | 0.9911 | 106 | 1.000 |
| **expert** | **9.18** | 47.6% | **0.9909** | 109 | **1.004** |
| scatter | 9.53 | 49.5% | 0.0002 | 12,509 | 1.043 |

**The gather is free**, and the reason is a number the plan already contained
but had never applied here: **an expert is 5.96 MB of contiguous weights**,
93,112 cache lines. The MoE "gather" is four jumps per layer between
megabyte-scale sequential runs, not a scatter of small reads, so the row-buffer
hit rate is at parity with a pure sweep — 0.9909 against 0.9911.

The `scatter` control is the pattern the flat derate was implicitly feared to
be. It is 4% *faster* than sequential, which is worth understanding rather than
dismissing: with `address_mapping = rochrababgco` the column bits are last, so a
sequential sweep serialises on one bank while random addresses spread across
banks and ranks. At this queue depth the device is bank-parallelism-limited, not
row-hit-limited. So the specific fear behind the flat derate — that the gather
would thrash the row buffer — does not describe this device at all.

**What this does not license.** The absolute 47% of peak is a property of this
LPDDR4-2400 model and its controller (`cmd_queue_size = 8`), not of Sonic.
`dram_eff` stays at 1.00 rather than being replaced by a number measured on the
wrong device. DRAMsim3 ships no LPDDR5X config, so closing the absolute half
needs either an authored LPDDR5X model or vendor data. That is now the whole of
what P1-3 has left, and it is a narrower question than the one it started with.

Note `dram_eff` is load-bearing: decode is 68.2 tok/s at 1.00 against a 70 tok/s
gate, so any realistic derate fails it. That is a real finding for the SKU
ladder, not a reason to keep the optimistic value.

## Open items

1. ~~Charge for sub-tiling in `sonic/roofline.area`~~ — **done, above.** It costs
   ~2% of die and does not change the recommendation. Model the *bandwidth* cost
   of sub-tiling instead: that is where the penalty must be, if there is one.
2. ~~Hook up a real DRAM timing model (DRAMsim3)~~ — **done for the pattern
   half; see below.** The expert gather costs nothing. What remains unmeasured
   is absolute device efficiency, which needs an LPDDR5X model DRAMsim3 does not
   ship.
3. ~~Re-run every sweep against measured routing~~ — **done**; P0-2 landed and
   the numbers above are measured, not modelled.
4. **Model the fewer-larger-experts variant** (8 experts top-1) — it makes every
   occupancy number better and is a co-design ask that must reach the model team
   before their next architecture freeze.
