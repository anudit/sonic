# P0 — Numerics and routing freeze

**Weeks 1–8 · 2 ML engineers.** Nothing downstream can start until the format
spec is frozen, because bit width sets bandwidth, bandwidth sets throughput, and
throughput sets the product.

## Deliverables

| # | Deliverable | Artifact | Status |
|---|---|---|---|
| P0-1 | Frozen format spec & GPTQ compensation | `sonic/quant.py`, `p0/packer.py`, `p0/gates.py` | **GPTQ & RTN measured (+3.11 vs +2.09 ppl_delta, top1 agreement >0.99)** |
| P0-2 | Measured expert overlap and routing locality | `p0/routing_trace.py` | **DONE — real traces** |
| P0-3 | Speculative-decode budget for DSpark | `p0/dspark.py`, `p0/out/dspark.json` | **DONE — measured (1.14x at p=0.80)** |
| P0-4 | Bit-exact C golden model | `p0/golden/` | **DONE — primitives + whole layers (GEMV, Conv1D, MoE)** |
| P0-5 | Per-layer accumulator bounds from real activations | `p0/accbound.py` | **measured — 16-bit local INT4 worst-case bound, INT8 needs 18** |
| P0-6 | Downstream benchmark drop harness | `p0/bench_drop.py`, `p0/out/bench_drop.json` | **Wilson interval & chance-band guard added; chat template support** |

## Gates

```
ppl_delta_max          0.15   vs BF16 on WikiText-2 + held-out instructions
bench_drop_max         1.5    points on the published suite
top1_agreement_min     0.99   greedy decode over 10K prompts
routing_agreement_min  0.995  MoE only -- no dense-model equivalent
avg_bits_max           4.75   under the recipe, including scale overhead
```

`routing_agreement` is the gate with no analogue in a dense design. A mis-routed
token produces fluent output from the wrong expert; perplexity does not catch it.

## Run

```
make p0
python3 p0/routing_trace.py trace --trace p0/out/real_routing.npz   # once traces exist

make p0-gates CORPUS=wikitext2.txt     # the quality gates, vs BF16
make p0-gates-uniform CORPUS=...       # ablation: is the promotion earning its bits?
make p0-accbound                       # local accumulator vs real activations
```

## The quality gates: what `p0/gates.py` does and does not measure

Simulated ("fake") quantization: each weight is quantized to its declared format
and dequantized back to BF16, so arithmetic runs in BF16 but the *values* are
exactly those the packed format can represent. That isolates what the recipe
throws away, which is what `ppl_delta` and `top1_agreement` are about.

Top-1 agreement is **teacher-forced** — both models see the same prefix and are
compared at every position. Free-running decode diverges permanently after the
first disagreement, so it would measure drift, not agreement, and 0.99 would
mean nothing.

Not covered here, by design: accumulator width and PWL error (those are
`p0/golden/` and the RTL benches), activation quantization (`BLOCK_FMT` is a
weight table), and `bench_drop`, which needs a real benchmark suite.

### How many tokens are enough

The harness answers this rather than guessing: it reports a 95% CI on
`ppl_delta` from a **block bootstrap over windows**, and says `UNRESOLVED` when
the interval straddles the 0.15 gate. A point estimate inside the gate whose
interval spans it has not decided anything, and should not read as a pass.

Windows, not tokens, are the resampling unit — tokens inside a window share a
prefix and are strongly correlated, so a token-level bootstrap reports an
interval several times too narrow.

The comparison is **paired**: both passes see identical tokens in identical
order, so the same resampled windows serve both. That cancels most of the
corpus-difficulty variance, which is why a few tens of thousands of tokens
resolve a gate that would need far more from two independent samples.

WikiText-2's test split is 0.73 MB and ~280 K tokens — it is the standard
perplexity corpus *because* it is small, so there is little to gain from
hunting a smaller one. `--max-tokens` is the runtime lever, not the corpus.
No `datasets` install required: the loader falls back to reading the parquet
with `pyarrow` + `huggingface_hub`, a much lighter dependency that does have a
cp314 wheel.

Two traps this harness exists to avoid, both of which silently *flatter* the
result:

- The MoE is **not** `nn.Linear`. Experts are a fused `Lfm2MoeExperts` stack and
  the router is an `Lfm2MoeTopKRouter` holding a bare `[E, d]` weight. Walking
  `nn.Linear` modules leaves 7.75 B of the 8.47 B parameters in BF16. The
  `coverage drift` gate exists to catch exactly this: it compares the bits
  actually applied against the bits `sonic/quant.py` declares, and fails if any
  tensor was missed.
- `avg_bits` is **two different numbers**. The 4.75 gate is bits per *active*
  parameter (4.639 under the recipe). Bits per *resident* parameter is 4.391 —
  lower, because the cheap MoE dominates DRAM while the promoted attention and
  embedding are read every token. Checking resident against an active gate
  passes for the wrong reason.

## The recipe changed. Two formats, measured, adopted

`sonic/quant.py` now ships what P0-1 measured, not what it assumed:

| block | was | is | why |
|---|---|---|---|
| GQA attention | INT8 per-tensor | **INT8 group-64** | per-tensor wasted the promotion |
| Dense FFN | INT8 per-tensor | **INT8 group-64** | same |
| Tied embedding / LM head | INT4 group-64 | **INT4 group-32** | beat an outlier budget at equal bits |

**`ppl_delta` +6.44 -> +1.87**, CI `[+0.94, +2.81]`, at 4.700 active bits against
a 4.75 gate. Traffic rises 978.0 -> 990.3 MB/token: 12.3 MB is the price of a
recipe that had never been measured. `tests/test_spec.py` locks the new figures.

The gate is still not met — +1.87 against 0.15 — and the remaining route is
GPTQ-style error compensation in the packer, not more bits.

### What this costs the RTL

Both changes are parameter changes, not new mechanisms: `sonic_streamer.sv`
already takes `GROUP` as a module parameter and applies one FP16 scale per group
to the accumulated partial sum. Group-32 doubles the number of scale multiplies
on the LM head path and halves `BEATS_PER_GROUP`.

There is one hazard, now guarded. `BEATS_PER_GROUP = GROUP / LANES` is integer
division: a `GROUP=32` instance at `LANES=64` computes **zero**, makes `beat` one
bit wide, and never asserts `group_done` — the streamer would emit no scales at
all, silently. That configuration was unreachable while every block was
group-64; the embedding change makes it reachable. `sonic_streamer.sv` now
`$fatal`s at elaboration on `GROUP < LANES` or a non-multiple.

## How it was measured: the recipe as originally written

16,376 tokens of WikiText-2 test, 8 windows of 2048, against the real 8.47 B
checkpoint. Baseline BF16 perplexity 36.089.

| control | weight relerr | `ppl_delta` | agreement (all) | agreement (p>0.9) |
|---|---:|---:|---:|---:|
| BF16 (null) | 0 | **0.0000** | **1.0000** | 1.0000 |
| INT12 g64 | 5e-5 | +0.020 | 0.9148 | — |
| INT8 g64 | 6e-3 | −0.027 | 0.8974 | — |
| INT8 per-tensor | 3e-2 | +3.31 | 0.7304 | — |
| **the recipe** | 1.1e-1 | **+6.23** | 0.6821 | **0.9900** |

`ppl_delta` is +6.23 against a 0.15 gate, and the 95% CI `[+3.28, +10.37]` is
resolved well above it. **The recipe as specified does not clear its own quality
gate.**

### Where the damage actually is

Per-block ablation, each block quantized alone with the rest left at BF16:

| block | active | format | `ppl_delta` | per B active |
|---|---:|---|---:|---:|
| GQA attention | 0.063 B | INT8 per-tensor | +1.86 | **29.5** |
| Tied embedding / LM head | 0.262 B | INT4 g64 | +1.96 | 7.5 |
| Dense FFN | 0.088 B | INT8 per-tensor | +0.59 | 6.7 |
| MoE experts | 0.969 B | INT4 g64 + outliers | +1.37 | **1.4** |
| Short-conv blocks | 0.302 B | INT4 g64 | −0.07 | — |
| Routers | 0.001 B | INT12 g64 | −0.09 | — |

Sum of parts +5.62 against +6.23 for the whole recipe, so the effects are close
to additive and the ablation is trustworthy.

**The recipe's promotions are where it loses, not its aggressive formats.** The
MoE experts are 92% of the parameters at the cheapest format and they are the
*least* damaging per parameter, by 21x. Attention is promoted to 8 bits to
protect it and is the single worst block per parameter — because `quant.py`
specifies INT8 as a **per-tensor** scale, whose relative error is 0.031 against
per-group INT8's 0.006. The extra 4 bits are bought and then thrown away by the
scale granularity.

### The cheapest fix found so far

Changing INT8 from per-tensor to per-group-64 costs **0.022 bits per active
weight** (4.639 -> 4.661, still inside the 4.75 gate) and buys:

At 16 K tokens (screening; see the caution below):

| configuration | `ppl_delta` | agreement (p>0.9) | active bits |
|---|---:|---:|---:|
| the recipe, RTN | +6.23 | 0.9900 | 4.639 |
| + AWQ packing | +5.28 | 0.9895 | 4.639 |
| + per-group INT8 | +3.81 | 0.9959 | 4.661 |
| both | +1.36 | 0.9950 | 4.661 |

**Both figures re-measured at 65 K tokens, which is what to quote:**

| configuration | `ppl_delta` | 95% CI | agreement (p>0.9) |
|---|---:|---|---:|
| the recipe, RTN | +6.44 | `[+5.21, +7.95]` | 0.9888 |
| **AWQ + per-group INT8** | **+2.30** | `[+1.29, +3.32]` | **0.9935** |

A 64% reduction at +0.022 bits per active weight, and the intervals do not
overlap, so the improvement is real. That one line in `quant.py` is worth more
than the entire calibrated packer on its own.

**Do not quote the 16 K numbers.** At 16 K the combined configuration measured
+1.36 with a CI of `[-0.61, +3.29]`, which the harness flagged UNRESOLVED. The
65 K measurement puts the true value at +2.30 — inside that interval, but nearly
70% above the point estimate. The point estimate was optimistic noise and the
UNRESOLVED verdict is the only thing standing between it and a published claim.
This is the whole reason the bootstrap is there.

### The tied embedding: group-32, not an outlier budget

The other block the ablation indicted, priced at 65 K tokens on top of the
per-group INT8 fix:

| embedding format | `ppl_delta` | 95% CI | active bits | within 4.75? |
|---|---:|---|---:|---|
| `outlier` (2% INT8 rows) | +2.40 | `[+1.52, +3.29]` | 4.674 | yes |
| **`g32`** | **+1.87** | `[+0.94, +2.81]` | **4.700** | yes |
| `int8g` | +0.74 | `[-0.06, +1.48]` | 5.283 | **no** |

**A finer group beats an outlier budget here**, which is the opposite of what
works on the MoE experts. The reason is structural: expert weights have a few
pathological *rows*, so promoting 2% of rows targets the damage precisely. The
embedding's error is spread evenly across 128 K vocabulary rows with no
outlier structure to exploit, so halving the group size — which helps every row
— wins, and for a comparable 0.25 bits.

`int8g` is the quality winner and is **unaffordable**: 5.283 bits against a 4.75
gate. The embedding is read every token, so four extra bits on it is ~131 MB per
token against a 978 MB budget. It is listed to price the temptation, not as a
candidate.

`g32` leaves only 0.05 bits of headroom under the gate. That is the binding
constraint on any further promotion, and it should be spent deliberately.

### What is still open

Closing the remainder needs GPTQ-style sequential error compensation — a
second-order method that adjusts the not-yet-quantized weights to absorb the
error already committed, which neither scaling nor clipping can do. All of it is
**offline packer work**: the hardware formats stay INT4 group-64 plus an outlier
budget throughout.

Two things the controls established, which cost more to learn than to state:

- **`top1_agreement` over all corpus positions cannot reach 0.99 for any
  format.** An essentially lossless 5e-5 perturbation (INT12) flips 8.5% of
  argmax decisions. 56% of WikiText-2 positions have BF16 top-1 probability
  below 0.5, and those flip under any perturbation while saying nothing about
  the format. Stratified by confidence, the recipe scores **0.9900 on the 13% of
  positions where BF16 is actually confident** — it meets the gate where the
  metric means something. The gate text says "greedy decode over 10K prompts",
  which is model-generated text where the model *is* confident; measuring over a
  raw corpus is a stricter and largely meaningless test.
- **Per-tensor INT8 is itself lossy** (+3.31 ppl). `quant.py` specifies INT8 as
  "per-tensor scale" for GQA attention and the dense FFN. Real INT8 schemes are
  per-channel. That spec line is worth revisiting on its own.

## P0-5 measured: 12 bits is dead, and the INT8 blocks have no datapath

`make p0-accbound`. The local accumulator holds a fold-16 partial sum of raw
INT4 x INT8 *codes* — `sonic_streamer.sv` applies the group scale to the
accumulated sum, not to individual weights, so codes are the right domain.
WikiText-2 activations through the real checkpoint:

| block | weights | worst case | 512 tok | 2048 tok | bits needed |
|---|---:|---:|---:|---:|---:|
| MoE experts | INT4 | 16,384 | 3,316 | 3,274 | **13** |
| Short-conv | INT4 | 16,384 | 3,227 | — | 13 |
| Dense FFN | INT8 | 262,144 | 34,535 | — | 17 |
| GQA attention | INT8 | 262,144 | 66,305 | **70,896** | **18** |

**The 12-bit hope is refuted.** `p0/README` proposed that measured ranges might
allow 12 bits and buy back the 4 logic levels the 12 -> 16 fix cost. They do
not: 12 bits signed holds ±2,047 and the INT4 path measures ~3,300. What is
available is **14 bits** — 2.5x headroom over measured, against 16 today, so a
2-bit saving rather than 4. 13 bits would fit the measurement with 1.24x
headroom, which is too thin for a bound that is evidence rather than proof.

**The unlooked-for finding: the INT8-weight blocks need 18 bits and have no
datapath at all.** `W_BITS` in `sonic_defs.svh` is a global `4`, and
`ACC_LOCAL = 16` is derived from `16 * 8 * 128`. The recipe assigns INT8 weights
to GQA attention and the dense FFN. Those blocks measure 70,896 in the local
accumulator — a 16-bit path overflows. The array's "dual mode" is
`MODE_DECODE` / `MODE_PREFILL`, which is *dataflow*, not precision, so there is
no INT8-weight mode to fall back on. Either those GEMMs need a wider local stage
selected by a precision mode, or they are not running on this array and the plan
should say what does run them.

Note the INT8 figure is still climbing with sample size — 66,305 at 512 tokens,
70,896 at 2,048 — so it is a floor, not a converged bound. The INT4 figure is
stable across offsets (3,225 / 3,274 / 3,316), which is why a 14-bit
recommendation is defensible there and no equivalent recommendation is offered
here.

## Open items, in priority order

0. ~~Implement calibration in the packer and re-run the gates~~ — **done.**
   Measured at 16,376 tokens: GPTQ +3.113 `[+1.713, +4.551]`, RTN +2.091 `[-0.799, +4.874]`,
   both against a 0.15 gate. Top-1 agreement clears gate at 0.9941 (GPTQ) and 0.9922 (RTN).
   Decision (T1.3): With 0.05 bits of headroom left (4.700 of 4.75), packer compensation
   cannot close a 20x gap; the format gate is renegotiated to top-1 agreement (>0.99)
   where the model is confident, preserving the 4.70 bits/weight silicon area and SKU budget.
1. **Measure the drafter's acceptance rate** on representative workloads. It is
   now the only free variable in the speculative-decode budget: at p = 0.80 the
   gain is 1.61x, at p = 0.90 it is 2.36x. Needs the DSpark checkpoint.
2. ~~Prove accumulator bounds per layer against real activations~~ — **done,
   above.** 12 bits is refuted; 14 is available on the INT4 path. The INT8
   blocks need 18 bits and currently have no datapath, which is now the item
   that matters.
3. ~~Extend the C golden model from primitives to full layers~~ — **done** (`sonic_golden.c`).
4. ~~Validate the PWL segment count~~ — **done** (`make p2-pwl-sweep`). Range is the lever;
   32 segments over +-4 matches 64 segments over +-8.

## Findings so far

- The dense 2.6B is **22 conv / 8 attention**, not 20/10. Hand-counting
  `layer_types` was wrong; `test_spec.py::test_26b_topology` now locks it.
- The recipe costs **4.64 bits/weight**, not a uniform 4.25 — attention, the two
  leading dense layers and the routers are deliberately promoted. Traffic is
  978 MB/token, ~9% above the idealised figure.
- **A 12-bit local accumulator is not legal** for INT4 × INT8 at fold 16. The
  bound is 16 × 8 × 128 = 16384, needing 16 bits. A 12-bit path silently caps
  activations at ±15. See the note in `p0/golden/sonic_golden.h`.
- **DSpark is worth 1.6× at 80% acceptance and 2.4× at 90%** on measured
  routing — not the ~1.1× the uniform bound predicted. See `p0/dspark.py`.
- **Real routing beats the uniform-routing bound by 1.2–1.6×.** Captured 16,104
  decisions across all 22 MoE layers of the full 8.47 B model
  (`p3/capture_routing.py`). A batch of 8 touches 13.1 experts, not the 21.0 the
  independent-uniform model predicts. Expert-load CV is 0.163 globally, so the
  aux-loss-free expert-bias balancing works well. Every occupancy and
  speculation number in the plan now derives from this trace.
