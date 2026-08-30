# P2 — Unit RTL and the automated PPA loop

**Weeks 12–30 · 4–5 RTL engineers.** Every unit gets a differential bench
against the C golden model and sits in an automated synthesis loop from its
first commit.

## Deliverables

| # | Deliverable | Artifact | Status |
|---|---|---|---|
| P2-1 | Hierarchical accumulator | `rtl/sonic_acc.sv` | **done, verified** |
| P2-2 | Dual-mode PE with banked accumulators | `rtl/sonic_pe.sv` | **done, verified** |
| P2-6 | MoE top-k router | `rtl/sonic_router.sv` | **done, verified vs the 8.47B model** |
| P2-3 | Differential bench vs. the golden model | `tb/tb_*.cpp` | **9 of 9 units** |
| P2-4 | Automated PPA loop | `ppa/loop.py` | done |
| P2-5a | 64x64 dual-mode systolic sub-tile | `rtl/sonic_tile.sv` | **done, verified** |
| P2-5b | Weight streamer + expert gather | `rtl/sonic_streamer.sv` | **done, verified** |
| P2-5c | Short-conv unit (k<=7, double-gated) | `rtl/sonic_conv.sv` | **done, verified — bug found** |
| P2-5d | Online-softmax attention accumulator | `rtl/sonic_softmax.sv` | **done, verified — two bugs found** |
| P2-5e | Streaming LM head + top-K | `rtl/sonic_lmhead.sv` | **done, verified** |
| P2-5f | Descriptor-ring sequencer | `rtl/sonic_seq.sv` | **done, verified — bug found** |

## Verification approach: Verilator + the C model in one process

`cocotb` has no cp314 wheel and its source build fails, so the benches link
`p0/golden/sonic_golden.c` directly into a Verilator C++ testbench. This is
better than the cocotb route anyway — the RTL and the reference see byte-identical
stimulus in the same process, with no Python marshalling and no separate model of
the model. Add cocotb later if a Python-side constrained-random layer is wanted.

`tb_acc.cpp` covers the failures that actually happen in this class of design:

- **every reduction length 1…300 at worst-case operands** — the K3 team proved a
  bound at K=64 that broke at K=32; sweep, never assume
- **the ragged tail** — a K that is not a multiple of the fold depth loses its
  partial local accumulator if the epilogue forgets it
- **double accumulation** — an enable spanning two pipeline phases counts each
  product twice; the bench drives idle cycles between products and proves the
  result is unchanged
- **`clr` isolation** — a partial local stage leaking into the next dot product

## Run

```
make p2             # lint, Verilator bench, Icarus bench, PPA loop
make p2-sweep       # accumulator width and fold-depth sweeps
make vectors        # export real tensors from the 8.47B checkpoint
make p2-router      # router RTL vs the model's own expert selections
make p2-pwl-sweep   # size the sigmoid PWL against routing agreement
make iv             # Icarus Verilog + VCD
make wave           # ...and open it in GTKWave
```

## Verifying against the model, not against random numbers

`tb_router.cpp` is the bench that matters. It loads real layer-5 hidden states,
router weights and expert-bias values from the 8.47 B checkpoint
(`p3/export_vectors.py`) together with the expert selections PyTorch made, and
requires the RTL to reproduce that decision. A bench built on random stimulus
proves the design is self-consistent; this one proves it computes the model.

```
tokens checked      512
top-1 expert match  1.0000
top-4 set match     0.9961   (gate >= 0.995)
exact order match   0.9902
```

Two simulators run the same RTL. Verilator drives the C++ differential benches;
Icarus Verilog (`make iv`) runs an independent SystemVerilog testbench and dumps
VCD for GTKWave. They have different front ends and different schedulers, so
agreement between them is evidence the design does not depend on one tool's
interpretation.

## Findings

**The router needs INT12, not INT4 and not BF16.** Measured on 512 real hidden
states, sweeping both operands:

| act \ weights | INT4 | INT8 | INT12 | INT16 |
|---|---|---|---|---|
| INT8 | 0.760 | 0.971 | 0.973 | 0.971 |
| INT12 | 0.760 | 0.986 | **0.998** | 0.996 |
| INT16 | 0.760 | 0.986 | 1.000 | 1.000 |

INT8 weights cap at 0.986 no matter how wide the activations get, and INT4
collapses to 0.76. The original recipe said BF16, which clears the gate but is
over-specified; INT12 x INT12 is the cheapest point that passes. **The floorplan
consequence is real: the router cannot consume the shared INT8 activation bus,
it needs a wider tap taken before the INT8 requantization.**

**The sigmoid PWL needs 64 segments, 4x the FFN's SiLU.** Sized the same way:

| Segments | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| top-4 agreement | 0.971 | 0.990 | **0.996** | 0.996 |

**Minimax-centre the PWL intercept, don't interpolate the endpoints.** A chord
through a convex segment sits entirely above the curve, which biased every score
by about +0.005 and cost ~3% of routing agreement. Centring the error band is a
different constant in the same table -- no extra hardware.

## Synthesis found four bugs simulation had accepted

Every unit passed Verilator lint and simulation before any of these surfaced.
Synthesizability is a separate property from correctness, and the gap is wide:

- **Unpacked array ports.** `input logic signed [31:0] bias [E]` simulates in
  both Verilator and Icarus and is rejected outright by Yosys. Every port is now
  a packed vector, so one set of sources covers simulation, synthesis and P&R.
- **Async reset folded together with a sync clear.** `if (!rst_n || clr)` inside
  `always_ff @(posedge clk or negedge rst_n)` makes the reset condition depend on
  a level signal absent from the sensitivity list. Simulators accept it; Yosys
  calls it ambiguous and stops. Three units had it.
- **A negated size cast.** `-DW'(expr)` parses in Yosys as a cast of a
  negative-sized expression.
- **A comment beginning with the simulator's own name** is parsed as a lint
  pragma. So are backticks inside comments.

## Operand width is a hardware decision, not a formatting one

SystemVerilog sizes a multiply from its operands, which makes width choices
silently expensive. Two cases, both caught by putting the units through the
PPA loop rather than by reading them:

- The router's epilogue used 64-bit operands for the scale multiply, asking for
  a **64x64 multiplier** -- ~4096 partial products. Yosys and abc simply never
  finished. Narrowed to 32x32 -> 64, it synthesises in minutes.
- `sonic_conv` widened its tap operands to `ACC_MID` before multiplying, asking
  for KMAX 24x24 multipliers per channel instead of 8x8: **688,016 cells** for
  64 channels. Multiplying at natural operand width and accumulating wide gives
  330,640 -- half the block, identical arithmetic.

**Two RTL bugs the model-driven bench caught that random stimulus would not
have.** Both produced a clean 0.0000 rather than a subtle error:
a Q16 scale port where `act_scale * weight_scale` is ~1e-6 and rounded to zero,
and a scale multiply sized 32-bit by SystemVerilog's operand rules that
overflowed before its shift. Then a third: shifting by 32 instead of 16 yielded
the *integer* logit where the PWL expected Q16, so every sigmoid returned 0.5.


**The overflow checker was 26% of the critical path.** Logic depth went 95 → 70
levels and cell count 1,564 → 1,274 when `SONIC_NO_OVF_CHECK` strips it. It is
essential in simulation and must not reach silicon; it is now `ifdef`-guarded.
Chasing f_max before finding this would have optimised debug logic.

**With the checker stripped, fold depth is not a frequency lever at all** —
`ACC_FOLD` of 4, 8, 16 and 32 all give 70 levels. The K3 write-up's 468 → 988 MHz
came from hierarchical accumulation at *their* operand widths; at INT4 × INT8 the
fold is free either way. Keep it at 16 for the accumulator-width bound, not for
timing.

**The 12 → 16 bit correctness fix costs 4 logic levels (66 → 70, ~6%).** That is
the real price of the P0 finding that a 12-bit local path is illegal for
INT4 × INT8. Cheap.

## Finding 16: `sonic_router` cannot be clocked, in any process

P4 sent block-level P&R through post-CTS timing repair and it reported
**WNS -1086.6 ns against a 10 ns clock** on the top-k `sel[]` path — an actual
critical-path delay of roughly 1,097 ns. That number is large enough to look
like a units error, so it was checked independently with Yosys, at the same
configuration P4 was running (`LANES=4`, `ROUTER_PWL_SEGS=8`):

| unit | combinational depth |
|---|---:|
| `sonic_acc` | 59 |
| **`sonic_router`** | **2,518** |

2,518 levels at roughly 0.43 ns per level on Sky130 is ~1,090 ns. It reproduces
the WNS almost exactly, from a completely separate tool. **The timing violation
is real.**

Three consequences, in order of how much they hurt:

1. **A faster process does not save it.** Sky130 at 130 nm to 14 nm is roughly
   10-15x per gate, so ~1,097 ns becomes ~75-110 ns. Against S1's 1 GHz target —
   a 1 ns period — the router is still 75-110x too slow. This is not a Sky130
   artefact and it is not something P&R can fix.
2. **The measured configuration is the small one.** `LANES=4` and `SEGS=8`. The
   shipping `LANES=64`, `SEGS=64` has a wider segment-select mux and a wider
   top-k network, so its path is *deeper* than 2,518, never shallower.
3. **Raising `CLOCK_PERIOD` to 1500 ns is the right call to unblock P&R and the
   wrong thing to record as a result.** It lets the flow reach routing, which is
   what P4 is for. It does not make the design meet timing, and the resulting
   layout must not be quoted as evidence that it does.

### The cause, and the fix: a linear scan that should have been a tree

The top-K block extracted each maximum with a **serial scan**:

```systemverilog
best = tmp[0];
for (int i = 1; i < E; i++)                    // 31 DEPENDENT compares
  if (tmp[i] > best) begin best = tmp[i]; bi = EW'(i); end
```

`best` carries from `i` to `i+1`, so one pass is E-1 = 31 dependent 32-bit
comparisons, and the K passes are serially dependent through `tmp[bi]`. The
comment defended this on area — "K*E comparators rather than a sorting network"
— and the area claim is true but irrelevant: **a tournament tree compares every
element exactly once too**, so it uses the same E-1 comparators while being
log2(E) deep instead of E-1 deep.

Measured, `LANES=4`, `SEGS=8`:

| top-K structure / flow | depth | cells |
|---|---:|---:|
| serial scan unpipelined (`-DROUTER_TOPK_SERIAL`) | 2,518 | 116,236 |
| tournament tree unpipelined (`-DROUTER_TOPK_COMB`) | 461 | 107,387 |
| **fully pipelined (S1 MAC, S2 acc, E1-E3 PWL, 2-cycle top-K)** | **114** | **67,467** |

**Depth reconciliation:** 2,518 was the unpipelined serial-scan cone; 461 was the
unpipelined tournament tree combinational cone. Pipelining the MAC tree,
epilogue scales, PWL evaluation, and splitting top-K across two cycles reduces
the longest critical path through the block to **114 levels** under `abc -g cmos2 -dff`
(67,467 cells, 1,864 DFFs).

**Why this was not caught sooner:** `sonic_router` was missing from `UNITS` in
`ppa/loop.py`. The one unit with a pathological critical path was the one unit
excluded from the loop built to measure critical paths. It is in the table now,
run via:

```
python3 p2/ppa/loop.py --unit sonic_router
```

## Finding 17: two bugs in `sonic_softmax`, both found by its first bench

The module's own comment named the bug it meant to avoid — "rescaling only one
of them is the classic online-softmax bug". It had that bug, and another.

**One `exp_val` was used for both factors.** `exp_arg` is cleverly built so that
exactly one of the two factors is always `exp(0) = 1`: on a new maximum the
lookup is the rescale and the incoming weight is 1; otherwise the lookup is the
weight and the rescale is 1. The update then used `exp_val` for *both*
(`l_q*exp_val + exp_val`), which loses the first score outright — at
`m = -inf` the rescale underflows to 0 and the new term is multiplied by that
same 0 — and thereafter scales the running sum by the incoming weight instead of
by the correction. Fixed by muxing `corr` and `wgt`, which costs one mux and no
extra PWL lookup.

**The Q16 multiply was truncating before its shift.** `l_q * corr` with two
`DW`-wide operands is a `DW`-wide multiply in SystemVerilog — self-determined
width — so at `DW = 32` the product overflows and truncates *before* the
`>>> 16` meant to rescale it. `1.0 * 1.0` evaluated to `0.0`, pinning the
running sum at 1.0 forever. This is the same family as findings 12-15: sizing a
multiply off its operand width rather than its result width. Fixed with an
explicit double-width intermediate; the multiply is 32x32 -> 64, the same size
finding 12 settled on for the router epilogue.

Both bugs were in RTL that linted clean, synthesized, and had been reported as
"done". `tb_softmax.cpp` failed all four score patterns — flat, rising,
falling, mixed — against `p0/golden/sonic_golden.c` before the fix and passes
all of them after.

## Finding 18: `sonic_pe` is correct, and the worst-case bound now holds in RTL

39 checks, no failures. Bank isolation under interleaved access, the
`MODE_PREFILL` / `MODE_DECODE` weight select including a mid-reduction switch,
and ungated one-cycle systolic pass-through all hold.

The bench also drives the pattern the accumulator widths were sized against —
every product at maximum magnitude and identical sign, to the full `D = 2048`
reduction depth — and confirms no stage overflows. P0-5 proved that bound in the
C model; it is now proved in the RTL as well.

One design question is pinned rather than answered: `ovf` is `|bank_ovf`, so an
overflow in any bank raises it on every read. Nothing here overflows, so the
bench records the scope rather than exercising it.

## Finding 19: a ternary made a signed multiply unsigned

`sonic_conv` failed 241 of 554 checks on its first bench. Waveform tracing named
it in one pass:

```
prod0 = 46    = 23 * 2      correct
prod1 = 2540  = 10 * 254    WRONG -- should be 10 * (-2) = -20
```

**254 is `(uint8)(-2)`.** The tap was being multiplied as unsigned. The cause is
visible in why `prod[0]` escaped:

```systemverilog
prod[0] = xv[c] * tap[0];                                    // no ternary -- correct
prod[t] = (t < int'(k_taps)) ? hist[c][t-1] * tap[t] : '0;   // ternary -- unsigned
```

`'0` is an **unsigned** unsized literal, and SystemVerilog makes an entire
expression unsigned if any operand is. So the conditional silently evaluated a
signed x signed multiply as unsigned. Both operands are declared `signed`; the
declaration is not what decides. Fixed with an `if/else`, which assigns each
branch independently and never unifies operand types. 554/554 pass.

This is the same family as findings 12-15 -- an expression rule quietly
changing the hardware -- but about **signedness** rather than width, and it is
nastier: a width mistake shows up as an implausible cell count, while this one
synthesized smaller and simulated cleanly. It was only visible because a bench
compared against a reference.

**It also corrects finding 12.** Signed multiplication needs sign-extension
logic that unsigned does not, so the fix costs cells:

| `sonic_conv`, CH=8 | cells |
|---|---:|
| with the unsigned ternary | 77,687 |
| correct, signed | 87,652 (+12.8%) |

Finding 12's 330,640-cell figure was measured on arithmetic that was wrong. The
correct block is roughly 373,000 at CH=64. Still half the 688,016 that the
operand-widening bug cost, so the finding's conclusion holds -- but the number
should be restated.

Separately pinned by the same bench: `y_out` is combinational from `x_in` while
`out_vld` is registered, so when `out_vld` is high `y_out` already reflects the
*next* sample. Consumers must sample `y_out` with `in_vld`.

## Finding 20: the SiLU table spans the wrong range

`make p3-layer` runs a whole MoE layer -- route, gather, gate_up GEMM, SiLU
gate, down GEMM, combine -- on the real `sonic_tile` against the hidden state
PyTorch produces from the same weights. It converged only after the activation
table was fixed:

| SiLU table | layer cosine | rel L1 |
|---|---:|---:|
| golden, 16 seg over ±8 | 0.9632 | 0.399 |
| refit, 16 seg over ±8 | 0.9822 | 0.265 |
| refit, 16 seg over ±2 | 0.9911 | 0.177 |
| **refit, 16 seg over ±1** | **0.9912** | **0.176** |
| refit, **8** seg over ±1 | 0.9911 | 0.177 |
| refit, 16 seg over ±0.5 | 0.9837 | 0.185 |
| exact SiLU | 0.9911 | 0.175 |

**The range is the lever, not the segment count.** `gate_up` activations have
mean magnitude 0.105, so a table spanning [-8, 8) puts every one of them inside
the two segments straddling zero — exactly where SiLU curves hardest. Narrowing
to ±1 reaches the accuracy of exact SiLU, and at ±1 **eight segments match
sixteen**: the table can shrink, not grow. ±0.5 is worse again, because it
starts clipping.

This is the mirror of the router finding. There the sigmoid needed *more*
segments (64 against the SiLU's 16); here the SiLU needs a *narrower range*.
Same root cause both times — PWL resolution has to match the data's dynamic
range — and neither is visible without running real activations through it.

Separately, at the same ±8 range, refitting improved 0.9632 to 0.9822 on its own.
That gap is the minimax-centring this repo already records as a finding for the
router; `sonic_pwl_fit_silu` does not appear to get the same treatment.

Both changes are free: `quant.SILU` marks the coefficients firmware-loadable.

## Finding 21: the streamer's scale/group_done contract

54 checks, no failures. Pass-through latency, hold-through-stall, the beat
counter refusing to advance on a stall, `group_done` landing on a group's last
beat, and expert latching independent of the weight stream all hold.

One behaviour is pinned rather than judged. A scale loaded in the **same cycle**
as a group's final beat lands on that group's `group_done`, so `scale_out`
carries the *next* group's scale:

```
scale loaded on the final beat -> scale_out = 1792 at group_done
   (256 is this group's scale, 1792 is the next one's)
```

Whether that is a bug depends on the producer's contract, which does not exist
yet, so the bench records it instead of asserting either way. It matters because
scales are applied to the **accumulated partial sum** — one misaligned scale
mis-scales all 64 weights in a group at once, and the result stays plausible.
Whoever writes the descriptor-ring sequencer (P3-3) has to guarantee the scale
is loaded at least one cycle before the final beat.

Also worth reconciling: the port comment says the scale is "FP16-as-Q16", but the
reset value is `16'sh0100` = 256, which is 1.0 only in Q8.8. A 16-bit Q16 cannot
represent 1.0 at all. One of the two is wrong.

## Finding 22: the sequencer issued every descriptor one slot late

`sonic_seq` was the last unbenched unit and the one `ppa/loop.py` files as
"control, not datapath; f_max here is irrelevant" — which is precisely why the
thing it needed was a bench, not a timing number.

`desc` was driven combinationally from the program counter while `desc_vld` was
registered:

```systemverilog
desc_vld <= 1'b1;          // registered
pc       <= pc + 1'b1;     // registered
...
assign desc = ring[pc];    // combinational -- reads the NEW pc
```

Both non-blocking assignments land on the same edge, so on the cycle `desc_vld`
goes high `pc` has already advanced. Measured by `tb_seq.cpp`:

| valid cycle | emitted | should have emitted |
|---|---|---|
| 0 | `ring[1]` | `ring[0]` |
| 1 | `ring[2]` | `ring[1]` |
| … | | |
| 7 (`done`) | `ring[8]` — **past the end** | `ring[7]` |

Three distinct failures out of one line. Descriptor 0 is never issued. Every
descriptor runs one slot ahead of the valid that announces it. And the last
cycle reads `ring[len]`, emitting whatever firmware happened to leave in the
next ring slot — the bench poisons that slot with `0xDEADBEEF…` and catches it
directly.

**Nothing downstream can detect this.** Every descriptor is well-formed. The
fetch it issues is a correct fetch of the wrong tensor, in a chip whose whole
premise is that one static schedule replays per token. It is the same failure
class the streamer bench pinned a contract for one block over: right data,
wrong cycle.

The fix registers `desc` alongside `desc_vld`, reading `ring[pc]` before the
increment. Same cycle count, same latency, no extra state beyond the descriptor
register itself. `tb_seq.cpp` also caught that `pc == len - 1` with `len == 0`
wraps to `DEPTH-1` and replays all 256 ring slots; a zero-length program is now
an ignored start.

64 checks, 0 failures. P2-3 is 9 of 9 units.

## Finding 23: the router's sigmoid table spanned the wrong range too

Finding 20 established, on the FFN's SiLU, that the **input range** of a PWL
table is the lever and the segment count is not. That lesson was never applied
to the router's sigmoid, which had been sized on the segment axis alone: 16
segments give 0.971 routing agreement, 32 give 0.990, 64 give 0.996, so 64 it
was — 4,096 flip-flops of firmware-writable table behind a 64-way 32-bit mux,
and the structure P4 could not get through ABC in either of two attempts (2 h
and ~11 h of CPU).

`make p2-pwl-sweep` now sweeps both axes against the same 512 real hidden
states. Top-4 set agreement, gate ≥ 0.995:

| range \ segs | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|
| ±8 | 0.9102 | 0.9707 | 0.9902 | **0.9961** |
| **±4** | 0.9707 | 0.9902 | **0.9961** | 0.9961 |
| ±2 | 0.9590 | 0.9844 | 0.9902 | 0.9941 |

**Halving the range is worth exactly one doubling of the segment count** — the
±4 row is the ±8 row shifted one column left. So 32 segments over ±4 buys the
same 0.9961 as 64 over ±8, with half the table flops and half the read mux, and
`make p2-router` reproduces top-1 1.0000 / top-4 0.9961 / exact 0.9902 bit for
bit at the new recipe. That is the default now.

±2 is where it stops paying. The logits genuinely reach past 2, so a ±2 table
clips rather than resolving, and every column gets worse. **Range is a lever
down to the actual dynamic range and no further** — which is why this has to be
measured on real activations and cannot be reasoned about from the curve.

Two things this makes concrete. First, the same mistake was made twice in this
chip, on two different tables, and both times the symptom was "we need more
segments". Second, the cost was not abstract: it is the difference between a
synthesis run that converges and two that did not.

## Finding 24: Pipelined tree reductions across `sonic_tile`, `sonic_router`, and `sonic_conv`

Three blocks previously suffered from serial accumulators in combinational cones:
- **`sonic_tile`**: 64 dependent 32-bit ripple adders per lane, missing from synthesis.
  Rewritten as an elaboration-checked 3-stage hierarchical adder tree (S1 fold at
  `ACC_LOCAL=16`, S2 fold-sum at `ACC_MID=24`, S3 `ACC_OUT=32` bank accumulate).
  Measured at `TILE=8`: **depth 79 levels, 79,339 cells, 2,568 DFFs**.
- **`sonic_conv`**: Tap sum chain converted to balanced reduction tree.
  Measured: **depth 102 → 92 levels, cells 43,291 → 42,816**.
- **`sonic_router`**: S1 MAC converted to balanced operand-width tree.
  Measured: **depth down to 114 levels, cells 67,467**.

All three rewrites are **100% numerically transparent** (`make golden`, `make test`,
`make p2-router` (top-4 0.9961), `make p3-dense` (0.99363), `make p3-layer` (0.99119),
and `make p3-top` (0.99119)).

## Finding 25: `sonic_seq`'s ring wasn't the reduction-tree bug it looked like, and `sonic_softmax`'s 132 levels isn't a tree at all

A post-router-run pass over PPA and power/size/speed follow-ups surfaced two
mismatches between what the numbers looked like and what the RTL actually is.

- **`sonic_softmax` has no reduction chain to rebalance.** Unlike `sonic_acc`,
  `sonic_conv` and `sonic_router` (Finding 24), it is a single-sample-per-cycle
  online recurrence: one `mulq16` (32x32→64 Q16 multiply) feeding one
  accumulate, once per cycle. Its 132 levels is the combinational depth of
  that one multiply-accumulate, gated by the `score > m_q` comparator ahead of
  it -- not a serial fold across N lanes. The real lever is pipelining the MAC
  itself, which is a materially riskier change here: `m_q`/`corr`/`wgt` feed
  back into the *same* cycle's computation, so naively registering the
  multiply breaks the online-softmax recurrence (the same family of bug as
  Finding 17) and needs a deliberate redesign plus a new differential check,
  not a mechanical rewrite. Left undone, on purpose, pending that.
- **`sonic_seq`'s outsized cell count was a memory-inference gap, not a
  clock-gating gap.** `p2/ppa/out/ppa.json`'s `area` field is actually a sum
  of `$_DFF` cell instances (mislabeled) -- 32,849 for `sonic_seq`, which is
  32,768 (the 512x64 descriptor ring) plus ~80 control/FSM bits. The ring was
  declared as a plain array with a **combinational** read (`ring[pc]`) feeding
  a mux before ever reaching a register -- a pattern most memory-inference and
  SRAM-macro-mapping passes don't recognize, so it flattened to a flip-flop
  array plus a large address-decode/read-mux instead of mapping to an SRAM.
  Fixed by registering the read directly (`raw_q <= ring[pc_next]`, with
  `pc_next` computed one cycle ahead to land on the exact same cycle the old
  combinational read did -- a synthesis-mapping change only, verified
  bit-identical against `tb_seq.cpp`, 367/367 checks, and `make test` 40/40).
  This doesn't move the needle in `p2/ppa`'s generic `abc -g cmos2` flow --
  that flow has no SRAM macro library at all, so it flattens any array to
  flops regardless of RTL style -- the payoff is at a real P4 Sky130 run,
  which now has a config (`p4/openlane/seq/config.json`, `make p4-seq`) but
  hasn't been executed yet.
- **SRAM bank power-gating was fully unwired, not partially extended.**
  `sonic_sram_bank.sv` has always supported per-bank retention sleep, but
  `sonic_top.sv` tied `sonic_sram`'s `bank_sleep` input to `'0` -- every one of
  the 16 512 KB banks stayed awake unconditionally. No code in this repo
  implements or measures any existing bank-level gating. Fixed with a new
  `sonic_sram_gate` (per-bank idle counter, sleep after 256 idle cycles,
  `bank_ce` forces sleep low combinationally the same cycle so a real access
  is never dropped -- see its own header comment for the safety argument),
  wired into `sonic_top` in place of the tie-off. Verified in isolation
  (`tb_sram_gate.cpp`, 261/261 checks: no premature sleep, sleep exactly at
  the idle threshold, glitch-free wake, bank independence) and functionally
  unchanged end-to-end (`make p3-top`).

## Caveats

- Depth and cell counts come from `abc -g cmos2` against a **generic** library
  with no timing. They are relative signals for the loop, not f_max. Absolute
  numbers need the foundry liberty at P4.
- `loop.py --sweep NAME=0,1` passes `-DNAME=0` and `-DNAME=1`, both of which
  *define* the macro. Boolean defines must be compared against a separate
  undefined baseline run.
- Every unit now has a bench. `sonic_conv`, `sonic_softmax` and `sonic_seq` each
  shipped a real bug that survived synthesis, lint and code review, and
  `sonic_router` shipped a critical path P&R could not close; findings 16-23 are
  what "synthesizes" is worth without a bench.

## macOS toolchain note

Homebrew's `binutils` puts **GNU `ar` ahead of Apple's on the PATH**, and GNU
`ar` produces archives that macOS `ld` rejects with
`archive member '/' not a mach-o file`. Verilator builds fail at link. The
Makefile passes `AR=/usr/bin/ar` to work around it.
