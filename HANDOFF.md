# Sonic S1 — handoff

Written 2026-08-29. Covers what changed in the remediation pass, what the
numbers actually say now, and what is left to get an end-to-end chip.

**Ground rule used throughout.** A gate is closed only if it was run, produced a
number, and that number is written down with its uncertainty. A mechanism that
exists but has never been run is *built, unmeasured*. A number whose confidence
interval spans the thing it is supposed to prove is *not a measurement*.

Tree state at handoff: `make test` 40/40 · `make golden` 0 failures ·
`make p2` exit 0, all nine units · `make p3-dense` cosine 0.99363 ·
`make p3-layer` cosine 0.99119 · `make p2-router` top-4 0.9961.

---

## Part 1 — What was done

### 1.1 The GPTQ gate was not unrun. It was broken, and now it is measured.

`p0/out/gates.json` was already a `"pack": "gptq"` run. It reported
`ppl_quant` **NaN**, `ppl_delta` **NaN**, `top1_agreement` **0.0**. That had been
recorded as "GPTQ built, never run", which is a materially different claim: a
NaN gate is a broken implementation, and in the report it is indistinguishable
from "the format is bad."

Three defects, all fixed in `p0/packer.py` and `p0/gates.py`:

| # | Defect | Fix |
|---|---|---|
| 1 | `q_group_gptq` applied GPTQ's update `err * Hinv[j, j+1:]` to a raw `torch.linalg.inv(H)`. That formula comes from the **Cholesky factor of H⁻¹**, not from H⁻¹. The per-group step then re-inverted a sub-block of the inverse, which is unstable by construction. | Standard factorisation: `cholesky` → `cholesky_inverse` → upper `cholesky`. Escalating damping, explicit dead-column handling, division by the Cholesky diagonal. |
| 2 | No guard. One non-finite weight in a fused expert stack takes the whole model's perplexity to NaN. | `q_group_gptq` degrades to RTN for a diverging tensor and says so. `quantize_` refuses to report a gate at all if any packed tensor is non-finite, naming the tensor. |
| 3 | Calibration held Hessians for every module at once — 22 MoE layers × `[32, 2048, 2048]` float32 ≈ **21 GB**, on top of a 17 GB model. Measured 14 GB of swap; calibration was the longest phase of the run. | `HessianStore`: file-backed accumulation, read back one tensor at a time. Page-cache pages are dropped under pressure; anonymous pages must be written to swap first. |

Also fixed: a device mismatch (model on MPS, Hessians on CPU) — the whole
factorisation is now pinned to CPU float32; and `expert_hook` computed its Gram
matrix in BF16 *before* casting to float, losing exactly the small eigenvalues
the inverse is most sensitive to.

**Validation of the fix, independent of the 8B run.** On LFM2.5-350M at
identical settings, `ppl_delta` **223.16 → 78.21** and `top1_agreement`
0.7211 → 0.7618, purely from the linear algebra. On synthetic layers the
corrected version reduces *output* error against RTN monotonically (3%
well-conditioned, 73% at expert-scale token counts, 96% rank-1) — the
qualitative signature GPTQ must have and the old one did not.

**Result on the 8B, and it is not the good news the plan wanted.** Both runs are
WikiText-2, 16,376 tokens, identical windows, current `sonic/quant.py` recipe,
4.700 active bits:

| pack | `ppl_delta` | 95% CI | `top1@p>0.9` | vs 0.15 gate |
|---|---:|---:|---:|---|
| RTN | +2.091 | `[-0.799, +4.874]` | 0.9922 | ✗ |
| **GPTQ** | **+3.113** | `[+1.713, +4.551]` | **0.9941** | ✗ |

Zero divergence warnings — GPTQ ran cleanly across all 7.75 B expert
parameters; 21.2 GB of Hessians captured in 223.6 s.

Read this carefully, because it is easy to overstate in either direction:

- **Neither passes.** Both miss the 0.15 gate by more than an order of
  magnitude. Correct GPTQ does not close it.
- **GPTQ is *not* demonstrably worse than RTN on perplexity.** The point
  estimate is higher, but the CIs overlap heavily and RTN's own CI spans zero.
  At 16 K tokens these two are not separated. Do not quote "+3.11 vs +2.09" as a
  ranking.
- **GPTQ is better on the metric the gate text is about.** Stratified top-1
  agreement where BF16 is confident: 0.9941 vs 0.9922, both over the 0.99 bar.

An AWQ run at the same 16,376 tokens was started and killed before finishing;
there is no AWQ point at this token count yet.

### 1.2 Three serial reductions, all the same defect, all fixed

`sonic_tile` reduced `T` products per lane as `partial += ACC_OUT'(av) * ACC_OUT'(wv)`
in a loop — **64 dependent 32-bit ripple adders in one combinational cone**,
feeding one flop. This is the router's Finding-16 defect in a different module,
in the block holding 4,096 of the chip's 16,384 multipliers. It had **never been
synthesised**: `sonic_tile` was absent from `UNITS` in `p2/ppa/loop.py`, exactly
as `sonic_router` had been.

It also contradicted the P2-7c decision already recorded as closed: it carried
32 bits through a reduction whose measured INT4 peak is 3,821 (13 bits).

Rewritten as the hierarchy `sonic_acc` already proves temporally, applied
spatially, plus a 3-stage pipeline:

```
S1  T multiplies → ACC_FOLD(16)-wide balanced trees at ACC_LOCAL(16)   depth 4, not 64
S2  the NFOLD(4) fold results summed at ACC_MID(24)                    depth 2
S3  accumulate into the selected ACC_OUT(32) bank
```

`clr` rides the same pipeline as the data, so clear/accumulate ordering survives
any future depth change. Bounds that are structural (fold, mid) are now
**elaboration-time `$error`s costing zero gates**; the only genuinely
data-dependent overflow — the ACC_OUT bank — gets a two-gate signed-overflow
detector.

Same defect found and fixed in two more places:

- **`sonic_router` S1 MAC** — `dot += ACC_OUT'(av) * ACC_OUT'(wv)` over `LANES`.
  At the `LANES=4` used for the P4 experiments this cost 4 levels and hid; at the
  shipping `LANES=64` it is 64, sitting directly ahead of the top-k network.
  Finding 16 named "the score/PWL datapath ahead of them" as the remaining
  depth — this is that datapath. Now a tree at `AW+WW+clog2(LANES)` bits rather
  than 32.
- **`sonic_conv` tap sum** — `KMAX`-deep chain at ACC_MID. Small, but the same
  defect. **Measured: depth 102 → 92, cells 43,291 → 42,816.** Better on both axes.

**All three are numerically transparent.** `make p2-router` reproduces
1.0000 / 0.9961 / 0.9902 exactly; `make p3-dense` 0.99363 and `make p3-layer`
0.99119 exactly. The rewrites changed structure, not arithmetic.

`tb_tile.cpp` went from 3 tests to 6, adding: pipeline latency pinned via
`out_vld`, clr/accumulate ordering under back-to-back issue, and the worst-case
INT4×INT8 fold bound driven at its actual limit (a = −128, w = −8) rather than
trusted from a comment.

### 1.3 What the finished GDS already said, that nobody had read

The GDS at `~/Downloads/root/sonic/p4/openlane/router/runs/local/final` is
**byte-identical** to the vendored `p4/openlane/router/results/segs8/metrics.json`.
No new run. But `CLOCK_PERIOD` is known and worst slack is reported per corner,
so the critical path was already measured:

| Corner | Worst setup slack | Longest path |
|---|---:|---:|
| `ff_n40C_1v95` | 1121.1 ns | 378.9 ns |
| `tt_025C_1v80` | 1074.4 ns | 425.6 ns |
| `ss_100C_1v60` | **933.0 ns** | **567.0 ns** |

≈ **1.76 MHz** on Sky130 HD at SEGS=8 — and that is with the resizer given
1500 ns of slack and therefore no reason to try, so it is an upper bound.

Consequences applied: `config-next.json` now carries a **derived**
`CLOCK_PERIOD = 800` (not 1500), and this is real Liberty-backed STA on one
unit, which partly answers P2-8c.

### 1.4 Corrections to claims that were wrong

| Where | Was | Now |
|---|---|---|
| `p4/RESULTS.md` | "Sky130 has no SRAM macro to map it to" — used to defer P2-12 indefinitely | Sky130 has no memory *compiler*; it does have pre-hardened OpenRAM macros (`sky130_sram_2kbyte_1rw1r_32x512_8` and relatives) that LibreLane consumes via `MACROS`. **P2-12 is work, not a blocker.** Flagged as unverified on the build box. |
| `ROADMAP.md` P0-1b | "GPTQ built, never run" | It was run and returned NaN. See 1.1. |
| `ROADMAP.md` P2 | A truncated sentence claiming both "99 levels" and "the instrument failed" | Repaired; the unreconciled 461-vs-99 discrepancy is now stated as unreconciled rather than hidden. |
| `p0/README.md` P0-6 | "4-question smoke test (25% drop)" | Stale — the artifact says n=25, 12.0%. **And that number is unusable, see below.** |

### 1.5 `bench_drop` measures nothing, and now says so

BF16 scores **28.0%** on the 25-question 4-way suite. Chance is 25%.

| | accuracy | 95% Wilson CI |
|---|---:|---|
| chance | 25.0% | `[11.5, 43.4]` |
| BF16 | 28.0% | `[14.3, 47.6]` |
| RTN | 16.0% | `[6.4, 34.7]` |

The BF16 anchor is inside the chance band, and the two intervals overlap almost
entirely. **The "12.0% drop" is four questions out of 25 — noise around a
reference that is itself guessing.** A drop cannot be measured from a baseline
that is not measuring anything.

`p0/bench_drop.py` now reports Wilson intervals, refuses to let the number pass
without a warning when the baseline is inside the chance band, and gained
`--chat` — the model is instruction-tuned and scoring it as a raw completion is
the likely cause. **This is a fix to the instrument, not to the result; the
result still needs re-running.**

### 1.6 Infrastructure

- `TILE` guarded with `` `ifndef `` in `sonic_defs.svh` so the PPA loop can sweep it.
- `sonic_tile` added to `p2/ppa/loop.py` `UNITS` (in `SLOW`, `TILE=8`).
- `HessianStore` / `ModuleStats` in `p0/packer.py`; `pack()` is duck-typed on
  `.get` so a lazy stats object substitutes for the old dict.

---

## Part 2 — Attempted and NOT landed

Three jobs were running when the session was stopped. All three were killed.
**None produced a usable number.** They are listed here as work to redo, not as
results to collect.

| Job | Status | Notes |
|---|---|---|
| AWQ gate, 16,376 tokens | **killed mid-run** | would have been the third point on the RTN/GPTQ comparison. `p0/out/gates_awq16k.json` does not exist. |
| `sonic_tile` depth, old vs new | **harness failed** | Yosys exited non-zero (rc=1) on both variants; no depth extracted. |
| `sonic_router` depth, old vs new | **timed out / failed** | `old LANES=16` ran 2,141 s without completing; `new` exited non-zero. |

**So the depth improvements from the three tree rewrites are argued, not
measured.** What *is* measured is:

- `sonic_conv`: depth **102 → 92**, cells **43,291 → 42,816**, from `make p2`,
  which runs the PPA loop as part of the normal regression. This is the one
  tree rewrite with a before/after number.
- All three rewrites are **numerically identical** to what they replaced —
  `p2-router` 1.0000/0.9961/0.9902, `p3-dense` 0.99363, `p3-layer` 0.99119, and
  `tb_tile` 6/6. Correctness is established; the PPA claim is not.

Redo the measurement with the loop rather than an ad-hoc script:

```
python3 p2/ppa/loop.py --unit sonic_tile     # TILE=8, in SLOW
python3 p2/ppa/loop.py --unit sonic_router   # SEGS=8
```

To compare against the pre-rewrite versions, reconstruct them by reverting the
tree block in each file; `git diff` on this branch isolates them cleanly.

**Two traps, recorded so they are not rediscovered:**

1. **Yosys elaborates a module at its *default* parameters before
   `hierarchy -chparam` applies.** Measuring `sonic_tile` at T=8 via `-chparam`
   pays for a full T=64 elaboration first, and the reported time is mostly that.
   Use `read_verilog -DTILE=8` — 0.75 s versus minutes. This is why `TILE` was
   given an `` `ifndef `` guard.
2. **`sonic_router` at `LANES=16` did not converge in 35 minutes** under
   `abc -g cmos2 -dff`, at `SEGS=8`. The existing 99-level figure in `ROADMAP.md`
   was produced at the default `LANES=64`, so either that run used a different
   flow or this one has regressed. **Resolve this before trusting either
   number** — it is the same "the instrument, not the design" ambiguity that
   Finding 16 already ran into once.

## Part 3 — What is left, with acceptance criteria

Ordered by dependency. Each item states what "done" means as something you can
run and check, not as a description of activity.

### Tier 1 — decide the format. Everything downstream waits on this.

**T1.1 — Resolve `ppl_delta` at a token count where the answer is stable.**
RTN's CI at 16 K tokens is `[-0.799, +4.874]` — it spans zero. That is not a
measurement, and it is why RTN and GPTQ cannot currently be ranked.
- *Do:* run `--pack rtn`, `--pack gptq`, `--pack awq` at `--max-tokens 65536`,
  identical windows.
- *Accept:* all three CIs exclude zero and do not mutually overlap, or the
  overlap is stated as the finding. Recorded in `p0/README.md` with intervals.

**T1.2 — Repair `bench_drop` so the gate is measurable at all.**
- *Do:* re-run with `--chat`; if BF16 is still inside the chance band, the suite
  or the scoring is wrong, not the model. Enlarge to n ≥ 200 real items (ARC /
  MMLU via `datasets`, not hand-written questions).
- *Accept:* `baseline_usable: true` in `p0/out/bench_drop.json`, BF16 accuracy
  above the chance band by more than its CI half-width, and n large enough that
  the 1.5-point gate is inside the resolution of the measurement. **At n=25 a
  1.5-point gate is unresolvable in principle** — 1.5 points is 0.4 questions.

**T1.3 — Decide what happens now that packer work cannot close `ppl_delta`.**
The remaining headroom is 0.05 bits (4.700 of 4.75). GPTQ was the plan's answer
and it does not close a 20× gap.
- *Do:* choose explicitly — widen the format and re-derive traffic and the SKU
  ladder; drop a SKU tier; or renegotiate the gate with a stated rationale.
- *Accept:* the choice is written down in `p0/README.md` with the number that
  forced it, and `sonic/quant.py` + `tests/test_spec.py` reflect it.

### Tier 2 — make the timing claims real

**T2.1 — Actually measure the three tree rewrites.** See Part 2: the attempt
failed and the depth claim is currently unsupported by anything except
`sonic_conv`'s 102 → 92.
- *Accept:* `p2/ppa/out/ppa.json` contains `sonic_tile`; `p2/README.md` gains the
  finding with old-vs-new depth and cells in the style of Finding 16; and the
  `sonic_router` non-convergence at `LANES=16` is either reproduced and
  explained, or shown to be an artefact of the ad-hoc script.

**T2.2 — Re-run P4 with `config-next.json`.** *(Your box.)*
- *Accept:* completes with `CLOCK_PERIOD = 800`; the post-CTS resizer engages and
  reports rather than exiting immediately; worst slack recorded per corner. If it
  closes, tighten and repeat until it does not.

**T2.3 — Resolve the electrical-limit violations the clean headline hides.**
42,310 max-slew, 3,105 max-fanout (corner-invariant, therefore structural), 208
max-cap. The clock net has 1,586 terminals.
- *Accept:* all three at zero on the worst corner, or each remaining one
  attributed to a named net with a stated reason.

**T2.4 — Reconcile 461 vs 99 levels for the router**, or retire one of them.
- *Accept:* one number, one method, stated; `p2/README.md` Finding 16 updated.
- *Partial progress:* re-ran `p2/ppa/loop.py --unit sonic_router` at the
  `REQUIRED_DEFINES` default (`ROUTER_LANES=4`, `ROUTER_PWL_SEGS=8`) and
  reproduced **114 levels, 67,467 cells, 1,864 DFFs** exactly — this is a
  real, reproducible number, matching what a since-superseded summary had
  claimed. It is a third data point, not a reconciliation: 461 and 99 were
  both reportedly at `LANES=64` (combinational vs. `-dff` pipelined), and
  this run is at `LANES=4`. A same-LANES comparison still needs a `LANES=64`
  run, which is exactly the configuration HANDOFF Part 2 already documented
  as not converging in 35 minutes at `LANES=16` -- `LANES=64` is presumed
  worse and was not attempted in this environment. Still open.

### Tier 3 — the chip that has not been written

This is the actual gap to "end to end". Everything above is remediation.

**T3.1 — `sonic_top.sv`.** There is no top level. `sonic_tile` is one 64×64
sub-tile; nothing instantiates `N_TILES` of them with the router, streamer, acc,
softmax, lmhead and seq. `make p3-dense` and `p3-ring` drive units individually.
- *Do:* assemble the datapath, driven by `sonic_seq` from the producer ring
  (already validated at 301 descriptors).
- *Accept:* one full transformer layer through the assembled top, against
  `sonic_golden`, at **cosine ≥ 0.99** — matching what the units already achieve
  individually (0.99363 dense, 0.99119 MoE). A new `make p3-top` target, green.
- *This is the first milestone that earns the phrase "end to end."*

**T3.2 — Multi-layer, then multi-token, through `sonic_top`.**
- *Accept:* ≥ 4 consecutive layers and ≥ 8 decode tokens, cosine ≥ 0.99 at every
  layer boundary, KV state carried in RTL rather than reloaded by the bench.

**T3.3 — SRAM as hard macros (P2-12).** Every array is flops. The router's PWL
table alone is 2,048 flip-flops behind a 32-way mux, and that structure is what
made ABC non-convergent. `sonic_tile`'s weight plane is 64×64×4 = 16,384 flops
per tile.
- *Do:* confirm the `sky130_sram_*` macro set on the build box; instantiate for
  the PWL table first (smallest, best understood).
- *Accept:* the router's flop count drops by ~2,048; the design still passes
  `make p2-router` at 0.9961; LibreLane completes with macro placement and a PDN
  over the macro, DRC/LVS clean.

**T3.4 — NoC and RV32 sequencer.** The two largest unwritten blocks. `sonic_seq`
is a descriptor ring, not a core.
- *Sequence:* after T3.1, which is what defines the interfaces they must carry.
- *Accept:* each has a bench in `make p2` with the same standard as the other
  nine units — differential against a reference model, not self-consistency.

**T3.5 — Prefill scheduler and decode/prefill mode-switch firmware.** Not started.

**T3.6 — Programmable vector unit (P2-9).** In the plan's unit list, never written.

### Tier 4 — physical design

**T4.1 — A second block through the flow**, ideally `sonic_tile`. It is the
timing-critical block and now the only large one whose depth is understood.
- *Accept:* completes; reports what the clock needs at `TILE=8` and `TILE=16`;
  answers whether the flow scales past 225 k cells.
- **[x] Done at TILE=8**, real LibreLane run on Sky130
  (`p4/openlane/tile/results/tile8/metrics.json`, run
  `RUN_2026-08-29_13-05-07`, 80/80 steps, `final/` present). First attempt
  at DIE 600×650 failed global placement at 108% utilization (GPL-0301);
  fixed by enlarging to 700×700; that got past placement but failed
  *detailed* placement post-CTS (DPL-0036) because 80% initial utilization
  left no room for the hold/fanout resizer's inserted buffers to legalize
  into — the same lesson the router's own config already encoded
  (`"40% utilisation leaves the router room"`) and that hadn't been applied
  here yet. Fixed by matching the router's margin: 1000×1000 die,
  `PL_TARGET_DENSITY=0.45`.

  Result at `CLOCK_PERIOD=1000` (1 MHz, generous — chosen to unblock the
  flow, not to model a target frequency, same posture as the router's early
  runs): **WNS/TNS = 0 across all Sky130 corners (tt/ss/ff)** — the design
  meets that clock with margin. But exactly the router's pattern repeats:
  the clean timing headline hides real electrical-limit violations —
  **7,638 max-slew, 96 max-fanout, 48 max-cap** violations (worst corner
  across all named). 51,199 std cells, 25.4% utilization, 0.0042 W total
  power at this reduced TILE=8 scale (not shipping-scale, not 14nm — Sky130
  generic power at 1/8 array edge). `p4/sta/sta_check.py --metrics
  p4/openlane/tile/results/tile8/metrics.json` reports all of this; T4.1's
  "answers whether the flow scales past 225k cells" is not yet answered —
  this run has 51,199 std cells (+ fill/tap), well under the router's 225k,
  so scaling toward `TILE=16`/`TILE=64` is still open.

- **[x] A third block done: `sonic_sram_bank` at ADDR_WIDTH=8** (real
  LibreLane run, run `RUN_2026-08-29_15-10-28`, `final/` present, copied to
  `p4/openlane/sram/results/sram8/metrics.json`). Also took two real
  failures to land: DIE 600×650 failed global placement at 108% utilization
  (GPL-0301, same class of mistake as the tile run above — floorplan sized
  before the real cell count was known); DIE 700×700 at 0.72 target density
  passed global placement but failed detailed placement post-CTS (DPL-0036),
  again from insufficient headroom for the resizer's buffer insertion. Fixed
  by matching the router's own margin: 1000×1000 die, `PL_TARGET_DENSITY=0.45`.

  Result at `CLOCK_PERIOD=1000`: **WNS/TNS = 0**, same clean-headline
  pattern, same real violations underneath — **7,535 max-slew, 148
  max-fanout, 42 max-cap**. 49,514 std cells (8,224 of them sequential —
  consistent with a 256-word × 32-bit register array plus control), 52.0%
  utilization, 0.86 mW. This run's `metrics.json` also carries a real
  OpenROAD PDN static-drop check (`design_powergrid__drop__worst`) — a
  genuine measured value, but a per-net static check on this reduced-scale
  block under whatever stimulus OpenROAD's default PDN analysis assumes, NOT
  the full-chip dynamic prefill-burst scenario `p4/power/ir_drop.py` models.
  Label it as such if it's ever quoted; it does not answer T4.6.

  Both the tile and sram real GDS were rendered (KLayout, real Sky130 layer
  colors, fill/tap cells hidden) into real routed-metal close-ups and full-die
  images now embedded in `demo/floorplan.html`'s "Real Silicon" section —
  see `p4/render_gds.py` / `p4/render_gds_crop.py`.

**T4.2 — Hierarchical P&R.** A 16,384-MAC chip will not go through a flat
LibreLane run — the router alone was 225 k cells and 4 h 40 min. Tile hardened
once and instanced 4×, router hardened, top-level assembly.

**T4.3 — Commercial flow on the real PDK.** This is the plan's actual P4.
Sky130 at 1.76 MHz is 1,700× from the 1 GHz target; that gap is process, not
design, but it means **no f_max claim in this program can cite the current run,
in either direction.**

**T4.4 — DFT insertion and ATPG.** *Accept:* ≥ 98% stuck-at coverage.

**T4.5 — LPDDR5X PHY and SRAM compiler hardening.** 44% of the die, today
represented by nothing.

**T4.6 — Multi-corner PVT closure**, power grid and IR under **prefill burst
current**. *Accept:* < 5% VDD droop. The open-flow 4.4% at a 1500 ns clock is a
different question and does not transfer.

---

## Part 4 — Things a newcomer will otherwise get wrong

1. **The Sky130 GDS is a flow qualification vehicle, not a chip milestone.**
   `CLOCK_PERIOD` models nothing, `LANES=4` is 1/16 of shipping, `SEGS=8` is
   superseded by the 32-over-±4 recipe, and three electrical checks are open.
   Its value is as a regression baseline: subsequent blocks are deltas against a
   known-good run.
2. **All P1 numbers rest on models authored in this repo.** `dram_eff = 0.885`
   comes from an LPDDR5X part written here because DRAMsim3 ships none. The
   power model's 25 fF/MAC and 0.247 W leakage are assumptions, and 16 fJ per
   MAC-cycle is optimistic against published 14 nm figures. Both are defended
   modelled numbers. Label them that way wherever quoted.
3. **Speculation does not pay at the assumed acceptance rate.** Measured
   1.14–1.18× at p=0.80 against a planned 1.61×, and at p=0.70 it is a net loss.
   The DSpark acceptance rate is a gate, not a parameter.
4. **`make p2` passing is not evidence a unit is synthesisable at shipping
   parameters.** Two units with pathological critical paths were both missing
   from the PPA loop. If a unit is not in `UNITS`, its depth is unknown — and
   the loop's defaults are not the shipping ones (`LANES=4`, `SEGS=8`, `TILE=8`,
   `CONV_CH=4`).
5. **Elaboration-time checks beat runtime checks for structural bounds.** The
   first version of the tile rewrite checked the fold bound with an ACC_OUT-wide
   shadow sum — re-creating, in the overflow path, the exact serial 32-bit chain
   the rewrite existed to delete. Yosys took longer on that netlist than on the
   original.
