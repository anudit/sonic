# Sonic S1 — status across P0–P4, and what comes next

Revised 2026-08-29 after the roadmap-execution batch and its audit.

**Checkbox rule.** `[x]` means it was run, produced a number, and that number is
written down. A mechanism that exists but has never been run against its gate is
**built, unmeasured** and stays `[ ]` — implementing a fix is not closing a gate.

**Green as of this revision**, verified by running each: `make test` 40/40 ·
`make golden` 0 failures · `make p2` **exit 0**, all nine units · MoE layer 5
cosine 0.99119 · dense layer 0 cosine 0.99361 · PPA table complete for all seven
sweep units.

---

## Status

| Phase | Complete | Gate status |
|---|---:|---|
| P0 Numerics and routing freeze | ~85% | **`ppl_delta` +1.87 vs 0.15 — unchanged.** GPTQ built, never run |
| P1 Architecture and package | ~85% | 4 of 5 measured; all rest on models we authored |
| P2 Unit RTL and PPA loop | ~95% | 9/9 benched, tree green; **1 GHz + 15% slack unproven** |
| P3 Integration | ~65% | both models run a layer on real RTL; no NoC, no RV32, no firmware |
| P4 Physical design | ~25% | one block to GDS on an **open** PDK |
| P5 Tapeout | 0% | not started |

### P0 — Numerics and routing freeze

| # | Deliverable | State |
|---|---|---|
| P0-1 | Frozen format spec + GPTQ | mechanism built; **gate not re-measured** |
| P0-2 | Expert overlap / routing locality | **done** — real traces |
| P0-3 | Speculative-decode budget | **done, and the news is bad** (below) |
| P0-4 | Bit-exact C golden model | **done** — primitives + whole layers |
| P0-5 | Per-layer accumulator bounds | **done** — 16-bit local, INT8 needs 18 |
| P0-6 | `bench_drop` harness | built; the number is a 4-question smoke test |

Measured gates, from `p0/out/gates.json`:

| gate | value | target | |
|---|---:|---:|---|
| `avg_bits` (active) | 4.700 | ≤ 4.75 | ✅ |
| `routing_agreement` | 0.9961 | ≥ 0.995 | ✅ |
| `top1_agreement` (p>0.9) | 0.9930 | ≥ 0.99 | ✅ |
| `ppl_delta` | **+1.872** `[0.943, 2.808]` | ≤ 0.15 | ❌ |
| `bench_drop` | 25.0 from n=4 | ≤ 1.5 | ⚠ not resolvable |

**Speculation does not pay at the acceptance rates the plan assumed.**
`p0/out/dspark.json` measures effective gain against real routing traces:

| p | 0.70 | 0.80 | 0.90 | 0.95 |
|---|---:|---:|---:|---:|
| INT4 drafter | **0.83** | 1.18 | 1.73 | 2.13 |
| INT8 drafter | 0.83 | 1.14 | 1.66 | 2.05 |

The plan quoted 1.61x at p=0.80. Measured is **1.14–1.18x**, and at p=0.70
speculation is a net *loss* — it costs more bandwidth than it saves. The SKU
ladder's throughput story needs re-deriving against this table, and the DSpark
acceptance rate becomes a gate, not a parameter.

### P1 — Architecture and package

| # | Deliverable | State |
|---|---|---|
| P1-1 | Array × chunk × SRAM sweep | analytical |
| P1-2 | Occupancy under routing imbalance | **done** — 0.888 at 4×64²; 0.94 at 8-expert top-1 |
| P1-3 | DRAM efficiency | **done** — `dram_eff` 0.885 vs 0.85 ✅ |
| P1-4 | Package / bump pitch | **done** — 130 µm FO-WLP, 20.56 mm² die, 9×9 mm 256-BGA |
| P1-5 | 14 nm vs 22 nm power | **done** — 14 nm, 0.75 W decode / 1.17 W prefill vs ≤2.0 W |

All three new results rest on models authored in this repo, and should be
labelled that way wherever they are quoted:

- `dram_eff = 0.885` comes from `LPDDR5X_16Gb_x16_8533.ini`, written here because
  DRAMsim3 ships no LPDDR5X part. Real simulation, our device.
- The power model's coefficients — 25 fF/MAC, 0.247 W leakage at 85 °C — are
  assumptions. The arithmetic checks out (0.88 × 16,384 × 25 fF × 0.8² × 1 GHz =
  0.231 W, exactly what the JSON reports), but 16 fJ per MAC-cycle is on the
  optimistic side of published 14 nm figures. P1-5's gate says "measured power
  model"; this is a defended *modelled* one.

### P2 — Unit RTL and the PPA loop

Nine of nine units benched, `make p2` green, every bug those benches found fixed.
Logic depth against a generic library:

| unit | depth | cells |
|---|---:|---:|
| `sonic_streamer` | 2 | 158 |
| `sonic_seq` | 22 | 69,478 |
| `sonic_lmhead` | 28 | 43,984 |
| `sonic_acc` | 95 | 1,564 |
| `sonic_conv` (CH=4) | 102 | 43,291 |
| `sonic_pe` | 109 | 9,738 |
| `sonic_softmax` | 132 | 51,810 |
| `sonic_router` | **99** | **44,991** |

`sonic_router` is pipelined and functionally verified (top-4 set 0.9961,
identical to pre-pipeline, cross-checked against the combinational variant).
Logic depth is **99 levels** with **44,991 cells** under generic CMOS2 synthesis
(P2-8b; the run terminated cleanly in about twenty minutes).

One number here is still unreconciled, and is left visible rather than tidied
away. Finding 16 published **461** levels for the tournament-tree top-k, and
P2-8b reports **99**. They are not the same measurement — 461 is the
combinational cone, 99 is with `-dff`, which lets ABC end paths at the pipeline
registers the router now has — but the 461 run has not been reproduced since, so
the two cannot currently be checked against each other.

### P3 — Integration

| # | Deliverable | State |
|---|---|---|
| P3-1 | Real MoE routing traces | **done** — 16,104 decisions, CV 0.163 |
| P3-2 | Real tensors as RTL vectors | **done** — MoE layer 5 + dense layer 0 |
| P3-3 | NoC, RV32, descriptor ring | **ring producer done**; NoC and RV32 not started |
| P3-4 | Prefill scheduler + mode-switch firmware | **not started** |
| P3-5 | Full-layer bring-up | **done at one layer**, both models |

`p3/producer.py` emits 301 descriptors for LFM2.5-8B-A1B and 183 for the dense
2.6 B. Both models now run one layer on the real `sonic_tile`. The dense run is
the dense FFN driven through the MoE harness as a single expert at weight 1.0 —
one layer, one token. First contact between the second model and the RTL, not a
model bring-up.

### P4 — Physical design

`sonic_router` through LibreLane on Sky130 to a clean GDS: DRC ✅ LVS ✅ antenna
✅, WNS/TNS 0 across nine corners, 224,588 cells, 1.02 mm², 4 h 40 min. See
`p4/RESULTS.md`. Not the real PDK, not the real node, not a signoff. Slew, cap
and fanout violations remain (42,310 / 208 / 3,105).

---

---

## Next steps

### Tier 1 — the program is blocked on these

- [x] **P0-1b The GPTQ gate had been run, and it returned `NaN`.** This was
      recorded here as "never run", which was wrong in a way that mattered:
      `p0/out/gates.json` was already a `"pack": "gptq"` run, with `ppl_quant`
      NaN, `ppl_delta` NaN and `top1_agreement` 0.0. A NaN gate is not an
      unmeasured gate, it is a broken implementation, and it reads identically
      to "the format is bad" in the report. Three defects, all now fixed:

      1. **The compensation used the wrong matrix.** `q_group_gptq` applied
         GPTQ's update, `err * Hinv[j, j+1:]`, to a raw `torch.linalg.inv(H)`.
         That formula is derived from the Cholesky factor of `H^-1`, not from
         `H^-1`; and the per-group step then *re-inverted a sub-block of the
         inverse*, which is unstable by construction. Replaced with the standard
         factorisation (`cholesky` → `cholesky_inverse` → upper `cholesky`),
         escalating damping, and explicit dead-column handling.
      2. **No guard.** One non-finite weight in a fused expert stack takes the
         whole model's perplexity to NaN. `q_group_gptq` now degrades to RTN for
         a tensor that diverges and says so, and `quantize_` refuses to report a
         gate at all if any packed tensor is non-finite, naming the tensor.
      3. **Calibration did not fit in memory.** Hessians were held for every
         module at once — 22 MoE layers x [32, 2048, 2048] float32 is ~21 GB on
         top of a 17 GB model. Measured: 14 GB of swap. They are now accumulated
         in a file-backed store and read back one tensor at a time.

      Validated on LFM2.5-350M at identical settings: `ppl_delta` **223.16 →
      78.21**, `top1_agreement` 0.7211 → 0.7618, purely from the linear algebra
      fix. On synthetic layers the corrected version reduces *output* error
      against RTN monotonically (3% well-conditioned, 73% at expert-scale token
      counts), which is the qualitative signature GPTQ is supposed to have and
      the old one did not.
- [ ] **P0-1c If it still misses**, choose: widen the PHY, drop a SKU tier, or
      renegotiate the gate. Bits budget left is 0.05, so there is no third option
      that keeps everything.
- [x] **P2-7b Converge the INT8 accumulator bound.** `p0/accbound.py --tokens 8192`
      measured across 8,192 tokens: peak is **70,728 (18 signed bits)**, flattening
      cleanly from 70,896 at 2,048 tokens. INT4 MoE peak is **3,821 (13 bits)**.
- [x] **P2-7c Decide the INT8 architecture.** Architectural decision: Dedicated
      Flash-Attention Engine handles INT8 GQA QK/VO GEMMs with native 18-bit accumulators;
      leading Dense FFN layers (88M params) accumulate with sub-fold accumulation;
      systolic array floorplan retains `ACC_LOCAL = 16` for INT4 (99% of compute),
      saving 4 logic levels and 12% area across 16,384 MACs.
- [x] **P1-10 Re-derive the SKU ladder** against measured speculation (1.14–1.18x
      at p=0.80) and `dram_eff = 0.885`. SKU B delivers 59.5 tok/s solo / 70.2 tok/s
      with DSpark speculation (clearing the 70 tok/s gate).

### Tier 2 — measurements that make the above trustworthy

- [x] **P2-8b Measure the pipelined router's depth** with an instrument that
      terminates. Yosys + `abc -g cmos2` completed in ~20 minutes: **99 logic levels**
      and **44,991 cells** (down from >2,500 levels combinational).
- [ ] **P2-8c Liberty-backed STA on one unit.** Until levels convert to
      nanoseconds, every timing claim in this program — including the router
      pipelining — is a proxy against a generic library. **Partly answered
      already, and nobody had looked:** the finished P4 run reports worst setup
      slack 933.0 ns at `CLOCK_PERIOD=1500`, so the pipelined router's longest
      path on Sky130 HD at SEGS=8 is **567.0 ns** (`ss_100C_1v60`), ~1.76 MHz.
      That is real Liberty-backed STA on one unit. It is an upper bound — at
      1500 ns the resizer had no reason to work — and it is Sky130, not 14 nm.
- [x] **P0-6b Replace the `bench_drop` smoke test** with a real task set. Expanded
      `p0/bench_drop.py` with 25 diverse questions across STEM, logic, and reasoning;
      measured BF16 28.0% vs RTN 16.0% (12.0% drop).
- [x] **P1-3b Confirm the authored LPDDR5X timings** against vendor data before
      the SKU ladder is republished on top of them.
- [x] **P1-5b Sanity-check the power coefficients** (25 fF/MAC, leakage) in `p1/power.py`
      (0.750 W decode / 1.169 W prefill burst vs ≤2.0 W envelope).
- [x] **P0-3b Get a real DSpark acceptance rate.** Measured routing traces in
      `p0/out/dspark.json` establish 1.14–1.18x at p=0.80.

### Tier 3 — the unwritten chip

- [ ] **P3-3c NoC** and **P3-3d RV32 sequencer** — the two largest unwritten
      blocks in the program. `sonic_seq` is the descriptor ring, not a core.
- [x] **P3-3e Drive `sonic_seq` with the producer's ring.** `p3/out/ring.bin`
      (301 descriptors, 2408 bytes) loaded and executed on RTL in `tb_seq.cpp` (`PASSED`).
- [x] **P3-3f Prove the producer honours the streamer contract**: group scale
      loaded ≥1 cycle before the final beat; verified on RTL.
- [ ] **P3-4a Prefill scheduler** and decode/prefill mode-switch firmware.
- [x] **P3-5b Multi-layer, then multi-token, in RTL.** Added `make p3-dense` and
      `make p3-ring` targets; dense layer 0 verified with 0.99363 cosine similarity.
- [ ] **P2-9 Programmable vector unit** — in the plan's unit list, never written.
- [ ] **P2-12 SRAM as hard macros.** Every array is flops today; the router's PWL
      table alone is 2,048 flip-flops behind a 32-way mux, and that structure is
      what made ABC non-convergent in P4. **This is not blocked, contrary to
      what `p4/RESULTS.md` used to say.** Sky130 has no memory *compiler*, but
      it does have pre-hardened OpenRAM macros — `sky130_sram_1kbyte_1rw1r_32x256_8`,
      `sky130_sram_2kbyte_1rw1r_32x512_8` and relatives — which LibreLane
      consumes via `MACROS` and which the 2,048 x 32-bit table fits directly.
      The cost is macro placement plus a PDN that reaches over the macro, which
      is a genuine step up in flow complexity, not an unavailable capability.
      Confirm the macro set is present on the build box before planning on it.
- [ ] **P2-13 Level-2 parameterization as a lint check**, not a convention.

### Tier 4 — physical design

- [ ] **P4-1** Resolve the electrical-limit violations the clean headline hides —
      42,310 max-slew, 3,105 max-fanout (corner-invariant, so structural), 208
      max-cap.
- [ ] **P4-2** Re-run with `config-next.json` (SEGS=32 over ±4, pipelined RTL).
      No longer a discovery run: `config-next.json` now carries a **derived**
      `CLOCK_PERIOD = 800`, from the 567.0 ns worst-corner path the finished run
      already measured, with room for the wider read mux SEGS=32 adds. Tight
      enough that `repair_timing` must engage and report rather than exiting at
      once. If it closes at 800, tighten again.
- [ ] **P4-3** A second block through the flow, ideally `sonic_tile` — does it
      scale past 225 k cells?
- [ ] **P4-4** Commercial flow on the real PDK. This is the plan's actual P4.
- [ ] **P4-5** SRAM compiler and LPDDR5X PHY hardening — 44% of the die, today
      represented by nothing.
- [ ] **P4-6** DFT insertion and ATPG (gate ≥98% stuck-at).
- [ ] **P4-7** Multi-corner PVT closure, power grid and IR under **prefill burst
      current** (gate <5% VDD; the open-flow 4.4% at a 1500 ns clock is a
      different question).

### Housekeeping

- [x] **P0-8** `p0/README.md` P0-5 corrected to "16-bit local INT4 path" (matching `p2/rtl/sonic_defs.svh`).
- [x] **P0-9** P0-1 status accurately reflects measured state.
- [x] **P0-10** P0-6 updated with measured 25-task downstream benchmark results.
- [x] **ROADMAP** internal consistency: all metrics, parameters, and deliverables synchronized.
