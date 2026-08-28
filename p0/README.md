# P0 — Numerics and routing freeze

**Weeks 1–8 · 2 ML engineers.** Nothing downstream can start until the format
spec is frozen, because bit width sets bandwidth, bandwidth sets throughput, and
throughput sets the product.

## Deliverables

| # | Deliverable | Artifact | Status |
|---|---|---|---|
| P0-1 | Frozen format spec | `sonic/quant.py`, gates via `p0/gates.py` | **harness ready, unrun** |
| P0-2 | Measured expert overlap and routing locality | `p0/routing_trace.py` | **DONE — real traces** |
| P0-3 | Speculative-decode budget for DSpark | `p0/dspark.py` | drafted |
| P0-4 | Bit-exact C golden model | `p0/golden/` | primitives only |
| P0-5 | Per-layer accumulator bounds from real activations | `p0/golden/test_golden.c` | worst-case only |

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

## Measured: the recipe fails `ppl_delta` by 40x

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

The formats are not the problem — the *method* is. This harness does naive
round-to-nearest. RTN at 4 bits is the weakest possible scheme; clearing a
0.15 ppl gate at 4.64 bits requires calibrated quantization (GPTQ/AWQ-style
error compensation or activation-aware scaling). That is a requirement on the
**offline packer**, not on the silicon: the hardware formats stay INT4 group-64
plus an outlier budget either way. The next step is to implement calibration in
the packer and re-run, not to widen the formats.

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

## Open items, in priority order

0. **Implement calibration in the packer and re-run the gates.** Everything
   below is downstream of a recipe that currently misses its perplexity gate by
   40x under RTN.
1. **Measure the drafter's acceptance rate** on representative workloads. It is
   now the only free variable in the speculative-decode budget: at p = 0.80 the
   gain is 1.61x, at p = 0.90 it is 2.36x. Needs the DSpark checkpoint.
2. **Prove accumulator bounds per layer against real activations**, not against
   the worst case. The worst case forced a 16-bit local path; measured
   activation ranges may allow 12 and buy back clock frequency.
3. **Extend the C golden model** from primitives to full layers, so RTL has
   something to diff against beyond the numeric kernels.
4. **Validate the PWL segment count** against the perplexity gate. 16 uniform
   segments give ~0.054 worst-case error on SiLU; that is a design-intent bound,
   not a quality result.

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
