# Sonic S1

A 20.6 mm² edge inference ASIC with a dedicated prefill engine, targeting
**LiquidAI LFM2.5-8B-A1B** with the dense **LFM2.5-2.6B** on the same silicon.

The full program plan is `sonic-s1-plan.html`. This repo is the tooling behind
it: every number the plan publishes is computed here and locked by tests, so a
figure cannot drift without a test changing with it.

```
make            # tests + all headline numbers
make p0         # numerics and routing freeze
make p1         # architecture sweeps
make numbers    # reproduce the plan's tables
```

## Layout

```
sonic/        design-space library -- the single source of truth
  modelspec   parse config.json into a parameter and traffic budget
  quant       the frozen format recipe; RTL and the C model implement this
  chipspec    S1 hardware parameters and the SKU ladder
  moe         routing economics: expert overlap, speculation, occupancy
  roofline    decode, prefill, power, area
  report      prints every headline number
p0/           numerics and routing freeze -- see p0/README.md
  golden/     bit-exact C reference for the numeric primitives
p1/           architecture model and sweeps -- see p1/README.md
tests/        locks the published numbers
configs/      cached model configs (8B-A1B, 2.6B, DSpark drafter)
```

## The one fact the design turns on

Batch-1 decode is purely memory-bound. At SKU B's 68.3 GB/s the array needs
**119 lanes** to keep up; it has 16,384. Everything above 119 exists for
prefill, which is the only compute-bound phase and the only latency a user
feels. That is why the array is sized for time-to-first-token and power-gated
during generation.

## Requirements

Python 3.11+ and numpy for the analysis; a C compiler for the golden model.
No other dependencies. `pytest` is optional — `tests/run.py` works without it.
