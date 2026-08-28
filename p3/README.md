# P3 — Integration and FPGA bring-up

**Weeks 26–46 · full team.** Everything before this is a hypothesis. P3 is where
the full 8.47 B model runs end to end and the plan's numbers meet reality.

## Deliverables

| # | Deliverable | Artifact | Status |
|---|---|---|---|
| P3-1 | Real MoE routing traces from the 8B model | `capture_routing.py` | **written, running** |
| P3-2 | Real tensors exported as RTL vectors | `export_vectors.py` | **done** |
| P3-3 | NoC, RV32 sequencer, descriptor ring | — | not started |
| P3-4 | Prefill scheduler + decode/prefill mode switch firmware | — | not started |
| P3-5 | FPGA port and full-model bring-up | — | blocked, see below |

## P3-1 is the one that unblocks P1

Every occupancy and speculation number in `p1/` rests on a synthetic routing
model. The gate is marginal under it — 0.790 against a 0.80 threshold — so the
real distribution decides whether the prefill engine works as specified.

```
python3 p3/capture_routing.py --tokens 4096
python3 p0/routing_trace.py trace --trace p0/out/real_routing.npz
python3 p1/occupancy.py            # re-run with the measured CV
```

Status: **done.** 16,104 decisions captured; measured expert-load CV 0.163, and
real routing beats the uniform bound by 1.2-1.6x. Three numbers came out of it:

1. **Load CV** — feeds `p1/sweep.py --imbalance`, which sets the sub-tile size.
2. **Expert overlap vs. batch** — if real locality beats the uniform bound, the
   speculative gain rises above 1.15× and the SKU table improves.
3. **Tokens-per-expert distribution** — the ragged-GEMM scheduler is sized
   against its tail, not its mean.

## Bring-up is simulation-only, by choice

There is no board, and there does not need to be one. FPGA buys simulation
*speed* for long soak tests; it does not buy correctness, and every functional
question this program has can be answered by a simulator. The toolchain:

| Tool | Role |
|---|---|
| **Verilator** | C++ differential benches. Links `p0/golden/sonic_golden.c` directly, so RTL and reference see identical stimulus in one process. Fast enough for full-model vector sweeps. |
| **Icarus Verilog** | Independent SystemVerilog front end and scheduler. Runs the same RTL from a native SV testbench (`make iv`). Agreement between the two is evidence the design does not depend on one tool's interpretation. |
| **GTKWave** | Waveform inspection (`make wave`). Both benches dump VCD. |
| **Yosys** | Synthesis for the PPA loop — cell counts and logic depth. |

What is genuinely lost without hardware: wall-clock soak testing, real DRAM
timing behaviour, and power measurement. All three are P4/P5 concerns. The
plan's "full model on FPGA" gate should read **"full model in Verilator against
exported golden tensors"**, which is what `make p2-router` already does for the
router and what P3-2 extends to the remaining units.

If a board is ever wanted, an open-source flow (Yosys + nextpnr) on a Lattice or
Gowin part exercises individual units without Vivado. Vivado itself is x86
Linux/Windows only and will not run on Apple Silicon.
