// Sonic S1 -- shared constants.
//
// Plain `define rather than a SystemVerilog package on purpose: Yosys's SV
// front end does not accept package imports in a module header, and the PPA
// loop needs the same sources Verilator simulates. One set of sources, two
// tools, no sv2v step.
//
// These MUST agree with sonic/quant.py and p0/golden/sonic_golden.h.
`ifndef SONIC_DEFS_SVH
`define SONIC_DEFS_SVH

// --- numerics --------------------------------------------------------------
`define W_BITS     4     // INT4 weights, group-64 scaled
`define A_BITS     8     // INT8 activations, per-token dynamic
`ifndef ACC_LOCAL
  `define ACC_LOCAL 16   // fast path -- 16, NOT 12; see the note below
`endif
`ifndef ACC_FOLD
  `define ACC_FOLD  16   // products before folding into mid
`endif
`define ACC_MID    24
`define ACC_OUT    32

// Local-stage bound: ACC_FOLD * max|w| * max|a| = 16 * 8 * 128 = 16384.
// That needs 16 bits. A 12-bit local path -- the width the Kimi K3 write-up
// quotes for its own operand widths -- silently caps |activation| at 15.
// p2/ppa/loop.py sweeps ACC_LOCAL so the cost of the extra bit is measured
// rather than assumed.

// --- systolic geometry -----------------------------------------------------
// P1 (p1/README.md) says sub-tiling beats one monolithic 128x128 array under
// realistic routing imbalance: occupancy 0.889 vs 0.790 at the same die size.
`define TILE       64
`define N_TILES    4          // 4 x 64^2 = 16,384 lanes
`define ACC_BANKS  8          // decode 1 | speculative verify ~10 | prefill 2048

// --- dataflow modes --------------------------------------------------------
`define MODE_DECODE  1'b0     // weight-streaming, output-stationary
`define MODE_PREFILL 1'b1     // weight-stationary, activations stream

// --- quantization ----------------------------------------------------------
`define GROUP      64         // weights per FP16 scale
`define SCALE_BITS 16

// --- MoE -------------------------------------------------------------------
`define N_EXPERTS  32
`define TOP_K      4
`define EXPERT_BITS 5   // clog2(N_EXPERTS)

// Router datapath width. NOT the INT4/INT8 the rest of the chip uses --
// measured on real layer-5 hidden states, INT8 weights cap routing agreement at
// 0.986 and INT4 collapses it to 0.76. INT12 x INT12 is the cheapest point that
// clears the 0.995 gate. See the table in sonic_router.sv.
`define ROUTER_A_BITS 12
`define ROUTER_W_BITS 12
// Sigmoid PWL resolution. 64, not the 16 the FFN's SiLU uses: measured against
// 512 real hidden states, 16 segments give 0.971 routing agreement and 32 give
// 0.990, both under the 0.995 gate. 64 gives 0.996 and 128 buys nothing.
// Chord fits are also minimax-centred, not endpoint-interpolated -- an endpoint
// chord sits entirely above a convex segment and biases every score.
`ifndef ROUTER_PWL_SEGS
  `define ROUTER_PWL_SEGS 64
`endif

`endif
