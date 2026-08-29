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
  `define ACC_LOCAL 16   // fast path -- 16 bits (guarantees zero overflow on worst-case 16*8*128=16384)
`endif
`ifndef ACC_FOLD
  `define ACC_FOLD  16   // products before folding into mid
`endif
`define ACC_MID    24
`define ACC_OUT    32

// Local-stage bound: ACC_FOLD * max|w| * max|a| = 16 * 8 * 128 = 16384.
// That strictly needs 16 bits signed (+-32768) to guarantee zero overflow under
// any legal INT4 x INT8 operands.

// --- systolic geometry -----------------------------------------------------
// P1 (p1/README.md) says sub-tiling beats one monolithic 128x128 array under
// realistic routing imbalance: occupancy 0.889 vs 0.790 at the same die size.
`ifndef TILE
  `define TILE     64
`endif
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
`ifndef ROUTER_LANES
  `define ROUTER_LANES 64
`endif
// Sigmoid PWL: 32 segments over [-4, 4). Measured on 512 real hidden states,
// `make p2-pwl-sweep`, top-4 set agreement against the 8.47 B checkpoint:
//
//     range \ segs      8       16       32       64
//     +-8            0.9102   0.9707   0.9902   0.9961
//     +-4            0.9707   0.9902   0.9961   0.9961
//     +-2            0.9590   0.9844   0.9902   0.9941
//
// Halving the range is worth exactly one doubling of the segment count -- the
// +-4 row is the +-8 row shifted one column left -- so 32 segments over +-4
// buys the same 0.9961 as 64 over +-8 with HALF the table. That halving is not
// cosmetic: the table is a firmware-writable flop array with a SEGS-way 32-bit
// read mux, and it is what made ABC non-convergent at 64 in P4 (p4/RESULTS.md).
//
// +-2 is where it stops paying: the logits genuinely reach past 2, so the
// narrower table clips instead of resolving. Range is a lever down to the
// actual dynamic range and no further -- the same shape finding 20 found on the
// FFN's SiLU, measured rather than assumed.
//
// Chord fits are minimax-centred, not endpoint-interpolated -- an endpoint
// chord sits entirely above a convex segment and biases every score.
`ifndef ROUTER_PWL_SEGS
  `define ROUTER_PWL_SEGS 32
`endif
// Half-width of the PWL's input range, in units of the logit. The table spans
// [-RANGE, RANGE). Must be a power of two -- segment select is a bare shift.
`ifndef ROUTER_PWL_RANGE
  `define ROUTER_PWL_RANGE 4
`endif

`endif
