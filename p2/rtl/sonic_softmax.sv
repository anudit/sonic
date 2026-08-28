// Online (flash) softmax accumulator for the attention unit.
//
// Keeps a running max and running sum so attention never materialises a full
// score row. That matters twice over here: the KV cache is paged from DRAM, and
// at 128K context the quadratic term is 66% of prefill -- a two-pass softmax
// would double the KV reads.
//
// The tile order MUST match the RTL's KV page walk. p0/golden/sonic_golden.c
// proves the result is invariant to tiling for every tile size from 1 to 97;
// this module has to preserve that, or RTL and reference diverge on page
// boundaries only, which is the worst kind of bug to find.
`include "sonic_defs.svh"

module sonic_softmax #(
  parameter int DW = 32          // Q16 scores and accumulators
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  clr,

  input  logic signed [DW-1:0]  score,       // Q16
  input  logic signed [DW-1:0]  value,       // Q16, one lane of V
  input  logic                  in_vld,

  // exp() via the shared PWL: the sequencer loads a table for exp the same way
  // it loads SiLU or sigmoid. One PWL block, three functions.
  output logic signed [DW-1:0]  exp_arg,
  input  logic signed [DW-1:0]  exp_val,

  output logic signed [DW-1:0]  run_max,
  output logic signed [DW-1:0]  run_sum,
  output logic signed [DW-1:0]  acc
);

  localparam logic signed [DW-1:0] ONE_Q16 = DW'(1 << 16);
  localparam int DW2 = 2 * DW;

  // Q16 multiply at DOUBLE width. `l_q * corr` with two DW-wide operands is a
  // DW-wide multiply in SystemVerilog -- self-determined width -- so the
  // product overflows and truncates BEFORE the >>>16 that was supposed to
  // rescale it. At DW=32 that makes 1.0 * 1.0 evaluate to 0.0, which pins the
  // running sum at 1.0 forever. Same family as findings 12-15: sizing a
  // multiply off its operand width rather than its result width.
  function automatic logic signed [DW-1:0] mulq16(
      input logic signed [DW-1:0] x, input logic signed [DW-1:0] y);
    logic signed [DW2-1:0] p;
    begin
      p = DW2'(x) * DW2'(y);
      mulq16 = DW'(p >>> 16);
    end
  endfunction

  logic signed [DW-1:0] m_q, l_q, a_q;
  logic signed [DW-1:0] m_new;
  logic                 new_max;
  logic signed [DW-1:0] corr, wgt;

  // Rescale factor when a new maximum arrives: exp(m_old - m_new).
  assign new_max = in_vld && (score > m_q);
  assign m_new   = new_max ? score : m_q;
  assign exp_arg = new_max ? (m_q - m_new) : (score - m_new);

  // One PWL lookup serves both factors because exactly one of them is always
  // exp(0) = 1: on a new maximum the lookup is the RESCALE and the incoming
  // weight is 1; otherwise the lookup is the WEIGHT and the rescale is 1.
  //
  // Using exp_val for both -- which is what this module did -- is the bug its
  // own comment warns about. It loses the first score outright (at m = -inf the
  // rescale underflows to 0 and the new term is multiplied by that same 0) and
  // thereafter scales the running sum by the incoming weight instead of by the
  // correction. tb_softmax.cpp failed all four score patterns against
  // p0/golden/sonic_golden.c before this.
  assign corr = new_max ? exp_val : ONE_Q16;
  assign wgt  = new_max ? ONE_Q16 : exp_val;

  always_ff @(posedge clk or negedge rst_n) begin
    // Async reset and sync clear are separate branches on purpose -- see the
    // note in sonic_conv.sv.
    if (!rst_n) begin
      m_q <= {1'b1, {(DW-1){1'b0}}};    // -inf
      l_q <= '0;
      a_q <= '0;
    end else if (clr) begin
      m_q <= {1'b1, {(DW-1){1'b0}}};
      l_q <= '0;
      a_q <= '0;
    end else if (in_vld) begin
      m_q <= m_new;
      // Both the running sum and the accumulator rescale by the same factor,
      // so their ratio -- the softmax output -- is unchanged. Rescaling only
      // one of them is the classic online-softmax bug.
      l_q <= mulq16(l_q, corr) + wgt;
      a_q <= mulq16(a_q, corr) + mulq16(value, wgt);
    end
  end

  assign run_max = m_q;
  assign run_sum = l_q;
  assign acc     = a_q;

endmodule
