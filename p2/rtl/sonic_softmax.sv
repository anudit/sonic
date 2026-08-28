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

  logic signed [DW-1:0] m_q, l_q, a_q;
  logic signed [DW-1:0] m_new;

  // Rescale factor when a new maximum arrives: exp(m_old - m_new).
  assign m_new   = (in_vld && score > m_q) ? score : m_q;
  assign exp_arg = (in_vld && score > m_q) ? (m_q - m_new) : (score - m_new);

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
      l_q <= DW'(((l_q * exp_val) >>> 16) + exp_val);
      a_q <= DW'(((a_q * exp_val) >>> 16) + ((value * exp_val) >>> 16));
    end
  end

  assign run_max = m_q;
  assign run_sum = l_q;
  assign acc     = a_q;

endmodule
