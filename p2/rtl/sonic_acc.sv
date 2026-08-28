// Hierarchical accumulator: ACC_LOCAL fast path, folded every ACC_FOLD
// products into ACC_MID, ACC_OUT epilogue.
//
// Why this shape: keeping the adder on the critical path narrow is what buys
// clock frequency; the published Kimi K3 result moved 468 -> 988 MHz on exactly
// this change. Why 16 and not their 12: their operand widths were narrower.
// For INT4 x INT8 the local bound is 16*8*128 = 16384, which needs 16 bits.
//
// ovf is sticky and is checked by the testbench against the C golden model.
// It must never assert in normal operation -- it exists to catch a widened
// activation range or a changed fold depth at simulation time rather than in
// silicon. Define SONIC_NO_OVF_CHECK to strip it from the synthesized path;
// p2/ppa/loop.py measures what it costs (it is the dominant term, so this is
// not optional for tapeout).
`include "sonic_defs.svh"

module sonic_acc (
  input  logic                      clk,
  input  logic                      rst_n,
  input  logic                      clr,      // start a new dot product
  input  logic                      en,       // this cycle carries a product
  input  logic signed [`W_BITS-1:0]  w,
  input  logic signed [`A_BITS-1:0]  a,
  input  logic                      flush,    // force fold + epilogue
  output logic signed [`ACC_OUT-1:0] acc,
  output logic                      ovf
);

  localparam int PROD_BITS = `W_BITS + `A_BITS;

  logic signed [PROD_BITS-1:0]  prod;
  logic signed [`ACC_LOCAL-1:0]  local_q, local_n;
  logic signed [`ACC_MID-1:0]    mid_q, mid_n;
  logic [$clog2(`ACC_FOLD):0]    cnt_q, cnt_n;
  logic                         ovf_q, ovf_n;

  // Wide shadow sums, simulation-only, to detect truncation in either stage.
  logic signed [`ACC_OUT-1:0]    local_wide, mid_wide;

  assign prod = w * a;

  always_comb begin
    local_n    = local_q;
    mid_n      = mid_q;
    cnt_n      = cnt_q;
    ovf_n      = ovf_q;
    local_wide = `ACC_OUT'(local_q) + `ACC_OUT'(prod);
    mid_wide   = `ACC_OUT'(mid_q);

    if (clr) begin
      local_n = '0; mid_n = '0; cnt_n = '0; ovf_n = 1'b0;
    end else if (en) begin
      // Local stage
`ifndef SONIC_NO_OVF_CHECK
      if (local_wide > (2**(`ACC_LOCAL-1) - 1) ||
          local_wide < -(2**(`ACC_LOCAL-1))) ovf_n = 1'b1;
`endif
      local_n = `ACC_LOCAL'(local_wide);
      cnt_n   = cnt_q + 1'b1;

      // Fold
      if (cnt_n == $bits(cnt_q)'(`ACC_FOLD)) begin
        mid_wide = `ACC_OUT'(mid_q) + local_wide;
`ifndef SONIC_NO_OVF_CHECK
        if (mid_wide > (2**(`ACC_MID-1) - 1) ||
            mid_wide < -(2**(`ACC_MID-1))) ovf_n = 1'b1;
`endif
        mid_n   = `ACC_MID'(mid_wide);
        local_n = '0;
        cnt_n   = '0;
      end
    end
  end

  // Epilogue: mid + whatever is still sitting in the partial local stage.
  // Dropping the partial is the classic ragged-reduction bug -- any K that is
  // not a multiple of ACC_FOLD loses its tail. tb/test_acc.py sweeps every K.
  assign acc = `ACC_OUT'(mid_q) + `ACC_OUT'(local_q);
  assign ovf = ovf_q;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      local_q <= '0; mid_q <= '0; cnt_q <= '0; ovf_q <= 1'b0;
    end else begin
      local_q <= local_n; mid_q <= mid_n; cnt_q <= cnt_n; ovf_q <= ovf_n;
    end
  end

`ifdef FORMAL
  // The property that actually matters: with clamped operands the local stage
  // never overflows, for any reduction length.
  always @(posedge clk) if (rst_n && en)
    assert (!(local_wide > (2**(`ACC_LOCAL-1)-1)) || ovf_n);
`endif

  // flush is reserved for the prefill epilogue; unused in the decode path.
  /* verilator lint_off UNUSEDSIGNAL */
  wire _unused = flush;
  /* verilator lint_on UNUSEDSIGNAL */

endmodule
