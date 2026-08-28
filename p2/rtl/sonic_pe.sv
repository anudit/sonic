// One processing element: an INT4 x INT8 MAC with a hierarchical accumulator
// and ACC_BANKS output banks.
//
// The banks are what make one array serve three batch regimes:
//   decode          batch 1     -- 1 bank used, 15 idle, array power-gated
//   speculative     batch ~10   -- one bank per candidate token
//   prefill         batch 2048  -- banks cycle as activation tiles stream
//
// In MODE_PREFILL the weight register holds still and activations stream past
// (weight-stationary); in MODE_DECODE the weight streams and the accumulator
// holds still (output-stationary). Mode select must not land on the decode
// critical path -- it is registered a stage upstream in sonic_tile.
`include "sonic_defs.svh"

module sonic_pe #(
  parameter int ACC_BANKS = `ACC_BANKS
) (
  input  logic                        clk,
  input  logic                        rst_n,
  input  logic                        mode,
  input  logic                        clr,
  input  logic                        en,
  input  logic [$clog2(ACC_BANKS)-1:0] bank,
  input  logic signed [`W_BITS-1:0]    w_in,
  input  logic                        w_load,   // latch weight (prefill)
  input  logic signed [`A_BITS-1:0]    a_in,
  output logic signed [`W_BITS-1:0]    w_out,    // systolic pass-through
  output logic signed [`A_BITS-1:0]    a_out,
  output logic signed [`ACC_OUT-1:0]   acc,
  output logic                        ovf
);

  logic signed [`W_BITS-1:0] w_held;
  logic signed [`W_BITS-1:0] w_eff;
  logic [ACC_BANKS-1:0]     bank_ovf;
  logic signed [`ACC_OUT-1:0] bank_acc [ACC_BANKS];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n)      w_held <= '0;
    else if (w_load) w_held <= w_in;
  end

  assign w_eff = (mode == `MODE_PREFILL) ? w_held : w_in;

  genvar b;
  generate
    for (b = 0; b < ACC_BANKS; b++) begin : g_bank
      sonic_acc u_acc (
        .clk(clk), .rst_n(rst_n),
        .clr  (clr && (bank == b[$clog2(ACC_BANKS)-1:0])),
        .en   (en  && (bank == b[$clog2(ACC_BANKS)-1:0])),
        .w(w_eff), .a(a_in), .flush(1'b0),
        .acc(bank_acc[b]), .ovf(bank_ovf[b])
      );
    end
  endgenerate

  assign acc = bank_acc[bank];
  assign ovf = |bank_ovf;

  // Systolic pass-through, one cycle each.
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin w_out <= '0; a_out <= '0; end
    else        begin w_out <= w_in; a_out <= a_in; end
  end

endmodule
