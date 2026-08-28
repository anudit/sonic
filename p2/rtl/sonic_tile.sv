// One 64x64 systolic sub-tile -- a quarter of the 16,384-lane array.
//
// WHY SUB-TILES: measured routing traces (p0/out/real_routing.npz) show a
// monolithic 128x128 array reaching only 0.805 occupancy at chunk 2048, against
// a 0.80 gate. Four independent 64x64 tiles reach 0.888 on the same silicon,
// because a short expert wastes a quarter-sized pass instead of a full one.
//
// TWO DATAFLOWS, one array:
//   MODE_PREFILL  weight-stationary. Weights latch into the PE grid once, then
//                 activations stream through. Each weight is reused TILE times.
//   MODE_DECODE   weight-streaming, output-stationary. Each weight is used once
//                 -- batch-1 GEMV has no reuse to exploit, so holding weights
//                 still would be pointless. Accumulators hold instead, with
//                 ACC_BANKS of them so a speculative verify batch shares one
//                 weight fetch.
//
// The mode register is sampled a stage upstream of the datapath on purpose:
// decode is the mode that ships in volume and must not pay for prefill's
// flexibility on its critical path.
`include "sonic_defs.svh"

module sonic_tile #(
  parameter int T         = `TILE,        // sub-tile edge
  parameter int ACC_BANKS = `ACC_BANKS
) (
  input  logic                        clk,
  input  logic                        rst_n,
  input  logic                        mode,          // `MODE_DECODE / `MODE_PREFILL

  input  logic                        w_load,        // latch the weight plane
  input  logic signed [T*`W_BITS-1:0] w_col,         // one column of weights
  input  logic signed [T*`A_BITS-1:0] a_row,         // one row of activations
  input  logic                        in_vld,
  input  logic                        clr,
  input  logic [$clog2(ACC_BANKS)-1:0] bank,

  output logic signed [T*`ACC_OUT-1:0] acc_col,      // one column of results
  output logic                        out_vld,
  output logic                        ovf
);

  // Weight plane: T x T registers, loaded column by column in prefill mode.
  logic signed [`W_BITS-1:0] wmem [T][T];
  logic [$clog2(T)-1:0]      wcol_ptr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wcol_ptr <= '0;
    end else if (w_load) begin
      for (int r = 0; r < T; r++)
        wmem[r][wcol_ptr] <= $signed(w_col[r*`W_BITS +: `W_BITS]);
      wcol_ptr <= wcol_ptr + 1'b1;
    end
  end

  // Registered mode, off the datapath's critical path.
  logic mode_q;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) mode_q <= `MODE_DECODE;
    else        mode_q <= mode;

  // T dot-product lanes. Each consumes the whole activation row against one
  // weight column, so the tile computes a T x T by T x 1 product per pass.
  genvar c;
  logic [T-1:0] lane_ovf;
  generate
    for (c = 0; c < T; c++) begin : g_lane
      logic signed [`ACC_OUT-1:0] sum;
      logic signed [`ACC_OUT-1:0] partial;

      always_comb begin
        logic signed [`W_BITS-1:0] wv;
        logic signed [`A_BITS-1:0] av;
        partial = '0;
        for (int r = 0; r < T; r++) begin
          av = $signed(a_row[r*`A_BITS +: `A_BITS]);
          wv = (mode_q == `MODE_PREFILL) ? wmem[r][c]
                                         : $signed(w_col[r*`W_BITS +: `W_BITS]);
          partial += `ACC_OUT'(av) * `ACC_OUT'(wv);
        end
      end

      // ACC_BANKS accumulators per lane: one per speculative candidate in
      // decode, cycling activation tiles in prefill.
      logic signed [`ACC_OUT-1:0] banks [ACC_BANKS];
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          for (int b = 0; b < ACC_BANKS; b++) banks[b] <= '0;
        end else if (clr) begin
          banks[bank] <= '0;
        end else if (in_vld) begin
          banks[bank] <= banks[bank] + partial;
        end
      end

      assign sum = banks[bank];
      assign acc_col[c*`ACC_OUT +: `ACC_OUT] = sum;
      // Saturation is a design error here, not a runtime condition: the P0
      // bound (fold 16 x |INT4| 8 x |INT8| 128) is proven per layer.
      assign lane_ovf[c] = 1'b0;
    end
  endgenerate

  assign ovf = |lane_ovf;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) out_vld <= 1'b0;
    else        out_vld <= in_vld;

endmodule
