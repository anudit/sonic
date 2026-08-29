// Short-conv unit: depthwise causal convolution with the LFM double gate.
//
//   in_proj(x) -> B, C, x'        (3d wide, done in the array, not here)
//   y = C * conv1d_causal(B * x') (this unit)
//   out_proj(y)                   (array again)
//
// This is the block that makes the whole chip viable. Conv state is
// 18 layers x 2048 channels x (k-1) taps = 72 KB and, crucially, is CONSTANT
// regardless of context length -- unlike a KV cache. It lives on die and is
// SRAM-retained across idle, so a conversation survives inter-token gaps with
// no DRAM traffic at all.
//
// k is parameterised to 7 rather than hardwired to the model's 3: widening the
// kernel is nearly free here (O(P), three cycles per layer per token) whereas
// adding an attention layer is quadratic. If a future LFM wants more mixing
// capacity, this is where it should come from.
`include "sonic_defs.svh"

`ifndef CONV_CH
  `define CONV_CH 64
`endif

module sonic_conv #(
  parameter int CH    = `CONV_CH,    // channels processed per pass
  parameter int KMAX  = 7,
  parameter int DW    = `A_BITS
) (
  input  logic                       clk,
  input  logic                       rst_n,

  input  logic [$clog2(KMAX+1)-1:0]  k_taps,        // active kernel size
  input  logic                       state_clr,     // sequence position 0

  input  logic signed [CH*DW-1:0]    x_in,          // gated input, B * x'
  input  logic signed [CH*DW-1:0]    c_gate,        // output gate C
  input  logic signed [KMAX*8-1:0]   kern,          // INT8 taps, shared per channel
  input  logic                       in_vld,

  output logic signed [CH*DW-1:0]    y_out,
  output logic                       out_vld
);

  // Rolling state: KMAX-1 previous samples per channel. This is the 72 KB.
  logic signed [DW-1:0] hist [CH][KMAX-1];

  logic signed [DW-1:0] xv   [CH];
  logic signed [DW-1:0] cv   [CH];
  logic signed [7:0]    tap  [KMAX];

  always_comb begin
    for (int c = 0; c < CH; c++) begin
      xv[c] = $signed(x_in[c*DW +: DW]);
      cv[c] = $signed(c_gate[c*DW +: DW]);
    end
    for (int t = 0; t < KMAX; t++) tap[t] = $signed(kern[t*8 +: 8]);
  end

  // Causal: tap 0 multiplies the current sample, tap t the sample t steps back.
  // Taps beyond k_taps are forced to zero rather than masked at the adder, so a
  // shorter kernel costs nothing in the reduction tree.
  genvar c;
  generate
    for (c = 0; c < CH; c++) begin : g_ch
      // Products are DW x 8 -> DW+8 bits, accumulated in ACC_MID. Widening the
      // OPERANDS to ACC_MID first, as the first draft did, asks the synthesiser
      // for KMAX 24x24 multipliers per channel instead of KMAX 8x8 ones -- 688k
      // cells for 64 channels, about four times what this block should cost.
      // SystemVerilog sizes a multiply from its operands, so operand width is a
      // hardware decision, not a formatting one.
      localparam int PW = DW + 8;
      logic signed [`ACC_MID-1:0] sum;
      logic signed [PW-1:0]       prod [KMAX];
      logic signed [`ACC_MID-1:0] tsum [KMAX];

      always_comb begin
        prod[0] = xv[c] * tap[0];
        // if/else, NOT a ternary. `'0` is an unsigned unsized literal, and
        // SystemVerilog makes an entire expression unsigned if any operand is,
        // so `cond ? (signed * signed) : '0` silently evaluates the multiply
        // UNSIGNED. Measured: tap = -2 multiplied as 254, so hist 10 * tap -2
        // gave +2540 instead of -20. prod[0] escaped only because it is not in
        // a ternary. Same family as findings 12-15 -- an expression rule
        // quietly changing the hardware -- but signedness rather than width.
        for (int t = 1; t < KMAX; t++) begin
          if (t < int'(k_taps)) prod[t] = hist[c][t-1] * tap[t];
          else                  prod[t] = '0;
        end
        // Balanced tree, for the same reason as sonic_tile's fold and the
        // router's MAC: a running sum over KMAX taps is KMAX dependent
        // ACC_MID-wide adders. KMAX is only 7, so this is 3 levels instead of
        // 7 rather than 6 instead of 64 -- small, but it is the same defect and
        // it costs nothing to not have it.
        for (int t = 0; t < KMAX; t++) tsum[t] = `ACC_MID'(prod[t]);
        for (int step = 1; step < KMAX; step = step * 2)
          for (int t = 0; t + step < KMAX; t = t + 2*step)
            tsum[t] = tsum[t] + tsum[t + step];
        sum = tsum[0];
      end

      // Output gate, then requantize back to DW with symmetric rounding.
      // Round-half-away-from-zero, written without a negated size cast:
      // -DW'(expr) parses as a cast of a negative-sized expression in Yosys
      // and is rejected, even though Verilator accepts it.
      // Output gate: ACC_MID x DW, not ACC_OUT x ACC_OUT.
      /* verilator lint_off UNUSEDSIGNAL */
      logic signed [`ACC_MID+DW-1:0] gated;
      /* verilator lint_on UNUSEDSIGNAL */
      logic signed [DW-1:0]          rounded;
      assign gated   = sum * cv[c];
      // Round half away from zero, written without a negated size cast --
      // -DW'(expr) parses as a cast of a negative-sized expression in Yosys.
      assign rounded = DW'(gated[`ACC_MID+DW-1]
                     ? -(((-gated) + (1 << 13)) >>> 14)
                     :  ((  gated  + (1 << 13)) >>> 14));
      assign y_out[c*DW +: DW] = rounded;

      // Async reset and sync clear must be SEPARATE branches. Folding them
      // into one condition (!rst_n || state_clr) makes the reset condition
      // depend on a level signal that is not in the sensitivity list. Simulation
      // accepts that; Yosys rejects it outright as ambiguous.
      // (A comment whose first word is the simulator's own name is parsed as a
      // lint pragma, which is why this sentence is phrased around it.)
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          for (int t = 0; t < KMAX-1; t++) hist[c][t] <= '0;
        end else if (state_clr) begin
          for (int t = 0; t < KMAX-1; t++) hist[c][t] <= '0;
        end else if (in_vld) begin
          hist[c][0] <= xv[c];
          for (int t = 1; t < KMAX-1; t++) hist[c][t] <= hist[c][t-1];
        end
      end
    end
  endgenerate

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) out_vld <= 1'b0;
    else        out_vld <= in_vld;

endmodule
