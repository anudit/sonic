// Weight streamer: INT4 group-64 unpack and dequantize, with expert gather.
//
// Sits between the memory controller and the array and is the chip's actual
// critical path -- if it stalls, everything stops. Two jobs:
//
//   1. UNPACK. Weights arrive as INT4 nibbles with one FP16 scale per group of
//      64. Dequantizing is inline, never a separate pass: a dequantize-to-SRAM
//      step would double memory traffic and halve the chip.
//
//   2. GATHER. MoE expert fetch is indirect -- the router picks 4 of 32 experts
//      per token, and their base addresses are patched into the descriptor at
//      runtime. Each expert is 5.85 MB of contiguous DRAM, so the gather is
//      coarse: thousands of sequential pages, not a scatter. That is why MoE at
//      this granularity is DRAM-friendly and a 128-tiny-expert model would not
//      be.
//
// Scales are applied to the ACCUMULATED partial sum, not to individual weights:
// one multiply per group of 64 instead of 64 of them. The array therefore sees
// raw INT4 codes and stays an integer datapath.
`include "sonic_defs.svh"

module sonic_streamer #(
  parameter int LANES = 32,
  parameter int GROUP = `GROUP
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // Packed nibble stream from DRAM: LANES INT4 codes per beat.
  input  logic [LANES*`W_BITS-1:0]      dram_data,
  input  logic                          dram_vld,
  output logic                          dram_rdy,

  // One FP16-as-Q16 scale per group of 64 weights.
  input  logic signed [15:0]            grp_scale,
  input  logic                          grp_scale_vld,

  // Expert gather: the router patches this before each MoE layer.
  input  logic [`EXPERT_BITS-1:0]       expert_id,
  input  logic                          expert_vld,
  output logic [`EXPERT_BITS-1:0]       expert_active,

  // To the array: raw INT4 codes plus the scale for the group just finished.
  output logic signed [LANES*`W_BITS-1:0] w_out,
  output logic                          w_vld,
  output logic signed [15:0]            scale_out,
  output logic                          group_done
);

  localparam int BEATS_PER_GROUP = GROUP / LANES;

  logic [$clog2(BEATS_PER_GROUP+1)-1:0] beat;
  logic signed [15:0]                   scale_q;

  assign dram_rdy = 1'b1;   // the array never backpressures; see the note above

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      beat <= '0; w_vld <= 1'b0; group_done <= 1'b0;
      w_out <= '0; scale_q <= 16'sh0100; expert_active <= '0;
    end else begin
      group_done <= 1'b0;

      if (expert_vld) expert_active <= expert_id;
      if (grp_scale_vld) scale_q <= grp_scale;

      w_vld <= dram_vld;
      if (dram_vld) begin
        // Pass codes through untouched -- dequantization happens once per
        // group at the accumulator, not once per weight here.
        w_out <= dram_data;
        if (beat == $bits(beat)'(BEATS_PER_GROUP - 1)) begin
          beat       <= '0;
          group_done <= 1'b1;
        end else begin
          beat <= beat + 1'b1;
        end
      end
    end
  end

  assign scale_out = scale_q;

endmodule
