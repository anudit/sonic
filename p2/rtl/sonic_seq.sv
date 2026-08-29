// Descriptor-ring sequencer.
//
// There is no graph compiler on this chip. One model, one static schedule: the
// offline packer emits a ring of descriptors and this block replays it per
// token. Twenty-four layers times a handful of descriptors each.
//
// The one runtime-variable field is the expert base address, which the router
// patches between the routing decision and the fetch. Everything else -- loop
// bounds, dimensions, modes -- is baked at pack time, which is what lets the
// same silicon run both LFM2.5-8B-A1B and the dense 2.6B.
//
// Every dimension arrives here rather than being hardwired in a datapath. That
// is a design rule, not a convenience: it is the whole reason two models with
// different layer counts, attention ratios and RoPE thetas share one die.
`include "sonic_defs.svh"

module sonic_seq #(
  parameter int DW    = 64,      // descriptor width
  parameter int DEPTH = 512      // supports full 301-descriptor ring for 8B model
) (
  input  logic              clk,
  input  logic              rst_n,

  input  logic              wr_en,       // firmware loads the ring
  input  logic [$clog2(DEPTH)-1:0] wr_addr,
  input  logic [DW-1:0]     wr_data,

  input  logic              start,
  input  logic [$clog2(DEPTH)-1:0] len,

  // Router patch: applied to the next descriptor whose opcode is EXPERT.
  input  logic [`EXPERT_BITS-1:0] patch_expert,
  input  logic              patch_vld,

  output logic [DW-1:0]     desc,
  output logic              desc_vld,
  output logic              busy,
  output logic              done
);

  // Opcode lives in the top nibble.
  localparam logic [3:0] OP_EXPERT = 4'h3;

  logic [DW-1:0] ring [DEPTH];
  logic [$clog2(DEPTH)-1:0] pc;
  logic [`EXPERT_BITS-1:0]  patch_q;

  always_ff @(posedge clk) if (wr_en) ring[wr_addr] <= wr_data;

  // `desc` is REGISTERED alongside desc_vld, not driven combinationally from
  // ring[pc].
  //
  // The first version did the latter, and p2/tb/tb_seq.cpp caught what that
  // costs: desc_vld is set in the same cycle pc increments, so the k-th cycle
  // with desc_vld high presented ring[k+1]. Descriptor 0 was never issued at
  // all, every descriptor ran one slot early against its own valid, and the
  // last valid cycle read ring[len] -- past the end of the program, emitting
  // whatever firmware happened to leave in the next ring slot.
  //
  // Nothing downstream can detect this. Every descriptor is well-formed; they
  // are just the wrong ones, and the fetch they issue is a correct fetch of the
  // wrong tensor. It is the exact failure class the streamer bench pinned a
  // contract for one block over: right data, wrong cycle.
  logic [DW-1:0] raw, patched;
  assign raw     = ring[pc];
  assign patched = (raw[DW-1:DW-4] == OP_EXPERT)
                 ? {raw[DW-1:`EXPERT_BITS], patch_q}
                 : raw;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pc <= '0; busy <= 1'b0; desc_vld <= 1'b0; done <= 1'b0;
      patch_q <= '0; desc <= '0;
    end else begin
      done     <= 1'b0;
      desc_vld <= 1'b0;

      if (patch_vld) patch_q <= patch_expert;

      // len == 0 is a firmware error. Ignoring the start beats replaying the
      // whole ring, which is what `pc == len - 1` would do once len-1 wraps to
      // DEPTH-1.
      if (start && len != '0) begin
        pc <= '0; busy <= 1'b1;
      end else if (busy) begin
        // Splice the routed expert in on the way out, so the fetch issues in
        // the same cycle it would have without MoE. The router is a 2048->32
        // GEMV and its decision is invisible against the 343 us it takes to
        // stream one layer's experts, so no prefetch bubble appears here.
        // patch_q is read before this cycle's patch_vld lands, so a patch
        // arriving with a descriptor applies from the NEXT descriptor.
        desc_vld <= 1'b1;
        desc     <= patched;
        if (pc == len - 1) begin
          busy <= 1'b0;
          done <= 1'b1;
        end
        pc <= pc + 1'b1;
      end
    end
  end

endmodule
