// Sonic S1 Top-Level Module: sonic_top.sv
//
// Instantiates and integrates the core Sonic S1 datapath and control blocks:
//   1. sonic_seq      -- 512-entry descriptor ring sequencer with router patch port
//   2. sonic_streamer -- INT4 group-64 weight streamer & expert gather
//   3. sonic_router   -- MoE top-k router (INT12xINT12, PWL sigmoid, pipelined top-k)
//   4. sonic_tile     -- 4x 64x64 dual-mode systolic sub-tiles (16,384 MACs total)
//   5. sonic_conv     -- 64-channel 1D causal short-convolution with double-gating
//   6. sonic_softmax  -- online flash-softmax attention accumulator
//   7. sonic_lmhead   -- streaming top-k vocab reduction & sampler
//
// Enables end-to-end layer and multi-token execution under unified sequencer control.
`include "sonic_defs.svh"

module sonic_top #(
  parameter int T                = `TILE,              // sub-tile edge (64)
  parameter int N_TILES          = `N_TILES,           // 4 sub-tiles = 16,384 lanes
  parameter int ACC_BANKS        = `ACC_BANKS,         // 8 accumulator banks
  parameter int ACC_LOCAL        = `ACC_LOCAL,         // 16-bit local stage
  parameter int ACC_FOLD         = `ACC_FOLD,          // 16 products/fold
  parameter int ACC_MID          = `ACC_MID,           // 24-bit intermediate
  parameter int D                = 2048,               // hidden dimension
  parameter int E                = `N_EXPERTS,         // 32 experts
  parameter int K                = `TOP_K,             // top-4 experts
  parameter int ROUTER_LANES     = `ROUTER_LANES,      // router MAC lanes (64)
  parameter int ROUTER_PWL_SEGS  = `ROUTER_PWL_SEGS,   // 32 segments
  parameter int ROUTER_PWL_RANGE = `ROUTER_PWL_RANGE,  // +-4 logit range
  parameter int CONV_CH          = `CONV_CH,           // 64 conv channels
  parameter int CONV_KMAX        = 7,                  // max conv kernel size
  parameter int LMHEAD_K         = 64,                 // LM head top-k candidates
  parameter int LMHEAD_VW        = 17,                 // clog2(128000 vocab)
  parameter int SEQ_DW           = 64,                 // descriptor width
  parameter int SEQ_DEPTH        = 512                 // descriptor ring depth
) (
  input  logic                                  clk,
  input  logic                                  rst_n,

  // =========================================================================
  // 1. Sequencer Interface (sonic_seq)
  // =========================================================================
  input  logic                                  seq_wr_en,
  input  logic [$clog2(SEQ_DEPTH)-1:0]          seq_wr_addr,
  input  logic [SEQ_DW-1:0]                     seq_wr_data,
  input  logic                                  seq_start,
  input  logic [$clog2(SEQ_DEPTH)-1:0]          seq_len,
  output logic [SEQ_DW-1:0]                     desc_out,
  output logic                                  desc_vld_out,
  output logic                                  seq_busy,
  output logic                                  seq_done,

  // =========================================================================
  // 2. Weight Streamer & DRAM Interface (sonic_streamer)
  // =========================================================================
  input  logic [T*`W_BITS-1:0]                  dram_data,
  input  logic                                  dram_vld,
  output logic                                  dram_rdy,
  input  logic signed [15:0]                    grp_scale,
  input  logic                                  grp_scale_vld,
  output logic signed [15:0]                    stream_scale_out,
  output logic                                  stream_group_done,
  output logic [`EXPERT_BITS-1:0]               expert_active,

  // =========================================================================
  // 3. MoE Top-K Router Interface (sonic_router)
  // =========================================================================
  input  logic                                  router_start,
  input  logic signed [31:0]                    router_logit_scale,
  input  logic signed [E*32-1:0]                router_bias,
  input  logic signed [ROUTER_LANES*`ROUTER_A_BITS-1:0] router_a_in,
  input  logic signed [ROUTER_LANES*`ROUTER_W_BITS-1:0] router_w_in,
  input  logic                                  router_in_vld,
  input  logic                                  router_tbl_we,
  input  logic [$clog2(ROUTER_PWL_SEGS)-1:0]   router_tbl_addr,
  input  logic signed [31:0]                    router_tbl_slope,
  input  logic signed [31:0]                    router_tbl_icept,
  output logic                                  router_done,
  output logic [K*$clog2(E)-1:0]                router_sel,
  output logic signed [E*32-1:0]                router_score,

  // =========================================================================
  // 4. Systolic Sub-Tiles Interface (4x sonic_tile = 16,384 MACs)
  // =========================================================================
  input  logic                                  tile_mode,
  input  logic [N_TILES-1:0]                    tile_w_load,
  input  logic signed [N_TILES*T*`W_BITS-1:0]   tile_w_col,
  input  logic signed [T*`A_BITS-1:0]           tile_a_row,
  input  logic [N_TILES-1:0]                    tile_in_vld,
  input  logic [N_TILES-1:0]                    tile_clr,
  input  logic [$clog2(ACC_BANKS)-1:0]          tile_bank,
  output logic signed [N_TILES*T*`ACC_OUT-1:0]  tile_acc_col,
  output logic [N_TILES-1:0]                    tile_out_vld,
  output logic [N_TILES-1:0]                    tile_ovf,

  // =========================================================================
  // 5. Short-Conv Unit Interface (sonic_conv)
  // =========================================================================
  input  logic [$clog2(CONV_KMAX+1)-1:0]        conv_k_taps,
  input  logic                                  conv_state_clr,
  input  logic signed [CONV_CH*`A_BITS-1:0]     conv_x_in,
  input  logic signed [CONV_CH*`A_BITS-1:0]     conv_c_gate,
  input  logic signed [CONV_KMAX*8-1:0]         conv_kern,
  input  logic                                  conv_in_vld,
  output logic signed [CONV_CH*`A_BITS-1:0]     conv_y_out,
  output logic                                  conv_out_vld,

  // =========================================================================
  // 6. Online Softmax Attention Interface (sonic_softmax)
  // =========================================================================
  input  logic                                  softmax_clr,
  input  logic signed [31:0]                    softmax_score,
  input  logic signed [31:0]                    softmax_value,
  input  logic                                  softmax_in_vld,
  output logic signed [31:0]                    softmax_exp_arg,
  input  logic signed [31:0]                    softmax_exp_val,
  output logic signed [31:0]                    softmax_run_max,
  output logic signed [31:0]                    softmax_run_sum,
  output logic signed [31:0]                    softmax_acc,

  // =========================================================================
  // 7. Streaming LM Head Interface (sonic_lmhead)
  // =========================================================================
  input  logic                                  lmhead_clr,
  input  logic signed [31:0]                    lmhead_logit,
  input  logic [LMHEAD_VW-1:0]                  lmhead_token_id,
  input  logic                                  lmhead_in_vld,
  input  logic                                  lmhead_last,
  input  logic signed [31:0]                    lmhead_prune_bound,
  output logic                                  lmhead_exact,
  output logic [LMHEAD_K*LMHEAD_VW-1:0]         lmhead_top_id,
  output logic [LMHEAD_K*32-1:0]                lmhead_top_score,
  output logic                                  lmhead_done
);

  // -------------------------------------------------------------------------
  // Internal Sequencer Patch Signal
  // -------------------------------------------------------------------------
  logic [`EXPERT_BITS-1:0] patch_expert_wire;
  logic                    patch_vld_wire;

  // The router's top expert selection (sel[0]) patches the sequencer and streamer
  assign patch_expert_wire = router_sel[`EXPERT_BITS-1:0];
  assign patch_vld_wire    = router_done;

  // -------------------------------------------------------------------------
  // 1. Descriptor Ring Sequencer
  // -------------------------------------------------------------------------
  sonic_seq #(
    .DW(SEQ_DW),
    .DEPTH(SEQ_DEPTH)
  ) u_seq (
    .clk(clk),
    .rst_n(rst_n),
    .wr_en(seq_wr_en),
    .wr_addr(seq_wr_addr),
    .wr_data(seq_wr_data),
    .start(seq_start),
    .len(seq_len),
    .patch_expert(patch_expert_wire),
    .patch_vld(patch_vld_wire),
    .desc(desc_out),
    .desc_vld(desc_vld_out),
    .busy(seq_busy),
    .done(seq_done)
  );

  // -------------------------------------------------------------------------
  // 2. Weight Streamer
  // -------------------------------------------------------------------------
  /* verilator lint_off UNUSEDSIGNAL */
  logic signed [T*`W_BITS-1:0] stream_w_out;
  logic                        stream_w_vld;
  /* verilator lint_on UNUSEDSIGNAL */

  sonic_streamer #(
    .LANES(T),
    .GROUP(`GROUP)
  ) u_streamer (
    .clk(clk),
    .rst_n(rst_n),
    .dram_data(dram_data),
    .dram_vld(dram_vld),
    .dram_rdy(dram_rdy),
    .grp_scale(grp_scale),
    .grp_scale_vld(grp_scale_vld),
    .expert_id(patch_expert_wire),
    .expert_vld(patch_vld_wire),
    .expert_active(expert_active),
    .w_out(stream_w_out),
    .w_vld(stream_w_vld),
    .scale_out(stream_scale_out),
    .group_done(stream_group_done)
  );

  // -------------------------------------------------------------------------
  // 3. MoE Top-K Router
  // -------------------------------------------------------------------------
  sonic_router #(
    .D(D),
    .E(E),
    .K(K),
    .AW(`ROUTER_A_BITS),
    .WW(`ROUTER_W_BITS),
    .LANES(ROUTER_LANES),
    .SEGS(ROUTER_PWL_SEGS),
    .RANGE(ROUTER_PWL_RANGE)
  ) u_router (
    .clk(clk),
    .rst_n(rst_n),
    .tbl_we(router_tbl_we),
    .tbl_addr(router_tbl_addr),
    .tbl_slope(router_tbl_slope),
    .tbl_icept(router_tbl_icept),
    .start(router_start),
    .logit_scale(router_logit_scale),
    .bias(router_bias),
    .a_in(router_a_in),
    .w_in(router_w_in),
    .in_vld(router_in_vld),
    .done(router_done),
    .sel(router_sel),
    .score(router_score)
  );

  // -------------------------------------------------------------------------
  // 4. Systolic Sub-Tiles (4 x 64x64 = 16,384 MACs)
  // -------------------------------------------------------------------------
  genvar tile_idx;
  generate
    for (tile_idx = 0; tile_idx < N_TILES; tile_idx++) begin : g_tiles
      // Tile weight input: can come either from direct tile_w_col or streamer
      logic signed [T*`W_BITS-1:0] w_col_mux;
      assign w_col_mux = tile_w_col[tile_idx*T*`W_BITS +: T*`W_BITS];

      sonic_tile #(
        .T(T),
        .ACC_BANKS(ACC_BANKS),
        .ACC_LOCAL(ACC_LOCAL),
        .ACC_FOLD(ACC_FOLD),
        .ACC_MID(ACC_MID)
      ) u_tile (
        .clk(clk),
        .rst_n(rst_n),
        .mode(tile_mode),
        .w_load(tile_w_load[tile_idx]),
        .w_col(w_col_mux),
        .a_row(tile_a_row),
        .in_vld(tile_in_vld[tile_idx]),
        .clr(tile_clr[tile_idx]),
        .bank(tile_bank),
        .acc_col(tile_acc_col[tile_idx*T*`ACC_OUT +: T*`ACC_OUT]),
        .out_vld(tile_out_vld[tile_idx]),
        .ovf(tile_ovf[tile_idx])
      );
    end
  endgenerate

  // -------------------------------------------------------------------------
  // 5. Short-Conv Unit
  // -------------------------------------------------------------------------
  sonic_conv #(
    .CH(CONV_CH),
    .KMAX(CONV_KMAX),
    .DW(`A_BITS)
  ) u_conv (
    .clk(clk),
    .rst_n(rst_n),
    .k_taps(conv_k_taps),
    .state_clr(conv_state_clr),
    .x_in(conv_x_in),
    .c_gate(conv_c_gate),
    .kern(conv_kern),
    .in_vld(conv_in_vld),
    .y_out(conv_y_out),
    .out_vld(conv_out_vld)
  );

  // -------------------------------------------------------------------------
  // 6. Online Softmax Unit
  // -------------------------------------------------------------------------
  sonic_softmax #(
    .DW(32)
  ) u_softmax (
    .clk(clk),
    .rst_n(rst_n),
    .clr(softmax_clr),
    .score(softmax_score),
    .value(softmax_value),
    .in_vld(softmax_in_vld),
    .exp_arg(softmax_exp_arg),
    .exp_val(softmax_exp_val),
    .run_max(softmax_run_max),
    .run_sum(softmax_run_sum),
    .acc(softmax_acc)
  );

  // -------------------------------------------------------------------------
  // 7. Streaming LM Head Unit
  // -------------------------------------------------------------------------
  sonic_lmhead #(
    .K(LMHEAD_K),
    .VW(LMHEAD_VW),
    .SW(32)
  ) u_lmhead (
    .clk(clk),
    .rst_n(rst_n),
    .clr(lmhead_clr),
    .logit(lmhead_logit),
    .token_id(lmhead_token_id),
    .in_vld(lmhead_in_vld),
    .last(lmhead_last),
    .prune_bound(lmhead_prune_bound),
    .exact(lmhead_exact),
    .top_id(lmhead_top_id),
    .top_score(lmhead_top_score),
    .done(lmhead_done)
  );

endmodule
