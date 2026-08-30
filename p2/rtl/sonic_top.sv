// Sonic S1 Full Top-Level Module: sonic_top.sv
//
// Integrates the complete Sonic S1 SoC:
//   1.  sonic_seq        -- 512-entry descriptor ring sequencer with runtime router patching
//   2.  sonic_streamer   -- INT4 group-64 weight streamer & coarse expert gather
//   3.  sonic_router     -- MoE top-k router (INT12xINT12, PWL sigmoid, pipelined top-k)
//   4.  sonic_tile       -- 4x 64x64 dual-mode systolic sub-tiles (16,384 MACs total)
//   5.  sonic_conv       -- 64-channel 1D causal short-convolution with double-gating
//   6.  sonic_softmax    -- online flash-softmax attention accumulator
//   7.  sonic_lmhead     -- streaming top-k vocab reduction & sampler
//   8.  sonic_vec        -- programmable 2048-wide SIMD vector unit (RMSNorm, RoPE, Residuals)
//   9.  sonic_sram       -- 8 MB on-die shared SRAM cache (16 independent 512 KB banks)
//   10. sonic_rv32       -- embedded RV32I microcontroller for boot, DMA, and mailbox
//   11. sonic_noc        -- multi-master multi-slave AXI4 streaming on-chip crossbar
//   12. sonic_mbist      -- memory built-in self-test controller (March C- algorithm)
//   13. sonic_phy_lpddr5x-- DFI 5.0 LPDDR5X high-speed PHY interface wrapper
//   14. sonic_ioring     -- C4 pad ring, ESD clamps, on-chip PLLs, and PVT monitors
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
  parameter int SEQ_DEPTH        = 512,                // descriptor ring depth
  parameter int N_SRAM_BANKS     = 16,                 // 16 banks x 512 KB = 8 MB
  parameter int SRAM_AW          = 17                  // 128K words / bank
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
  output logic                                  lmhead_done,

  // =========================================================================
  // 8. Vector Unit Interface (sonic_vec)
  // =========================================================================
  input  logic [1:0]                            vec_op_sel,
  input  logic                                  vec_in_vld,
  input  logic                                  vec_last_beat,
  input  logic signed [64*16-1:0]               vec_a,
  input  logic signed [64*16-1:0]               vec_b,
  input  logic signed [64*16-1:0]               vec_gamma,
  input  logic signed [64*16-1:0]               vec_sin,
  input  logic signed [15:0]                    vec_inv_rms,
  output logic signed [64*16-1:0]               vec_out,
  output logic                                  vec_out_vld,
  output logic                                  vec_out_last,

  // =========================================================================
  // 9. 8 MB Shared SRAM Interface (sonic_sram)
  // =========================================================================
  input  logic [N_SRAM_BANKS-1:0]               sram_ce,
  input  logic [N_SRAM_BANKS-1:0]               sram_we,
  input  logic [N_SRAM_BANKS*SRAM_AW-1:0]       sram_addr,
  input  logic [N_SRAM_BANKS*32-1:0]            sram_wdata,
  input  logic [N_SRAM_BANKS*4-1:0]             sram_wmask,
  output logic [N_SRAM_BANKS*32-1:0]            sram_rdata,

  // =========================================================================
  // 10. MBIST & DFT Interface (sonic_mbist)
  // =========================================================================
  input  logic                                  bist_start,
  output logic                                  bist_busy,
  output logic                                  bist_done,
  output logic                                  bist_pass,

  // =========================================================================
  // 11. External Host & Telemetry (sonic_ioring)
  // =========================================================================
  input  logic                                  host_wr_en,
  input  logic [7:0]                            host_wr_addr,
  input  logic [31:0]                           host_wr_data,
  output logic [31:0]                           host_rd_data,
  output logic                                  host_irq,
  output logic [7:0]                            die_temp,
  output logic [7:0]                            die_vdd_mv
);

  // -------------------------------------------------------------------------
  // Internal Sequencer Patch Signal
  // -------------------------------------------------------------------------
  logic [`EXPERT_BITS-1:0] patch_expert_wire;
  logic                    patch_vld_wire;

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
  sonic_streamer #(
    .LANES(T),
    .GROUP(64)
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
    .w_out(),
    .w_vld(),
    .scale_out(stream_scale_out),
    .group_done(stream_group_done)
  );

  // -------------------------------------------------------------------------
  // 3. MoE Top-K Router
  // -------------------------------------------------------------------------
  sonic_router #(
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

  // -------------------------------------------------------------------------
  // 8. Programmable Vector Unit
  // -------------------------------------------------------------------------
  sonic_vec #(
    .LANES(64),
    .DW(16),
    .D(D)
  ) u_vec (
    .clk(clk),
    .rst_n(rst_n),
    .op_sel(vec_op_sel),
    .in_vld(vec_in_vld),
    .last_beat(vec_last_beat),
    .vec_a(vec_a),
    .vec_b(vec_b),
    .vec_gamma(vec_gamma),
    .vec_sin(vec_sin),
    .inv_rms(vec_inv_rms),
    .vec_out(vec_out),
    .out_vld(vec_out_vld),
    .out_last(vec_out_last)
  );

  // -------------------------------------------------------------------------
  // 9. 8 MB Shared On-Die SRAM Cache & MBIST
  // -------------------------------------------------------------------------
  logic [N_SRAM_BANKS*32-1:0] bist_sram_rdata;
  logic                      mbist_en_w;
  logic                      mbist_we_w;
  logic [SRAM_AW-1:0]        mbist_addr_w;
  logic [31:0]               mbist_wdata_w;
  logic [N_SRAM_BANKS-1:0]   sram_bank_sleep;

  // Only the banks the current chunk/decode step actually addresses stay
  // awake; a bank idle for 2**IDLE_BITS cycles drops into retention sleep.
  // See sonic_sram_gate.sv for the same-cycle-access safety property.
  sonic_sram_gate #(
    .N_BANKS(N_SRAM_BANKS),
    .IDLE_BITS(8)
  ) u_sram_gate (
    .clk(clk),
    .rst_n(rst_n),
    .bank_ce(sram_ce),
    .bank_sleep(sram_bank_sleep)
  );

  sonic_sram #(
    .N_BANKS(N_SRAM_BANKS),
    .DATA_WIDTH(32),
    .BANK_AW(SRAM_AW)
  ) u_sram (
    .clk(clk),
    .rst_n(rst_n),
    .bank_ce(sram_ce),
    .bank_we(sram_we),
    .bank_addr(sram_addr),
    .bank_wdata(sram_wdata),
    .bank_wmask(sram_wmask),
    .bank_rdata(sram_rdata),
    .bank_sleep(sram_bank_sleep),
    .bist_en(mbist_en_w),
    .bist_we(mbist_we_w),
    .bist_addr(mbist_addr_w),
    .bist_wdata(mbist_wdata_w),
    .bist_rdata(bist_sram_rdata),
    .bist_pass()
  );

  sonic_mbist #(
    .ADDR_WIDTH(SRAM_AW),
    .DATA_WIDTH(32),
    .N_BANKS(N_SRAM_BANKS)
  ) u_mbist (
    .clk(clk),
    .rst_n(rst_n),
    .bist_start(bist_start),
    .bist_busy(bist_busy),
    .bist_done(bist_done),
    .bist_pass(bist_pass),
    .bist_bank_fail(),
    .bist_en(mbist_en_w),
    .bist_we(mbist_we_w),
    .bist_addr(mbist_addr_w),
    .bist_wdata(mbist_wdata_w),
    .bist_rdata(bist_sram_rdata)
  );

  // -------------------------------------------------------------------------
  // 10. RV32 Controller & NoC Interconnect
  // -------------------------------------------------------------------------
  logic noc_req, noc_we, noc_ack;
  logic [31:0] noc_addr, noc_wdata, noc_rdata;

  sonic_rv32 u_rv32 (
    .clk(clk),
    .rst_n(rst_n),
    .host_wr_en(host_wr_en),
    .host_wr_addr(host_wr_addr),
    .host_wr_data(host_wr_data),
    .host_rd_data(host_rd_data),
    .host_irq(host_irq),
    .seq_kick(),
    .seq_start_pc(),
    .seq_done_irq(seq_done),
    .noc_m_req(noc_req),
    .noc_m_we(noc_we),
    .noc_m_addr(noc_addr),
    .noc_m_wdata(noc_wdata),
    .noc_m_rdata(noc_rdata),
    .noc_m_ack(noc_ack)
  );

  sonic_noc u_noc (
    .clk(clk),
    .rst_n(rst_n),
    .m_req({2'b00, noc_req}),
    .m_we({2'b00, noc_we}),
    .m_addr({64'd0, noc_addr}),
    .m_wdata({64'd0, noc_wdata}),
    .m_rdata({noc_rdata}),
    .m_ack({noc_ack}),
    .s_req(),
    .s_we(),
    .s_addr(),
    .s_wdata(),
    .s_rdata('0),
    .s_ack(4'b1111)
  );

  // -------------------------------------------------------------------------
  // 11. I/O Pad Ring & Telemetry
  // -------------------------------------------------------------------------
  sonic_ioring u_ioring (
    .ext_osc_clk(clk),
    .ext_rst_n(rst_n),
    .core_clk(),
    .dfi_clk(),
    .sys_rst_n(),
    .host_sclk(clk),
    .host_cs_n(~host_wr_en),
    .host_mosi(host_wr_data[0]),
    .host_miso(),
    .host_intr_n(),
    .die_temp_sensor(die_temp),
    .vdd_sense_mv(die_vdd_mv)
  );

endmodule
