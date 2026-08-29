// 8 MB On-Die Shared SRAM Cache: sonic_sram.sv
//
// Integrates 16 independent 512 KB hard SRAM banks (8 MB total).
//
// In prefill mode:
//   Holds a 2048-token chunk residual stream on-die, allowing each expert to be
//   fetched from DRAM once per chunk rather than once per token (reducing DRAM traffic 6.2x).
// In decode mode:
//   Provides zero-latency staging buffers for speculative verify batches, KV pages,
//   and intermediate activation states.
`include "sonic_defs.svh"

module sonic_sram #(
  parameter int N_BANKS    = 16,        // 16 banks x 512 KB = 8 MB total
  parameter int DATA_WIDTH = 32,
  parameter int BANK_AW    = 17,        // 128K words per bank
  parameter int BYTE_WIDTH = 8
) (
  input  logic                                  clk,
  input  logic                                  rst_n,

  // Multi-Bank Functional Ports (one per bank or crossbar routed)
  input  logic [N_BANKS-1:0]                    bank_ce,
  input  logic [N_BANKS-1:0]                    bank_we,
  input  logic [N_BANKS*BANK_AW-1:0]            bank_addr,
  input  logic [N_BANKS*DATA_WIDTH-1:0]         bank_wdata,
  input  logic [N_BANKS*(DATA_WIDTH/BYTE_WIDTH)-1:0] bank_wmask,
  output logic [N_BANKS*DATA_WIDTH-1:0]         bank_rdata,

  // Bank Power Management (Per-Bank Sleep Retention)
  input  logic [N_BANKS-1:0]                    bank_sleep,

  // Global MBIST Test Interface
  input  logic                                  bist_en,
  input  logic                                  bist_we,
  input  logic [BANK_AW-1:0]                    bist_addr,
  input  logic [DATA_WIDTH-1:0]                 bist_wdata,
  output logic [N_BANKS*DATA_WIDTH-1:0]         bist_rdata,
  output logic                                  bist_pass
);

  localparam int NUM_BYTES = DATA_WIDTH / BYTE_WIDTH;
  logic [N_BANKS-1:0] bank_match;

  genvar b;
  generate
    for (b = 0; b < N_BANKS; b++) begin : g_sram_banks
      logic [DATA_WIDTH-1:0] rdata_w;
      logic [DATA_WIDTH-1:0] bist_rdata_w;

      sonic_sram_bank #(
        .DATA_WIDTH(DATA_WIDTH),
        .ADDR_WIDTH(BANK_AW),
        .BYTE_WIDTH(BYTE_WIDTH)
      ) u_bank (
        .clk(clk),
        .rst_n(rst_n),
        .ce(bank_ce[b]),
        .we(bank_we[b]),
        .addr(bank_addr[b*BANK_AW +: BANK_AW]),
        .wdata(bank_wdata[b*DATA_WIDTH +: DATA_WIDTH]),
        .wmask(bank_wmask[b*NUM_BYTES +: NUM_BYTES]),
        .rdata(rdata_w),
        .sleep(bank_sleep[b]),
        .bist_en(bist_en),
        .bist_we(bist_we),
        .bist_addr(bist_addr),
        .bist_wdata(bist_wdata),
        .bist_rdata(bist_rdata_w)
      );

      assign bank_rdata[b*DATA_WIDTH +: DATA_WIDTH] = rdata_w;
      assign bist_rdata[b*DATA_WIDTH +: DATA_WIDTH] = bist_rdata_w;
      assign bank_match[b] = (bist_rdata_w == bist_wdata);
    end
  endgenerate

  // MBIST verification signature
  assign bist_pass = bist_en & (&bank_match);

endmodule
