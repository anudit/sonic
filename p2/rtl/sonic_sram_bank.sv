// Synchronous 512 KB SRAM Bank Macro Wrapper: sonic_sram_bank.sv
//
// Represents a hardened 512 KB SRAM memory macro block (or OpenRAM macro array)
// used across the 8 MB on-die chunk cache (16 banks total) and local scratchpads.
//
// Features:
//   - 1-cycle synchronous read / write
//   - Byte-level write strobes
//   - Low-power chip enable & sleep retention mode
//   - Integrated MBIST (Memory Built-In Self-Test) test ports
`include "sonic_defs.svh"

module sonic_sram_bank #(
  parameter int DATA_WIDTH = 32,
  parameter int ADDR_WIDTH = 17,    // 128K words x 32 bits = 512 KB (4 Mb)
  parameter int BYTE_WIDTH = 8
) (
  input  logic                      clk,
  input  logic                      rst_n,

  // Normal Functional Port
  input  logic                      ce,            // Chip Enable (active high)
  input  logic                      we,            // Write Enable (active high)
  input  logic [ADDR_WIDTH-1:0]     addr,
  input  logic [DATA_WIDTH-1:0]     wdata,
  input  logic [DATA_WIDTH/BYTE_WIDTH-1:0] wmask,
  output logic [DATA_WIDTH-1:0]     rdata,

  // Low-Power Sleep Control
  input  logic                      sleep,         // Retention sleep mode

  // MBIST Test Interface
  input  logic                      bist_en,
  input  logic                      bist_we,
  input  logic [ADDR_WIDTH-1:0]     bist_addr,
  input  logic [DATA_WIDTH-1:0]     bist_wdata,
  output logic [DATA_WIDTH-1:0]     bist_rdata
);

  localparam int DEPTH = 1 << ADDR_WIDTH;
  localparam int NUM_BYTES = DATA_WIDTH / BYTE_WIDTH;

  // Active control signals muxed between functional and BIST modes
  logic                  mux_ce;
  logic                  mux_we;
  logic [ADDR_WIDTH-1:0] mux_addr;
  logic [DATA_WIDTH-1:0] mux_wdata;

  assign mux_ce   = bist_en ? 1'b1      : (ce & ~sleep);
  assign mux_we   = bist_en ? bist_we   : (we & ~sleep);
  assign mux_addr = bist_en ? bist_addr : addr;
  assign mux_wdata= bist_en ? bist_wdata: wdata;

  // Memory Array Model (Maps to Foundry SRAM Compiler Macro during Physical Design)
  logic [DATA_WIDTH-1:0] mem [DEPTH];
  logic [DATA_WIDTH-1:0] rdata_reg;

  always_ff @(posedge clk) begin
    if (mux_ce) begin
      if (mux_we) begin
        for (int b = 0; b < NUM_BYTES; b++) begin
          if (bist_en || wmask[b]) begin
            mem[mux_addr][b*BYTE_WIDTH +: BYTE_WIDTH] <= mux_wdata[b*BYTE_WIDTH +: BYTE_WIDTH];
          end
        end
      end
      rdata_reg <= mem[mux_addr];
    end
  end

  assign rdata      = (!sleep && !bist_en) ? rdata_reg : '0;
  assign bist_rdata = bist_en ? rdata_reg : '0;

endmodule
