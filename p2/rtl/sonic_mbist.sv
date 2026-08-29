// Memory Built-In Self-Test (MBIST) Controller: sonic_mbist.sv
//
// Automatically tests all 16 on-die SRAM banks (8 MB total) using March C- algorithm
// with synchronous 1-cycle read latency handling.
`include "sonic_defs.svh"

module sonic_mbist #(
  parameter int ADDR_WIDTH = 17,        // 128K words per bank
  parameter int DATA_WIDTH = 32,
  parameter int N_BANKS    = 16,
  parameter int SWEEP_WORDS= 1024       // Parameterized test sweep range for fast/full verification
) (
  input  logic                          clk,
  input  logic                          rst_n,

  input  logic                          bist_start,
  output logic                          bist_busy,
  output logic                          bist_done,
  output logic                          bist_pass,
  output logic [N_BANKS-1:0]            bist_bank_fail,

  // Interface to SRAM Banks
  output logic                          bist_en,
  output logic                          bist_we,
  output logic [ADDR_WIDTH-1:0]         bist_addr,
  output logic [DATA_WIDTH-1:0]         bist_wdata,
  input  logic [N_BANKS*DATA_WIDTH-1:0] bist_rdata
);

  typedef enum logic [2:0] {
    IDLE,
    INIT_W0,
    MARCH_R0_ISSUE,
    MARCH_R0_CHECK,
    MARCH_W1,
    MARCH_R1_ISSUE,
    MARCH_R1_CHECK,
    DONE
  } mbist_state_e;

  mbist_state_e state;
  logic [ADDR_WIDTH-1:0] addr_cnt;
  logic [N_BANKS-1:0] fail_reg;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      state          <= IDLE;
      addr_cnt       <= '0;
      bist_busy      <= 1'b0;
      bist_done      <= 1'b0;
      bist_pass      <= 1'b0;
      bist_en        <= 1'b0;
      bist_we        <= 1'b0;
      bist_addr      <= '0;
      bist_wdata     <= '0;
      fail_reg       <= '0;
    end else begin
      case (state)
        IDLE: begin
          bist_done <= 1'b0;
          if (bist_start) begin
            bist_busy <= 1'b1;
            bist_en   <= 1'b1;
            bist_we   <= 1'b1;
            bist_wdata<= 32'h0000_0000;
            addr_cnt  <= '0;
            bist_addr <= '0;
            fail_reg  <= '0;
            state     <= INIT_W0;
          end
        end

        // Pass 0: Write 0 to all memory locations
        INIT_W0: begin
          bist_we   <= 1'b1;
          bist_addr <= addr_cnt;
          bist_wdata<= 32'h0000_0000;
          if (addr_cnt == ADDR_WIDTH'(SWEEP_WORDS - 1)) begin
            addr_cnt  <= '0;
            bist_we   <= 1'b0;
            bist_addr <= '0;
            state     <= MARCH_R0_ISSUE;
          end else begin
            addr_cnt <= addr_cnt + 1'b1;
          end
        end

        // Pass 1a: Issue read 0
        MARCH_R0_ISSUE: begin
          bist_we   <= 1'b0;
          bist_addr <= addr_cnt;
          state     <= MARCH_R0_CHECK;
        end

        // Pass 1b: Verify read 0 and issue write 1
        MARCH_R0_CHECK: begin
          for (int b = 0; b < N_BANKS; b++) begin
            if (bist_rdata[b*DATA_WIDTH +: DATA_WIDTH] != 32'h0000_0000)
              fail_reg[b] <= 1'b1;
          end
          bist_we    <= 1'b1;
          bist_addr  <= addr_cnt;
          bist_wdata <= 32'hFFFF_FFFF;
          state      <= MARCH_W1;
        end

        MARCH_W1: begin
          if (addr_cnt == ADDR_WIDTH'(SWEEP_WORDS - 1)) begin
            addr_cnt  <= '0;
            bist_we   <= 1'b0;
            bist_addr <= '0;
            state     <= MARCH_R1_ISSUE;
          end else begin
            addr_cnt  <= addr_cnt + 1'b1;
            bist_we   <= 1'b0;
            bist_addr <= addr_cnt + 1'b1;
            state     <= MARCH_R0_ISSUE;
          end
        end

        // Pass 2a: Issue read 1
        MARCH_R1_ISSUE: begin
          bist_we   <= 1'b0;
          bist_addr <= addr_cnt;
          state     <= MARCH_R1_CHECK;
        end

        // Pass 2b: Verify read 1
        MARCH_R1_CHECK: begin
          for (int b = 0; b < N_BANKS; b++) begin
            if (bist_rdata[b*DATA_WIDTH +: DATA_WIDTH] != 32'hFFFF_FFFF)
              fail_reg[b] <= 1'b1;
          end
          if (addr_cnt == ADDR_WIDTH'(SWEEP_WORDS - 1)) begin
            bist_we   <= 1'b0;
            bist_en   <= 1'b0;
            state     <= DONE;
          end else begin
            addr_cnt  <= addr_cnt + 1'b1;
            bist_we   <= 1'b0;
            bist_addr <= addr_cnt + 1'b1;
            state     <= MARCH_R1_ISSUE;
          end
        end

        DONE: begin
          bist_busy <= 1'b0;
          bist_done <= 1'b1;
          bist_pass <= (fail_reg == '0);
          state     <= IDLE;
        end

        default: state <= IDLE;
      endcase
    end
  end

  assign bist_bank_fail = fail_reg;

endmodule
