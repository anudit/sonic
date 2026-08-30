// Embedded RV32I RISC-V Controller: sonic_rv32.sv
//
// A real (if simple) single-cycle RV32I integer core -- fetch/decode/execute/
// writeback every cycle, with a multi-cycle stall only for accesses that miss
// internal IMEM/DMEM and have to go out over the NoC master bus. Used for
// system boot, DMA sequencing, and mailbox communication with the host
// (SPI/I3C). No M/F/C extension, no CSR/interrupt controller beyond the
// mailbox IRQ already wired at the top: this is deliberately the smallest
// core that actually executes a program, not a mailbox register file wearing
// a core's docstring.
`include "sonic_defs.svh"

module sonic_rv32 #(
  parameter int IMEM_DEPTH = 2048,      // words -> 8 KB Instruction Memory
  parameter int DMEM_DEPTH = 2048       // words -> 8 KB Data Memory
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // Boot-load port: writes IMEM directly (mirrors how a real chip's boot ROM
  // / JTAG / mailbox-driven DMA would stage a program before release-from-reset).
  input  logic                          imem_ld_en,
  input  logic [$clog2(IMEM_DEPTH)-1:0] imem_ld_addr,
  input  logic [31:0]                   imem_ld_data,

  // External Host Mailbox / Control Interface
  input  logic                          host_wr_en,
  input  logic [7:0]                    host_wr_addr,
  input  logic [31:0]                   host_wr_data,
  output logic [31:0]                   host_rd_data,
  output logic                          host_irq,

  // System DMA & Sequencer Trigger
  output logic                          seq_kick,
  output logic [31:0]                   seq_start_pc,
  input  logic                          seq_done_irq,

  // NoC Master Bus Interface: word address >= DMEM_DEPTH is routed here.
  output logic                          noc_m_req,
  output logic                          noc_m_we,
  output logic [31:0]                   noc_m_addr,
  output logic [31:0]                   noc_m_wdata,
  input  logic [31:0]                   noc_m_rdata,
  input  logic                          noc_m_ack,

  // Debug/verification observability (read-only; not part of the ISA -- a
  // real chip would expose this over JTAG, not a bench-only port)
  output logic [31:0]                   dbg_pc,
  output logic                          dbg_trap,
  input  logic [4:0]                    dbg_rf_addr,
  output logic [31:0]                   dbg_rf_data,
  input  logic [$clog2(DMEM_DEPTH)-1:0] dbg_dmem_addr,
  output logic [31:0]                   dbg_dmem_data
);

  localparam int DMEM_BYTES = DMEM_DEPTH * 4;

  // -----------------------------------------------------------------------
  // State
  // -----------------------------------------------------------------------
  logic [31:0] imem [IMEM_DEPTH];
  logic [31:0] dmem [DMEM_DEPTH];
  logic [31:0] rf   [32];
  logic [31:0] pc;
  logic [31:0] mbox [16];
  logic [31:0] csr_status;
  logic        trap;

  typedef enum logic [1:0] {ST_FETCH, ST_NOC_WAIT, ST_HALT} state_t;
  state_t state;

  // -----------------------------------------------------------------------
  // Fetch / decode (combinational on the instruction at `pc`)
  // -----------------------------------------------------------------------
  logic [31:0] instr;
  assign instr = imem[pc[$clog2(IMEM_DEPTH)+1:2]];

  wire [6:0] opcode = instr[6:0];
  wire [4:0] rd     = instr[11:7];
  wire [2:0] funct3 = instr[14:12];
  wire [4:0] rs1    = instr[19:15];
  wire [4:0] rs2    = instr[24:20];
  wire       funct7_b5 = instr[30]; // the only funct7 bit RV32I decode needs

  wire [31:0] imm_i = {{20{instr[31]}}, instr[31:20]};
  wire [31:0] imm_s = {{20{instr[31]}}, instr[31:25], instr[11:7]};
  wire [31:0] imm_b = {{19{instr[31]}}, instr[31], instr[7], instr[30:25], instr[11:8], 1'b0};
  wire [31:0] imm_u = {instr[31:12], 12'b0};
  wire [31:0] imm_j = {{11{instr[31]}}, instr[31], instr[19:12], instr[20], instr[30:21], 1'b0};

  wire [31:0] rv1 = (rs1 == 5'd0) ? 32'd0 : rf[rs1];
  wire [31:0] rv2 = (rs2 == 5'd0) ? 32'd0 : rf[rs2];

  localparam logic [6:0] OP_LUI    = 7'b0110111;
  localparam logic [6:0] OP_AUIPC  = 7'b0010111;
  localparam logic [6:0] OP_JAL    = 7'b1101111;
  localparam logic [6:0] OP_JALR   = 7'b1100111;
  localparam logic [6:0] OP_BRANCH = 7'b1100011;
  localparam logic [6:0] OP_LOAD   = 7'b0000011;
  localparam logic [6:0] OP_STORE  = 7'b0100011;
  localparam logic [6:0] OP_IMM    = 7'b0010011;
  localparam logic [6:0] OP_REG    = 7'b0110011;
  localparam logic [6:0] OP_MISC   = 7'b0001111;
  localparam logic [6:0] OP_SYSTEM = 7'b1110011;

  // ALU (shared by OP-IMM and OP)
  wire        alu_is_imm = (opcode == OP_IMM);
  wire [31:0] alu_b      = alu_is_imm ? imm_i : rv2;
  wire [2:0]  alu_f3     = funct3;
  wire        alu_sub    = (opcode == OP_REG) && funct7_b5;
  wire        alu_arith_shift = funct7_b5;
  logic [31:0] alu_y;

  always_comb begin
    unique case (alu_f3)
      3'b000:  alu_y = alu_sub ? (rv1 - alu_b) : (rv1 + alu_b);
      3'b001:  alu_y = rv1 << alu_b[4:0];
      3'b010:  alu_y = {31'b0, $signed(rv1) < $signed(alu_b)};
      3'b011:  alu_y = {31'b0, rv1 < alu_b};
      3'b100:  alu_y = rv1 ^ alu_b;
      3'b101:  alu_y = alu_arith_shift ? ($signed(rv1) >>> alu_b[4:0]) : (rv1 >> alu_b[4:0]);
      3'b110:  alu_y = rv1 | alu_b;
      3'b111:  alu_y = rv1 & alu_b;
      default: alu_y = 32'b0;
    endcase
  end

  // Branch condition
  logic branch_taken;
  always_comb begin
    unique case (funct3)
      3'b000:  branch_taken = (rv1 == rv2);
      3'b001:  branch_taken = (rv1 != rv2);
      3'b100:  branch_taken = ($signed(rv1) < $signed(rv2));
      3'b101:  branch_taken = ($signed(rv1) >= $signed(rv2));
      3'b110:  branch_taken = (rv1 < rv2);
      3'b111:  branch_taken = (rv1 >= rv2);
      default: branch_taken = 1'b0;
    endcase
  end

  // Load/store address and internal-vs-NoC routing
  wire [31:0] mem_addr    = (opcode == OP_LOAD) ? (rv1 + imm_i) : (rv1 + imm_s);
  wire        mem_is_ld   = (opcode == OP_LOAD);
  wire        mem_is_st   = (opcode == OP_STORE);
  wire        mem_is_op   = mem_is_ld | mem_is_st;
  wire        mem_ext     = mem_is_op && (mem_addr >= DMEM_BYTES);
  wire [$clog2(DMEM_DEPTH)-1:0] dmem_word_addr = mem_addr[$clog2(DMEM_DEPTH)+1:2];

  logic [31:0] dmem_rword;
  assign dmem_rword = dmem[dmem_word_addr];

  logic [31:0] ld_data_int, ld_data;
  always_comb begin
    unique case (funct3)
      3'b000:  ld_data_int = {{24{dmem_rword[{mem_addr[1:0],3'b0}+:8][7]}}, dmem_rword[{mem_addr[1:0],3'b0}+:8]};
      3'b001:  ld_data_int = {{16{dmem_rword[{mem_addr[1],4'b0}+:16][15]}}, dmem_rword[{mem_addr[1],4'b0}+:16]};
      3'b010:  ld_data_int = dmem_rword;
      3'b100:  ld_data_int = {24'b0, dmem_rword[{mem_addr[1:0],3'b0}+:8]};
      3'b101:  ld_data_int = {16'b0, dmem_rword[{mem_addr[1],4'b0}+:16]};
      default: ld_data_int = dmem_rword;
    endcase
  end
  assign ld_data = mem_ext ? noc_m_rdata : ld_data_int;

  // Writeback source select
  logic [31:0] wb_data;
  always_comb begin
    unique case (opcode)
      OP_LUI:            wb_data = imm_u;
      OP_AUIPC:          wb_data = pc + imm_u;
      OP_JAL, OP_JALR:   wb_data = pc + 32'd4;
      OP_LOAD:           wb_data = ld_data;
      OP_IMM, OP_REG:    wb_data = alu_y;
      OP_MISC:           wb_data = 32'b0; // FENCE: treated as nop
      default:           wb_data = 32'b0;
    endcase
  end
  wire rd_writes = ((opcode==OP_LUI)||(opcode==OP_AUIPC)||(opcode==OP_JAL)||(opcode==OP_JALR)||(opcode==OP_LOAD)||(opcode==OP_IMM)||(opcode==OP_REG)) && (rd != 5'd0);

  wire [31:0] next_pc_seq  = pc + 32'd4;
  wire [31:0] next_pc_jal  = pc + imm_j;
  wire [31:0] next_pc_jalr = (rv1 + imm_i) & 32'hFFFF_FFFE;
  wire [31:0] next_pc_br   = pc + imm_b;

  logic [31:0] next_pc;
  always_comb begin
    unique case (opcode)
      OP_JAL:    next_pc = next_pc_jal;
      OP_JALR:   next_pc = next_pc_jalr;
      OP_BRANCH: next_pc = branch_taken ? next_pc_br : next_pc_seq;
      default:   next_pc = next_pc_seq;
    endcase
  end

  wire mem_write_now = mem_is_st && !mem_ext; // internal store completes same cycle

  // -----------------------------------------------------------------------
  // Sequencing
  // -----------------------------------------------------------------------
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      pc           <= 32'h0000_0000;
      state        <= ST_FETCH;
      seq_kick     <= 1'b0;
      seq_start_pc <= '0;
      host_irq     <= 1'b0;
      noc_m_req    <= 1'b0;
      noc_m_we     <= 1'b0;
      noc_m_addr   <= '0;
      noc_m_wdata  <= '0;
      csr_status   <= '0;
      trap         <= 1'b0;
      for (int i = 0; i < 16; i++) mbox[i] <= '0;
      for (int i = 0; i < 32; i++) rf[i]   <= '0;
    end else begin
      seq_kick <= 1'b0;
      host_irq <= seq_done_irq;
      if (seq_done_irq) csr_status[0] <= 1'b1;

      if (host_wr_en && host_wr_addr < 16) begin
        mbox[host_wr_addr[3:0]] <= host_wr_data;
        if (host_wr_addr == 0 && host_wr_data[0]) begin
          seq_kick     <= 1'b1;
          seq_start_pc <= mbox[1];
        end
      end

      unique case (state)
        ST_FETCH: begin
          if (opcode == OP_SYSTEM) begin
            // ECALL/EBREAK: halt the core rather than silently no-op-ing --
            // a real trap controller is out of scope, but pretending nothing
            // happened is not.
            trap  <= 1'b1;
            state <= ST_HALT;
          end else if (mem_ext) begin
            noc_m_req   <= 1'b1;
            noc_m_we    <= mem_is_st;
            noc_m_addr  <= mem_addr;
            noc_m_wdata <= rv2;
            state       <= ST_NOC_WAIT;
          end else begin
            if (rd_writes)     rf[rd] <= wb_data;
            if (mem_write_now) begin
              unique case (funct3)
                3'b000: dmem[dmem_word_addr][{mem_addr[1:0],3'b0}+:8]  <= rv2[7:0];
                3'b001: dmem[dmem_word_addr][{mem_addr[1],4'b0}+:16]   <= rv2[15:0];
                default: dmem[dmem_word_addr] <= rv2;
              endcase
            end
            pc <= next_pc;
          end
        end
        ST_NOC_WAIT: begin
          if (noc_m_ack) begin
            noc_m_req <= 1'b0;
            if (rd_writes) rf[rd] <= ld_data; // ld_data reflects noc_m_rdata while mem_ext
            pc    <= next_pc;
            state <= ST_FETCH;
          end
        end
        ST_HALT: begin
          // core stays parked until reset; mailbox/IRQ path above still runs.
        end
        default: state <= ST_FETCH;
      endcase
    end
  end

  always_ff @(posedge clk) begin
    if (imem_ld_en) imem[imem_ld_addr] <= imem_ld_data;
  end

  assign host_rd_data = (host_wr_addr < 16) ? mbox[host_wr_addr[3:0]] : csr_status;
  assign dbg_rf_data   = (dbg_rf_addr == 5'd0) ? 32'd0 : rf[dbg_rf_addr];
  assign dbg_dmem_data = dmem[dbg_dmem_addr];
  assign dbg_pc   = pc;
  assign dbg_trap = trap;

endmodule
