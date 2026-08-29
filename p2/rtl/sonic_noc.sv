// On-Chip Network (NoC) Crossbar Interconnect: sonic_noc.sv
//
// Multi-master, multi-slave low-latency on-chip crossbar interconnecting:
//   Masters:
//     M0: RV32 Core / Mailbox Controller
//     M1: Weight Streamer / DRAM DMA
//     M2: Sequencer / Router Control
//   Slaves:
//     S0: Systolic Tile Array
//     S1: 8 MB Shared SRAM Cache (16 banks)
//     S2: Programmable Vector Unit
//     S3: Output LM Head Unit
//
// Every master decodes its own target slave from its own address (top 2
// bits); every slave picks among the masters requesting it that cycle with a
// round-robin arbiter, so two masters contending for the same slave get
// serviced fairly instead of one silently starving the other. A master whose
// request is not granted this cycle simply sees m_ack low and must hold
// m_req/m_addr/m_wdata steady, which is the contract sonic_rv32's NoC master
// port and the streamer/sequencer masters already assume.
`include "sonic_defs.svh"

module sonic_noc #(
  parameter int N_MASTERS = 3,
  parameter int N_SLAVES  = 4,
  parameter int ADDR_W    = 32,
  parameter int DATA_W    = 32
) (
  input  logic                                  clk,
  input  logic                                  rst_n,

  // Masters Interface
  input  logic [N_MASTERS-1:0]                  m_req,
  input  logic [N_MASTERS-1:0]                  m_we,
  input  logic [N_MASTERS*ADDR_W-1:0]           m_addr,
  input  logic [N_MASTERS*DATA_W-1:0]           m_wdata,
  output logic [N_MASTERS*DATA_W-1:0]           m_rdata,
  output logic [N_MASTERS-1:0]                  m_ack,

  // Slaves Interface
  output logic [N_SLAVES-1:0]                   s_req,
  output logic [N_SLAVES-1:0]                   s_we,
  output logic [N_SLAVES*ADDR_W-1:0]            s_addr,
  output logic [N_SLAVES*DATA_W-1:0]            s_wdata,
  input  logic [N_SLAVES*DATA_W-1:0]            s_rdata,
  input  logic [N_SLAVES-1:0]                   s_ack
);

  // Address decoding (top bits select slave): 0x0.. tiles, 0x1.. SRAM,
  // 0x2.. vector unit, 0x3.. LM head. Same map every master uses.
  localparam int SEL_W = $clog2(N_SLAVES);
  function automatic logic [SEL_W-1:0] decode(input logic [ADDR_W-1:0] addr);
    /* verilator lint_off UNUSEDSIGNAL */
    return addr[ADDR_W-1 -: SEL_W];
    /* verilator lint_on UNUSEDSIGNAL */
  endfunction

  logic [SEL_W-1:0] m_target [N_MASTERS];
  always_comb
    for (int m = 0; m < N_MASTERS; m++)
      m_target[m] = decode(m_addr[m*ADDR_W +: ADDR_W]);

  // Per-slave round-robin priority pointer: index of the master with top
  // priority this cycle. Advances to (granted_master + 1) after any grant.
  logic [$clog2(N_MASTERS)-1:0] rr_ptr [N_SLAVES];

  logic [N_MASTERS-1:0] grant [N_SLAVES]; // one-hot per slave

  always_comb begin
    for (int s = 0; s < N_SLAVES; s++) begin
      grant[s] = '0;
      for (int off = 0; off < N_MASTERS; off++) begin
        automatic int m = (int'(rr_ptr[s]) + off) % N_MASTERS;
        if (grant[s] == '0 && m_req[m] && (m_target[m] == s[SEL_W-1:0]))
          grant[s][m] = 1'b1;
      end
    end
  end

  always_comb begin
    s_req   = '0;
    s_we    = '0;
    s_addr  = '0;
    s_wdata = '0;
    m_rdata = '0;
    m_ack   = '0;

    for (int s = 0; s < N_SLAVES; s++) begin
      for (int m = 0; m < N_MASTERS; m++) begin
        if (grant[s][m]) begin
          s_req[s]                    = 1'b1;
          s_we[s]                     = m_we[m];
          s_addr[s*ADDR_W +: ADDR_W]  = m_addr[m*ADDR_W +: ADDR_W];
          s_wdata[s*DATA_W +: DATA_W] = m_wdata[m*DATA_W +: DATA_W];
          m_rdata[m*DATA_W +: DATA_W] = s_rdata[s*DATA_W +: DATA_W];
          m_ack[m]                    = s_ack[s];
        end
      end
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      for (int s = 0; s < N_SLAVES; s++) rr_ptr[s] <= '0;
    end else begin
      for (int s = 0; s < N_SLAVES; s++) begin
        for (int m = 0; m < N_MASTERS; m++) begin
          if (grant[s][m] && s_ack[s])
            rr_ptr[s] <= $clog2(N_MASTERS)'((m + 1) % N_MASTERS);
        end
      end
    end
  end

endmodule
