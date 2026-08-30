// Per-bank SRAM idle-sleep gate: sonic_sram_gate.sv
//
// sonic_sram_bank.sv has always supported per-bank retention sleep (the
// `sleep` port), but sonic_top.sv tied sonic_sram's bank_sleep input to '0 --
// every one of the 16 512 KB banks stays awake regardless of whether the
// current chunk touches it. This module is the missing control: one idle
// counter per bank, sleep asserted only after a bank has gone unaddressed
// for 2**IDLE_BITS cycles, so only the banks the current chunk/decode step
// actually touches stay powered.
//
// Safety property, not just intent: `bank_sleep[b]` is ANDed with
// `!bank_ce[b]` combinationally, so a real access this cycle always forces
// sleep low THIS cycle regardless of the counter's state -- an access is
// never dropped by a stale sleep bit. sonic_sram_bank.sv's own
// `mux_ce = bist_en ? 1 : (ce & ~sleep)` is what makes this safe: sleep only
// ever masks a cycle with no real ce asserted.
`include "sonic_defs.svh"

module sonic_sram_gate #(
  parameter int N_BANKS  = 16,
  parameter int IDLE_BITS = 8     // banks sleep after 2**IDLE_BITS idle cycles
) (
  input  logic                 clk,
  input  logic                 rst_n,
  input  logic [N_BANKS-1:0]   bank_ce,
  output logic [N_BANKS-1:0]   bank_sleep
);

  localparam logic [IDLE_BITS-1:0] IDLE_MAX = '1;

  logic [IDLE_BITS-1:0] idle_cnt [N_BANKS];

  genvar b;
  generate
    for (b = 0; b < N_BANKS; b++) begin : g_bank
      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          idle_cnt[b] <= '0;
        end else if (bank_ce[b]) begin
          idle_cnt[b] <= '0;
        end else if (idle_cnt[b] != IDLE_MAX) begin
          idle_cnt[b] <= idle_cnt[b] + 1'b1;
        end
      end

      assign bank_sleep[b] = (idle_cnt[b] == IDLE_MAX) && !bank_ce[b];
    end
  endgenerate

endmodule
