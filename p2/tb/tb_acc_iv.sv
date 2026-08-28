// Icarus Verilog testbench for the hierarchical accumulator.
//
// A second, independent simulator running the same RTL. Verilator and Icarus
// have different SystemVerilog front ends and different scheduling
// implementations, so agreement between them is real evidence the design does
// not depend on one tool's interpretation. Icarus also dumps VCD for GTKWave
// without a C++ harness.
//
//   iverilog -g2012 -I p2/rtl -o build/tb_acc_iv p2/tb/tb_acc_iv.sv p2/rtl/sonic_acc.sv
//   ./build/tb_acc_iv && gtkwave build/acc.vcd
`timescale 1ns/1ps
`include "sonic_defs.svh"

module tb_acc_iv;

  logic clk = 0, rst_n = 0, clr = 0, en = 0, flush = 0;
  logic signed [`W_BITS-1:0] w = 0;
  logic signed [`A_BITS-1:0] a = 0;
  logic signed [`ACC_OUT-1:0] acc;
  logic ovf;

  int fails = 0;

  sonic_acc dut (.clk(clk), .rst_n(rst_n), .clr(clr), .en(en),
                 .w(w), .a(a), .flush(flush), .acc(acc), .ovf(ovf));

  always #5 clk = ~clk;

  task automatic reset();
    rst_n = 0; clr = 0; en = 0; @(posedge clk); @(posedge clk);
    rst_n = 1; @(posedge clk);
  endtask

  // One dot product of length K with constant operands, checked against the
  // closed form. Sweeping K is the point: the fold boundary and the ragged
  // tail are where this class of design breaks.
  task automatic dot_const(input int K, input int wv, input int av,
                           output logic signed [`ACC_OUT-1:0] result);
    clr = 1; en = 0; @(posedge clk);
    clr = 0; en = 1; w = wv[`W_BITS-1:0]; a = av[`A_BITS-1:0];
    repeat (K) @(posedge clk);
    en = 0; @(posedge clk);
    result = acc;
  endtask

  logic signed [`ACC_OUT-1:0] r;
  int ovf_at;

  initial begin
    $dumpfile("build/acc.vcd");
    $dumpvars(0, tb_acc_iv);

    $display("Icarus Verilog: sonic_acc");
    $display("  ACC_LOCAL=%0d ACC_FOLD=%0d ACC_MID=%0d ACC_OUT=%0d",
             `ACC_LOCAL, `ACC_FOLD, `ACC_MID, `ACC_OUT);
    reset();

    // Every reduction length across two fold boundaries.
    for (int K = 1; K <= 64; K++) begin
      dot_const(K, 3, 5, r);
      if (r !== K * 15) begin
        $display("  FAIL K=%0d: got %0d want %0d", K, r, K * 15);
        fails++;
      end
    end
    $display("  ragged reduction 1..64: %s", fails ? "FAIL" : "ok");

    // Worst-case operands must not overflow at any depth. This is the property
    // that forced ACC_LOCAL from 12 to 16.
    begin
      ovf_at = -1;
      for (int K = 1; K <= 512; K++) begin
        dot_const(K, -8, -128, r);
        if (ovf && ovf_at < 0) ovf_at = K;
      end
      if (ovf_at >= 0) begin
        $display("  FAIL: overflow at K=%0d on legal INT4 x INT8", ovf_at);
        fails++;
      end else begin
        $display("  worst-case INT4 x INT8 to K=512: no overflow  ok");
      end
    end

    // clr must clear a partial local stage, not just the folded total.
    dot_const(37, 7, 100, r);
    dot_const(5, 1, 1, r);
    if (r !== 5) begin
      $display("  FAIL: leak across clr, got %0d want 5", r);
      fails++;
    end else $display("  clr isolation: ok");

    $display("\n%s (%0d failures)", fails ? "FAILED" : "PASSED", fails);
    $display("wrote build/acc.vcd -- open with: gtkwave build/acc.vcd");
    $finish;
  end

endmodule
