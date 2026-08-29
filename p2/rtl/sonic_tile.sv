// One 64x64 systolic sub-tile -- a quarter of the 16,384-lane array.
//
// WHY SUB-TILES: measured routing traces (p0/out/real_routing.npz) show a
// monolithic 128x128 array reaching only 0.805 occupancy at chunk 2048, against
// a 0.80 gate. Four independent 64x64 tiles reach 0.888 on the same silicon,
// because a short expert wastes a quarter-sized pass instead of a full one.
//
// TWO DATAFLOWS, one array:
//   MODE_PREFILL  weight-stationary. Weights latch into the PE grid once, then
//                 activations stream through. Each weight is reused TILE times.
//   MODE_DECODE   weight-streaming, output-stationary. Each weight is used once
//                 -- batch-1 GEMV has no reuse to exploit, so holding weights
//                 still would be pointless. Accumulators hold instead, with
//                 ACC_BANKS of them so a speculative verify batch shares one
//                 weight fetch.
//
// The mode register is sampled a stage upstream of the datapath on purpose:
// decode is the mode that ships in volume and must not pay for prefill's
// flexibility on its critical path.
//
// THE REDUCTION IS A PIPELINED TREE, NOT A CHAIN. Each lane reduces T products
// per pass. Written as a sequential `partial += ...` loop over ACC_OUT-wide
// operands -- which is what this module used to be -- that elaborates to T
// dependent 32-bit ripple adders in one combinational cone: 64 deep, at the
// widest accumulator in the design, feeding a single flop. That is the same
// class of defect the router's serial top-k had (2,518 levels, WNS -1086.6 ns
// in p4/RESULTS.md), and no amount of gate resizing closes it, because logic
// depth is not drive strength.
//
// So the reduction is hierarchical, spatially, in exactly the widths sonic_acc
// proves temporally:
//
//   S1  T multiplies, then ACC_FOLD-wide balanced adder trees at ACC_LOCAL.
//       Depth log2(ACC_FOLD) = 4 adders of 16 bits, not 64 of 32.
//   S2  the NFOLD fold results summed at ACC_MID.
//   S3  accumulate into the selected ACC_OUT bank.
//
// The bound is P0-5's and P2-7c's, unchanged: ACC_FOLD * max|w| * max|a| =
// 16 * 8 * 128 = 16,384, which fits ACC_LOCAL = 16 signed bits exactly. The
// old code carried 32 bits through the whole reduction and so paid for range
// it had already proven it could not use.
//
// LATENCY: a pass presented with in_vld high is visible in acc_col three
// cycles later. clr travels down the same pipeline as the data it precedes,
// so clear/accumulate ordering is preserved as issued regardless of depth.
`include "sonic_defs.svh"

module sonic_tile #(
  parameter int T         = `TILE,        // sub-tile edge
  parameter int ACC_BANKS = `ACC_BANKS,
  parameter int ACC_LOCAL = `ACC_LOCAL,
  parameter int ACC_FOLD  = `ACC_FOLD,
  parameter int ACC_MID   = `ACC_MID
) (
  input  logic                        clk,
  input  logic                        rst_n,
  input  logic                        mode,          // `MODE_DECODE / `MODE_PREFILL

  input  logic                        w_load,        // latch the weight plane
  input  logic signed [T*`W_BITS-1:0] w_col,         // one column of weights
  input  logic signed [T*`A_BITS-1:0] a_row,         // one row of activations
  input  logic                        in_vld,
  input  logic                        clr,
  input  logic [$clog2(ACC_BANKS)-1:0] bank,

  output logic signed [T*`ACC_OUT-1:0] acc_col,      // one column of results
  output logic                        out_vld,
  output logic                        ovf
);

  localparam int PROD_BITS = `W_BITS + `A_BITS;          // 12: |w||a| <= 1024
  localparam int NFOLD     = (T + ACC_FOLD - 1) / ACC_FOLD;
  localparam int BANK_W    = $clog2(ACC_BANKS);

  // Weight plane: T x T registers, loaded column by column in prefill mode.
  logic signed [`W_BITS-1:0] wmem [T][T];
  logic [$clog2(T)-1:0]      wcol_ptr;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      wcol_ptr <= '0;
    end else if (w_load) begin
      for (int r = 0; r < T; r++)
        wmem[r][wcol_ptr] <= $signed(w_col[r*`W_BITS +: `W_BITS]);
      wcol_ptr <= wcol_ptr + 1'b1;
    end
  end

  // Registered mode, off the datapath's critical path.
  logic mode_q;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) mode_q <= `MODE_DECODE;
    else        mode_q <= mode;

  // ---------------------------------------------------------------- control
  // clr and bank ride alongside the data so that a clear issued before a pass
  // still lands before it, whatever the pipeline depth.
  logic              vld_s1, vld_s2;
  logic              clr_s1, clr_s2;
  logic [BANK_W-1:0] bank_s1, bank_s2;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      vld_s1 <= 1'b0; vld_s2 <= 1'b0;
      clr_s1 <= 1'b0; clr_s2 <= 1'b0;
      bank_s1 <= '0;  bank_s2 <= '0;
    end else begin
      vld_s1 <= in_vld;  vld_s2 <= vld_s1;
      clr_s1 <= clr;     clr_s2 <= clr_s1;
      bank_s1 <= bank;   bank_s2 <= bank_s1;
    end
  end

  // ---------------------------------------------- S1..S3, one block per lane
  //
  // Everything per-lane lives in ONE generate block and every cross-stage
  // signal is a PACKED vector. That is not a style preference. The first
  // version declared `fold_q [T][NFOLD]` and `mid_q [T]` at module scope and
  // wrote them from T*NFOLD separate always_ff blocks inside the generate --
  // which Yosys reads as one memory with T*NFOLD write ports, and builds the
  // decode logic to match. Measured at T=8: the front end alone (before ABC)
  // went from 36 s to over 300 s, on a module the rewrite was supposed to make
  // cheaper. Packed vectors are inferred as plain registers and the front end
  // returns to a few tens of seconds.
  logic [T-1:0] bank_ovf;

  genvar c, g;
  generate
    for (c = 0; c < T; c++) begin : g_lane

      // -- S1: ACC_FOLD-wide balanced adder trees, one per fold group --------
      logic signed [NFOLD*ACC_LOCAL-1:0] folds;

      for (g = 0; g < NFOLD; g++) begin : g_fold
        // The halving loop unrolls to log2(ACC_FOLD) levels; each level depends
        // only on the one before it, which is what makes it a tree.
        logic signed [ACC_LOCAL-1:0] tree [ACC_FOLD];
        logic signed [ACC_LOCAL-1:0] fold_q;

        always_comb begin
          logic signed [`W_BITS-1:0]   wv;
          logic signed [`A_BITS-1:0]   av;
          logic signed [PROD_BITS-1:0] pv;
          int r;

          for (int i = 0; i < ACC_FOLD; i++) begin
            r = g*ACC_FOLD + i;
            if (r < T) begin
              av = $signed(a_row[r*`A_BITS +: `A_BITS]);
              wv = (mode_q == `MODE_PREFILL) ? wmem[r][c]
                                             : $signed(w_col[r*`W_BITS +: `W_BITS]);
              pv = PROD_BITS'(av) * PROD_BITS'(wv);
            end else begin
              av = '0;
              wv = '0;
              pv = '0;                              // zero-pad a ragged tail
            end
            tree[i] = ACC_LOCAL'(pv);
          end
          for (int sp = 1; sp < ACC_FOLD; sp = sp * 2)
            for (int i = 0; i + sp < ACC_FOLD; i = i + 2*sp)
              tree[i] = tree[i] + tree[i + sp];
        end

        always_ff @(posedge clk or negedge rst_n)
          if (!rst_n)      fold_q <= '0;
          else if (in_vld) fold_q <= tree[0];

        assign folds[g*ACC_LOCAL +: ACC_LOCAL] = fold_q;
      end

      // -- S2: the NFOLD folds, summed as a tree as well ---------------------
      logic signed [ACC_MID-1:0] mtree [NFOLD];
      logic signed [ACC_MID-1:0] mid_q;

      always_comb begin
        for (int gg = 0; gg < NFOLD; gg++)
          mtree[gg] = ACC_MID'($signed(folds[gg*ACC_LOCAL +: ACC_LOCAL]));
        for (int sp = 1; sp < NFOLD; sp = sp * 2)
          for (int gg = 0; gg + sp < NFOLD; gg = gg + 2*sp)
            mtree[gg] = mtree[gg] + mtree[gg + sp];
      end

      always_ff @(posedge clk or negedge rst_n)
        if (!rst_n)      mid_q <= '0;
        else if (vld_s1) mid_q <= mtree[0];

      // -- S3: accumulate into the selected ACC_OUT bank ---------------------
      // ACC_BANKS accumulators per lane: one per speculative candidate in
      // decode, cycling activation tiles in prefill.
      logic signed [`ACC_OUT-1:0] banks [ACC_BANKS];
      logic signed [`ACC_OUT-1:0] cur, add, nxt;

      assign cur = banks[bank_s2];
      assign add = `ACC_OUT'(mid_q);
      assign nxt = cur + add;
      // Same-signed operands producing a differently-signed result is the only
      // signature of signed overflow, and it needs no wider shadow adder.
      assign bank_ovf[c] = vld_s2 & ~clr_s2
                         & (cur[`ACC_OUT-1] == add[`ACC_OUT-1])
                         & (nxt[`ACC_OUT-1] != cur[`ACC_OUT-1]);

      always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
          for (int b = 0; b < ACC_BANKS; b++) banks[b] <= '0;
        end else if (clr_s2) begin
          banks[bank_s2] <= '0;
        end else if (vld_s2) begin
          banks[bank_s2] <= nxt;
        end
      end

      assign acc_col[c*`ACC_OUT +: `ACC_OUT] = banks[bank];
    end
  endgenerate

  // The fold and mid bounds are STRUCTURAL, so they are checked at elaboration
  // and cost nothing in silicon.
  //
  // sonic_acc carries a runtime shadow-sum check because it reduces serially
  // and its fold depth is a runtime counter. Here the depth is ACC_FOLD
  // exactly, spatially, and the operand widths are fixed by the ports: the
  // largest |sum| is ACC_FOLD * 2^(PROD_BITS-1) by construction. So the only
  // way this bound can break is a config edit, and an elaboration error catches
  // that on every build rather than on whichever simulation happens to hit it.
  //
  // The first version of this rewrite checked it at runtime with a shadow sum
  // in ACC_OUT width -- which re-created, in the overflow path, the same
  // 16-deep 32-bit serial adder chain the rewrite existed to delete.
  if (PROD_BITS + $clog2(ACC_FOLD) > ACC_LOCAL) begin : g_bound_local
    $error("ACC_LOCAL=%0d cannot hold ACC_FOLD=%0d products of %0d bits",
           ACC_LOCAL, ACC_FOLD, PROD_BITS);
  end
  if (ACC_LOCAL + $clog2(NFOLD) > ACC_MID) begin : g_bound_mid
    $error("ACC_MID=%0d cannot hold NFOLD=%0d folds of %0d bits",
           ACC_MID, NFOLD, ACC_LOCAL);
  end

  // What is genuinely data-dependent is the ACC_OUT bank: how many passes are
  // accumulated before a read is a firmware decision, not a structural one.
  logic ovf_q;
  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n)         ovf_q <= 1'b0;
    else if (clr_s2)    ovf_q <= 1'b0;
    else if (|bank_ovf) ovf_q <= 1'b1;

  assign ovf = ovf_q;

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) out_vld <= 1'b0;
    else        out_vld <= vld_s2;

endmodule
