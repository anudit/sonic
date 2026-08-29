// MoE top-k router.
//
//   logits  = W . h                       INT12 x INT12 -> INT32
//   weights = sigmoid(logits * scale)     16-segment PWL, firmware-loadable
//   scores  = weights + expert_bias       aux-loss-free load balancing
//   select  = top_k(scores)
//
// Matches transformers' Lfm2MoeTopKRouter.forward, which returns
// (router_logits, routing_weights, selected_experts).
//
// WIDTH: INT12, not the BF16 the first draft of the plan specified and not the
// INT4 the rest of the chip uses. Measured on 512 real hidden states from
// layer 5 of the 8.47 B checkpoint (p3/export_vectors.py):
//
//     act \ w      INT4     INT8    INT12    INT16
//     INT8       0.7598   0.9707   0.9727   0.9707
//     INT12      0.7598   0.9863   0.9980   0.9961
//     INT16      0.7598   0.9863   1.0000   1.0000
//
// INT8 weights cap at 0.986 no matter how wide the activations are, and INT4
// is catastrophic at 0.76. INT12 x INT12 is the cheapest point that clears the
// 0.995 routing-agreement gate. A mis-routed token produces fluent output from
// the wrong expert, so perplexity never catches this.
//
// Consequence for the floorplan: the router CANNOT consume the shared INT8
// activation bus. It needs a wider tap, taken before the INT8 requantization.
`include "sonic_defs.svh"

module sonic_router #(
  parameter int D       = 2048,
  parameter int E       = `N_EXPERTS,
  parameter int K       = `TOP_K,
  parameter int AW      = `ROUTER_A_BITS,   // activation width
  parameter int WW      = `ROUTER_W_BITS,   // weight width
  parameter int LANES   = `ROUTER_LANES,    // MACs per cycle
  // PWL sigmoid resolution. Segment select stays a bare shift for any
  // power-of-two SEGS. `make p2-pwl-sweep` sizes SEGS x RANGE against measured
  // routing agreement.
  parameter int SEGS    = `ROUTER_PWL_SEGS,
  // Table spans [-RANGE, RANGE). Power of two, so segment select stays a shift.
  parameter int RANGE   = `ROUTER_PWL_RANGE,
  parameter int SEGSH   = $clog2(SEGS),
  parameter int SHIFT   = 16 + $clog2(2*RANGE) - SEGSH
) (
  input  logic                    clk,
  input  logic                    rst_n,

  // 16-segment PWL sigmoid table, loaded by firmware.
  input  logic                    tbl_we,
  input  logic [$clog2(SEGS)-1:0] tbl_addr,
  input  logic signed [31:0]      tbl_slope,   // Q16
  input  logic signed [31:0]      tbl_icept,   // Q16

  input  logic                    start,       // begin a token
  // Q32, not Q16: act_scale * w_scale is ~1e-6 for INT12 operands, which
  // rounds to ZERO in Q16 and silently zeroes every logit. The first version of
  // this port was Q16 and the RTL scored 0.0000 against the model.
  //
  // 32 bits, not 64: SystemVerilog sizes a multiply from its operands, so a
  // 64-bit scale port makes the epilogue a 64x64 multiplier -- ~4096 partial
  // products, and Yosys/abc will not finish synthesising it. 32x32 -> 64 is one
  // ordinary multiplier and is all the precision the sigmoid input needs.
  input  logic signed [31:0]      logit_scale, // Q32

  // Ports are PACKED vectors, not unpacked arrays. Verilator and Icarus accept
  // unpacked array ports; Yosys does not, so an unpacked interface simulates
  // perfectly and then cannot be synthesized. Packing here keeps one set of
  // sources across simulation, synthesis and place-and-route.
  input  logic signed [E*32-1:0]      bias,    // Q16, expert-major
  input  logic signed [LANES*AW-1:0]  a_in,    // LANES elements per cycle
  input  logic signed [LANES*WW-1:0]  w_in,
  input  logic                        in_vld,

  output logic                        done,
  output logic [K*$clog2(E)-1:0]      sel,
  output logic signed [E*32-1:0]      score    // Q16, post-sigmoid + bias
);

  localparam int STEPS = D / LANES;
  localparam int EW    = $clog2(E);

  // --- PWL table --------------------------------------------------------
  logic signed [31:0] slope [SEGS], icept [SEGS];
  always_ff @(posedge clk) if (tbl_we) begin
    slope[tbl_addr] <= tbl_slope;
    icept[tbl_addr] <= tbl_icept;
  end

  // --- pipeline ----------------------------------------------------------
  // Everything below this line used to be ONE combinational cone hanging off a
  // single always_ff: the LANES-wide MAC tree, the 32x32 epilogue scale, the
  // segment select, the second 32x32 PWL multiply, and then K serially
  // dependent tournament passes over the 1024-bit score register. P4 measured
  // what that costs -- WNS -1086.6 ns at a 10 ns clock on the sel[] path -- and
  // the depth probe put it at 2,518 levels before the tournament-tree fix and
  // 461 after. 461 levels is still ~35-45 ns at 14 nm against a 1 ns period.
  //
  // The tree fix removed redundant depth. The rest is real work, and the only
  // way to shorten real work is to put registers through it. Nothing here
  // changes what the block computes -- p2-router still checks the same
  // selections against the 8.47 B checkpoint -- it changes only when the
  // answer appears, and `done` is what says so.
  //
  //   S0  input beat            step / e_idx advance
  //   S1  MAC                   d1_dot = sum(a_in * w_in)
  //   S2  accumulate            acc += d1_dot; on the last beat, hand off
  //   E1  epilogue scale        e1_scaled = acc_final * logit_scale
  //   E2  segment select        e2_slope/e2_icept = table[seg(x)]
  //   E3  PWL + bias            score[e]
  //   T*  top-k                 2 cycles per k, K*2 cycles total
  //
  // Cost: ~13 cycles of added latency per token. The router decides once per
  // token and the layer it gates takes 343 us to stream, so this is invisible.

  // --- S0: sequencing ----------------------------------------------------
  logic [EW-1:0]              e_idx;
  logic [$clog2(STEPS+1)-1:0] step;
  logic                       busy;
  logic                       beat, last_beat;

  assign beat      = busy && in_vld;
  assign last_beat = beat && (step == $bits(step)'(STEPS - 1));

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; e_idx <= '0; step <= '0;
    end else if (start) begin
      busy <= 1'b1; e_idx <= '0; step <= '0;
    end else if (beat) begin
      if (last_beat) begin
        step <= '0;
        if (e_idx == EW'(E - 1)) busy  <= 1'b0;
        else                     e_idx <= e_idx + 1'b1;
      end else begin
        step <= step + 1'b1;
      end
    end
  end

  // --- S1: the MAC tree, registered --------------------------------------
  logic signed [`ACC_OUT-1:0] dot;
  // S1's MAC is a balanced tree, not a running sum.
  //
  // It used to be `dot += ACC_OUT'(av) * ACC_OUT'(wv)` over LANES iterations,
  // which elaborates to LANES dependent 32-bit adders in one cone. At the
  // LANES=4 used for the P4 experiments that cost 4 levels and hid; at the
  // shipping LANES=64 it is 64, and it sits directly ahead of the top-k network
  // whose depth finding 16 is about. Finding 16 named "the score/PWL datapath
  // ahead of them" as the remaining depth after the tournament tree -- this is
  // that datapath.
  //
  // Width follows the operands rather than the output register: a product is
  // AW+WW bits and LANES of them need $clog2(LANES) more, so the tree is DOTW
  // wide, not ACC_OUT. Sign-extending once at the end is free; carrying 32 bits
  // through every level is not.
  localparam int RPROD = AW + WW;
  localparam int DOTW  = RPROD + $clog2(LANES);

  logic signed [DOTW-1:0] dtree [LANES];

  always_comb begin
    logic signed [AW-1:0] av;
    logic signed [WW-1:0] wv;
    for (int i = 0; i < LANES; i++) begin
      av = a_in[i*AW +: AW];
      wv = w_in[i*WW +: WW];
      dtree[i] = DOTW'(RPROD'(av) * RPROD'(wv));
    end
    for (int sp = 1; sp < LANES; sp = sp * 2)
      for (int i = 0; i + sp < LANES; i = i + 2*sp)
        dtree[i] = dtree[i] + dtree[i + sp];
    dot = `ACC_OUT'(dtree[0]);
  end

  logic signed [`ACC_OUT-1:0] d1_dot;
  logic                       d1_vld, d1_last;
  logic [EW-1:0]              d1_e;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      d1_vld <= 1'b0; d1_last <= 1'b0; d1_dot <= '0; d1_e <= '0;
    end else begin
      d1_vld  <= beat;
      d1_last <= last_beat;
      d1_dot  <= dot;
      d1_e    <= e_idx;
    end
  end

  // --- S2: accumulate ----------------------------------------------------
  logic signed [`ACC_OUT-1:0] acc;
  logic signed [`ACC_OUT-1:0] a2_val;
  logic                       a2_vld;
  logic [EW-1:0]              a2_e;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      acc <= '0; a2_vld <= 1'b0; a2_val <= '0; a2_e <= '0;
    end else begin
      a2_vld <= 1'b0;
      if (start) begin
        acc <= '0;
      end else if (d1_vld) begin
        if (d1_last) begin
          a2_val <= acc + d1_dot;
          a2_e   <= d1_e;
          a2_vld <= 1'b1;
          acc    <= '0;
        end else begin
          acc <= acc + d1_dot;
        end
      end
    end
  end

  // --- E1: fold both operand scales in -----------------------------------
  // Both operands are 32-bit and the lvalue is 64-bit, which is the idiom that
  // gets a 32x32 -> 64 multiply rather than a truncated one.
  logic signed [63:0] e1_scaled;
  logic               e1_vld;
  logic [EW-1:0]      e1_e;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      e1_vld <= 1'b0; e1_scaled <= '0; e1_e <= '0;
    end else begin
      e1_vld    <= a2_vld;
      e1_scaled <= $signed(a2_val) * $signed(logit_scale);
      e1_e      <= a2_e;
    end
  end

  // --- E2: segment select ------------------------------------------------
  // Shift by 16, not 32. acc is an integer and logit_scale is Q32, so the
  // product is the true logit scaled by 2^32; >>> 32 yields the INTEGER logit
  // and >>> 16 yields Q16, which is what the PWL expects. Shifting 32 here made
  // every logit round to ~0, every sigmoid come out at 0.5, and the RTL score
  // 0.0000 against the model.
  //
  // Segment select is a bare shift for any power-of-two SEGS and RANGE: map
  // [-RANGE, RANGE) onto [0, SEGS) by adding RANGE in Q16 and shifting right by
  // 16 + log2(2*RANGE) - log2(SEGS).
  logic signed [31:0] e2_x, e2_slope, e2_icept;
  logic               e2_vld;
  logic [EW-1:0]      e2_e;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      e2_vld <= 1'b0; e2_x <= '0; e2_slope <= '0; e2_icept <= '0; e2_e <= '0;
    end else begin
      logic signed [31:0] x;
      logic signed [31:0] xs;
      int                 idx;
      x  = 32'(e1_scaled >>> 16);
      xs = (x + (RANGE <<< 16)) >>> SHIFT;
      idx = xs;
      if (idx < 0)        idx = 0;
      if (idx > SEGS - 1) idx = SEGS - 1;

      e2_vld   <= e1_vld;
      e2_x     <= x;
      e2_slope <= slope[idx];
      e2_icept <= icept[idx];
      e2_e     <= e1_e;
    end
  end

  // --- E3: PWL evaluate + expert bias ------------------------------------
  logic score_last;

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      score <= '0; score_last <= 1'b0;
    end else begin
      score_last <= e2_vld && (e2_e == EW'(E - 1));
      if (e2_vld)
        score[e2_e*32 +: 32] <= 32'(($signed(e2_x) * $signed(e2_slope)) >>> 16)
                                + e2_icept
                                + $signed(bias[e2_e*32 +: 32]);
    end
  end

  // --- top-K selection ---------------------------------------------------
  // K passes over E scores, ties broken toward the lowest index. A tournament
  // tree and a linear scan both compare every element exactly once, so both
  // cost E-1 comparators; the tree's dependency chain is log2(E) deep instead
  // of E-1. The serial version this replaced paid 31 dependent 32-bit compares
  // per pass for no area saving at all -- see p2/README.md finding 16.
  //
  // Exclusion uses an E-bit `taken` mask rather than a writable copy of the
  // scores: 32 flops instead of 1024, and the mask folds into the leaf mux.
  localparam logic signed [31:0] NEG_INF = 32'sh8000_0000;
  localparam int CAND = 4;   // candidates carried between the two half-cycles

`ifndef ROUTER_TOPK_COMB
  // Pipelined: two cycles per pass. Cycle A reduces E -> CAND (log2(E/CAND)
  // dependent compares), cycle B reduces CAND -> 1 and commits sel[k]. The
  // split exists because one full 5-level chain of 32-bit compares was itself
  // becoming the critical path once the epilogue stopped dominating.
  typedef enum logic [1:0] { TK_IDLE, TK_A, TK_B } tk_e;
  tk_e                tk_state;
  logic [E-1:0]       taken;
  logic [$clog2(K+1)-1:0] kcnt;
  logic signed [31:0] cand_v [CAND];
  logic [EW-1:0]      cand_x [CAND];

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      tk_state <= TK_IDLE; taken <= '0; kcnt <= '0; sel <= '0; done <= 1'b0;
    end else begin
      logic signed [31:0] v [E];
      logic [EW-1:0]      x [E];
      done <= 1'b0;

      // A `start` mid-selection abandons the pass in flight. The scores it was
      // reading are about to be overwritten by the new token, so finishing it
      // would emit a selection over a mix of two tokens' scores -- plausible,
      // wrong, and undetectable downstream.
      if (start) tk_state <= TK_IDLE;
      else case (tk_state)
        TK_IDLE: if (score_last) begin
          taken <= '0; kcnt <= '0; tk_state <= TK_A;
        end

        TK_A: begin
          for (int i = 0; i < E; i++) begin
            v[i] = taken[i] ? NEG_INF : $signed(score[i*32 +: 32]);
            x[i] = EW'(i);
          end
          for (int s = E >> 1; s >= CAND; s = s >> 1)
            for (int i = 0; i < s; i++)
              if (v[i + s] > v[i]) begin v[i] = v[i + s]; x[i] = x[i + s]; end
          for (int i = 0; i < CAND; i++) begin
            cand_v[i] <= v[i];
            cand_x[i] <= x[i];
          end
          tk_state <= TK_B;
        end

        TK_B: begin
          logic signed [31:0] w [CAND];
          logic [EW-1:0]      y [CAND];
          for (int i = 0; i < CAND; i++) begin w[i] = cand_v[i]; y[i] = cand_x[i]; end
          for (int s = CAND >> 1; s > 0; s = s >> 1)
            for (int i = 0; i < s; i++)
              if (w[i + s] > w[i]) begin w[i] = w[i + s]; y[i] = y[i + s]; end
          sel[kcnt*EW +: EW] <= y[0];
          taken[y[0]]        <= 1'b1;
          if (kcnt == $bits(kcnt)'(K - 1)) begin
            done     <= 1'b1;
            tk_state <= TK_IDLE;
          end else begin
            kcnt     <= kcnt + 1'b1;
            tk_state <= TK_A;
          end
        end

        default: tk_state <= TK_IDLE;
      endcase
    end
  end
`else
  // Single-cycle combinational reference, kept for differential comparison:
  // select with -DROUTER_TOPK_COMB. Same result, K dependent tournament trees
  // in one cycle -- which is exactly the depth the pipelined version removes.
  // `done` still marks the cycle sel is valid, so one bench drives both.
  always_comb begin
    logic signed [31:0] tmp [E];
    logic signed [31:0] v [E];
    logic [EW-1:0]      x [E];
    for (int i = 0; i < E; i++) tmp[i] = $signed(score[i*32 +: 32]);
    for (int k = 0; k < K; k++) begin
      for (int i = 0; i < E; i++) begin
        v[i] = tmp[i];
        x[i] = EW'(i);
      end
      for (int s = E >> 1; s > 0; s = s >> 1)
        for (int i = 0; i < s; i++)
          if (v[i + s] > v[i]) begin v[i] = v[i + s]; x[i] = x[i + s]; end
      sel[k*EW +: EW] = x[0];
      tmp[x[0]] = NEG_INF;   // exclude from the next pass
    end
  end

  always_ff @(posedge clk or negedge rst_n)
    if (!rst_n) done <= 1'b0; else done <= score_last;
`endif

endmodule
