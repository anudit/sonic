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
  parameter int LANES   = 64,               // MACs per cycle
  // PWL sigmoid resolution over [-8, 8). Segment select stays a bare shift for
  // any power of two. p2/ppa/pwl_sweep.sh sizes this against routing agreement.
  parameter int SEGS    = `ROUTER_PWL_SEGS,
  parameter int SEGSH   = $clog2(SEGS)      // shift = 20 - log2(SEGS)
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

  function automatic logic signed [31:0] pwl(input logic signed [31:0] x_q16);
    logic signed [31:0] xs;
    int idx;
    begin
      // Segment select is a bare shift for any power-of-two SEGS: map
      // [-8, 8) onto [0, SEGS) by adding 8 in Q16 and shifting right by
      // 20 - log2(SEGS).
      xs  = (x_q16 + (8 <<< 16)) >>> (20 - SEGSH);
      idx = xs;
      if (idx < 0)        idx = 0;
      if (idx > SEGS - 1) idx = SEGS - 1;
      pwl = 32'(($signed({{32{x_q16[31]}}, x_q16}) * slope[idx]) >>> 16) + icept[idx];
    end
  endfunction

  // --- accumulate one expert at a time -----------------------------------
  logic signed [`ACC_OUT-1:0] acc;
  logic [EW-1:0]              e_idx;
  logic [$clog2(STEPS+1)-1:0] step;
  logic                       busy;

  // Epilogue scaling. Both operands are 32-bit and the lvalue is 64-bit, which
  // is the idiom that gets a 32x32 -> 64 multiply rather than a truncated one.
  logic signed [`ACC_OUT-1:0] acc_final;
  logic signed [63:0]         scaled;
  assign acc_final = acc + dot;
  assign scaled    = $signed(acc_final) * $signed(logit_scale);

  logic signed [`ACC_OUT-1:0] dot;
  always_comb begin
    logic signed [AW-1:0] av;
    logic signed [WW-1:0] wv;
    dot = '0;
    for (int i = 0; i < LANES; i++) begin
      av = a_in[i*AW +: AW];
      wv = w_in[i*WW +: WW];
      dot += `ACC_OUT'(av) * `ACC_OUT'(wv);
    end
  end

  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      busy <= 1'b0; e_idx <= '0; step <= '0; acc <= '0; done <= 1'b0;
    end else begin
      done <= 1'b0;
      if (start) begin
        busy <= 1'b1; e_idx <= '0; step <= '0; acc <= '0;
      end else if (busy && in_vld) begin
        if (step == $bits(step)'(STEPS - 1)) begin
          // Epilogue for this expert: scale to Q16, sigmoid, add bias.
          // Shift by 16, not 32. acc is an integer and logit_scale is Q32, so
          // the product is the true logit scaled by 2^32; >>> 32 yields the
          // INTEGER logit and >>> 16 yields Q16, which is what pwl() expects.
          // Shifting 32 here made every logit round to ~0, every sigmoid come
          // out at 0.5, and the RTL score 0.0000 against the model.
          score[e_idx*32 +: 32] <= pwl(32'(scaled >>> 16))
                                   + $signed(bias[e_idx*32 +: 32]);
          acc  <= '0;
          step <= '0;
          if (e_idx == EW'(E - 1)) begin
            busy <= 1'b0;
            done <= 1'b1;
          end else begin
            e_idx <= e_idx + 1'b1;
          end
        end else begin
          acc  <= acc + dot;
          step <= step + 1'b1;
        end
      end
    end
  end

  // --- top-K selection ---------------------------------------------------
  // Sequential max-extract: K passes over E scores. K and E are both small, so
  // this costs K*E comparators once rather than a sorting network.
`ifndef ROUTER_TOPK_SERIAL
  // Tournament tree. Same E-1 comparators as the serial scan below -- a tree
  // and a linear scan both compare every element exactly once -- but the
  // dependency chain is log2(E) deep instead of E-1 deep. The serial version
  // paid 31 dependent 32-bit compares per pass for no area saving at all.
  //
  // P4 measured the cost of getting this wrong: WNS -1086.6 ns at a 10 ns clock
  // on this exact sel[] path, and 2,518 combinational levels against
  // sonic_acc's 59. See p2/README.md finding 16.
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
      // Pairwise reduce: E -> E/2 -> ... -> 1, carrying the argmax alongside.
      for (int s = E >> 1; s > 0; s = s >> 1)
        for (int i = 0; i < s; i++)
          if (v[i + s] > v[i]) begin
            v[i] = v[i + s];
            x[i] = x[i + s];
          end
      sel[k*EW +: EW] = x[0];
      tmp[x[0]] = 32'sh8000_0000;   // exclude from the next pass
    end
  end
`else
  // Original serial max-extract, kept for differential comparison. Functionally
  // identical: both return the K largest scores, ties broken toward the lowest
  // index. Select with -DROUTER_TOPK_SERIAL.
  always_comb begin
    logic signed [31:0] tmp [E];
    logic signed [31:0] best;
    logic [EW-1:0]      bi;
    for (int i = 0; i < E; i++) tmp[i] = $signed(score[i*32 +: 32]);
    for (int k = 0; k < K; k++) begin
      best = tmp[0]; bi = '0;
      for (int i = 1; i < E; i++)
        if (tmp[i] > best) begin best = tmp[i]; bi = EW'(i); end
      sel[k*EW +: EW] = bi;
      tmp[bi] = 32'sh8000_0000;   // exclude from the next pass
    end
  end
`endif

endmodule
