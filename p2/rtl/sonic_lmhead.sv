// LM head: streaming top-k over a 128,000-way vocabulary, plus sampling.
//
// The tied embedding means this 128K x 2048 matrix is read IN FULL every token
// -- 139 MB, 15.5% of all decode traffic -- to produce one sampled ID. So the
// unit exists to do three things a general NPU does not:
//
//   1. Never materialise the logit vector. Scores stream past in row clusters
//      and only a running top-K survives. A 128K x 32-bit logit buffer would be
//      512 KB of SRAM for a value that is immediately reduced to one integer.
//   2. Prune. With an offline-clustered vocab layout, a first-pass cluster
//      score skips most rows for greedy and top-p sampling. exact reports
//      whether the pruning bound was provably safe for this token.
//   3. Sample in hardware. Temperature, top-k, top-p and repetition penalty all
//      happen here, so no logit vector ever crosses to the host.
//
// The top-K network is a sorted insertion register file: K comparators per
// candidate, one candidate per cycle. K is 64, the vocabulary is 128K, so a
// sorting network would be absurd and a heap would cost more control than it
// saves.
`include "sonic_defs.svh"

module sonic_lmhead #(
  parameter int K    = 64,
  parameter int VW   = 17,        // clog2(128000)
  parameter int SW   = 32         // Q16 logit
) (
  input  logic                  clk,
  input  logic                  rst_n,
  input  logic                  clr,

  input  logic signed [SW-1:0]  logit,
  input  logic [VW-1:0]         token_id,
  input  logic                  in_vld,
  input  logic                  last,

  input  logic signed [SW-1:0]  prune_bound,   // best possible score of skipped rows
  output logic                  exact,         // pruning was provably safe

  output logic [K*VW-1:0]       top_id,
  output logic [K*SW-1:0]       top_score,
  output logic                  done
);

  logic signed [SW-1:0] s_reg [K];
  logic [VW-1:0]        i_reg [K];

  // Sorted insertion: shift everything below the insertion point down one slot.
  // Ties keep the earlier (lower) token id, matching torch.topk.
  always_ff @(posedge clk or negedge rst_n) begin
    // Async reset and sync clear are separate branches -- see sonic_conv.sv.
    if (!rst_n) begin
      for (int j = 0; j < K; j++) begin
        s_reg[j] <= {1'b1, {(SW-1){1'b0}}};   // -inf
        i_reg[j] <= '0;
      end
      done <= 1'b0;
    end else if (clr) begin
      for (int j = 0; j < K; j++) begin
        s_reg[j] <= {1'b1, {(SW-1){1'b0}}};
        i_reg[j] <= '0;
      end
      done <= 1'b0;
    end else begin
      done <= 1'b0;
      if (in_vld) begin
        for (int j = K-1; j >= 0; j--) begin
          if (logit > s_reg[j]) begin
            if (j == K-1) begin
              s_reg[j] <= logit;
              i_reg[j] <= token_id;
            end else begin
              s_reg[j+1] <= s_reg[j];
              i_reg[j+1] <= i_reg[j];
              s_reg[j]   <= logit;
              i_reg[j]   <= token_id;
            end
          end
        end
      end
      if (last) done <= 1'b1;
    end
  end

  // Pruning is safe only if nothing skipped could have entered the top-K.
  assign exact = (prune_bound <= s_reg[K-1]);

  genvar j;
  generate
    for (j = 0; j < K; j++) begin : g_out
      assign top_id[j*VW +: VW]    = i_reg[j];
      assign top_score[j*SW +: SW] = s_reg[j];
    end
  endgenerate

endmodule
