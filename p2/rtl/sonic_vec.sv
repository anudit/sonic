// Programmable Vector Unit: sonic_vec.sv
//
// 2048-wide SIMD vector processing unit for non-matrix transformer operations:
//   1. RMSNorm: y = (x / RMS(x)) * gamma
//   2. RoPE: Rotary Position Embedding rotation on Q and K heads
//   3. Residual Stream Addition: out = x + residual
//   4. Elementwise Gating: out = x * act_gate
//
// Operates on INT8 / INT16 / FP16 vectors, providing architectural insurance
// for operators the systolic array cannot directly express.
`include "sonic_defs.svh"

module sonic_vec #(
  parameter int LANES = 64,             // 64 SIMD vector lanes per beat (2048 / 64 = 32 beats)
  parameter int DW    = 16,             // 16-bit intermediate fixed-point / Q8.8
  parameter int D     = 2048
) (
  input  logic                          clk,
  input  logic                          rst_n,

  // Operation Select:
  //   00: NOP
  //   01: RESIDUAL_ADD (a + b)
  //   10: ROPE_ROTATE (rotate pairs using cos_val and sin_val)
  //   11: RMSNORM_SCALE (a * inv_rms * gamma)
  input  logic [1:0]                    op_sel,
  input  logic                          in_vld,
  input  logic                          last_beat,

  // Vector Operands (LANES elements per cycle)
  input  logic signed [LANES*DW-1:0]    vec_a,
  input  logic signed [LANES*DW-1:0]    vec_b,
  input  logic signed [LANES*DW-1:0]    vec_gamma,     // RMSNorm weights or RoPE cos
  input  logic signed [LANES*DW-1:0]    vec_sin,       // RoPE sin values
  input  logic signed [DW-1:0]          inv_rms,       // 1 / RMS(x) scale factor

  // Result Output
  output logic signed [LANES*DW-1:0]    vec_out,
  output logic                          out_vld,
  output logic                          out_last
);

  localparam logic [1:0] OP_NOP     = 2'b00;
  localparam logic [1:0] OP_ADD     = 2'b01;
  localparam logic [1:0] OP_ROPE    = 2'b10;
  localparam logic [1:0] OP_RMSNORM = 2'b11;

  logic signed [LANES*DW-1:0] out_comb;
  logic signed [DW-1:0] a_elem, b_elem, g_elem, x0_elem, x1_elem, c0_elem, s0_elem;
  logic signed [2*DW-1:0] r0_calc, r1_calc, norm_step_calc, scaled_calc;

  // Combinational Vector Compute Pipeline
  always_comb begin
    out_comb = '0;
    a_elem = '0;
    b_elem = '0;
    g_elem = '0;
    x0_elem = '0;
    x1_elem = '0;
    c0_elem = '0;
    s0_elem = '0;
    r0_calc = '0;
    r1_calc = '0;
    norm_step_calc = '0;
    scaled_calc = '0;

    case (op_sel)
      OP_ADD: begin
        for (int i = 0; i < LANES; i++) begin
          a_elem = $signed(vec_a[i*DW +: DW]);
          b_elem = $signed(vec_b[i*DW +: DW]);
          out_comb[i*DW +: DW] = a_elem + b_elem;
        end
      end

      OP_ROPE: begin
        // RoPE rotation across adjacent 2-element pairs
        for (int i = 0; i < LANES; i += 2) begin
          x0_elem = $signed(vec_a[i*DW +: DW]);
          x1_elem = $signed(vec_a[(i+1)*DW +: DW]);
          c0_elem = $signed(vec_gamma[i*DW +: DW]);
          s0_elem = $signed(vec_sin[i*DW +: DW]);

          // (x0 * cos - x1 * sin), (x0 * sin + x1 * cos) in Q8.8
          r0_calc = (x0_elem * c0_elem - x1_elem * s0_elem) >>> 8;
          r1_calc = (x0_elem * s0_elem + x1_elem * c0_elem) >>> 8;

          out_comb[i*DW +: DW]     = DW'(r0_calc);
          out_comb[(i+1)*DW +: DW] = DW'(r1_calc);
        end
      end

      OP_RMSNORM: begin
        // Normalization scaling: a * inv_rms * gamma in Q8.8
        for (int i = 0; i < LANES; i++) begin
          a_elem = $signed(vec_a[i*DW +: DW]);
          g_elem = $signed(vec_gamma[i*DW +: DW]);
          norm_step_calc = (a_elem * inv_rms) >>> 8;
          scaled_calc    = (norm_step_calc[DW-1:0] * g_elem) >>> 8;
          out_comb[i*DW +: DW] = DW'(scaled_calc);
        end
      end

      default: out_comb = vec_a;
    endcase
  end

  // Registered Output Stage
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) begin
      vec_out  <= '0;
      out_vld  <= 1'b0;
      out_last <= 1'b0;
    end else begin
      out_vld  <= in_vld;
      out_last <= in_vld & last_beat;
      if (in_vld) begin
        vec_out <= out_comb;
      end
    end
  end

endmodule
