/* Sonic S1 bit-exact golden model -- numeric primitives.
 *
 * This is the SPEC. RTL implements this file; when they disagree, this wins.
 * Only the primitives where bugs actually live are here: the hierarchical
 * accumulator, the piecewise-linear activation, and online softmax. Layer
 * plumbing belongs in the Python fake-quant reference, which this must match.
 *
 * Every function is deterministic and free of floating point except where the
 * format spec says BF16.
 */
#ifndef SONIC_GOLDEN_H
#define SONIC_GOLDEN_H

#include <stdint.h>
#include <stddef.h>

/* --- hierarchical accumulator (sonic/quant.py ACCUM) ---------------------
 * 12-bit local fast path, folded into 24-bit every 16 additions, 32-bit
 * epilogue. Keeping the fast path short is what buys the clock frequency;
 * getting the fold depth wrong is what silently corrupts long reductions.
 *
 * The local bound must hold for the WORST CASE product at the fold depth:
 *   |acc_local| <= FOLD * max|w| * max|a|
 *
 * FINDING (P0, this file): 12 bits is NOT enough. For INT4 weights in [-8,7]
 * and INT8 activations in [-128,127] at fold 16 the bound is 16*8*128 = 16384,
 * which needs 16 bits. A 12-bit local path caps |activation| at 15 -- i.e. it
 * silently demands 5-bit activations, which the format spec does not provide.
 *
 * The published Kimi K3 result quotes 12 -> 24 -> 32; that holds for their
 * operand widths, not for INT4 x INT8. We use 16 -> 24 -> 32. The frequency
 * cost is one extra adder bit on the fast path, which is cheap; the cost of
 * getting it wrong is silent corruption on long reductions.
 *
 * Alternative if 12 bits is required for timing: drop the fold depth to 2.
 * test_accum_bound_all_depths() proves the bound either way -- do not assume it.
 */
#define SONIC_ACC_LOCAL_BITS 16
#define SONIC_ACC_FOLD       16
#define SONIC_ACC_MID_BITS   24
#define SONIC_ACC_OUT_BITS   32

typedef struct {
    int32_t local;   /* holds SONIC_ACC_LOCAL_BITS, checked */
    int32_t mid;     /* holds SONIC_ACC_MID_BITS,   checked */
    int32_t out;
    int     n;       /* products since last fold */
    int     overflow;/* sticky: set if any stage exceeded its declared width */
} sonic_acc_t;

void    sonic_acc_init(sonic_acc_t *a);
void    sonic_acc_mac(sonic_acc_t *a, int8_t w, int8_t x); /* w is INT4 sign-extended */
int32_t sonic_acc_final(sonic_acc_t *a);

/* Max |activation| that keeps the 12-bit local path legal at a given fold
 * depth and weight bound. Returns 0 if no positive bound works. */
int sonic_acc_max_activation(int fold, int w_absmax);

/* --- piecewise-linear activation (16 segments, firmware-loadable) --------
 * SiLU by default; the same table covers GELU and ReLU^2. Segments are uniform
 * in x over [-8, 8) with saturating ends, so segment select is a shift.
 */
#define SONIC_PWL_SEGS 16
typedef struct { int32_t slope_q15[SONIC_PWL_SEGS], icept_q15[SONIC_PWL_SEGS]; } sonic_pwl_t;

void    sonic_pwl_fit_silu(sonic_pwl_t *t);
int32_t sonic_pwl_eval(const sonic_pwl_t *t, int32_t x_q15);

/* --- online softmax (flash ordering) -------------------------------------
 * Running max and sum, updated per tile. Tile order must match the RTL's KV
 * page walk or the result differs in the last bits.
 */
typedef struct { float m, l; } sonic_softmax_t;
void  sonic_softmax_init(sonic_softmax_t *s);
void  sonic_softmax_update(sonic_softmax_t *s, const float *scores, size_t n,
                           float *acc, const float *values, size_t d);

/* --- whole-layer golden models (P0-4a integration reference) ------------
 * End-to-end layer references combining routing, GEMV, PWL, and accumulation.
 */
typedef struct {
    int   num_experts;
    int   top_k;
    int   hidden_dim;
    int   intermediate_dim;
} sonic_layer_config_t;

/* INT4 group-64 quantized matrix-vector multiply with FP16 scale */
void sonic_golden_gemv_int4(const int8_t *x, const int8_t *w, const float *scales,
                            float *y_out, size_t M, size_t K, size_t group);

/* 1D short convolution (depthwise across channels) */
void sonic_golden_conv1d(const int8_t *x, const int8_t *k, int32_t *y_out,
                         size_t channels, size_t kernel_size);

/* Full MoE layer golden forward: router -> gate/up -> SiLU PWL -> down -> combine */
void sonic_golden_moe_layer(const int8_t *x, const int8_t *router_w,
                            const int8_t *gate_up_w, const float *gate_up_scales,
                            const int8_t *down_w, const float *down_scales,
                            float *y_out, const sonic_layer_config_t *cfg);

#endif /* SONIC_GOLDEN_H */
