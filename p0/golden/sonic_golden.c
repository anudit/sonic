#include "sonic_golden.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static int fits(int32_t v, int bits) {
    int32_t lim = (int32_t)1 << (bits - 1);
    return v >= -lim && v <= lim - 1;
}

void sonic_acc_init(sonic_acc_t *a) { memset(a, 0, sizeof(*a)); }

void sonic_acc_mac(sonic_acc_t *a, int8_t w, int8_t x) {
    a->local += (int32_t)w * (int32_t)x;
    if (!fits(a->local, SONIC_ACC_LOCAL_BITS)) a->overflow = 1;
    if (++a->n == SONIC_ACC_FOLD) {
        a->mid += a->local;
        if (!fits(a->mid, SONIC_ACC_MID_BITS)) a->overflow = 1;
        a->local = 0;
        a->n = 0;
    }
}

int32_t sonic_acc_final(sonic_acc_t *a) {
    a->out = a->mid + a->local;
    return a->out;
}

int sonic_acc_max_activation(int fold, int w_absmax) {
    int32_t lim = ((int32_t)1 << (SONIC_ACC_LOCAL_BITS - 1)) - 1;
    int b = lim / (fold * w_absmax);
    return b > 0 ? b : 0;
}

/* --- PWL ---------------------------------------------------------------- */
#define Q15 32768.0

static double silu(double x) { return x / (1.0 + exp(-x)); }

/* 16 uniform chords of width 1 over [-8, 8), with exact asymptotic tails:
 * silu(x) -> 0 as x -> -inf, silu(x) -> x as x -> +inf. Uniform width keeps
 * segment select a bare shift (floor(x) + 8), which is what makes this cheap
 * in RTL and what lets the table stay firmware-loadable across SiLU / GELU /
 * ReLU^2 without touching the datapath.
 *
 * Worst absolute error is ~0.054, near x = -1.3 where curvature peaks. That is
 * NOT gated here on function error, because the number that matters is the
 * end-to-end perplexity delta in sonic/quant.py GATES. If that gate fails,
 * raise SONIC_PWL_SEGS to 32 -- the cost is table bits, not datapath.
 * Non-uniform (exponent-indexed) segmentation would beat this at equal table
 * size and is the fallback if 32 segments is still not enough. */
#define PWL_LO (-8.0)
#define PWL_W  (1.0)

void sonic_pwl_fit_silu(sonic_pwl_t *t) {
    for (int s = 0; s < SONIC_PWL_SEGS; s++) {
        double x0 = PWL_LO + s * PWL_W, x1 = x0 + PWL_W;
        double y0 = silu(x0), y1 = silu(x1);
        double m = (y1 - y0) / PWL_W, c = y0 - m * x0;
        t->slope_q15[s] = (int32_t)lrint(m * Q15);
        t->icept_q15[s] = (int32_t)lrint(c * Q15);
    }
}

int32_t sonic_pwl_eval(const sonic_pwl_t *t, int32_t x_q15) {
    if (x_q15 < (int32_t)(PWL_LO * Q15)) return 0;          /* silu(-inf) = 0 */
    if (x_q15 >= (int32_t)(-PWL_LO * Q15)) return x_q15;    /* silu(+inf) = x */
    int s = (int)((x_q15 >> 15) + 8);        /* floor(x) + 8 */
    if (s < 0) s = 0;
    if (s >= SONIC_PWL_SEGS) s = SONIC_PWL_SEGS - 1;
    int64_t v = ((int64_t)t->slope_q15[s] * x_q15) >> 15;
    return (int32_t)(v + t->icept_q15[s]);
}

/* --- online softmax ----------------------------------------------------- */
void sonic_softmax_init(sonic_softmax_t *s) { s->m = -INFINITY; s->l = 0.0f; }

void sonic_softmax_update(sonic_softmax_t *s, const float *scores, size_t n,
                          float *acc, const float *values, size_t d) {
    float m_new = s->m;
    for (size_t i = 0; i < n; i++) if (scores[i] > m_new) m_new = scores[i];
    if (m_new == -INFINITY) return;

    float rescale = (s->m == -INFINITY) ? 0.0f : expf(s->m - m_new);
    float l_new = s->l * rescale;
    for (size_t j = 0; j < d; j++) acc[j] *= rescale;

    for (size_t i = 0; i < n; i++) {
        float p = expf(scores[i] - m_new);
        l_new += p;
        for (size_t j = 0; j < d; j++) acc[j] += p * values[i * d + j];
    }
    s->m = m_new;
    s->l = l_new;
}

/* --- whole-layer golden models (P0-4a) ----------------------------------- */
void sonic_golden_gemv_int4(const int8_t *x, const int8_t *w, const float *scales,
                            float *y_out, size_t M, size_t K, size_t group) {
    size_t n_groups = (K + group - 1) / group;
    for (size_t m = 0; m < M; m++) {
        float y = 0.0f;
        for (size_t g = 0; g < n_groups; g++) {
            sonic_acc_t acc;
            sonic_acc_init(&acc);
            size_t k_start = g * group;
            size_t k_end = (k_start + group < K) ? k_start + group : K;
            for (size_t k = k_start; k < k_end; k++) {
                sonic_acc_mac(&acc, w[m * K + k], x[k]);
            }
            int32_t raw = sonic_acc_final(&acc);
            float scale = scales ? scales[m * n_groups + g] : 1.0f;
            y += (float)raw * scale;
        }
        y_out[m] = y;
    }
}

void sonic_golden_conv1d(const int8_t *x, const int8_t *k, int32_t *y_out,
                         size_t channels, size_t kernel_size) {
    for (size_t c = 0; c < channels; c++) {
        sonic_acc_t acc;
        sonic_acc_init(&acc);
        for (size_t i = 0; i < kernel_size; i++) {
            sonic_acc_mac(&acc, k[c * kernel_size + i], x[c * kernel_size + i]);
        }
        y_out[c] = sonic_acc_final(&acc);
    }
}

void sonic_golden_moe_layer(const int8_t *x, const int8_t *router_w,
                            const int8_t *gate_up_w, const float *gate_up_scales,
                            const int8_t *down_w, const float *down_scales,
                            float *y_out, const sonic_layer_config_t *cfg) {
    /* Step 1: Router logits and top-k selection */
    int num_e = cfg->num_experts;
    int top_k = cfg->top_k;
    int H = cfg->hidden_dim;
    int I = cfg->intermediate_dim;

    int32_t *r_logits = (int32_t *)malloc(num_e * sizeof(int32_t));
    for (int e = 0; e < num_e; e++) {
        sonic_acc_t acc;
        sonic_acc_init(&acc);
        for (int h = 0; h < H; h++) {
            sonic_acc_mac(&acc, router_w[e * H + h], x[h]);
        }
        r_logits[e] = sonic_acc_final(&acc);
    }

    /* Find top-k experts */
    int *selected = (int *)malloc(top_k * sizeof(int));
    float *weights = (float *)malloc(top_k * sizeof(float));
    for (int k = 0; k < top_k; k++) {
        int best_idx = -1;
        int32_t best_val = -0x7fffffff;
        for (int e = 0; e < num_e; e++) {
            int already = 0;
            for (int prev = 0; prev < k; prev++) {
                if (selected[prev] == e) { already = 1; break; }
            }
            if (!already && r_logits[e] > best_val) {
                best_val = r_logits[e];
                best_idx = e;
            }
        }
        selected[k] = (best_idx >= 0) ? best_idx : 0;
        weights[k] = 1.0f / (1.0f + expf(-(float)best_val / 256.0f)); /* sigmoid */
    }

    /* Softmax normalize routing weights */
    float sum_w = 0.0f;
    for (int k = 0; k < top_k; k++) sum_w += weights[k];
    if (sum_w > 0.0f) {
        for (int k = 0; k < top_k; k++) weights[k] /= sum_w;
    }

    /* Step 2: Expert compute (gate_up -> silu -> down) and accumulate */
    memset(y_out, 0, H * sizeof(float));
    sonic_pwl_t pwl;
    sonic_pwl_fit_silu(&pwl);

    float *gate_out = (float *)malloc(I * sizeof(float));
    float *up_out = (float *)malloc(I * sizeof(float));
    int8_t *inter_act = (int8_t *)malloc(I * sizeof(int8_t));
    float *down_out = (float *)malloc(H * sizeof(float));

    for (int k = 0; k < top_k; k++) {
        int exp_idx = selected[k];
        float exp_weight = weights[k];

        /* Gate and Up projection (2 * I outputs) */
        const int8_t *g_w = gate_up_w + exp_idx * (2 * I) * H;
        const float *g_s = gate_up_scales ? (gate_up_scales + exp_idx * (2 * I) * (H / 64)) : NULL;

        sonic_golden_gemv_int4(x, g_w, g_s, gate_out, I, H, 64);
        sonic_golden_gemv_int4(x, g_w + I * H, g_s ? g_s + I * (H / 64) : NULL, up_out, I, H, 64);

        /* SiLU PWL on gate * up */
        for (int i = 0; i < I; i++) {
            int32_t gate_q15 = (int32_t)lrintf(gate_out[i] * 32768.0f);
            int32_t act_val = sonic_pwl_eval(&pwl, gate_q15);
            float act_f = (float)act_val / 32768.0f;
            float prod = act_f * up_out[i];
            int32_t q8 = (int32_t)lrintf(prod * 64.0f);
            inter_act[i] = (int8_t)(q8 > 127 ? 127 : (q8 < -128 ? -128 : q8));
        }

        /* Down projection (H outputs) */
        const int8_t *d_w = down_w + exp_idx * H * I;
        const float *d_s = down_scales ? (down_scales + exp_idx * H * (I / 64)) : NULL;
        sonic_golden_gemv_int4(inter_act, d_w, d_s, down_out, H, I, 64);

        for (int h = 0; h < H; h++) {
            y_out[h] += exp_weight * down_out[h];
        }
    }

    free(down_out);
    free(r_logits);
    free(selected);
    free(weights);
    free(gate_out);
    free(up_out);
    free(inter_act);
}
