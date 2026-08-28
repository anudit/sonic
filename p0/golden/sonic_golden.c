#include "sonic_golden.h"
#include <math.h>
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
