/* Directed tests for the failure modes that actually happen in this class of
 * design. Each one corresponds to a line in the plan's verification section. */
#include "sonic_golden.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: "); printf(__VA_ARGS__); \
                                       printf("\n"); fails++; } } while (0)

/* The Kimi K3 bug: a bound proven at K=64 that breaks at K=32. Sweep every
 * reduction depth rather than trusting a symbolic argument. */
static void test_accum_bound_all_depths(void) {
    printf("accumulator bound across reduction depths\n");
    int amax = sonic_acc_max_activation(SONIC_ACC_FOLD, 8);
    printf("  %db local path supports |activation| <= %d at fold %d\n",
           SONIC_ACC_LOCAL_BITS, amax, SONIC_ACC_FOLD);
    CHECK(amax >= 127, "local path cannot hold full INT8 activations "
                       "(bound %d); widen it or reduce the fold depth", amax);

    for (int K = 1; K <= 4096; K++) {
        sonic_acc_t a; sonic_acc_init(&a);
        for (int i = 0; i < K; i++) sonic_acc_mac(&a, -8, (int8_t)-128);
        sonic_acc_final(&a);
        CHECK(!a.overflow, "overflow at K=%d on worst-case INT4 x INT8", K);
    }
    /* Regression guard: a 12-bit local path -- the width the Kimi K3 write-up
     * quotes -- must be shown to FAIL here, so nobody narrows it back. */
    int amax12 = 2047 / (SONIC_ACC_FOLD * 8);
    printf("  a 12b local path would cap |activation| at %d -- rejected\n", amax12);
    CHECK(amax12 < 127, "12b unexpectedly sufficient; re-derive the bound");
}

/* Reductions that are not a multiple of the fold depth drop the tail if the
 * epilogue forgets the partial local accumulator. */
static void test_ragged_reduction(void) {
    printf("non-multiple-of-fold reductions\n");
    for (int K = 1; K <= 200; K++) {
        sonic_acc_t a; sonic_acc_init(&a);
        int32_t ref = 0;
        for (int i = 0; i < K; i++) { sonic_acc_mac(&a, 3, 5); ref += 15; }
        int32_t got = sonic_acc_final(&a);
        CHECK(got == ref, "K=%d: got %d want %d", K, got, ref);
    }
}

static void test_pwl(void) {
    printf("PWL SiLU accuracy and monotonicity of segment select\n");
    sonic_pwl_t t; sonic_pwl_fit_silu(&t);
    double worst = 0.0;
    for (double x = -8.0; x < 8.0; x += 1.0 / 256) {
        int32_t q = (int32_t)lrint(x * 32768.0);
        double got = sonic_pwl_eval(&t, q) / 32768.0;
        double ref = x / (1.0 + exp(-x));
        double e = fabs(got - ref);
        if (e > worst) worst = e;
    }
    printf("  worst absolute error over [-8,8): %.4f\n", worst);
    /* Implementation-intent bound, not a quality gate. The quality gate is the
     * perplexity delta in sonic/quant.py GATES -- see the note in sonic_golden.c. */
    CHECK(worst < 0.06, "PWL error %.4f exceeds the 16-segment design intent", worst);
    /* Saturating ends must not wrap. */
    CHECK(sonic_pwl_eval(&t, (int32_t)lrint(-40.0 * 32768.0)) == 0, "left tail wrapped");
    int32_t big = (int32_t)lrint(40.0 * 32768.0);
    CHECK(sonic_pwl_eval(&t, big) == big, "right tail is not identity");
}

/* Online softmax must equal the naive two-pass result regardless of how the KV
 * pages are tiled -- otherwise RTL and reference diverge on page boundaries. */
static void test_softmax_tiling(void) {
    printf("online softmax invariance to KV tiling\n");
    enum { N = 97, D = 8 };
    float sc[N], v[N * D], ref[D] = {0};
    for (int i = 0; i < N; i++) {
        sc[i] = (float)((i * 37 % 61) - 30) * 0.3f;
        for (int j = 0; j < D; j++) v[i * D + j] = (float)((i + j) % 11) * 0.1f;
    }
    float m = -INFINITY, l = 0;
    for (int i = 0; i < N; i++) if (sc[i] > m) m = sc[i];
    for (int i = 0; i < N; i++) {
        float p = expf(sc[i] - m); l += p;
        for (int j = 0; j < D; j++) ref[j] += p * v[i * D + j];
    }
    for (int j = 0; j < D; j++) ref[j] /= l;

    for (int tile = 1; tile <= N; tile++) {
        sonic_softmax_t s; sonic_softmax_init(&s);
        float acc[D] = {0};
        for (int i = 0; i < N; i += tile) {
            int n = (i + tile <= N) ? tile : N - i;
            sonic_softmax_update(&s, sc + i, n, acc, v + i * D, D);
        }
        for (int j = 0; j < D; j++) {
            float got = acc[j] / s.l;
            CHECK(fabsf(got - ref[j]) < 1e-5f,
                  "tile=%d lane=%d: %.7f vs %.7f", tile, j, got, ref[j]);
        }
    }
}

static void test_golden_gemv(void) {
    printf("golden INT4 GEMV and 1D convolution\n");
    enum { M = 8, K = 64 };
    int8_t x[K], w[M * K];
    float scales[M];
    for (int k = 0; k < K; k++) x[k] = (int8_t)((k % 15) - 7);
    for (int m = 0; m < M; m++) {
        scales[m] = 0.5f;
        for (int k = 0; k < K; k++) w[m * K + k] = (int8_t)(((m + k) % 9) - 4);
    }
    float y[M];
    sonic_golden_gemv_int4(x, w, scales, y, M, K, 64);
    for (int m = 0; m < M; m++) {
        int32_t exp = 0;
        for (int k = 0; k < K; k++) exp += (int32_t)w[m * K + k] * x[k];
        float exp_f = (float)exp * 0.5f;
        CHECK(fabsf(y[m] - exp_f) < 1e-4f, "GEMV m=%d: got %.2f want %.2f", m, y[m], exp_f);
    }

    int8_t conv_k[4 * 3] = {1, 2, 1,  2, -1, 1,  -2, 0, 2,  1, 1, 1};
    int8_t conv_x[4 * 3] = {3, 2, 1,  1,  1, 1,   2, 2, 2,  0, 1, 2};
    int32_t conv_y[4];
    sonic_golden_conv1d(conv_x, conv_k, conv_y, 4, 3);
    CHECK(conv_y[0] == 8, "conv 0: got %d want 8", conv_y[0]);
    CHECK(conv_y[1] == 2, "conv 1: got %d want 2", conv_y[1]);
    CHECK(conv_y[2] == 0, "conv 2: got %d want 0", conv_y[2]);
    CHECK(conv_y[3] == 3, "conv 3: got %d want 3", conv_y[3]);
}

static void test_golden_moe_layer_run(void) {
    printf("golden whole MoE layer forward\n");
    sonic_layer_config_t cfg = {
        .num_experts = 4,
        .top_k = 2,
        .hidden_dim = 64,
        .intermediate_dim = 64
    };
    int8_t x[64];
    int8_t router_w[4 * 64];
    int8_t gate_up_w[4 * (2 * 64) * 64];
    int8_t down_w[4 * 64 * 64];
    float gu_scales[4 * 128 * 1];
    float dn_scales[4 * 64 * 1];
    float y[64];

    for (int i = 0; i < 64; i++) x[i] = (int8_t)((i % 7) - 3 + 2);
    for (int i = 0; i < 4 * 64; i++) router_w[i] = (int8_t)((i % 5) - 2);
    for (int i = 0; i < 4 * 128 * 64; i++) gate_up_w[i] = (int8_t)((i % 9) - 4);
    for (int i = 0; i < 4 * 64 * 64; i++) down_w[i] = (int8_t)((i % 7) - 3);
    for (int i = 0; i < 4 * 128; i++) gu_scales[i] = 0.1f;
    for (int i = 0; i < 4 * 64; i++) dn_scales[i] = 0.1f;

    sonic_golden_moe_layer(x, router_w, gate_up_w, gu_scales, down_w, dn_scales, y, &cfg);
    int non_zero = 0;
    for (int i = 0; i < 64; i++) {
        if (fabsf(y[i]) > 1e-4f) non_zero++;
    }
    CHECK(non_zero > 0, "MoE layer output all zeros");
}

int main(void) {
    printf("Sonic S1 golden-model directed tests\n\n");
    test_accum_bound_all_depths();
    test_ragged_reduction();
    test_pwl();
    test_softmax_tiling();
    test_golden_gemv();
    test_golden_moe_layer_run();
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}

