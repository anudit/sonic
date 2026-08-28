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

int main(void) {
    printf("Sonic S1 golden-model directed tests\n\n");
    test_accum_bound_all_depths();
    test_ragged_reduction();
    test_pwl();
    test_softmax_tiling();
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
