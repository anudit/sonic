// Differential testbench: RTL accumulator vs the P0 C golden model, in one
// process. No marshalling, no Python -- the golden model IS the reference, so
// the two are compared on identical inputs at every step.
//
// This is the bench that has to catch:
//   * the tail dropped on a non-multiple-of-fold reduction
//   * the overflow bound at EVERY K, not just the convenient one
//   * a double-accumulated product from an enable spanning two phases
#include "Vsonic_acc.h"
#include "verilated.h"
extern "C" {
#include "sonic_golden.h"
}
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vsonic_acc *dut;
static vluint64_t  main_time = 0;
double sc_time_stamp() { return main_time; }

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->clr = 0; dut->en = 0; dut->flush = 0;
    dut->w = 0; dut->a = 0;
    tick(); tick();
    dut->rst_n = 1;
    tick();
}

static int fails = 0;
#define CHECK(c, ...) do { if (!(c)) { printf("  FAIL: "); printf(__VA_ARGS__); \
                                       printf("\n"); if (++fails > 20) { \
                                       printf("  (stopping after 20)\n"); return; } } } while (0)

// One dot product of length K, driven identically into RTL and golden model.
static void run_dot(const std::vector<int8_t>& w, const std::vector<int8_t>& a,
                    int32_t *rtl, int *rtl_ovf, int32_t *gold, int *gold_ovf) {
    sonic_acc_t g; sonic_acc_init(&g);

    dut->clr = 1; dut->en = 0; tick(); dut->clr = 0;
    for (size_t i = 0; i < w.size(); i++) {
        dut->en = 1; dut->w = w[i]; dut->a = a[i];
        tick();
        sonic_acc_mac(&g, w[i], a[i]);
    }
    dut->en = 0; tick();

    *rtl = (int32_t)dut->acc;  *rtl_ovf = dut->ovf;
    *gold = sonic_acc_final(&g); *gold_ovf = g.overflow;
}

// Every reduction length from 1 to 300. The tail bug and the fold-boundary bug
// both live in the lengths nobody tests.
static void test_every_length(void) {
    printf("every reduction length 1..300, worst-case operands\n");
    for (int K = 1; K <= 300; K++) {
        std::vector<int8_t> w(K, -8), a(K, -128);   // max |product|
        int32_t r, g; int ro, go;
        run_dot(w, a, &r, &ro, &g, &go);
        CHECK(r == g, "K=%d: rtl %d != golden %d", K, r, g);
        CHECK(ro == go, "K=%d: ovf rtl %d != golden %d", K, ro, go);
        CHECK(!ro, "K=%d: overflow on legal INT4 x INT8 operands", K);
    }
}

static void test_random(void) {
    printf("randomised dot products\n");
    srand(1);
    for (int t = 0; t < 2000; t++) {
        int K = 1 + rand() % 200;
        std::vector<int8_t> w(K), a(K);
        for (int i = 0; i < K; i++) {
            w[i] = (int8_t)((rand() % 16) - 8);      // INT4 range
            a[i] = (int8_t)((rand() % 256) - 128);   // INT8 range
        }
        int32_t r, g; int ro, go;
        run_dot(w, a, &r, &ro, &g, &go);
        CHECK(r == g, "trial %d K=%d: rtl %d != golden %d", t, K, r, g);
        CHECK(ro == go, "trial %d: ovf mismatch", t);
    }
}

// The K3 team's control bug: an enable asserted across two pipeline phases
// accumulates each element twice. Drive en low between products and prove the
// result is unchanged.
static void test_no_double_accumulate(void) {
    printf("enable gating -- no double accumulation\n");
    const int K = 100;
    std::vector<int8_t> w(K), a(K);
    for (int i = 0; i < K; i++) { w[i] = (int8_t)(i % 15 - 7); a[i] = (int8_t)(i * 7 % 255 - 127); }

    int32_t dense, g; int dovf, govf;
    run_dot(w, a, &dense, &dovf, &g, &govf);

    // Same stimulus, but with an idle cycle between every product.
    sonic_acc_t gg; sonic_acc_init(&gg);
    dut->clr = 1; dut->en = 0; tick(); dut->clr = 0;
    for (int i = 0; i < K; i++) {
        dut->en = 1; dut->w = w[i]; dut->a = a[i]; tick();
        dut->en = 0; dut->w = w[i]; dut->a = a[i]; tick();   // held, must not count
        sonic_acc_mac(&gg, w[i], a[i]);
    }
    tick();
    CHECK((int32_t)dut->acc == dense, "idle cycles changed the result: %d vs %d",
          (int32_t)dut->acc, dense);
    CHECK((int32_t)dut->acc == sonic_acc_final(&gg), "golden mismatch with idle cycles");
}

// clr must fully reset all three stages, including a partial local accumulator.
static void test_clr_between_dots(void) {
    printf("clr isolates consecutive dot products\n");
    for (int first = 1; first <= 40; first++) {
        std::vector<int8_t> w1(first, 7), a1(first, 100);
        int32_t r1, g1; int o1, og1;
        run_dot(w1, a1, &r1, &o1, &g1, &og1);
        std::vector<int8_t> w2(5, 1), a2(5, 1);
        int32_t r2, g2; int o2, og2;
        run_dot(w2, a2, &r2, &o2, &g2, &og2);
        CHECK(r2 == 5, "leak from a %d-long dot: got %d, want 5", first, r2);
        CHECK(r2 == g2, "golden disagrees after clr: %d vs %d", r2, g2);
    }
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsonic_acc;
    printf("sonic_acc: RTL vs C golden model\n\n");
    reset(); test_every_length();
    reset(); test_random();
    reset(); test_no_double_accumulate();
    reset(); test_clr_between_dots();
    dut->final(); delete dut;
    printf("\n%s (%d failures)\n", fails ? "FAILED" : "PASSED", fails);
    return fails ? 1 : 0;
}
