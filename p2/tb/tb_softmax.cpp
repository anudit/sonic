// Differential testbench: sonic_softmax vs the P0 C golden online softmax.
//
// The RTL's own comment names the bug it is trying to avoid -- "rescaling only
// one of them is the classic online-softmax bug" -- so this bench checks the
// invariant that claim rests on: for any tiling of any score sequence, the
// running (max, sum, acc) must match a reference that rescales BOTH.
//
// exp() is supplied by the host rather than by the PWL block, so a numeric
// disagreement here is the accumulator's, not the activation table's. The PWL
// has its own bound in p0/golden and its own sweep in p2-pwl-sweep.
#include "Vsonic_softmax.h"
#include "verilated.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vsonic_softmax *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const double Q = 65536.0;
static int32_t q16(double x) { return (int32_t)llround(x * Q); }
static double  unq(int32_t x) { return (double)x / Q; }

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->clr = 0; dut->in_vld = 0;
    dut->score = 0; dut->value = 0; dut->exp_val = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { printf("  FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// Drive one (score, value). exp_arg -> exp_val is combinational in the RTL, so
// evaluate, read the argument, answer it, then clock.
static void feed(double score, double value) {
    dut->score = q16(score); dut->value = q16(value); dut->in_vld = 1;
    dut->eval();
    double arg = unq((int32_t)dut->exp_arg);
    if (arg > 0) arg = 0;                     // both branches produce arg <= 0
    dut->exp_val = q16(std::exp(arg));
    dut->eval();
    tick();
    dut->in_vld = 0;
}

// Reference: the same recurrence sonic_golden.c implements, one score per step.
struct Ref {
    double m = -1e30, l = 0.0, a = 0.0;
    void step(double score, double value) {
        double m_new = std::max(m, score);
        double rescale = (m <= -1e29) ? 0.0 : std::exp(m - m_new);
        double p = std::exp(score - m_new);
        l = l * rescale + p;
        a = a * rescale + p * value;
        m = m_new;
    }
};

static void run_sequence(const char *name, const std::vector<double> &scores,
                         const std::vector<double> &values) {
    reset();
    dut->clr = 1; tick(); dut->clr = 0;
    Ref ref;
    for (size_t i = 0; i < scores.size(); i++) {
        feed(scores[i], values[i]);
        ref.step(scores[i], values[i]);
    }
    double rl = unq((int32_t)dut->run_sum), ra = unq((int32_t)dut->acc);
    double rm = unq((int32_t)dut->run_max);
    // Q16 with a >>>16 product each step: allow a per-step ulp budget, not an
    // exact match. The failures this bench is for are structural, not rounding.
    double tol_l = 1e-3 * std::max(1.0, std::fabs(ref.l)) * scores.size();
    double tol_a = 1e-3 * std::max(1.0, std::fabs(ref.a)) * scores.size();
    CHECK(std::fabs(rm - ref.m) < 1e-3, "%s: run_max %.5f != ref %.5f",
          name, rm, ref.m);
    CHECK(std::fabs(rl - ref.l) < tol_l, "%s: run_sum %.5f != ref %.5f",
          name, rl, ref.l);
    CHECK(std::fabs(ra - ref.a) < tol_a, "%s: acc %.5f != ref %.5f",
          name, ra, ref.a);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsonic_softmax;
    srand(7);

    // 1. Flat scores: every correction is exp(0)=1, so this isolates the
    //    weight path. If the two factors are conflated, this still passes.
    run_sequence("flat", {1.0, 1.0, 1.0, 1.0}, {1.0, 2.0, 3.0, 4.0});

    // 2. Monotonically rising: every step sets a new maximum, so every step
    //    takes the rescale branch.
    run_sequence("rising", {0.0, 1.0, 2.0, 3.0, 4.0}, {1.0, 1.0, 1.0, 1.0, 1.0});

    // 3. Monotonically falling: no step sets a new maximum after the first,
    //    so every step takes the weight branch with correction 1.
    run_sequence("falling", {4.0, 3.0, 2.0, 1.0, 0.0}, {1.0, 2.0, 1.0, 2.0, 1.0});

    // 4. Mixed, the realistic case: corrections and weights interleave.
    {
        std::vector<double> s, v;
        for (int i = 0; i < 24; i++) {
            s.push_back(((rand() % 800) - 400) / 100.0);
            v.push_back(((rand() % 400) - 200) / 100.0);
        }
        run_sequence("mixed", s, v);
    }

    dut->final();
    printf("  checks %d, failures %d\n", checks, fails);
    printf(fails ? "FAILED\n" : "PASSED\n");
    delete dut;
    return fails ? 1 : 0;
}
