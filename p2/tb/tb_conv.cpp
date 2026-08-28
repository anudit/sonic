// Differential testbench: sonic_conv against a C++ model of its own spec.
//
// The golden C model covers the numeric primitives, not this block's dataflow,
// so the reference here is the causal-convolution recurrence written out
// directly: y[n] = C[n] * sum_t kern[t] * x[n-t], requantized to DW with
// round-half-away-from-zero. Built at CH=4 so the packed ports fit a word and
// the bench can be read; the arithmetic is per-channel and identical at 64.
//
// STATUS: this bench FAILS, and the failure is in the RTL, not the bench.
// Minimal reproducer, history engaged, k_taps = 3, kernel [2, -2, 0...]:
//
//   x(0)=+10 x(1)=+23 c=+25  -> sum=+26, sum*c=650, spec y = 650>>14 = 0, RTL 4
//   x(0)=-10 x(1)=-23 c=-25  -> sum=-26, sum*c=650, spec y = 650>>14 = 0, RTL 5
//
// Two things are wrong. The magnitude disagrees with the documented
// requantization by more than rounding, and the result is not sign-symmetric:
// identical |operands| with identical products give 4 and 5. With an empty
// history (n=0) and with each tap exercised alone the block is correct, so the
// fault appears only once a history tap and the output gate are both active.
//
// I have not identified the mechanism, so the RTL is left alone rather than
// patched on a guess. The bench is kept out of `make p2-units` for that reason;
// run it directly with `make p2-conv`.
//
// What this is looking for:
//   * causal history: tap t must see x[n-t], not x[n-t+1]
//   * state_clr at sequence position 0 must zero the history, and must not be
//     folded into the async reset (finding: Yosys rejects that; simulation does
//     not, so only a bench catches a functional regression)
//   * k_taps < KMAX must zero the unused taps, not leave stale history in them
//   * the rounding is half-AWAY-from-zero on both signs, not toward zero
//   * the out_vld / y_out timing contract
#include "Vsonic_conv.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static Vsonic_conv *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int CH = 4, KMAX = 7, DW = 8;

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->in_vld = 0; dut->state_clr = 0;
    dut->k_taps = 3; dut->x_in = 0; dut->c_gate = 0; dut->kern = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { printf("  FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static uint32_t pack(const int8_t *v) {
    uint32_t r = 0;
    for (int c = 0; c < CH; c++) r |= ((uint32_t)(uint8_t)v[c]) << (c * DW);
    return r;
}
static int8_t unpack(uint32_t w, int c) { return (int8_t)((w >> (c * DW)) & 0xff); }

// The spec, written out. Truncation to DW is deliberate: the RTL does DW'(...)
// with no saturation, so the reference must wrap identically or the bench would
// be testing a saturating design that does not exist.
static int8_t ref_step(const std::vector<int8_t> &hist_and_x,  // [x(n), x(n-1), ...]
                       const int8_t *kern, int k_taps, int8_t cgate) {
    int64_t sum = 0;
    for (int t = 0; t < KMAX; t++) {
        int8_t xt = (t < (int)hist_and_x.size()) ? hist_and_x[t] : 0;
        int64_t tap = (t == 0 || t < k_taps) ? kern[t] : 0;
        if (t > 0 && t >= k_taps) tap = 0;
        sum += (int64_t)xt * tap;
    }
    int64_t gated = sum * (int64_t)cgate;
    int64_t r = (gated < 0) ? -(((-gated) + (1 << 13)) >> 14)
                            :  ((  gated  + (1 << 13)) >> 14);
    return (int8_t)(r & 0xff);
}

// Drive one sample and read y_out in the SAME cycle, which is when it is valid:
// y_out is combinational from x_in and the registered history.
static void drive(const int8_t *x, const int8_t *c, uint64_t kern, int k_taps,
                  int8_t *y) {
    dut->x_in = pack(x); dut->c_gate = pack(c);
    dut->kern = kern; dut->k_taps = k_taps; dut->in_vld = 1;
    dut->eval();
    for (int i = 0; i < CH; i++) y[i] = unpack(dut->y_out, i);
    tick();
    dut->in_vld = 0;
}

// --- 1. causal convolution over a sequence ------------------------------
static void test_causal(int k_taps) {
    reset();
    dut->state_clr = 1; tick(); dut->state_clr = 0;

    int8_t kern8[KMAX];
    uint64_t kern = 0;
    for (int t = 0; t < KMAX; t++) {
        kern8[t] = (int8_t)((rand() % 15) - 7);
        kern |= ((uint64_t)(uint8_t)kern8[t]) << (t * 8);
    }

    std::vector<std::vector<int8_t>> past(CH);   // newest first
    for (int n = 0; n < 24; n++) {
        int8_t x[CH], c[CH], y[CH];
        for (int i = 0; i < CH; i++) {
            x[i] = (int8_t)((rand() % 61) - 30);
            c[i] = (int8_t)((rand() % 61) - 30);
        }
        drive(x, c, kern, k_taps, y);
        for (int i = 0; i < CH; i++) {
            past[i].insert(past[i].begin(), x[i]);
            if ((int)past[i].size() > KMAX) past[i].pop_back();
            int8_t want = ref_step(past[i], kern8, k_taps, c[i]);
            CHECK(y[i] == want, "k=%d n=%d ch=%d: rtl %d != ref %d",
                  k_taps, n, i, (int)y[i], (int)want);
        }
    }
}

// --- 2. state_clr zeroes the history ------------------------------------
// A sequence restarted at position 0 must not see the previous sequence's tail.
static void test_state_clr() {
    reset();
    int8_t kern8[KMAX] = {1, 1, 1, 1, 1, 1, 1};
    uint64_t kern = 0;
    for (int t = 0; t < KMAX; t++) kern |= ((uint64_t)(uint8_t)kern8[t]) << (t * 8);

    int8_t big[CH], one[CH], y[CH];
    for (int i = 0; i < CH; i++) { big[i] = 100; one[i] = 64; }
    for (int n = 0; n < 6; n++) drive(big, one, kern, 7, y);   // fill history

    dut->state_clr = 1; tick(); dut->state_clr = 0;

    int8_t zero[CH]; for (int i = 0; i < CH; i++) zero[i] = 0;
    drive(zero, one, kern, 7, y);
    for (int i = 0; i < CH; i++)
        CHECK(y[i] == 0, "state_clr left history in ch %d: y=%d", i, (int)y[i]);
}

// --- 3. taps beyond k_taps must be dead ---------------------------------
// Fill the history with large values, then shrink the kernel. Anything the
// unused taps still see shows up immediately.
static void test_ktaps_masking() {
    reset();
    dut->state_clr = 1; tick(); dut->state_clr = 0;
    int8_t kern8[KMAX];
    uint64_t kern = 0;
    for (int t = 0; t < KMAX; t++) { kern8[t] = 8; kern |= ((uint64_t)8) << (t * 8); }

    int8_t x[CH], c[CH], y[CH];
    for (int i = 0; i < CH; i++) { x[i] = 100; c[i] = 64; }
    for (int n = 0; n < 7; n++) drive(x, c, kern, 7, y);

    // Now k_taps = 1: only the current sample counts, history is irrelevant.
    for (int i = 0; i < CH; i++) x[i] = 10;
    drive(x, c, kern, 1, y);
    for (int i = 0; i < CH; i++) {
        std::vector<int8_t> only = {10};
        int8_t want = ref_step(only, kern8, 1, c[i]);
        CHECK(y[i] == want, "k_taps=1 leaked history in ch %d: rtl %d != ref %d",
              i, (int)y[i], (int)want);
    }
}

// --- 4. rounding is half-away-from-zero on both signs -------------------
static void test_rounding_symmetry() {
    reset();
    dut->state_clr = 1; tick(); dut->state_clr = 0;
    uint64_t kern = 1;                       // tap0 = 1, rest 0
    int8_t kern8[KMAX] = {1, 0, 0, 0, 0, 0, 0};

    for (int trial = 0; trial < 40; trial++) {
        int8_t x[CH], c[CH], y[CH];
        int8_t v = (int8_t)((rand() % 255) - 127);
        for (int i = 0; i < CH; i++) { x[i] = v; c[i] = (int8_t)((rand() % 255) - 127); }
        dut->state_clr = 1; tick(); dut->state_clr = 0;
        drive(x, c, kern, 1, y);
        for (int i = 0; i < CH; i++) {
            std::vector<int8_t> only = {x[i]};
            int8_t want = ref_step(only, kern8, 1, c[i]);
            CHECK(y[i] == want, "rounding x=%d c=%d ch=%d: rtl %d != ref %d",
                  (int)x[i], (int)c[i], i, (int)y[i], (int)want);
        }
    }
}

// --- 5. the out_vld / y_out contract ------------------------------------
// y_out is combinational from x_in; out_vld is registered. They are therefore
// one cycle apart: when out_vld is high, y_out already reflects the NEXT
// input. Pinned here so a consumer written against out_vld is not silently
// sampling the wrong sample.
static void test_vld_timing() {
    reset();
    dut->state_clr = 1; tick(); dut->state_clr = 0;
    uint64_t kern = 1;
    int8_t a[CH], b[CH], c[CH], y[CH];
    // Values that requantize to DIFFERENT outputs, or 'advanced' is
    // unobservable: 100*127>>>14 = 1 and -100*127>>>14 = -1.
    for (int i = 0; i < CH; i++) { a[i] = 100; b[i] = -100; c[i] = 127; }

    dut->x_in = pack(a); dut->c_gate = pack(c); dut->kern = kern;
    dut->k_taps = 1; dut->in_vld = 1; dut->eval();
    int8_t y_same = unpack(dut->y_out, 0);
    tick();                                   // out_vld now reflects sample a
    dut->x_in = pack(b); dut->eval();
    int8_t y_when_vld = unpack(dut->y_out, 0);

    CHECK(dut->out_vld == 1, "out_vld did not assert one cycle after in_vld");
    CHECK(y_same != y_when_vld,
          "expected y_out to have advanced past the sample out_vld refers to");
    printf("  note: y_out is combinational, out_vld is registered -- when\n");
    printf("        out_vld is high y_out shows the NEXT sample (%d, not %d).\n",
           (int)y_when_vld, (int)y_same);
    printf("        Consumers must sample y_out with in_vld, not out_vld.\n");
    dut->in_vld = 0;
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsonic_conv;
    srand(11);

    for (int k : {1, 3, 5, 7}) test_causal(k);
    test_state_clr();
    test_ktaps_masking();
    test_rounding_symmetry();
    test_vld_timing();

    dut->final();
    printf("  checks %d, failures %d\n", checks, fails);
    printf(fails ? "FAILED\n" : "PASSED\n");
    delete dut;
    return fails ? 1 : 0;
}
