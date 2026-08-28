// Differential testbench: sonic_pe vs the P0 C golden accumulator.
//
// p2/README.md said of this unit: "drafted but has no bench yet. Do not trust
// its numbers." The PE wraps ACC_BANKS golden accumulators behind a bank mux
// and a dataflow-mode weight select, so the arithmetic is already covered by
// tb_acc. What is NOT covered, and what this bench exists for, is the wrapper:
//
//   * bank isolation -- accumulating into one bank must leave the other seven
//     bit-identical. This is the bug that makes speculative decode return one
//     candidate's logits for another.
//   * mode select -- MODE_PREFILL must use the LATCHED weight and ignore w_in;
//     MODE_DECODE must use w_in and ignore w_held. Getting these backwards
//     still produces plausible numbers in the mode you tested.
//   * a mode switch mid-reduction must not disturb the accumulator.
//   * systolic pass-through latency: exactly one cycle, unconditionally --
//     not gated by `en`, or a stalled column silently drops operands.
//   * overflow reporting per bank.
#include "Vsonic_pe.h"
#include "verilated.h"
extern "C" {
#include "sonic_golden.h"
}
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vsonic_pe *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int BANKS = 8;          // must match `ACC_BANKS
static const int MODE_DECODE = 0, MODE_PREFILL = 1;

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->mode = MODE_DECODE; dut->clr = 0; dut->en = 0;
    dut->bank = 0; dut->w_in = 0; dut->w_load = 0; dut->a_in = 0;
    tick(); tick();
    dut->rst_n = 1;
    tick();
}

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { printf("  FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static void clear_bank(int b) {
    dut->bank = b; dut->en = 0; dut->clr = 1; tick(); dut->clr = 0;
}

// Drive one MAC into `bank`. Weight reaches the array via w_in in DECODE and
// via the latched w_held in PREFILL, so the caller says which it means.
static void mac(int bank, int mode, int8_t w, int8_t a) {
    dut->bank = bank; dut->mode = mode;
    dut->w_in = (mode == MODE_DECODE) ? w : 0;   // PREFILL must ignore w_in
    dut->a_in = a; dut->en = 1;
    tick();
    dut->en = 0;
}

static void load_weight(int8_t w) {
    dut->w_load = 1; dut->w_in = w; tick(); dut->w_load = 0; dut->w_in = 0;
}

static int32_t read_bank(int b) {
    dut->bank = b; dut->eval();
    return (int32_t)dut->acc;
}

// --- 1. bank isolation --------------------------------------------------
// Fill every bank with a distinct dot product, interleaving the banks so a
// shared-state bug cannot hide behind sequential access.
static void test_bank_isolation() {
    reset();
    std::vector<int32_t> gold(BANKS, 0);
    std::vector<std::vector<int8_t>> ws(BANKS), as(BANKS);
    for (int b = 0; b < BANKS; b++) {
        clear_bank(b);
        for (int i = 0; i < 40; i++) {
            ws[b].push_back((int8_t)((rand() % 16) - 8));
            as[b].push_back((int8_t)((rand() % 255) - 127));
        }
    }
    // Interleave: step i of every bank, then step i+1 of every bank.
    for (int i = 0; i < 40; i++)
        for (int b = 0; b < BANKS; b++)
            mac(b, MODE_DECODE, ws[b][i], as[b][i]);

    for (int b = 0; b < BANKS; b++) {
        sonic_acc_t g; sonic_acc_init(&g);
        for (int i = 0; i < 40; i++) sonic_acc_mac(&g, ws[b][i], as[b][i]);
        gold[b] = sonic_acc_final(&g);
        CHECK(read_bank(b) == gold[b],
              "bank %d interleaved: rtl %d != golden %d", b, read_bank(b), gold[b]);
    }

    // Clearing one bank must not touch the others.
    int32_t before = read_bank(3);
    clear_bank(5);
    CHECK(read_bank(3) == before, "clearing bank 5 disturbed bank 3");
    CHECK(read_bank(5) == 0, "clearing bank 5 did not zero it");
}

// --- 2. dataflow mode ---------------------------------------------------
static void test_mode_select() {
    reset();
    clear_bank(0);
    load_weight(6);
    // PREFILL: w_in is driven to 0 by mac(); the held 6 must be used.
    sonic_acc_t g; sonic_acc_init(&g);
    for (int i = 0; i < 12; i++) {
        int8_t a = (int8_t)(10 + i);
        mac(0, MODE_PREFILL, 6, a);
        sonic_acc_mac(&g, 6, a);
    }
    CHECK(read_bank(0) == sonic_acc_final(&g),
          "PREFILL ignored the latched weight: rtl %d != golden %d",
          read_bank(0), sonic_acc_final(&g));

    // DECODE: w_in must be used and the stale w_held ignored.
    reset(); clear_bank(1);
    load_weight(-7);                       // stale value that must NOT be used
    sonic_acc_t g2; sonic_acc_init(&g2);
    for (int i = 0; i < 12; i++) {
        int8_t w = (int8_t)((i % 15) - 7), a = (int8_t)(20 - i);
        mac(1, MODE_DECODE, w, a);
        sonic_acc_mac(&g2, w, a);
    }
    CHECK(read_bank(1) == sonic_acc_final(&g2),
          "DECODE used the held weight: rtl %d != golden %d",
          read_bank(1), sonic_acc_final(&g2));
}

// --- 3. mode switch mid-reduction ---------------------------------------
static void test_mode_switch_midway() {
    reset(); clear_bank(2);
    load_weight(3);
    sonic_acc_t g; sonic_acc_init(&g);
    for (int i = 0; i < 20; i++) {
        int8_t a = (int8_t)(i - 10);
        if (i < 10) { mac(2, MODE_PREFILL, 3, a); sonic_acc_mac(&g, 3, a); }
        else        { mac(2, MODE_DECODE,  5, a); sonic_acc_mac(&g, 5, a); }
    }
    CHECK(read_bank(2) == sonic_acc_final(&g),
          "mode switch mid-reduction corrupted the accumulator: rtl %d != golden %d",
          read_bank(2), sonic_acc_final(&g));
}

// --- 4. systolic pass-through -------------------------------------------
// One cycle, and NOT gated by en: a column that stalls its accumulator must
// still forward operands, or the whole array shifts out of step.
static void test_passthrough() {
    reset();
    dut->en = 0;
    for (int i = 1; i <= 8; i++) {
        int8_t w = (int8_t)((i % 15) - 7), a = (int8_t)(i * 7);
        dut->w_in = w; dut->a_in = a;
        tick();
        CHECK((int8_t)dut->w_out == w, "w_out latency/gating: got %d want %d",
              (int)(int8_t)dut->w_out, (int)w);
        CHECK((int8_t)dut->a_out == a, "a_out latency/gating: got %d want %d",
              (int)(int8_t)dut->a_out, (int)a);
    }
}

// --- 5. the worst case must NOT overflow --------------------------------
// This is the property the whole 12 -> 16 bit argument exists to establish:
// at fold 16 with INT4 x INT8, the local bound is 16*8*128 = 16,384, which
// fits a 16-bit signed stage (+/-32,767) with 2x headroom. Driving the most
// extreme operand pattern the format permits must therefore leave ovf clear.
// An assertion that overflow DOES fire would be testing a bug, not a spec.
//
// It also pins reporting scope: `ovf` is `|bank_ovf`, so an overflow in any
// bank raises it on every read. Nothing here overflows, so that is recorded
// rather than exercised -- per-bank reporting would need a mux, and whether
// it is wanted is a design question this bench should not silently answer.
static void test_worst_case_does_not_overflow() {
    reset();
    for (int b = 0; b < BANKS; b++) clear_bank(b);

    // Every product at maximum magnitude and identical sign -- the pattern the
    // bound is computed against. 2048 is the deepest reduction the array sees
    // (D = 2048), which is also the mid-stage's worst case.
    sonic_acc_t g; sonic_acc_init(&g);
    for (int i = 0; i < 2048; i++) {
        mac(7, MODE_DECODE, -8, -128);
        sonic_acc_mac(&g, -8, -128);
    }
    dut->eval();
    CHECK(dut->ovf == 0,
          "worst-case operands overflowed a stage: the 16/24/32 widths do not "
          "hold the bound they were sized for");
    CHECK(g.overflow == 0, "golden model flagged overflow on the worst case");
    CHECK(read_bank(7) == sonic_acc_final(&g),
          "worst-case reduction: rtl %d != golden %d",
          read_bank(7), sonic_acc_final(&g));

    // The other banks must still be clean after 2048 saturating MACs next door.
    for (int b = 0; b < BANKS - 1; b++)
        CHECK(read_bank(b) == 0, "bank %d disturbed by saturating bank 7", b);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsonic_pe;
    srand(1);

    test_bank_isolation();
    test_mode_select();
    test_mode_switch_midway();
    test_passthrough();
    test_worst_case_does_not_overflow();

    dut->final();
    printf("  checks %d, failures %d\n", checks, fails);
    printf(fails ? "FAILED\n" : "PASSED\n");
    delete dut;
    return fails ? 1 : 0;
}
