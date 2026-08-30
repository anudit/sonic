// Testbench: sonic_sram_gate.
//
// sonic_sram_bank.sv has always had a per-bank retention `sleep` port; until
// this unit, sonic_top.sv tied it to '0 -- every one of the 16 banks stayed
// awake regardless of whether the current chunk touched it. This bench
// checks the two properties that make it safe to wire the gate in for real:
//
//   * a bank only sleeps after IDLE_BITS consecutive idle cycles -- it does
//     not sleep prematurely and does not stay awake forever
//   * bank_sleep is LOW on any cycle bank_ce is high, even the very cycle
//     after a bank has gone to sleep -- so a real access is never dropped.
//     This is the property that makes the gate safe to wire in without
//     touching sonic_sram_bank.sv's own sleep-masking logic.
//   * banks are independent: activity on one bank doesn't reset another's
//     idle counter
#include "Vsonic_sram_gate.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>

static Vsonic_sram_gate *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int N_BANKS   = 16;
static const int IDLE_BITS = 8;
static const int IDLE_MAX  = (1 << IDLE_BITS) - 1;   // cycles idle before sleep

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { printf("  FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->bank_ce = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

// --- 1. a freshly-reset bank does not sleep immediately -------------------
static void test_no_premature_sleep() {
    reset();
    CHECK(dut->bank_sleep == 0, "all banks asleep right after reset");
}

// --- 2. a bank idle for IDLE_MAX cycles sleeps; a busy bank never does ----
static void test_sleep_after_idle() {
    reset();
    dut->bank_ce = 0; // every bank idle from here
    for (int i = 0; i < IDLE_MAX - 1; i++) {
        CHECK((dut->bank_sleep & 1) == 0,
              "bank 0 slept early at idle cycle %d (need %d)", i, IDLE_MAX);
        tick();
    }
    tick(); // the IDLE_MAX-th idle cycle
    CHECK((dut->bank_sleep & 1) == 1, "bank 0 never slept after %d idle cycles", IDLE_MAX);
    CHECK(dut->bank_sleep == 0xFFFF, "not every idle bank slept: sleep=%04x", dut->bank_sleep);
}

// --- 3. bank_ce forces sleep low the SAME cycle, even right after sleeping -
static void test_wake_is_glitch_free() {
    reset();
    dut->bank_ce = 0;
    for (int i = 0; i < IDLE_MAX; i++) tick();
    CHECK((dut->bank_sleep & 1) == 1, "setup: bank 0 should be asleep");

    // Access bank 0 the very next cycle -- combinational override, no wake
    // latency to wait out.
    dut->bank_ce = 0x1;
    dut->eval();
    CHECK((dut->bank_sleep & 1) == 0,
          "bank 0 still reports sleep on the same cycle it is accessed -- "
          "sonic_sram_bank would drop this access");
    tick();
    dut->bank_ce = 0;
}

// --- 4. banks are independent ---------------------------------------------
static void test_bank_independence() {
    reset();
    dut->bank_ce = 0;
    for (int i = 0; i < IDLE_MAX + 2; i++) {
        // keep bank 3 continuously busy; every other bank idle
        dut->bank_ce = (1u << 3);
        tick();
    }
    CHECK((dut->bank_sleep & (1u << 3)) == 0, "bank 3 slept despite continuous ce");
    CHECK((dut->bank_sleep & ~(1u << 3) & 0xFFFF) == (0xFFFF & ~(1u << 3)),
          "an idle bank failed to sleep while bank 3 stayed busy: sleep=%04x",
          dut->bank_sleep);
}

int main() {
    dut = new Vsonic_sram_gate;
    printf("sonic_sram_gate: N_BANKS=%d, IDLE_BITS=%d\n\n", N_BANKS, IDLE_BITS);

    test_no_premature_sleep();
    test_sleep_after_idle();
    test_wake_is_glitch_free();
    test_bank_independence();

    printf("\n  checks %d, failures %d\n", checks, fails);
    printf(fails ? "FAILED\n" : "PASSED\n");
    delete dut;
    return fails ? 1 : 0;
}
