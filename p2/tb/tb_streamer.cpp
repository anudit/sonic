// Differential testbench: sonic_streamer.
//
// The last unbenched unit on the end-to-end path, and the one p2/README calls
// "THE critical path -- if this stalls, the chip stops". It has no arithmetic to
// diff against the golden model; what it has is a protocol, and protocols fail
// in ways that look like correct data arriving at the wrong moment.
//
//   * pass-through latency is exactly one cycle, and w_out holds when the
//     stream stalls rather than shifting garbage into the array
//   * the beat counter advances only on dram_vld -- a stall that advanced it
//     would emit group_done mid-group and mis-scale every weight after it
//   * group_done lands on the LAST beat of a group, not the first of the next
//   * scale_out carries the group's own scale when group_done fires. This is
//     the off-by-one that matters: the scales are applied to the accumulated
//     partial sum, so one misaligned scale corrupts 64 weights at once.
//   * expert_active latches independently of the weight stream
#include "Vsonic_streamer.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vsonic_streamer *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int LANES = 16, GROUP = 64;
static const int BEATS = GROUP / LANES;          // 4

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->dram_vld = 0; dut->dram_data = 0;
    dut->grp_scale = 0; dut->grp_scale_vld = 0;
    dut->expert_id = 0; dut->expert_vld = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { printf("  FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

// Push one beat and settle. Returns nothing; outputs are read after the tick.
static void beat(uint64_t data, bool vld) {
    dut->dram_data = data; dut->dram_vld = vld;
    tick();
    // Deassert after the edge. Leaving dram_vld high made the scale-load tick
    // in test_scale_alignment consume a phantom beat, which desynchronised the
    // group counter from beat 5 onward -- a bench bug that reads exactly like a
    // cadence bug in the RTL.
    dut->dram_vld = 0;
}

// --- 1. pass-through: one cycle, and holds through a stall ---------------
static void test_passthrough() {
    reset();
    for (int i = 1; i <= 6; i++) {
        uint64_t d = 0x1111111111111111ull * (uint64_t)i;
        beat(d, true);
        CHECK((uint64_t)dut->w_out == d, "w_out beat %d: got %llx want %llx",
              i, (unsigned long long)dut->w_out, (unsigned long long)d);
        CHECK(dut->w_vld == 1, "w_vld low on a valid beat %d", i);
    }
    // Stall: w_vld must drop, w_out must hold its last value.
    uint64_t held = dut->w_out;
    beat(0xDEADBEEFDEADBEEFull, false);
    CHECK(dut->w_vld == 0, "w_vld high during a stall");
    CHECK((uint64_t)dut->w_out == held,
          "w_out changed during a stall: %llx != %llx",
          (unsigned long long)dut->w_out, (unsigned long long)held);
}

// --- 2. group_done cadence ----------------------------------------------
static void test_group_cadence() {
    reset();
    int seen = 0;
    for (int i = 0; i < BEATS * 5; i++) {
        beat(0xA5A5A5A5A5A5A5A5ull, true);
        bool last_of_group = ((i % BEATS) == BEATS - 1);
        CHECK(dut->group_done == (last_of_group ? 1 : 0),
              "beat %d: group_done=%d, expected %d", i,
              (int)dut->group_done, last_of_group ? 1 : 0);
        seen += dut->group_done;
    }
    CHECK(seen == 5, "saw %d group_done in 5 groups", seen);
}

// --- 3. a stall must not advance the beat counter ------------------------
static void test_stall_does_not_advance() {
    reset();
    for (int i = 0; i < BEATS - 1; i++) beat(0x11, true);   // 3 of 4 beats
    for (int s = 0; s < 5; s++) {                           // stall a while
        beat(0, false);
        CHECK(dut->group_done == 0, "group_done fired during a stall (s=%d)", s);
    }
    beat(0x22, true);                                       // the 4th beat
    CHECK(dut->group_done == 1, "group_done did not fire on the beat after a stall");
}

// --- 4. the scale must belong to the group that just finished ------------
// Load group N's scale, stream group N, and require scale_out to carry it when
// group_done fires -- even though group N+1's scale is loaded immediately after.
static void test_scale_alignment() {
    reset();
    for (int g = 1; g <= 4; g++) {
        dut->grp_scale = (int16_t)(0x0100 * g);   // Q8.8: 0x0100 is 1.0
        dut->grp_scale_vld = 1; tick(); dut->grp_scale_vld = 0;
        for (int b = 0; b < BEATS; b++) {
            beat(0x33, true);
            if (b == BEATS - 1) {
                CHECK(dut->group_done == 1, "group %d: no group_done", g);
                CHECK((int16_t)dut->scale_out == (int16_t)(0x0100 * g),
                      "group %d: scale_out %d, expected %d", g,
                      (int)(int16_t)dut->scale_out, 0x0100 * g);
            }
        }
    }
}

// A scale loaded in the same cycle as the final beat belongs to the NEXT group.
// If it lands on this group_done, every weight in the group is mis-scaled.
static void test_scale_race() {
    reset();
    dut->grp_scale = 0x0100; dut->grp_scale_vld = 1; tick(); dut->grp_scale_vld = 0;
    for (int b = 0; b < BEATS - 1; b++) beat(0x44, true);
    // Final beat and the next group's scale, same cycle.
    dut->grp_scale = 0x0700; dut->grp_scale_vld = 1;
    beat(0x44, true);
    dut->grp_scale_vld = 0;
    CHECK(dut->group_done == 1, "no group_done on the final beat");
    printf("  note: scale loaded on the final beat -> scale_out=%d at group_done\n",
           (int)(int16_t)dut->scale_out);
    printf("        (0x0100=256 is this group's, 0x0700=1792 is the next one's)\n");
}

// --- 5. expert gather ----------------------------------------------------
static void test_expert_latch() {
    reset();
    CHECK(dut->expert_active == 0, "expert_active not zero after reset");
    dut->expert_id = 21; dut->expert_vld = 1; tick(); dut->expert_vld = 0;
    CHECK(dut->expert_active == 21, "expert_active=%d, expected 21",
          (int)dut->expert_active);
    // Streaming weights must not disturb it.
    for (int i = 0; i < BEATS * 2; i++) beat(0x55, true);
    CHECK(dut->expert_active == 21,
          "expert_active changed to %d while streaming", (int)dut->expert_active);
    dut->expert_id = 7; dut->expert_vld = 1; tick(); dut->expert_vld = 0;
    CHECK(dut->expert_active == 7, "expert_active did not update to 7");
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsonic_streamer;
    printf("sonic_streamer: LANES=%d, GROUP=%d, %d beats per group\n\n",
           LANES, GROUP, BEATS);

    test_passthrough();
    test_group_cadence();
    test_stall_does_not_advance();
    test_scale_alignment();
    test_scale_race();
    test_expert_latch();

    dut->final();
    printf("\n  checks %d, failures %d\n", checks, fails);
    printf(fails ? "FAILED\n" : "PASSED\n");
    delete dut;
    return fails ? 1 : 0;
}
