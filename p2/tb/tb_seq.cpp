// Differential testbench: sonic_seq.
//
// The last unbenched unit. p2/ppa/loop.py files it as "control, not datapath;
// f_max here is irrelevant" -- which is exactly why it needs a bench and not a
// timing measurement. A sequencer has no arithmetic to diff against the golden
// model; what it has is a contract about WHICH descriptor is on the bus WHEN,
// and every failure mode is a descriptor arriving one cycle off.
//
// The contract this asserts, in the order it matters:
//
//   * the k-th cycle with desc_vld high presents ring[k]. This is the whole
//     job. An off-by-one here issues layer k's descriptor for layer k+1's
//     work and nothing downstream can detect it -- the fetch is well-formed,
//     it is just fetching the wrong thing.
//   * desc_vld is high for exactly `len` cycles, and pc never reads past
//     len-1. Reading ring[len] emits whatever firmware happened to leave there.
//   * done pulses exactly once, on the last descriptor, and busy clears with it
//   * an EXPERT-opcode descriptor gets its low EXPERT_BITS replaced by the
//     latched patch; every other opcode passes through bit-identical
//   * the patch latches on patch_vld and PERSISTS -- the router decides once
//     per token and the ring replays several expert descriptors per decision
//   * back-to-back starts re-run the ring from 0 without a reset
//   * len == 0 is a no-op rather than a 256-descriptor replay
#include "Vsonic_seq.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static Vsonic_seq *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int DW          = 64;
static const int EXPERT_BITS = 5;
static const uint64_t OP_EXPERT = 0x3ull << (DW - 4);
static const uint64_t OP_GEMM   = 0x1ull << (DW - 4);

static int fails = 0, checks = 0;
#define CHECK(c, ...) do { checks++; if (!(c)) { printf("  FAIL: "); \
    printf(__VA_ARGS__); printf("\n"); fails++; } } while (0)

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static void reset() {
    dut->rst_n = 0; dut->wr_en = 0; dut->start = 0;
    dut->patch_vld = 0; dut->patch_expert = 0; dut->len = 0;
    tick(); tick();
    dut->rst_n = 1; tick();
}

static void load_ring(const std::vector<uint64_t> &prog) {
    for (size_t i = 0; i < prog.size(); i++) {
        dut->wr_en = 1; dut->wr_addr = (int)i; dut->wr_data = prog[i];
        tick();
    }
    dut->wr_en = 0; tick();
}

// Run the ring and capture every (desc, done) pair on a cycle with desc_vld.
struct Beat { uint64_t desc; bool done; bool busy; };
static std::vector<Beat> run(int len, int extra_cycles = 6) {
    std::vector<Beat> got;
    dut->len = len; dut->start = 1; tick();
    dut->start = 0;
    for (int c = 0; c < len + extra_cycles; c++) {
        if (dut->desc_vld)
            got.push_back({(uint64_t)dut->desc, (bool)dut->done, (bool)dut->busy});
        tick();
    }
    return got;
}

// --- 1. the k-th valid descriptor is ring[k] ----------------------------
static void test_descriptor_order() {
    reset();
    std::vector<uint64_t> prog;
    for (int i = 0; i < 8; i++) prog.push_back(OP_GEMM | (uint64_t)(0xA00 + i));
    // Poison one past the end: if the sequencer reads ring[len] it shows here
    // rather than looking like a plausible descriptor.
    prog.push_back(0xDEADBEEFDEADBEEFull);
    load_ring(prog);

    std::vector<Beat> got = run(8);
    CHECK(got.size() == 8, "desc_vld high for %zu cycles, want 8", got.size());
    for (size_t k = 0; k < got.size() && k < 8; k++)
        CHECK(got[k].desc == prog[k],
              "valid cycle %zu: desc=%016llx want ring[%zu]=%016llx",
              k, (unsigned long long)got[k].desc, k,
              (unsigned long long)prog[k]);
    for (size_t k = 0; k < got.size(); k++)
        CHECK(got[k].desc != 0xDEADBEEFDEADBEEFull,
              "valid cycle %zu read ring[len] -- past the end of the program", k);
}

// --- 2. done pulses once, on the last descriptor ------------------------
static void test_done_and_busy() {
    reset();
    std::vector<uint64_t> prog;
    for (int i = 0; i < 5; i++) prog.push_back(OP_GEMM | (uint64_t)i);
    load_ring(prog);

    std::vector<Beat> got = run(5);
    int dones = 0, last_done_at = -1;
    for (size_t k = 0; k < got.size(); k++)
        if (got[k].done) { dones++; last_done_at = (int)k; }
    CHECK(dones == 1, "done pulsed %d times, want exactly 1", dones);
    CHECK(last_done_at == (int)got.size() - 1,
          "done landed on valid cycle %d, want the last (%zu)",
          last_done_at, got.size() - 1);
    CHECK(dut->busy == 0, "busy still high after done");
    CHECK(dut->desc_vld == 0, "desc_vld still high after done");
}

// --- 3. expert patch splices, other opcodes pass through ----------------
static void test_expert_patch() {
    reset();
    std::vector<uint64_t> prog = {
        OP_GEMM   | 0x1F,          // low bits set; must survive untouched
        OP_EXPERT | 0x00,
        OP_GEMM   | 0x07,
        OP_EXPERT | 0x1F,          // low bits set; must be overwritten
    };
    load_ring(prog);

    dut->patch_expert = 19; dut->patch_vld = 1; tick();
    dut->patch_vld = 0;

    std::vector<Beat> got = run((int)prog.size());
    const uint64_t mask = (1ull << EXPERT_BITS) - 1;
    for (size_t k = 0; k < got.size() && k < prog.size(); k++) {
        bool is_expert = (prog[k] >> (DW - 4)) == 0x3;
        uint64_t want = is_expert ? ((prog[k] & ~mask) | 19u) : prog[k];
        CHECK(got[k].desc == want,
              "%s descriptor %zu: got %016llx want %016llx",
              is_expert ? "EXPERT" : "non-expert", k,
              (unsigned long long)got[k].desc, (unsigned long long)want);
    }
}

// --- 4. the patch persists across descriptors and is overridable --------
static void test_patch_persistence() {
    reset();
    std::vector<uint64_t> prog(4, OP_EXPERT);
    load_ring(prog);

    dut->patch_expert = 7; dut->patch_vld = 1; tick();
    dut->patch_vld = 0;
    std::vector<Beat> got = run(4);
    const uint64_t mask = (1ull << EXPERT_BITS) - 1;
    for (size_t k = 0; k < got.size(); k++)
        CHECK((got[k].desc & mask) == 7,
              "patch did not persist to expert descriptor %zu: got %llu",
              k, (unsigned long long)(got[k].desc & mask));

    // A new routing decision replaces it, with no reset in between.
    dut->patch_expert = 23; dut->patch_vld = 1; tick();
    dut->patch_vld = 0;
    got = run(4);
    for (size_t k = 0; k < got.size(); k++)
        CHECK((got[k].desc & mask) == 23,
              "second patch not applied to descriptor %zu: got %llu",
              k, (unsigned long long)(got[k].desc & mask));
}

// --- 5. re-start without a reset ----------------------------------------
static void test_restart() {
    reset();
    std::vector<uint64_t> prog;
    for (int i = 0; i < 4; i++) prog.push_back(OP_GEMM | (uint64_t)(0x500 + i));
    load_ring(prog);

    std::vector<Beat> a = run(4);
    std::vector<Beat> b = run(4);
    CHECK(a.size() == b.size(), "second run emitted %zu descriptors, first %zu",
          b.size(), a.size());
    for (size_t k = 0; k < a.size() && k < b.size(); k++)
        CHECK(a[k].desc == b[k].desc,
              "restart diverged at %zu: %016llx vs %016llx", k,
              (unsigned long long)a[k].desc, (unsigned long long)b[k].desc);
}

// --- 6. idle: nothing on the bus until start ----------------------------
static void test_idle() {
    reset();
    std::vector<uint64_t> prog(4, OP_GEMM);
    load_ring(prog);
    for (int c = 0; c < 8; c++) {
        CHECK(dut->desc_vld == 0, "desc_vld high while idle (cycle %d)", c);
        CHECK(dut->busy == 0, "busy high while idle (cycle %d)", c);
        CHECK(dut->done == 0, "done high while idle (cycle %d)", c);
        tick();
    }
}

// --- 7. len == 0 is a no-op, not a full-ring replay ---------------------
// `pc == len - 1` with len == 0 wraps to DEPTH-1, so an unguarded start on an
// empty program would replay all 256 ring slots.
static void test_zero_length() {
    reset();
    std::vector<uint64_t> prog(4, OP_GEMM);
    load_ring(prog);
    std::vector<Beat> got = run(0, 12);
    CHECK(got.empty(), "len=0 emitted %zu descriptors, want none", got.size());
    CHECK(dut->busy == 0, "len=0 left the sequencer busy");
}

// --- 8. real producer ring test ------------------------------------------
static void test_real_producer_ring() {
    FILE *f = fopen("p3/out/ring.bin", "rb");
    if (!f) {
        printf("  note: p3/out/ring.bin not found, skipping real-ring test\n");
        return;
    }
    std::vector<uint64_t> prog;
    uint64_t d;
    while (fread(&d, sizeof(uint64_t), 1, f) == 1) {
        prog.push_back(d);
    }
    fclose(f);

    reset();
    load_ring(prog);
    std::vector<Beat> got = run((int)prog.size());
    CHECK(got.size() == prog.size(), "real ring size %zu, emitted %zu",
          prog.size(), got.size());
    for (size_t k = 0; k < got.size() && k < prog.size(); k++) {
        CHECK(got[k].desc == prog[k], "desc mismatch at %zu: %016llx != %016llx",
              k, (unsigned long long)got[k].desc, (unsigned long long)prog[k]);
    }
    if (!got.empty()) {
        CHECK(got.back().done == true, "final descriptor missing done pulse");
    }
    printf("  real ring: loaded and verified %zu descriptors from producer\n", prog.size());
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    dut = new Vsonic_seq;
    printf("sonic_seq: DW=%d, EXPERT_BITS=%d\n\n", DW, EXPERT_BITS);

    test_descriptor_order();
    test_done_and_busy();
    test_expert_patch();
    test_patch_persistence();
    test_restart();
    test_idle();
    test_zero_length();
    test_real_producer_ring();

    dut->final();
    printf("\n  checks %d, failures %d\n", checks, fails);
    printf(fails ? "FAILED\n" : "PASSED\n");
    delete dut;
    return fails ? 1 : 0;
}

