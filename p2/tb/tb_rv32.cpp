// Differential bench for sonic_rv32: hand-assembled RV32I program, loaded
// via the boot IMEM port, run for a fixed cycle budget, then checked against
// expected architectural state computed independently in C++ -- not against
// the RTL's own behavior. This is the standard HANDOFF.md sets for every
// unit ("differential against a reference model, not self-consistency").
//
// Program under test:
//   sum = 0; for (i = 1; i <= 10; i++) sum += i;   // sum -> 55
//   dmem[0] = sum;
//   x4 = 5; x5 = 7; x6 = x5 - x4;                   // x6 -> 2
//   x7 = x4 << 2;                                    // x7 -> 20
//   infinite self-loop (parks the core; bench stops on a cycle budget)
#include "Vsonic_rv32.h"
#include "verilated.h"
#include <cstdint>
#include <cstdio>
#include <vector>

static Vsonic_rv32 *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static uint32_t enc_r(int f7, int rs2, int rs1, int f3, int rd, int op) {
    return (f7 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t enc_i(int imm, int rs1, int f3, int rd, int op) {
    return ((imm & 0xFFF) << 20) | (rs1 << 15) | (f3 << 12) | (rd << 7) | op;
}
static uint32_t enc_s(int imm, int rs2, int rs1, int f3, int op) {
    return (((imm >> 5) & 0x7F) << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) |
           ((imm & 0x1F) << 7) | op;
}
static uint32_t enc_b(int imm, int rs2, int rs1, int f3, int op) {
    uint32_t b12 = (imm >> 12) & 1, b11 = (imm >> 11) & 1, b10_5 = (imm >> 5) & 0x3F, b4_1 = (imm >> 1) & 0xF;
    return (b12 << 31) | (b10_5 << 25) | (rs2 << 20) | (rs1 << 15) | (f3 << 12) | (b4_1 << 8) | (b11 << 7) | op;
}
static uint32_t enc_j(int imm, int rd, int op) {
    uint32_t b20 = (imm >> 20) & 1, b19_12 = (imm >> 12) & 0xFF, b11 = (imm >> 11) & 1, b10_1 = (imm >> 1) & 0x3FF;
    return (b20 << 31) | (b10_1 << 21) | (b11 << 20) | (b19_12 << 12) | (rd << 7) | op;
}
static uint32_t ADDI(int rd, int rs1, int imm) { return enc_i(imm, rs1, 0b000, rd, 0x13); }
static uint32_t SLLI(int rd, int rs1, int sh)  { return enc_i(sh & 0x1F, rs1, 0b001, rd, 0x13); }
static uint32_t ADD(int rd, int rs1, int rs2)  { return enc_r(0x00, rs2, rs1, 0b000, rd, 0x33); }
static uint32_t SUB(int rd, int rs1, int rs2)  { return enc_r(0x20, rs2, rs1, 0b000, rd, 0x33); }
static uint32_t BEQ(int rs1, int rs2, int off) { return enc_b(off, rs2, rs1, 0b000, 0x63); }
static uint32_t JAL(int rd, int off)           { return enc_j(off, rd, 0x6F); }
static uint32_t SW(int rs1, int rs2, int off)  { return enc_s(off, rs2, rs1, 0b010, 0x23); }

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    printf("=== sonic_rv32 differential bench ===\n");

    // idx: 0..12, see program layout in file header.
    std::vector<uint32_t> prog(13);
    prog[0]  = ADDI(1, 0, 0);       // x1 = 0            (sum)
    prog[1]  = ADDI(2, 0, 1);       // x2 = 1            (i)
    prog[2]  = ADDI(3, 0, 11);      // x3 = 11           (limit)
    prog[3]  = BEQ(2, 3, (7 - 3) * 4);  // if i==limit goto idx7
    prog[4]  = ADD(1, 1, 2);        // sum += i
    prog[5]  = ADDI(2, 2, 1);       // i++
    prog[6]  = JAL(0, (3 - 6) * 4); // goto idx3
    prog[7]  = SW(0, 1, 0);         // dmem[0] = sum
    prog[8]  = ADDI(4, 0, 5);       // x4 = 5
    prog[9]  = ADDI(5, 0, 7);       // x5 = 7
    prog[10] = SUB(6, 5, 4);        // x6 = x5 - x4
    prog[11] = SLLI(7, 4, 2);       // x7 = x4 << 2
    prog[12] = JAL(0, 0);           // self-loop (park)

    dut = new Vsonic_rv32;
    dut->rst_n = 0;
    dut->imem_ld_en = 0;
    dut->host_wr_en = 0;
    dut->dbg_rf_addr = 0;
    dut->dbg_dmem_addr = 0;
    tick(); tick();

    for (size_t i = 0; i < prog.size(); i++) {
        dut->imem_ld_en = 1;
        dut->imem_ld_addr = i;
        dut->imem_ld_data = prog[i];
        tick();
    }
    dut->imem_ld_en = 0;
    tick();

    dut->rst_n = 1;
    // Run enough cycles: 3 setup + 10-iteration loop (3 instrs/iter) + tail.
    // Single-cycle core except NoC stalls (none used here), so cycles ~= instrs retired.
    for (int c = 0; c < 200; c++) tick();

    dut->dbg_rf_addr = 6; tick(); uint32_t x6 = dut->dbg_rf_data;
    dut->dbg_rf_addr = 7; tick(); uint32_t x7 = dut->dbg_rf_data;
    dut->dbg_rf_addr = 1; tick(); uint32_t x1 = dut->dbg_rf_data;
    dut->dbg_dmem_addr = 0; tick(); uint32_t mem0 = dut->dbg_dmem_data;

    // Independent expected values -- computed in C++, not derived from the RTL.
    uint32_t exp_sum = 0;
    for (int i = 1; i <= 10; i++) exp_sum += i;
    uint32_t exp_x6 = 7 - 5;
    uint32_t exp_x7 = 5u << 2;

    bool ok = true;
    auto check = [&](const char *name, uint32_t got, uint32_t exp) {
        bool pass = (got == exp);
        ok &= pass;
        printf("  %-8s got=%-6u expected=%-6u %s\n", name, got, exp, pass ? "PASS" : "FAIL");
    };
    check("x1(sum)", x1, exp_sum);
    check("dmem[0]", mem0, exp_sum);
    check("x6", x6, exp_x6);
    check("x7", x7, exp_x7);

    printf(ok ? "PASSED\n" : "FAILED\n");
    dut->final();
    delete dut;
    return ok ? 0 : 1;
}
