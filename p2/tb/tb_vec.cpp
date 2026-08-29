// Differential bench for sonic_vec: RESIDUAL_ADD, ROPE_ROTATE, RMSNORM_SCALE
// checked against an independent Q8.8 fixed-point reference computed in
// C++ (not derived from the RTL), plus a floating-point sanity check that
// the fixed-point RoPE recipe actually approximates a real rotation. This
// was previously untested beyond a single ADD case in tb_top.cpp -- ROADMAP
// P2-9 flagged the vector unit as "in the plan's unit list, never written";
// it has since been written but not verified against RoPE/RMSNorm math
// until this bench.
#include "Vsonic_vec.h"
#include "verilated.h"
#include <cmath>
#include <cstdint>
#include <cstdio>

static Vsonic_vec *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

static const int LANES = 64;

template <typename W>
static void pack16(W &sig, const int16_t *v, int n) {
    for (int w = 0; w < (n + 1) / 2; w++) sig[w] = 0;
    for (int i = 0; i < n; i++) {
        uint32_t u = (uint16_t)v[i];
        int wd = i / 2, off = (i % 2) * 16;
        sig[wd] |= (u << off);
    }
}
template <typename W>
static int16_t unpack16(const W &sig, int i) {
    int wd = i / 2, off = (i % 2) * 16;
    return (int16_t)((sig[wd] >> off) & 0xFFFF);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    printf("=== sonic_vec differential bench ===\n");

    dut = new Vsonic_vec;
    dut->rst_n = 0; dut->in_vld = 0; dut->op_sel = 0; dut->last_beat = 0;
    tick(); tick();
    dut->rst_n = 1;
    tick();

    bool ok = true;
    auto check = [&](const char *name, int16_t got, int16_t exp, int tol = 0) {
        bool pass = std::abs((int)got - (int)exp) <= tol;
        ok &= pass;
        printf("  %-20s got=%-6d expected=%-6d %s\n", name, got, exp, pass ? "PASS" : "FAIL");
    };

    // --- RESIDUAL_ADD ---
    {
        int16_t a[LANES], b[LANES];
        for (int i = 0; i < LANES; i++) { a[i] = 100 + i; b[i] = 50 - i; }
        pack16(dut->vec_a, a, LANES);
        pack16(dut->vec_b, b, LANES);
        dut->op_sel = 1; dut->in_vld = 1; dut->last_beat = 1;
        tick();
        dut->in_vld = 0;
        tick();
        for (int i = 0; i < LANES; i += 16) {
            int16_t got = unpack16(dut->vec_out, i);
            int16_t exp = (int16_t)(a[i] + b[i]);
            char name[32]; snprintf(name, sizeof name, "ADD[%d]", i);
            check(name, got, exp);
        }
    }

    // --- ROPE_ROTATE: x0,x1 pairs rotated by (cos,sin) in Q8.8 ---
    {
        int16_t a[LANES] = {0}, cosv[LANES] = {0}, sinv[LANES] = {0};
        // pair 0: x=(100,50) Q8.8, angle ~45deg (cos=sin=181/256=0.707)
        a[0] = 100; a[1] = 50; cosv[0] = 181; sinv[0] = 181;
        // pair 1: x=(256,0) Q8.8 (1.0, 0.0), angle 0 (cos=256,sin=0) -> identity
        a[2] = 256; a[3] = 0; cosv[2] = 256; sinv[2] = 0;
        pack16(dut->vec_a, a, LANES);
        pack16(dut->vec_gamma, cosv, LANES);
        pack16(dut->vec_sin, sinv, LANES);
        dut->op_sel = 2; dut->in_vld = 1; dut->last_beat = 1;
        tick();
        dut->in_vld = 0;
        tick();

        // Independent reference: same Q8.8 fixed-point recipe, computed here,
        // not copied from the RTL.
        auto rope_ref = [](int16_t x0, int16_t x1, int16_t c, int16_t s, int16_t &r0, int16_t &r1) {
            int32_t r0c = ((int32_t)x0 * c - (int32_t)x1 * s) >> 8;
            int32_t r1c = ((int32_t)x0 * s + (int32_t)x1 * c) >> 8;
            r0 = (int16_t)r0c; r1 = (int16_t)r1c;
        };
        int16_t r0, r1;
        rope_ref(a[0], a[1], cosv[0], sinv[0], r0, r1);
        check("ROPE pair0 r0", unpack16(dut->vec_out, 0), r0);
        check("ROPE pair0 r1", unpack16(dut->vec_out, 1), r1);
        rope_ref(a[2], a[3], cosv[2], sinv[2], r0, r1);
        check("ROPE identity r0", unpack16(dut->vec_out, 2), r0);
        check("ROPE identity r1", unpack16(dut->vec_out, 3), r1);

        // Floating-point sanity: pair0's fixed-point result should approximate
        // a real 45-degree rotation of (100/256, 50/256) to within Q8.8
        // quantization error (independent of the fixed-point recipe above).
        double x0f = 100.0 / 256, x1f = 50.0 / 256, cf = 181.0 / 256, sf = 181.0 / 256;
        double r0f = x0f * cf - x1f * sf, r1f = x0f * sf + x1f * cf;
        int16_t r0q = unpack16(dut->vec_out, 0), r1q = unpack16(dut->vec_out, 1);
        double err0 = std::fabs(r0q / 256.0 - r0f), err1 = std::fabs(r1q / 256.0 - r1f);
        bool float_ok = err0 < (2.0 / 256) && err1 < (2.0 / 256);  // within 2 Q8.8 LSBs
        ok &= float_ok;
        printf("  ROPE float sanity     err0=%.5f err1=%.5f (tol 0.00781) %s\n",
               err0, err1, float_ok ? "PASS" : "FAIL");
    }

    // --- RMSNORM_SCALE: a * inv_rms * gamma, Q8.8 ---
    {
        int16_t a[LANES] = {0}, gamma[LANES] = {0};
        a[0] = 200; gamma[0] = 128;   // 0.5 in Q8.8
        int16_t inv_rms = 64;         // 0.25 in Q8.8
        pack16(dut->vec_a, a, LANES);
        pack16(dut->vec_gamma, gamma, LANES);
        dut->inv_rms = inv_rms;
        dut->op_sel = 3; dut->in_vld = 1; dut->last_beat = 1;
        tick();
        dut->in_vld = 0;
        tick();

        int32_t step = ((int32_t)a[0] * inv_rms) >> 8;
        int16_t exp = (int16_t)(((int32_t)(int16_t)step * gamma[0]) >> 8);
        check("RMSNORM[0]", unpack16(dut->vec_out, 0), exp);
    }

    printf(ok ? "PASSED\n" : "FAILED\n");
    dut->final();
    delete dut;
    return ok ? 0 : 1;
}
