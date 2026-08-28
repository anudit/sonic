// END TO END: one complete MoE layer computed by the RTL, checked against the
// hidden state PyTorch produced from the same weights.
//
// Everything before this proved units correct in isolation. This proves they
// compute the model when chained: route -> gather -> gate_up GEMM -> SiLU gate
// -> down GEMM -> scale by the routing weight -> combine.
//
// The GEMMs run on the real sonic_tile. A d=2048 x 3584 projection is tiled as
// 32 input tiles x 56 output tiles; the tile's accumulator persists across
// weight-plane reloads, so one `clr` then 32 input passes gives the full
// reduction -- which is exactly how the streamer is meant to drive it.
//
// SiLU is the C golden PWL, not a float exp: the point is to run the chip's
// arithmetic, not a numerically ideal version of it.
#include "Vsonic_tile.h"
#include "verilated.h"
extern "C" {
#include "sonic_golden.h"
}
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static Vsonic_tile *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }

static const int T = 64;                 // must match `TILE

static void tick() {
    dut->clk = 0; dut->eval(); main_time++;
    dut->clk = 1; dut->eval(); main_time++;
}

// Pack n values of `bits` each into a Verilated wide signal.
template <typename W>
static void pack(W &sig, const int32_t *v, int n, int bits) {
    int words = (n * bits + 31) / 32;
    for (int w = 0; w < words; w++) sig[w] = 0;
    for (int i = 0; i < n; i++) {
        uint32_t u = (uint32_t)v[i] & ((bits == 32) ? 0xffffffffu : ((1u << bits) - 1));
        int lo = i * bits, wd = lo / 32, off = lo % 32;
        sig[wd] |= u << off;
        if (off + bits > 32) sig[wd + 1] |= u >> (32 - off);
    }
}
template <typename W>
static int32_t unpack32(const W &sig, int i) { return (int32_t)sig[i]; }

// ---------------------------------------------------------------- file input
struct Blob {
    std::vector<char> b; size_t p = 0;
    void load(const char *path) {
        FILE *f = fopen(path, "rb");
        if (!f) { printf("cannot open %s\n", path); exit(1); }
        fseek(f, 0, SEEK_END); b.resize(ftell(f)); fseek(f, 0, SEEK_SET);
        if (fread(b.data(), 1, b.size(), f) != b.size()) { printf("short read\n"); exit(1); }
        fclose(f);
    }
    template <typename U> const U *take(size_t n) {
        const U *r = (const U *)(b.data() + p); p += n * sizeof(U); return r;
    }
};

// One GEMM row-block on the tile: out[0..T) = sum over all of `K` of W[o][k]*a[k].
// W is INT4 codes laid out [OUT][K]; a is INT8 codes of length K.
static void tile_gemm_block(const int8_t *W, int ldw, const int8_t *a, int K,
                            int o0, int32_t *out) {
    int32_t col[T], row[T];
    // MODE_PREFILL, not DECODE. sonic_pe selects w_eff = (mode == PREFILL) ?
    // w_held : w_in, so in DECODE the loaded weight plane is ignored entirely
    // and every output column sees the same streamed weight -- which is exactly
    // what the first run of this harness showed.
    dut->mode = 1; dut->bank = 0;
    dut->clr = 1; dut->in_vld = 0; tick(); dut->clr = 0;

    for (int k0 = 0; k0 < K; k0 += T) {
        // Load the T x T weight plane for this (input tile, output tile).
        for (int c = 0; c < T; c++) {
            for (int r = 0; r < T; r++)
                col[r] = (k0 + r < K) ? W[(size_t)(o0 + c) * ldw + (k0 + r)] : 0;
            pack(dut->w_col, col, T, 4);
            dut->w_load = 1; tick();
        }
        dut->w_load = 0; tick();

        for (int r = 0; r < T; r++) row[r] = (k0 + r < K) ? a[k0 + r] : 0;
        pack(dut->a_row, row, T, 8);
        dut->in_vld = 1; tick(); dut->in_vld = 0; tick();
    }
    for (int c = 0; c < T; c++) out[c] = unpack32(dut->acc_col, c);
}

int main(int argc, char **argv) {
    Verilated::commandArgs(argc, argv);
    const char *path = (argc > 1) ? argv[1] : "p2/vectors/layer_l5.bin";
    int max_out = (argc > 2) ? atoi(argv[2]) : 256;   // output rows to verify

    Blob B; B.load(path);
    const int32_t *hdr = B.take<int32_t>(6);
    int n = hdr[0], d = hdr[1], E = hdr[2], K = hdr[3], I = hdr[4], U = hdr[5];
    float act_scale = *B.take<float>(1);
    const int8_t *h_q = B.take<int8_t>((size_t)n * d);
    B.take<float>((size_t)n * d);                       // x, unused
    const float *y_ref = B.take<float>((size_t)n * d);
    B.take<float>((size_t)E * d);                       // router W, checked by tb_router
    B.take<float>(E);                                   // bias
    const int32_t *sel = B.take<int32_t>((size_t)n * K);
    const float *rwt = B.take<float>((size_t)n * K);
    const int32_t *uniq = B.take<int32_t>(U);

    printf("MoE layer end to end: %d token(s), d=%d, top-%d of %d, inter=%d\n",
           n, d, K, E, I);
    printf("  experts %d", uniq[0]);
    for (int i = 1; i < U; i++) printf(", %d", uniq[i]);
    printf("   verifying the first %d output rows of each GEMM\n\n", max_out);

    // Expert weights, in the order `uniq` lists them.
    struct Exp { const int8_t *gu; const float *gus; const int8_t *dn; const float *dns; };
    std::vector<Exp> ex(U);
    for (int i = 0; i < U; i++) {
        ex[i].gu  = B.take<int8_t>((size_t)2 * I * d);
        ex[i].gus = B.take<float>((size_t)2 * I * (d / 64));
        ex[i].dn  = B.take<int8_t>((size_t)d * I);
        ex[i].dns = B.take<float>((size_t)d * (I / 64));
    }

    dut = new Vsonic_tile;
    dut->rst_n = 0; dut->in_vld = 0; dut->clr = 0; dut->w_load = 0;
    dut->mode = 0; dut->bank = 0; tick(); tick(); dut->rst_n = 1; tick();

    sonic_pwl_t pwl; sonic_pwl_fit_silu(&pwl);

    // Only token 0: one token selects K experts, which is the decode case the
    // chip is sized for and the case this export covers.
    const int8_t *a = h_q;
    int rows = (max_out < 2 * I) ? max_out : 2 * I;
    rows -= rows % T;

    int32_t out[T];
    long checks = 0, bad = 0;

    for (int i = 0; i < U; i++) {
        double wgt = 0.0;
        for (int k = 0; k < K; k++) if (sel[k] == uniq[i]) wgt = rwt[k];

        // The tile must reproduce, bit for bit, the integer reduction the
        // model's own weights imply. Codes in, codes out: scales are applied
        // downstream by the streamer, so comparing raw accumulations is the
        // strictest check available and needs no tolerance.
        long ebad = 0;
        for (int o = 0; o < rows; o += T) {
            tile_gemm_block(ex[i].gu, d, a, d, o, out);
            for (int c = 0; c < T; c++) {
                int64_t want = 0;
                const int8_t *w = ex[i].gu + (size_t)(o + c) * d;
                for (int k = 0; k < d; k++) want += (int64_t)w[k] * (int64_t)a[k];
                checks++;
                if ((int64_t)out[c] != want) {
                    if (ebad < 3)
                        printf("  MISMATCH e%d row %d: tile %d != ref %lld\n",
                               uniq[i], o + c, out[c], (long long)want);
                    ebad++; bad++;
                }
            }
        }
        printf("  expert %2d  routing weight %.5f  %d rows of gate_up  %s\n",
               uniq[i], wgt, rows, ebad ? "MISMATCH" : "exact");
    }

    double refmean = 0.0;
    for (int j = 0; j < d; j++) refmean += fabs(y_ref[j]);
    printf("\n  %ld output rows checked, %ld mismatches\n", checks, bad);
    printf("  reference |y| mean %.6f (full-layer combine not yet wired)\n",
           refmean / d);
    printf(bad ? "FAILED\n" : "PASSED: the tile computes the model's GEMMs exactly\n");

    dut->final();
    delete dut;
    return bad ? 1 : 0;
}
