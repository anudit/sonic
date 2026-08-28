// Drives the router RTL with REAL tensors from LFM2.5-8B-A1B and checks it
// selects the same experts PyTorch did.
//
// This is the test that matters. A bench built on random stimulus proves the
// design is self-consistent; this one proves it computes the model. The
// reference selections in the vector file came out of the 8.47 B checkpoint.
#include "Vsonic_router.h"
#include "verilated.h"
#ifdef DUMP_VCD
#include "verilated_vcd_c.h"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <vector>

static Vsonic_router *dut;
static vluint64_t main_time = 0;
#ifdef DUMP_VCD
static VerilatedVcdC *vcd = nullptr;
#endif
double sc_time_stamp() { return main_time; }

static void tick() {
  dut->clk = 0; dut->eval();
#ifdef DUMP_VCD
  if (vcd) vcd->dump(main_time);
#endif
  main_time++;
  dut->clk = 1; dut->eval();
#ifdef DUMP_VCD
  if (vcd) vcd->dump(main_time);
#endif
  main_time++;
}

static const int LANES = 64;
static const int Q16 = 65536;
static const int AW = 12;   // must match ROUTER_A_BITS / ROUTER_W_BITS

// Pack `n` signed AW-bit values, LSB-first, into a Verilator wide signal.
// Verilator models a port wider than 64 bits as VlWide<N>, which is indexable
// but is not a plain pointer -- hence the reference template.
template <typename W>
static void pack12(W &dst, const int16_t *src, int n) {
  const int words = (n * AW + 31) / 32;
  for (int w = 0; w < words; w++) dst[w] = 0;
  for (int i = 0; i < n; i++) {
    uint32_t v = (uint32_t)(src[i] & ((1 << AW) - 1));
    int bit = i * AW;
    dst[bit >> 5] |= v << (bit & 31);
    int spill = (bit & 31) + AW - 32;
    if (spill > 0) dst[(bit >> 5) + 1] |= v >> (AW - spill);
  }
}

struct Vec {
  int n, d, E, K;
  float act_scale;
  std::vector<int8_t> h_q, W_q;
  std::vector<float> W_s, bias;
  std::vector<int32_t> sel;
  std::vector<float> h_f, W_f;
};

static bool load(const char *path, Vec &v) {
  FILE *f = fopen(path, "rb");
  if (!f) { printf("cannot open %s -- run: make vectors\n", path); return false; }
  int hdr[4];
  if (fread(hdr, 4, 4, f) != 4) return false;
  v.n = hdr[0]; v.d = hdr[1]; v.E = hdr[2]; v.K = hdr[3];
  fread(&v.act_scale, 4, 1, f);
  v.h_q.resize((size_t)v.n * v.d);  fread(v.h_q.data(), 1, v.h_q.size(), f);
  v.W_q.resize((size_t)v.E * v.d);  fread(v.W_q.data(), 1, v.W_q.size(), f);
  v.W_s.resize((size_t)v.E * (v.d / 64)); fread(v.W_s.data(), 4, v.W_s.size(), f);
  v.bias.resize(v.E);               fread(v.bias.data(), 4, v.E, f);
  v.sel.resize((size_t)v.n * v.K);  fread(v.sel.data(), 4, v.sel.size(), f);
  v.h_f.resize((size_t)v.n * v.d);  fread(v.h_f.data(), 4, v.h_f.size(), f);
  v.W_f.resize((size_t)v.E * v.d);  fread(v.W_f.data(), 4, v.W_f.size(), f);
  fclose(f);
  return true;
}

// Symmetric per-tensor quantization to `bits`, returning codes and the scale.
template <typename T>
static float quantize(const std::vector<float> &x, std::vector<T> &out, int bits) {
  float m = 0;
  for (float v : x) m = std::max(m, std::fabs(v));
  const int lim = (1 << (bits - 1)) - 1;
  float s = m / lim;
  if (s == 0) s = 1;
  out.resize(x.size());
  for (size_t i = 0; i < x.size(); i++)
    out[i] = (T)std::max(-lim - 1, std::min(lim, (int)std::lrint(x[i] / s)));
  return s;
}

static void load_sigmoid_table() {
  // 16 segments of width 1 over [-8, 8), Q16 -- the firmware-loadable table.
  //
  // Slope from the endpoints, but the intercept is MINIMAX-CENTRED rather than
  // taken from the chord. A chord through a segment of a convex region sits
  // entirely above the curve, which gave every score a systematic +0.005 bias
  // and cost ~3% of routing agreement. Centring the error band halves the peak
  // error and removes the bias, for free -- it is a different constant in the
  // same table, not extra hardware.
#ifndef SEGS_OVERRIDE
#define SEGS_OVERRIDE 16
#endif
  const int SEGS = SEGS_OVERRIDE;   // must match ROUTER_PWL_SEGS in the RTL
  const double W = 16.0 / SEGS;
  for (int s = 0; s < SEGS; s++) {
    double x0 = -8.0 + s * W, x1 = x0 + W;
    double y0 = 1.0 / (1.0 + std::exp(-x0)), y1 = 1.0 / (1.0 + std::exp(-x1));
    double m = (y1 - y0) / W;
    double lo = 1e30, hi = -1e30;
    for (int k = 0; k <= 64; k++) {
      double x = x0 + (x1 - x0) * k / 64.0;
      double r = 1.0 / (1.0 + std::exp(-x)) - m * x;
      lo = std::min(lo, r); hi = std::max(hi, r);
    }
    double c = 0.5 * (lo + hi);
    dut->tbl_we = 1;
    dut->tbl_addr = s;
    dut->tbl_slope = (int32_t)std::lrint(m * Q16);
    dut->tbl_icept = (int32_t)std::lrint(c * Q16);
    tick();
  }
  dut->tbl_we = 0;
  tick();
  printf("  PWL sigmoid: %d segments of width %.4f\n", SEGS, W);
}

int main(int argc, char **argv) {
  Verilated::commandArgs(argc, argv);
  Vec v;
  const char *path = argc > 1 ? argv[1] : "p2/vectors/router_l5.bin";
  if (!load(path, v)) return 2;
  int limit = argc > 2 ? atoi(argv[2]) : 128;
  int n = std::min(v.n, limit);

  printf("router RTL vs the 8.47B model\n");
  printf("  vectors: %s  (%d tokens, d=%d, %d experts, top-%d)\n\n",
         path, v.n, v.d, v.E, v.K);

  // Quantize the float references to the router's INT12 datapath.
  std::vector<int16_t> hq, wq;
  float hs = quantize(v.h_f, hq, 12);
  float ws = quantize(v.W_f, wq, 12);

  dut = new Vsonic_router;
#ifdef DUMP_VCD
  Verilated::traceEverOn(true);
  vcd = new VerilatedVcdC; dut->trace(vcd, 99); vcd->open("build/router.vcd");
#endif
  dut->rst_n = 0; dut->start = 0; dut->in_vld = 0; dut->tbl_we = 0;
  tick(); tick();
  dut->rst_n = 1; tick();

  load_sigmoid_table();

  // Q32 scale folding both operand scales into the sigmoid's input. In Q16
  // this product underflows to zero -- see the note on the port.
  dut->logit_scale = (uint32_t)(int32_t)std::llrint((double)hs * ws * 4294967296.0);
  printf("  hs=%.3e ws=%.3e -> logit_scale(Q32)=%d\n",
         hs, ws, (int32_t)dut->logit_scale);
  // Ports are packed vectors; Verilator exposes wide signals as arrays of
  // 32-bit words, so an int32 field per expert maps one-to-one here.
  for (int e = 0; e < v.E; e++)
    dut->bias[e] = (uint32_t)(int32_t)std::lrint((double)v.bias[e] * Q16);

  int exact = 0, setmatch = 0, top1 = 0;
  for (int t = 0; t < n; t++) {
    dut->start = 1; dut->in_vld = 0; tick();
    dut->start = 0; dut->in_vld = 1;

    for (int e = 0; e < v.E; e++)
      for (int s = 0; s < v.d / LANES; s++) {
        // Pack LANES x 12-bit operands into the flat port words.
        pack12(dut->a_in, &hq[(size_t)t * v.d + s * LANES], LANES);
        pack12(dut->w_in, &wq[(size_t)e * v.d + s * LANES], LANES);
        tick();
      }
    dut->in_vld = 0;
    tick();

    if (t == 0) {
      // Reference: float logits -> sigmoid -> +bias, computed here so a
      // mismatch points at the RTL rather than at the vector file.
      printf("\n  --- token 0 debug ---\n");
      for (int e = 0; e < 6; e++) {
        double L = 0;
        for (int i = 0; i < v.d; i++)
          L += (double)v.h_f[(size_t)t * v.d + i] * v.W_f[(size_t)e * v.d + i];
        double ref = 1.0 / (1.0 + std::exp(-L)) + v.bias[e];
        printf("   e%-2d  rtl_score=%12d (%.5f)   ref=%.5f  logit=%.4f\n",
               e, (int)dut->score[e], (double)(int)dut->score[e] / Q16, ref, L);
      }
      printf("   rtl sel = %d %d %d %d   want = %d %d %d %d\n\n",
             (int)((dut->sel >> 0) & 31), (int)((dut->sel >> 5) & 31),
             (int)((dut->sel >> 10) & 31), (int)((dut->sel >> 15) & 31),
             v.sel[0], v.sel[1], v.sel[2], v.sel[3]);
    }
    std::vector<int> got(v.K), want(v.K);
    const int EW = 5;   // clog2(32)
    for (int k = 0; k < v.K; k++) {
      got[k]  = (dut->sel >> (k * EW)) & ((1 << EW) - 1);
      want[k] = v.sel[(size_t)t * v.K + k];
    }
    if (got[0] == want[0]) top1++;
    if (got == want) exact++;
    std::sort(got.begin(), got.end());
    std::sort(want.begin(), want.end());
    if (got == want) setmatch++;
  }

#ifdef DUMP_VCD
  if (vcd) { vcd->close(); printf("  wrote build/router.vcd -- open with gtkwave\n"); }
#endif
  dut->final(); delete dut;

  double agree = (double)setmatch / n;
  printf("  tokens checked      %d\n", n);
  printf("  top-1 expert match  %.4f\n", (double)top1 / n);
  printf("  top-%d set match     %.4f   (gate >= 0.995)\n", v.K, agree);
  printf("  exact order match   %.4f\n", (double)exact / n);
  printf("\n%s\n", agree >= 0.995 ? "PASSED" : "FAILED");
  return agree >= 0.995 ? 0 : 1;
}
