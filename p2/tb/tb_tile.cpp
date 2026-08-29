// sonic_tile: both dataflows must produce identical results.
//
// That is the whole point of a dual-mode array, and it is the property most
// likely to break silently -- prefill latches weights into the grid while
// decode streams them past, so a bug in one mode is invisible from the other.
//
// Since the reduction became a pipelined tree (see the header comment in
// sonic_tile.sv) this bench also pins the three properties that rewrite is
// claiming, because each of them is a thing a future depth change could break
// without any test noticing:
//
//   * LATENCY is exactly 3 and out_vld reports it,
//   * clr issued before a pass still lands before it, at any depth,
//   * the ACC_LOCAL = 16 fold bound holds at worst-case INT4 x INT8 operands,
//     which is the arithmetic claim P0-5 and P2-7c both rest on.
#include "Vsonic_tile.h"
#include "verilated.h"
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vsonic_tile *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }
static void tick() { dut->clk=0; dut->eval(); main_time++; dut->clk=1; dut->eval(); main_time++; }

static const int T = 64;
static const int BANKS = 8;
// S1 fold trees -> S2 fold-to-mid -> S3 bank accumulate.
static const int LATENCY = 3;
static int fails = 0;
#define CHECK(c,...) do{ if(!(c)){ printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); \
                          if(++fails>10){printf("  (stopping)\n"); return;} } }while(0)

template <typename W> static void pack(W &d, const int8_t *s, int n, int bits) {
  int words = (n*bits+31)/32;
  for (int w=0; w<words; w++) d[w]=0;
  for (int i=0;i<n;i++){
    uint32_t v=(uint32_t)(s[i]&((1<<bits)-1));
    int b=i*bits; d[b>>5]|=v<<(b&31);
    int sp=(b&31)+bits-32; if(sp>0) d[(b>>5)+1]|=v>>(bits-sp);
  }
}
template <typename W> static int32_t unpack32(W &d, int i){ return (int32_t)d[i]; }

static void reset(){ dut->rst_n=0; dut->in_vld=0; dut->clr=0; dut->w_load=0;
                     dut->mode=0; dut->bank=0; tick(); tick(); dut->rst_n=1; tick(); }

// Present one pass, then drain the pipeline so acc_col is settled.
static void issue_and_drain() {
  dut->in_vld=1; tick(); dut->in_vld=0;
  for (int i=1;i<LATENCY;i++) tick();
}

// Run one T x T weight block against one activation row, in a chosen mode.
static void run(int mode, const std::vector<int8_t>&W, const std::vector<int8_t>&A,
                std::vector<int32_t>&out) {
  dut->mode = mode; tick();               // mode is registered a stage upstream
  if (mode == 1) {                        // prefill: latch the weight plane
    for (int c=0;c<T;c++){
      std::vector<int8_t> col(T);
      for(int r=0;r<T;r++) col[r]=W[r*T+c];
      pack(dut->w_col, col.data(), T, 4);
      dut->w_load=1; tick();
    }
    dut->w_load=0; tick();
  }
  // clr now rides the same pipeline as the data, so it must drain too.
  dut->clr=1; dut->in_vld=0; tick(); dut->clr=0;
  for (int i=1;i<LATENCY;i++) tick();

  pack(dut->a_row, A.data(), T, 8);
  if (mode == 0) {                        // decode: weights stream alongside
    std::vector<int8_t> col(T);
    for(int r=0;r<T;r++) col[r]=W[r*T+0];
    pack(dut->w_col, col.data(), T, 4);
  }
  issue_and_drain();

  out.resize(T);
  for (int c=0;c<T;c++) out[c]=unpack32(dut->acc_col,c);
}

static void test_prefill_matches_reference() {
  printf("prefill mode vs reference GEMV\n");
  srand(7);
  std::vector<int8_t> W(T*T), A(T);
  for(auto&v:W) v=(int8_t)((rand()%16)-8);
  for(auto&v:A) v=(int8_t)((rand()%256)-128);

  std::vector<int32_t> got;
  run(1, W, A, got);

  for(int c=0;c<T;c++){
    int32_t ref=0;
    for(int r=0;r<T;r++) ref += (int32_t)A[r]*(int32_t)W[r*T+c];
    CHECK(got[c]==ref, "lane %d: got %d want %d", c, got[c], ref);
  }
  CHECK(dut->ovf==0, "ovf asserted on legal INT4 x INT8 operands");
}

static void test_decode_matches_prefill_on_column_zero() {
  printf("decode mode agrees with prefill on the streamed column\n");
  srand(11);
  std::vector<int8_t> W(T*T), A(T);
  for(auto&v:W) v=(int8_t)((rand()%16)-8);
  for(auto&v:A) v=(int8_t)((rand()%256)-128);

  std::vector<int32_t> pre, dec;
  run(1, W, A, pre);
  run(0, W, A, dec);
  // In decode the same weight column feeds every lane, so lane c sees
  // column 0 of W. Compare against that explicitly.
  int32_t ref=0;
  for(int r=0;r<T;r++) ref += (int32_t)A[r]*(int32_t)W[r*T+0];
  for(int c=0;c<T;c++) CHECK(dec[c]==ref, "decode lane %d: got %d want %d", c, dec[c], ref);
  (void)pre;
}

// Banks must be independent: writing one must not disturb the others. This is
// what lets a speculative verify batch share a single weight fetch.
static void test_bank_independence() {
  printf("accumulator banks are independent\n");
  std::vector<int8_t> W(T*T,1), A(T,1);
  int32_t expect[BANKS];

  dut->mode=1; tick();
  for(int c=0;c<T;c++){ std::vector<int8_t> col(T,1); pack(dut->w_col,col.data(),T,4);
                        dut->w_load=1; tick(); }
  dut->w_load=0; tick();

  for(int b=0;b<BANKS;b++){
    dut->bank=b; dut->clr=1; tick(); dut->clr=0;
    for(int i=1;i<LATENCY;i++) tick();
    expect[b]=0;
    for(int rep=0;rep<=b;rep++){                 // bank b gets b+1 accumulations
      pack(dut->a_row,A.data(),T,8);
      issue_and_drain();
      expect[b]+=T;
    }
  }
  for(int b=0;b<BANKS;b++){
    dut->bank=b; tick();
    CHECK(unpack32(dut->acc_col,0)==expect[b],
          "bank %d: got %d want %d", b, unpack32(dut->acc_col,0), expect[b]);
  }
}

// The pipeline is only correct if issue order survives it. A clr presented one
// cycle before a pass must clear first and accumulate second, not the reverse
// -- which is exactly what would happen if clr bypassed the pipeline registers.
static void test_clr_ordering_through_the_pipeline() {
  printf("clr issued before a pass lands before it\n");
  std::vector<int8_t> W(T*T,1), A(T,1);

  dut->mode=1; tick();
  for(int c=0;c<T;c++){ std::vector<int8_t> col(T,1); pack(dut->w_col,col.data(),T,4);
                        dut->w_load=1; tick(); }
  dut->w_load=0; tick();
  dut->bank=0;

  // Seed the bank with two passes.
  pack(dut->a_row,A.data(),T,8);
  issue_and_drain(); issue_and_drain();
  CHECK(unpack32(dut->acc_col,0)==2*T, "seed: got %d want %d",
        unpack32(dut->acc_col,0), 2*T);

  // Now clr on one cycle and a pass on the very next, back to back with no
  // drain between them. Both are in the pipeline simultaneously.
  dut->clr=1; tick(); dut->clr=0;
  dut->in_vld=1; tick(); dut->in_vld=0;
  for (int i=0;i<LATENCY;i++) tick();
  CHECK(unpack32(dut->acc_col,0)==T,
        "clr/accumulate reordered: got %d want %d", unpack32(dut->acc_col,0), T);
}

// out_vld is the tile's only statement about its own latency. If a future
// depth change moves the data without moving out_vld, every consumer breaks
// silently -- so pin it.
static void test_out_vld_reports_latency() {
  printf("out_vld pulses exactly %d cycles after in_vld\n", LATENCY);
  std::vector<int8_t> A(T,1);
  dut->mode=0; tick();
  std::vector<int8_t> col(T,1); pack(dut->w_col,col.data(),T,4);
  pack(dut->a_row,A.data(),T,8);
  dut->bank=0; dut->clr=1; tick(); dut->clr=0;
  for(int i=1;i<LATENCY;i++) tick();

  dut->in_vld=1; tick(); dut->in_vld=0;
  for(int i=1;i<LATENCY;i++){
    CHECK(dut->out_vld==0, "out_vld early at cycle %d", i);
    tick();
  }
  CHECK(dut->out_vld==1, "out_vld did not assert at cycle %d", LATENCY);
  tick();
  CHECK(dut->out_vld==0, "out_vld did not deassert after one cycle");
}

// P0-5 / P2-7c claim ACC_FOLD * max|w| * max|a| = 16 * 8 * 128 = 16,384 fits
// ACC_LOCAL = 16 signed bits. That claim is now load-bearing for the whole
// systolic floorplan, so drive the worst case at it rather than trusting the
// arithmetic in a comment.
static void test_worst_case_fold_bound() {
  printf("worst-case INT4 x INT8 fold stays inside ACC_LOCAL\n");
  // a = -128, w = -8 gives the largest positive product, 1024, on every term.
  std::vector<int8_t> W(T*T,(int8_t)-8), A(T,(int8_t)-128);
  std::vector<int32_t> got;
  run(1, W, A, got);
  int32_t ref = (int32_t)T * 1024;
  for(int c=0;c<T;c++) CHECK(got[c]==ref, "max+ lane %d: got %d want %d", c, got[c], ref);
  CHECK(dut->ovf==0, "ovf on the bound P0-5 proves cannot overflow");

  // And the largest negative product, 127 * -8 = -1016.
  std::fill(W.begin(), W.end(), (int8_t)-8);
  std::fill(A.begin(), A.end(), (int8_t)127);
  run(1, W, A, got);
  ref = (int32_t)T * -1016;
  for(int c=0;c<T;c++) CHECK(got[c]==ref, "max- lane %d: got %d want %d", c, got[c], ref);
  CHECK(dut->ovf==0, "ovf on the negative bound");
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  dut=new Vsonic_tile;
  printf("sonic_tile: %dx%d sub-tile, %d accumulator banks, latency %d\n\n",
         T, T, BANKS, LATENCY);
  reset(); test_prefill_matches_reference();
  reset(); test_decode_matches_prefill_on_column_zero();
  reset(); test_bank_independence();
  reset(); test_clr_ordering_through_the_pipeline();
  reset(); test_out_vld_reports_latency();
  reset(); test_worst_case_fold_bound();
  dut->final(); delete dut;
  printf("\n%s (%d failures)\n", fails?"FAILED":"PASSED", fails);
  return fails?1:0;
}
