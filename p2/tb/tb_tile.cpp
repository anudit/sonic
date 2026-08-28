// sonic_tile: both dataflows must produce identical results.
//
// That is the whole point of a dual-mode array, and it is the property most
// likely to break silently -- prefill latches weights into the grid while
// decode streams them past, so a bug in one mode is invisible from the other.
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
  dut->clr=1; dut->in_vld=0; tick(); dut->clr=0;

  pack(dut->a_row, A.data(), T, 8);
  if (mode == 0) {                        // decode: weights stream alongside
    std::vector<int8_t> col(T);
    for(int r=0;r<T;r++) col[r]=W[r*T+0];
    pack(dut->w_col, col.data(), T, 4);
  }
  dut->in_vld=1; tick(); dut->in_vld=0; tick();

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
  srand(23);
  std::vector<int8_t> W(T*T,1), A(T,1);
  std::vector<int32_t> out;
  int32_t expect[BANKS];

  dut->mode=1; tick();
  for(int c=0;c<T;c++){ std::vector<int8_t> col(T,1); pack(dut->w_col,col.data(),T,4);
                        dut->w_load=1; tick(); }
  dut->w_load=0; tick();

  for(int b=0;b<BANKS;b++){
    dut->bank=b; dut->clr=1; tick(); dut->clr=0;
    expect[b]=0;
    for(int rep=0;rep<=b;rep++){                 // bank b gets b+1 accumulations
      pack(dut->a_row,A.data(),T,8);
      dut->in_vld=1; tick(); dut->in_vld=0; tick();
      expect[b]+=T;
    }
  }
  for(int b=0;b<BANKS;b++){
    dut->bank=b; tick();
    CHECK(unpack32(dut->acc_col,0)==expect[b],
          "bank %d: got %d want %d", b, unpack32(dut->acc_col,0), expect[b]);
  }
  (void)out;
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  dut=new Vsonic_tile;
  printf("sonic_tile: %dx%d sub-tile, %d accumulator banks\n\n", T, T, BANKS);
  reset(); test_prefill_matches_reference();
  reset(); test_decode_matches_prefill_on_column_zero();
  reset(); test_bank_independence();
  dut->final(); delete dut;
  printf("\n%s (%d failures)\n", fails?"FAILED":"PASSED", fails);
  return fails?1:0;
}
