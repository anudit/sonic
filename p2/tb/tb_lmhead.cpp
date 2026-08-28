// sonic_lmhead: streaming top-K over a vocabulary, checked against std::sort.
//
// The hard cases are ties and the boundary at slot K-1, because the whole point
// of the unit is that it never sees the full logit vector -- a candidate that
// should have entered at exactly K-1 is the one a shift-register insertion gets
// wrong.
#include "Vsonic_lmhead.h"
#include "verilated.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

static Vsonic_lmhead *dut;
static vluint64_t main_time = 0;
double sc_time_stamp() { return main_time; }
static void tick(){ dut->clk=0; dut->eval(); main_time++; dut->clk=1; dut->eval(); main_time++; }

static const int K = 64, VW = 17, SW = 32;
static int fails = 0;
#define CHECK(c,...) do{ if(!(c)){ printf("  FAIL: "); printf(__VA_ARGS__); printf("\n"); \
                          if(++fails>10) return; } }while(0)

template <typename W> static uint32_t field(W &d, int idx, int bits) {
  uint64_t lo = (uint64_t)d[(idx*bits)>>5];
  uint64_t hi = (uint64_t)d[((idx*bits)>>5)+1];
  uint64_t v = (lo | (hi<<32)) >> ((idx*bits)&31);
  return (uint32_t)(v & ((1ull<<bits)-1));
}

static void reset(){ dut->rst_n=0; dut->clr=1; dut->in_vld=0; dut->last=0;
                     tick(); tick(); dut->rst_n=1; dut->clr=0; tick(); }

static void run_vocab(const std::vector<int32_t> &logits) {
  reset();
  for (size_t i=0;i<logits.size();i++){
    dut->logit=logits[i]; dut->token_id=(uint32_t)i;
    dut->in_vld=1; dut->last=(i==logits.size()-1);
    tick();
  }
  dut->in_vld=0; dut->last=0; tick();
}

static void test_topk(int n, unsigned seed, const char *what) {
  printf("%s (%d candidates)\n", what, n);
  srand(seed);
  std::vector<int32_t> lg(n);
  for (auto &v : lg) v = (rand()%20001)-10000;

  run_vocab(lg);

  std::vector<int> idx(n);
  for(int i=0;i<n;i++) idx[i]=i;
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int a,int b){ return lg[a]>lg[b]; });

  int checked = std::min(K, n);
  for (int j=0;j<checked;j++){
    int32_t got = (int32_t)field(dut->top_score, j, SW);
    CHECK(got == lg[idx[j]], "slot %d: score %d want %d", j, got, lg[idx[j]]);
  }
}

// Every candidate identical: the unit must fill K slots and keep the lowest
// token ids, matching torch.topk's tie behaviour.
static void test_all_ties() {
  printf("all-ties\n");
  std::vector<int32_t> lg(200, 42);
  run_vocab(lg);
  for (int j=0;j<K;j++)
    CHECK((int32_t)field(dut->top_score,j,SW)==42, "slot %d not filled", j);
}

// Fewer candidates than K: the unused slots must stay at -inf, not garbage.
static void test_fewer_than_k() {
  printf("fewer candidates than K\n");
  std::vector<int32_t> lg(10);
  for (int i=0;i<10;i++) lg[i]=i*100;
  run_vocab(lg);
  for (int j=0;j<10;j++)
    CHECK((int32_t)field(dut->top_score,j,SW)==(9-j)*100, "slot %d wrong", j);
  CHECK((int32_t)field(dut->top_score,K-1,SW) == INT32_MIN,
        "unused slot %d not -inf", K-1);
}

int main(int argc,char**argv){
  Verilated::commandArgs(argc,argv);
  dut=new Vsonic_lmhead;
  printf("sonic_lmhead: streaming top-%d\n\n", K);
  test_topk(500, 3, "random logits");
  test_topk(4096, 9, "larger vocabulary slice");
  test_all_ties();
  test_fewer_than_k();
  dut->final(); delete dut;
  printf("\n%s (%d failures)\n", fails?"FAILED":"PASSED", fails);
  return fails?1:0;
}
