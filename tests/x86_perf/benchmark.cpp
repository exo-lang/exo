#include <vector>

#include <benchmark/benchmark.h>
#include <test_avx2_sgemm_6x16.h>

#define restrict __restrict // good old BLIS uses the C keyword...
#include <blis.h>

static void benchmark_blis(benchmark::State &state) {
  size_t k = state.range(0);
  float alpha = 1.0f;
  float beta = 1.0f;

  std::vector<float> a_vec(6 * k);
  std::vector<float> b_vec(k * 16);
  std::vector<float> c_vec(6 * 16);

  float *a = a_vec.data();
  float *b = b_vec.data();
  float *c = c_vec.data();

  for (auto _ : state) {
    // bli_sgemm_haswell_asm_6x16(k, &alpha, a, b, &beta, c, 16, 1, nullptr,
    //                            nullptr);
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, 6, 16, k, &alpha, a, k, 1,
              b, 16, 1, &beta, c, 16, 1);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations() * 2 * 6 * 16 * k),
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(benchmark_blis)
    // ->DenseRange(1, 16)
    // ->DenseRange(32, 1024, 32)
    ->DenseRange(1536, 4096, 512)
    ->ReportAggregatesOnly();

static void benchmark_sys_atl(benchmark::State &state) {
  size_t k = state.range(0);
  float alpha = 1.0f;
  float beta = 1.0f;

  std::vector<float> a_vec(6 * k);
  std::vector<float> b_vec(k * 16);
  std::vector<float> c_vec(6 * 16);

  float *a = a_vec.data();
  float *b = b_vec.data();
  float *c = c_vec.data();

  for (auto _ : state) {
    avx2_sgemm_6x16(nullptr, k, c, a, b);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations() * 2 * 6 * 16 * k),
      benchmark::Counter::kIsRate, benchmark::Counter::kIs1000);
}

BENCHMARK(benchmark_sys_atl)
    // ->DenseRange(1, 16)
    // ->DenseRange(32, 1024, 32)
    ->DenseRange(1536, 4096, 512)
    ->ReportAggregatesOnly();

BENCHMARK_MAIN();
