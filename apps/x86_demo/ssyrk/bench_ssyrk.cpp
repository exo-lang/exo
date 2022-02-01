#include <benchmark/benchmark.h>
#include <mkl.h>

#include <random>
#include <vector>

#include <util.hpp>

// ----------------------------------------------------------------------------
// Utilities

// Source: http://www.netlib.org/lapack/lawnspdf/lawn41.pdf (p.120)
static double num_flops(long n, long k) {
  return static_cast<double>(k * n * (n + 1));
}

template <typename SsyrkFn>
static void BM_ssyrk(benchmark::State &state) {
  long n = state.range(0);
  long k = state.range(1);

  auto a = util::gen_matrix<float>(n, k);
  auto c = util::gen_matrix<float>(n, n);

  for (auto _ : state) {
    SsyrkFn{}(a.data(), c.data(), n, k);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * num_flops(n, k),  //
      benchmark::Counter::kIsRate,                                //
      benchmark::Counter::kIs1000                                 //
  );
}

// ----------------------------------------------------------------------------
// MKL SSYRK benchmark

struct mkl_functor {
  void operator()(const float *a, float *c, int n, int k) {
    cblas_ssyrk(CblasRowMajor, CblasUpper, CblasNoTrans,  // layout
                n, k,                                     // dimensions
                1.0,                                      // alpha
                a, k,                                     // A (lda)
                1.0,                                      // beta
                c, n                                      // C (ldc)
    );
  }
};

BENCHMARK_TEMPLATE(BM_ssyrk, mkl_functor)
    ->Name("ssyrk_mkl")
    ->ArgsProduct({
        benchmark::CreateRange(64, 2048, /* multi= */ 2),  // n
        benchmark::CreateRange(16, 2048, /* multi= */ 2)   // k
    });

BENCHMARK_TEMPLATE(BM_ssyrk, mkl_functor)
    ->Name("ssyrk_mkl_square")
    ->Args({64, 64})
    ->Args({192, 192})
    ->Args({221, 221})
    ->Args({256, 256})
    ->Args({320, 320})
    ->Args({397, 397})
    ->Args({412, 412})
    ->Args({448, 448})
    ->Args({512, 512})
    ->Args({576, 576})
    ->Args({704, 704})
    ->Args({732, 732})
    ->Args({832, 832})
    ->Args({911, 911})
    ->Args({960, 960})
    ->Args({1024, 1024})
    ->Args({1088, 1088})
    ->Args({1216, 1216})
    ->Args({1344, 1344})
    ->Args({1472, 1472})
    ->Args({1600, 1600})
    ->Args({1728, 1728})
    ->Args({1856, 1856})
    ->Args({1984, 1984})
    ->Args({2048, 2048});
