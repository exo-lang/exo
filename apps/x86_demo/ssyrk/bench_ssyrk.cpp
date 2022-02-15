#include <benchmark/benchmark.h>
#include <mkl.h>

#include <random>
#include <vector>

// ----------------------------------------------------------------------------
// Utilities

// Source: http://www.netlib.org/lapack/lawnspdf/lawn41.pdf (p.120)
static double num_flops(long n, long k) {
  return static_cast<double>(k * n * (n + 1));
}

static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

template <typename SsyrkFn>
static void BM_ssyrk(benchmark::State &state) {
  long n = state.range(0);
  long k = state.range(1);

  auto a = gen_matrix(n, k);
  auto c = gen_matrix(n, n);

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
        n, k,                                             // dimensions
        1.0,                                              // alpha
        a, k,                                             // A (lda)
        1.0,                                              // beta
        c, n                                              // C (ldc)
    );
  }
};

BENCHMARK_TEMPLATE(BM_ssyrk, mkl_functor)
    ->Name("ssyrk_mkl")
    ->ArgsProduct({
        benchmark::CreateRange(64, 2048, /* multi= */ 2),  // n
        benchmark::CreateRange(16, 2048, /* multi= */ 2)   // k
    });
