#include <benchmark/benchmark.h>
#include <cblas.h>
#include <sgemm.h>

#include <cassert>
#include <random>
#include <vector>

#include "alex_sgemm.h"

#ifndef CBLAS_NAME
#error Must set CBLAS_NAME
#endif

// ----------------------------------------------------------------------------
// Utilities

// Source: http://www.netlib.org/lapack/lawnspdf/lawn41.pdf (p.120)
static double num_flops(long m, long n, long k) {
  return static_cast<double>(2 * m * n * k);
}

static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

template <typename SgemmFn>
static void BM_sgemm(benchmark::State &state) {
  long m = state.range(0);
  long n = state.range(1);
  long k = state.range(2);

  auto a = gen_matrix(m, k);
  auto b = gen_matrix(k, n);
  auto c = gen_matrix(m, n);

  for ([[maybe_unused]] auto _ : state) {
    SgemmFn{}(a.data(), b.data(), c.data(), m, n, k);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations()) * num_flops(m, n, k),  //
      benchmark::Counter::kIsRate,                                   //
      benchmark::Counter::kIs1000                                    //
  );
}

// ----------------------------------------------------------------------------
// Sizes

#define ARG_SQUARE_SIZES                                                       \
  ->Args({64, 64, 64})                                                         \
      ->Args({192, 192, 192})                                                  \
      ->Args({221, 221, 221})                                                  \
      ->Args({256, 256, 256})                                                  \
      ->Args({320, 320, 320})                                                  \
      ->Args({397, 397, 397})                                                  \
      ->Args({412, 412, 412})                                                  \
      ->Args({448, 448, 448})                                                  \
      ->Args({512, 512, 512})                                                  \
      ->Args({576, 576, 576})                                                  \
      ->Args({704, 704, 704})                                                  \
      ->Args({732, 732, 732})                                                  \
      ->Args({832, 832, 832})                                                  \
      ->Args({911, 911, 911})                                                  \
      ->Args({960, 960, 960})                                                  \
      ->Args({1024, 1024, 1024})                                               \
      ->Args({1088, 1088, 1088})                                               \
      ->Args({1216, 1216, 1216})                                               \
      ->Args({1344, 1344, 1344})                                               \
      ->Args({1472, 1472, 1472})                                               \
      ->Args({1600, 1600, 1600})                                               \
      ->Args({1728, 1728, 1728})                                               \
      ->Args({1856, 1856, 1856})                                               \
      ->Args({1984, 1984, 1984})                                               \
      ->Args({2048, 2048, 2048})

#define ARG_CONST_512_SIZES                                                    \
  ->Args({32768, 8, 512})                                                      \
      ->Args({16384, 16, 512})                                                 \
      ->Args({8192, 32, 512})                                                  \
      ->Args({4096, 64, 512})                                                  \
      ->Args({2048, 128, 512})                                                 \
      ->Args({1024, 256, 512})                                                 \
      ->Args({256, 1024, 512})                                                 \
      ->Args({128, 2048, 512})                                                 \
      ->Args({64, 4096, 512})                                                  \
      ->Args({32, 8192, 512})                                                  \
      ->Args({16, 16384, 512})                                                 \
      ->Args({8, 32768, 512})

// ----------------------------------------------------------------------------
// MKL SGEMM benchmark

struct cblas_square {
  void operator()(float *a, float *b, float *c, long m, long n, long k) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  // layout
        m, n, k,                                            // m, n, k
        1.0,                                                // alpha
        a, k,                                               // A (lda)
        b, n,                                               // B (ldb)
        1.0,                                                // beta
        c, n                                                // C (ldc)
    );
  }
};

BENCHMARK_TEMPLATE(BM_sgemm, cblas_square)
    ->Name("sgemm_" CBLAS_NAME)  //
    ARG_SQUARE_SIZES             //
    ARG_CONST_512_SIZES;

// ----------------------------------------------------------------------------
// Handwritten C++ SGEMM benchmark

struct alex_square {
  void operator()(float *a, float *b, float *c, long m, long n, long k) {
    assert(m == n && n == k);
    sgemm_square(a, b, c, n);
  }
};

BENCHMARK_TEMPLATE(BM_sgemm, alex_square)
    ->Name("sgemm_alex")  //
    ARG_SQUARE_SIZES;

// ----------------------------------------------------------------------------
// Exo SGEMM benchmark

struct exo_square {
  void operator()(float *a, float *b, float *c, long m, long n, long k) {
    sgemm_exo(nullptr, m, n, k, a, b, c);
  }
};

BENCHMARK_TEMPLATE(BM_sgemm, exo_square)
    ->Name("sgemm_exo")  //
    ARG_SQUARE_SIZES     //
    ARG_CONST_512_SIZES;
