#include <benchmark/benchmark.h>
#include <mkl.h>
#include <sgemm.h>

#include <random>
#include <vector>

#include "alex_sgemm.h"

// ----------------------------------------------------------------------------
// Utilities

// Source: http://www.netlib.org/lapack/lawnspdf/lawn41.pdf (p.120)
static double num_flops(long m, long n, long k) { return 2 * m * n * k; }

static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

template <typename SgemmFn>
static void BM_square_sgemm(benchmark::State &state) {
  size_t n = state.range(0);
  auto a = gen_matrix(n, n);
  auto b = gen_matrix(n, n);
  auto c = gen_matrix(n, n);

  for (auto _ : state) {
    SgemmFn{}(a.data(), b.data(), c.data(), n);
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations() * num_flops(n, n, n)),  //
      benchmark::Counter::kIsRate,                                   //
      benchmark::Counter::kIs1000                                    //
  );
}

// ----------------------------------------------------------------------------
// Benchmarking just the inner kernel

static void BM_sys_atl_kernel(benchmark::State &state) {
  size_t k = state.range(0);
  auto a = gen_matrix(6, k);
  auto b = gen_matrix(k, 64);
  auto c = gen_matrix(6, 64);

  for (auto _ : state) {
    sgemm_kernel_avx512_6x4(nullptr, k, a.data(), b.data(),
                            {c.data(), {64, 1}});
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations() * num_flops(6, 64, k)),  //
      benchmark::Counter::kIsRate,                                    //
      benchmark::Counter::kIs1000                                     //
  );
}

BENCHMARK(BM_sys_atl_kernel)->Name("kernel_sys_atl")->Range(8, 8196);

// ----------------------------------------------------------------------------
// MKL SGEMM benchmark

struct mkl_square {
  void operator()(const float *a, const float *b, float *c, long n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,  // layout
                n, n, n,                                    // m, n, k
                1.0,                                        // alpha
                a, n,                                       // A (lda)
                b, n,                                       // B (ldb)
                1.0,                                        // beta
                c, n                                        // C (ldc)
    );
  }
};

BENCHMARK_TEMPLATE(BM_square_sgemm, mkl_square)
    ->Name("sgemm_mkl")
    ->DenseRange(64, 1984, 128)
    ->Arg(221)
    ->Arg(256)
    ->Arg(397)
    ->Arg(412)
    ->Arg(512)
    ->Arg(732)
    ->Arg(911)
    ->Arg(1024)
    ->Arg(2048);

// ----------------------------------------------------------------------------
// Handwritten C++ SGEMM benchmark

struct alex_square {
  void operator()(const float *a, const float *b, float *c, long n) {
    sgemm_square(a, b, c, n);
  }
};

BENCHMARK_TEMPLATE(BM_square_sgemm, alex_square)
    ->Name("sgemm_alex")
    ->DenseRange(64, 1984, 128)
    ->Arg(221)
    ->Arg(256)
    ->Arg(397)
    ->Arg(412)
    ->Arg(512)
    ->Arg(732)
    ->Arg(911)
    ->Arg(1024)
    ->Arg(2048);

// ----------------------------------------------------------------------------
// SYS_ATL SGEMM benchmark

struct sys_atl_square {
  void operator()(float *a, float *b, float *c, long n) {
    sgemm_sys_atl(nullptr, n, n, n, a, b, c);
  }
};

BENCHMARK_TEMPLATE(BM_square_sgemm, sys_atl_square)
    ->Name("sgemm_sys_atl")
    ->DenseRange(64, 1984, 128)
    ->Arg(221)
    ->Arg(256)
    ->Arg(397)
    ->Arg(412)
    ->Arg(512)
    ->Arg(732)
    ->Arg(911)
    ->Arg(1024)
    ->Arg(2048);
