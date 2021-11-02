#include <random>
#include <vector>

#include <benchmark/benchmark.h>
#include <mkl.h>

#include <sgemm.h>

double num_flops(long m, long n, long k) { return 2 * m * n * k; }

std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

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
      static_cast<double>(state.iterations() * num_flops(6, 64, k)), //
      benchmark::Counter::kIsRate,                                   //
      benchmark::Counter::kIs1000                                    //
  );
}

BENCHMARK(BM_sys_atl_kernel)->Range(8, 8196);

static void BM_mkl_kernel(benchmark::State &state) {
  size_t k = state.range(0);
  auto a = gen_matrix(6, k);
  auto b = gen_matrix(k, 64);
  auto c = gen_matrix(6, 64);

  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, // layout
                6, 64, k,                                  // m, n, k
                1.0,                                       // alpha
                a.data(), k,                               // A (lda)
                b.data(), 64,                              // B (ldb)
                1.0,                                       // beta
                c.data(), 64                               // C (ldc)
    );
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations() * num_flops(6, 64, k)), //
      benchmark::Counter::kIsRate,                                   //
      benchmark::Counter::kIs1000                                    //
  );
}

BENCHMARK(BM_mkl_kernel)->Range(8, 8196);

static void BM_mkl_sgemm(benchmark::State &state) {
  size_t n = state.range(0);
  auto a = gen_matrix(n, n);
  auto b = gen_matrix(n, n);
  auto c = gen_matrix(n, n);

  for (auto _ : state) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, // layout
                n, n, n,                                   // m, n, k
                1.0,                                       // alpha
                a.data(), n,                               // A (lda)
                b.data(), n,                               // B (ldb)
                1.0,                                       // beta
                c.data(), n                                // C (ldc)
    );
  }

  state.counters["flops"] = benchmark::Counter(
      static_cast<double>(state.iterations() * num_flops(n, n, n)), //
      benchmark::Counter::kIsRate,                                  //
      benchmark::Counter::kIs1000                                   //
  );
}

BENCHMARK(BM_mkl_sgemm)->DenseRange(64, 1984, 128);
