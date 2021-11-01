#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

#include <benchmark/benchmark.h>

#include <sgemm.h>

float num_flops(int m, int n, int k) { return 2 * m * n * k; }

std::vector<float> gen_matrix(int m, int n) {
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

BENCHMARK(BM_sys_atl_kernel)->Range(8, 8 << 10);
