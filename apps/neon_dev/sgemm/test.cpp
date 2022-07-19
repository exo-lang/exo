
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <chrono>

#include "sgemm.h"
//#include "alex_sgemm.h"
#include "naive_sgemm.h"
#include <Accelerate/Accelerate.h>

static std::vector<float> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<> rv{-1.0f, 1.0f};

  std::vector<float> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

int main(int argc, char *argv[]) {
  if (argc != 2) {
    printf("Usage: %s <n>\n", argv[0]);
    return 1;
  }
  int n = std::atoi(argv[1]);
  if (n < 1) {
    printf("n < 1!!\n");
    return 1;
  }

  auto a = gen_matrix(n, n);
  auto b = gen_matrix(n, n);
  auto c = gen_matrix(n, n);
  auto c2 = c;

  printf("\n\n\n\n");
  printf("Multiplying two %d x %d matrices\n", n, n);
  long FLOP_C = 2 * long(n) * long(n) * long(n);

  int N_TIMES_NAIVE = 1;
  auto begin = std::chrono::steady_clock::now();
  for (int times = 0; times < N_TIMES_NAIVE; times++) {
    naive_sgemm_square(a.data(), b.data(), c2.data(), n);
  }
  auto end = std::chrono::steady_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  double ms_per_gemm = duration / N_TIMES_NAIVE * 1.0e3;
  printf("-----------------------------------------------------------\n");
  printf("Naive SGEMM took %5.1lf ms, or %4.1lf GFLOPS\n", ms_per_gemm,
      (FLOP_C * 1.0e-6) / ms_per_gemm);

  int N_TIMES_ACCELERATE = 1000;
  begin = std::chrono::steady_clock::now();
  for (int times = 0; times < N_TIMES_ACCELERATE; times++) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n,  // M N K
        1.0,                                                         // alpha
        a.data(),
        n,  // M
        b.data(),
        n,    // K
        1.0,  // beta
        c.data(),
        n  // M
    );
  }
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  ms_per_gemm = duration / N_TIMES_ACCELERATE * 1.0e3;
  printf("-----------------------------------------------------------\n");
  printf("Apple SGEMM took %5.1lf ms, or %4.1lf GFLOPS\n", ms_per_gemm,
      (FLOP_C * 1.0e-6) / ms_per_gemm);

  int N_TIMES_EXO = 50;
  begin = std::chrono::steady_clock::now();
  for (int times = 0; times < N_TIMES_EXO; times++) {
    sgemm_exo(nullptr, n, n, n, a.data(), b.data(), c.data());
  }
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  ms_per_gemm = duration / N_TIMES_EXO * 1.0e3;
  printf("-----------------------------------------------------------\n");
  printf("  Exo SGEMM took %5.1lf ms, or %4.1lf GFLOPS\n", ms_per_gemm,
      (FLOP_C * 1.0e-6) / ms_per_gemm);
  printf("-----------------------------------------------------------\n");

  /*
    Notes for Apple M1 Mac
    Cache line size : 128 bytes = 32 floats
    L1 Cache size   : 64 KB
    L2 Cache size   :  4 MB


     453 M FMAdds per launch * 30 launches
     = 13.6 B FMAdds total
    576 KB of data per A, B, C matrix

    Old Information (30 runs)
    8.0  B L1 Data Cache Accesses
    5.7  M L1 Data Store Miss
    0.57 B L1 Data Load Miss

    Improved SGEMM (50 runs)
    2.7  B L1 Data Cache Accesses
    7.1  M L1 Data Store Miss
    2.7  B L1 Data Load Miss


  */

  /*
  for (int i = 0; i < c2.size(); i++) {
    float expected = c2[i];
    float actual = c[i];
    double relerr = fabsf(actual - expected) / expected;
    if (relerr > 1e-3) {
      printf("index %d: %.6f != %.6f (expected)\n", i, actual, expected);
    }
  }

  printf("both methods produced consistent output\n\n\n\n");
  */
}
