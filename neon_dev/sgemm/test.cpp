
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <chrono>

#include "sgemm.h"
//#include "alex_sgemm.h"
#include "naive_sgemm.h"

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
  long FLOP_C = long(n)*long(n)*long(n);
  int N_TIMES = 3;

  auto begin = std::chrono::steady_clock::now();
  for(int times = 0; times<N_TIMES; times++) {
    naive_sgemm_square(a.data(), b.data(), c2.data(), n);
  }
  auto end = std::chrono::steady_clock::now();
  double duration = std::chrono::duration<double>(end - begin).count();
  printf("-----------------------------------------------------------\n");
  printf("Naive SGEMM took %5.1lf ms, or %4.1lf GFLOPS\n",
         duration/N_TIMES*1.0e3, (FLOP_C*1.0e-9)/duration);
  
  begin = std::chrono::steady_clock::now();
  for(int times = 0; times<N_TIMES; times++) {
    sgemm_systl(nullptr, n, n, n, a.data(), b.data(), c.data());
  }
  end = std::chrono::steady_clock::now();
  duration = std::chrono::duration<double>(end - begin).count();
  printf("-----------------------------------------------------------\n");
  printf("SYSTL SGEMM took %5.1lf ms, or %4.1lf GFLOPS\n",
         duration/N_TIMES*1.0e3, (FLOP_C*1.0e-9)/duration);
  printf("-----------------------------------------------------------\n");
  

  for (int i = 0; i < c2.size(); i++) {
    float expected = c2[i];
    float actual = c[i];
    double relerr = fabsf(actual - expected) / expected;
    if (relerr > 1e-3) {
      printf("index %d: %.6f != %.6f (expected)\n", i, actual, expected);
    }
  }

  printf("both methods produced consistent output\n\n\n\n");
}
