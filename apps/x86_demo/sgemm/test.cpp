#include <sgemm.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "alex_sgemm.h"

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

  sgemm_exo(nullptr, n, n, n, a.data(), b.data(), c.data());
  sgemm_square(a.data(), b.data(), c2.data(), n);

  for (int i = 0; i < c2.size(); i++) {
    float expected = c2[i];
    float actual = c[i];
    double relerr = fabsf(actual - expected) / expected;
    if (relerr > 1e-3) {
      printf("index %d: %.6f != %.6f (expected)\n", i, actual, expected);
    }
  }

  printf("didn't crash, yay\n");
}
