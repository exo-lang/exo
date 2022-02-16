#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "conv_instance.hpp"
#include "exo_conv.hpp"
#include "halide_conv.hpp"
#include "onednn_conv.hpp"

bool check_output(const std::vector<float> &expected_vec,
    const std::vector<float> &actual_vec) {
  int err_count = 0;

  auto n = expected_vec.size();
  for (int i = 0; i < n; ++i) {
    double expected = expected_vec[i];
    double actual = actual_vec[i];
    double relerr = fabs((actual - expected) / expected);
    if (relerr > 1e-1) {
      fprintf(stderr,
          "Bad value at index %d - relative error = %.6f - actual = "
          "%.6f - "
          "expected = %.6f\n",
          i, relerr, actual, expected);
      err_count++;
    }
    if (err_count > 20) {
      fprintf(stderr, "Too many errors! Exiting early...\n");
      return false;
    }
  }
  return true;
}

int main() {
  conv_instance ci_onednn{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};
  conv_instance ci_exo{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};
  conv_instance ci_halide{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};

  printf("Running OneDNN...\n");
  OneDNN_Conv reference{ci_onednn};
  reference.run();

  printf("Running Exo...\n");
  exo_conv(ci_exo);
  printf("Checking Exo...\n");
  if (!check_output(ci_onednn.dst_data, ci_exo.dst_data)) {
    return 1;
  }

  printf("Running Halide...\n");
  halide_conv(ci_halide);
  printf("Checking Halide...\n");
  if (!check_output(ci_onednn.dst_data, ci_halide.dst_data)) {
    return 1;
  }

  return 0;
}
