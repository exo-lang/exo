#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "conv_instance.hpp"
#include "halide_conv.hpp"
#include "onednn_conv.hpp"
#include "sys_atl_conv.hpp"

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
  conv_instance ci_sys_atl{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};
  conv_instance ci_halide{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};

  printf("Running OneDNN...\n");
  OneDNN_Conv reference{ci_onednn};
  reference.run();

  printf("Running SYS_ATL...\n");
  sys_atl_conv(ci_sys_atl);
  printf("Checking SYS_ATL...\n");
  if (!check_output(ci_onednn.dst_data, ci_sys_atl.dst_data)) {
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
