#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "conv_instance.hpp"
#include "onednn_conv.hpp"
#include "sys_atl_conv.hpp"

int main() {
  conv_instance ci_onednn{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};
  conv_instance ci_sys_atl{5, 80 + 2, 100 + 2, 128, 128, 3, 0, 1};

  OneDNN_Conv reference{ci_onednn};
  reference.run();

  conv_SYS_ATL(ci_sys_atl);

  if (ci_onednn.dst_data.size() != ci_sys_atl.dst_data.size()) {
    fprintf(stderr, "Sizes do not match!\n");
    return 1;
  }

  int err_count = 0;

  auto n = ci_onednn.dst_data.size();
  for (int i = 0; i < n; ++i) {
    double expected = ci_onednn.dst_data[i];
    double actual = ci_sys_atl.dst_data[i];
    double relerr = fabs((actual - expected) / expected);
    if (relerr > 1e-1) {
      fprintf(stderr,
              "Bad value at index %d - relative error = %.6f - actual = "
              "%.6f - "
              "expected = %.6f\n",
              i, relerr, actual, expected);
      err_count++;
    }
    if (err_count > 100) {
      fprintf(stderr, "Too many errors! Exiting early...\n");
      return 1;
    }
  }
}
