#include <algorithm>
#include <cassert>
#include <cmath>
#include <numeric>
#include <vector>

#include "conv_instance.hpp"
#include "onednn_conv.hpp"
#include "conv.h"

void conv_SYS_ATL(conv_instance &ci) {
    assert(ci.IW == ci.IH);
    assert(ci.OW == ci.OH);
    assert(ci.KW == ci.KH);

    float scale = 1.0f;
    constexpr int batch_size = 4;

    conv(nullptr, (int)ci.OW, (int)ci.OC, (int)ci.KW, (int)ci.IC, (int)ci.IW,
         &scale, batch_size, ci.src_data.data(), ci.dst_data.data(),
         ci.weights_data.data(), ci.bias_data.data());
}

int main() {
    conv_instance ci_onednn{56, 64, 64, 3, 0, 1};
    conv_instance ci_sys_atl{56, 64, 64, 3, 0, 1};

    OneDNN_Conv reference{ci_onednn};
    reference.run();

    conv_SYS_ATL(ci_sys_atl);

    if (ci_onednn.dst_data.size() != ci_sys_atl.dst_data.size()) {
        fprintf(stderr, "Sizes do not match!\n");
        return 1;
    }

    auto n = ci_onednn.dst_data.size();
    for (int i = 0; i < n; ++i) {
        double expected = ci_onednn.dst_data[i];
        double actual = ci_sys_atl.dst_data[i];
        double relerr = fabs((actual - expected) / expected);
        if (relerr > 1e-3) {
            fprintf(stderr,
                    "Bad value at index %d - relative error = %.6f - actual = "
                    "%.6f - "
                    "expected = %.6f\n",
                    i, relerr, actual, expected);
        }
    }
}
