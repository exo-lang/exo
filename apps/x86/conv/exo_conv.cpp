#include "exo_conv.hpp"

#include <cassert>

#include "conv.h"

void exo_conv(conv_instance &ci) {
  if (ci.OH != 80 || ci.OW != 100 || ci.OC != 128 || ci.KW != 3 || ci.N != 5) {
    abort();
  }
#if 0
  conv(nullptr, (int)ci.OH, (int)ci.OW, (int)ci.OC, (int)ci.IH, (int)ci.IW,
       (int)ci.IC, (int)ci.KW, (int)ci.N, ci.src_data.data(),
       ci.dst_data.data(), ci.weights_data.data(), ci.bias_data.data());
#else
  conv_specialized(nullptr, ci.src_data.data(), ci.dst_data.data(),
      ci.weights_data.data(), ci.bias_data.data());
#endif
}
