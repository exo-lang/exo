#include "sys_atl_conv.hpp"

#include <cassert>

#include "conv.h"

void sys_atl_conv(conv_instance &ci) {
  float scale = 1.0f;
  conv(nullptr, (int)ci.OH, (int)ci.OW, (int)ci.OC, (int)ci.KW, (int)ci.IC,
       (int)ci.IH, (int)ci.IW, &scale, (int)ci.N, ci.src_data.data(),
       ci.dst_data.data(), ci.weights_data.data(), ci.bias_data.data());
}
