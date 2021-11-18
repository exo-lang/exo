#include "sys_atl_conv.hpp"

#include <cassert>

#include "conv.h"

void sys_atl_conv(conv_instance &ci) {
  conv(nullptr, (int)ci.OH, (int)ci.OW, (int)ci.OC, (int)ci.IH, (int)ci.IW,
       (int)ci.IC, (int)ci.KW, (int)ci.N, ci.src_data.data(),
       ci.dst_data.data(), ci.weights_data.data(), ci.bias_data.data());
}
