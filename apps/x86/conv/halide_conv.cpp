#include "halide_conv.hpp"

#include <HalideBuffer.h>
#include <halide_conv_kernel.h>

using namespace Halide::Runtime;

void halide_conv(conv_instance &ci) {
  Buffer<float> input(ci.src_data.data(), ci.IC, ci.IW, ci.IH, ci.N);
  Buffer<float> weights(ci.weights_data.data(), ci.OC, ci.KW, ci.KH, ci.IC);
  Buffer<float> bias(ci.bias_data.data(), ci.OC);
  Buffer<float> output(ci.dst_data.data(), ci.OC, ci.OW, ci.OH, ci.N);
  halide_conv_kernel(input, weights, bias, output);
}
