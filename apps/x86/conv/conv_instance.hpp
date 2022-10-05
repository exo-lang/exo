#ifndef CONV_INSTANCE_H
#define CONV_INSTANCE_H

#include <algorithm>
#include <cmath>
#include <numeric>
#include <vector>

struct conv_instance {
  const long N,  // batch size
      IC,        // input channels
      IH,        // input height
      IW,        // input width
      OC,        // output channels
      KH,        // weights height
      KW,        // weights width
      PH_L,      // height padding: left (top)
      PH_R,      // height padding: right (bottom)
      PW_L,      // width padding: left
      PW_R,      // width padding: right
      SH,        // height-wise stride
      SW,        // width-wise stride
      OH,        // output height
      OW;        // output width

  std::vector<float> src_data;
  std::vector<float> weights_data;
  std::vector<float> bias_data;
  std::vector<float> dst_data;

  conv_instance(long batch_size, long in_h, long in_w, long in_chan,
      long out_chan, long kern_sz, long pad, long stride)
      : N(batch_size),
        IC(in_chan),
        IH(in_h),
        IW(in_w),
        OC(out_chan),
        KH(kern_sz),
        KW(kern_sz),
        PH_L(pad),
        PH_R(pad),
        PW_L(pad),
        PW_R(pad),
        SH(stride),
        SW(stride),
        OH((IH - KH + PH_L + PH_R) / SH + 1),
        OW((IW - KW + PW_L + PW_R) / SW + 1),
        src_data(N * IC * IH * IW),
        weights_data(OC * IC * KH * KW),
        bias_data(OC),
        dst_data(N * OC * OH * OW) {
    // Initialize src, weights, and dst tensors.
    std::generate(src_data.begin(), src_data.end(),
        [i = 0.0f]() mutable { return std::cos(i++ / 10.f); });

    std::generate(weights_data.begin(), weights_data.end(),
        [i = 0.0f]() mutable { return std::sin(i++ * 2.f); });

    std::generate(bias_data.begin(), bias_data.end(),
        [i = 0.0f]() mutable { return std::tanh(i++); });
  }

private:
  inline long product(const std::vector<long> &dims) {
    return std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<>());
  }
};

#endif  // CONV_INSTANCE_H
