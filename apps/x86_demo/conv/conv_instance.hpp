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

  const std::vector<long> src_dims;
  const std::vector<long> weights_dims;
  const std::vector<long> bias_dims;
  const std::vector<long> dst_dims;

  const std::vector<long> strides_dims;
  const std::vector<long> padding_dims_l;
  const std::vector<long> padding_dims_r;

  std::vector<float> src_data;
  std::vector<float> weights_data;
  std::vector<float> bias_data;
  std::vector<float> dst_data;

  conv_instance(long batch_size, long in_dim, long in_chan, long out_chan,
                long kern_sz, long pad, long stride)
      : N(batch_size),
        IC(in_chan),
        IH(in_dim),
        IW(in_dim),
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
        src_dims({N, IC, IH, IW}),
        weights_dims({OC, IC, KH, KW}),
        bias_dims({OC}),
        dst_dims({N, OC, OH, OW}),
        strides_dims({SH, SW}),
        padding_dims_l({PH_L, PW_L}),
        padding_dims_r({PH_R, PW_R}),
        src_data(product(src_dims)),
        weights_data(product(weights_dims)),
        bias_data(product(bias_dims)),
        dst_data(product(dst_dims)) {
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
