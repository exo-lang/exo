#include "filter1D.h"



#include <stdio.h>
#include <stdlib.h>

#include <arm_neon.h>


// filter1D(
//     ow : size,
//     kw : size,
//     x : f32[ow + kw - 1] @DRAM,
//     y : f32[ow] @DRAM,
//     w : f32[kw] @DRAM
// )
void filter1D( void *ctxt, int_fast32_t ow, int_fast32_t kw, const float* x, float* y, const float* w ) {
for (int outXo = 0; outXo < ((ow) / (4)); outXo++) {
  float32x4_t sum;
  sum = vmovq_n_f32(0.0f);
  for (int k = 0; k < kw; k++) {
    float32x4_t xX4;
    xX4 = vld1q_f32(&x[(k + 4 * outXo + 0) * (1)]);
    sum = vmlaq_n_f32(sum, xX4, w[(k + 0) * (1)]);
  }
  vst1q_f32(&y[(4 * outXo + 0) * (1)], sum);
}
if (ow % 4 > 0) {
  for (int outXi = 0; outXi < ow % 4; outXi++) {
    float sum;
    sum = 0.0;
    for (int k = 0; k < kw; k++) {
      sum += x[(outXi + ((ow) / (4)) * 4 + k) * (1)] * w[(k) * (1)];
    }
    y[(outXi + ((ow) / (4)) * 4) * (1)] = sum;
  }
}
}


/* relying on the following instruction..."
neon_vfmadd_4xf32_1xf32(dst,lhs,rhs)
{dst_data} = vmlaq_n_f32({dst_data}, {lhs_data}, {rhs_data});
*/

/* relying on the following instruction..."
neon_vld_4xf32(dst,src)
{dst_data} = vld1q_f32(&{src_data});
*/

/* relying on the following instruction..."
neon_vst_4xf32(dst,src)
vst1q_f32(&{dst_data}, {src_data});
*/

/* relying on the following instruction..."
neon_zero_4xf32(dst)
{dst_data} = vmovq_n_f32(0.0f);
*/
