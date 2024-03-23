#include "unsharp.h"



#include <stdio.h>
#include <stdlib.h>



// unsharp(
//     W : size,
//     H : size,
//     output : f32[H, W] @DRAM,
//     input : f32[H + 6, W + 6, 3] @DRAM
// )
void unsharp( void *ctxt, int_fast32_t W, int_fast32_t H, float* output, const float* input ) {
EXO_ASSUME(H % 32 == 0);
EXO_ASSUME(W % 256 == 0);
for (int_fast32_t y = 0; y < ((H) / (32)); y++) {
  float *gray = (float*) malloc(38 * (6 + W) * sizeof(*gray));
  float *blur_y = (float*) malloc(32 * (6 + W) * sizeof(*blur_y));
  float *ratio = (float*) malloc(32 * W * sizeof(*ratio));
  for (int_fast32_t yi = 0; yi < 32; yi++) {
    for (int_fast32_t yii = 0; yii < 7; yii++) {
      for (int_fast32_t x = 0; x < 6 + W; x++) {
        gray[(yi + yii) * (6 + W) + x] = 0.299f * input[(yi + yii + 32 * y) * (W + 6) * 3 + x * 3] + 0.587f * input[(yi + yii + 32 * y) * (W + 6) * 3 + x * 3 + 1] + 0.114f * input[(yi + yii + 32 * y) * (W + 6) * 3 + x * 3 + 2];
      }
    }
    for (int_fast32_t x = 0; x < 6 + W; x++) {
      blur_y[yi * (6 + W) + x] = 0.2659615202676218f * gray[(3 + yi) * (6 + W) + x] + 0.2129653370149015f * (gray[(2 + yi) * (6 + W) + x] + gray[(4 + yi) * (6 + W) + x]) + 0.10934004978399575f * (gray[(1 + yi) * (6 + W) + x] + gray[(5 + yi) * (6 + W) + x]) + 0.035993977675458706f * (gray[yi * (6 + W) + x] + gray[(6 + yi) * (6 + W) + x]);
    }
    for (int_fast32_t x = 0; x < W; x++) {
      ratio[yi * W + x] = (2.0f * gray[(3 + yi) * (6 + W) + 3 + x] - (0.2659615202676218f * blur_y[yi * (6 + W) + 3 + x] + 0.2129653370149015f * (blur_y[yi * (6 + W) + 2 + x] + blur_y[yi * (6 + W) + 4 + x]) + 0.10934004978399575f * (blur_y[yi * (6 + W) + 1 + x] + blur_y[yi * (6 + W) + 5 + x]) + 0.035993977675458706f * (blur_y[yi * (6 + W) + x] + blur_y[yi * (6 + W) + 6 + x]))) / gray[(3 + yi) * (6 + W) + 3 + x];
    }
    for (int_fast32_t x = 0; x < W; x++) {
      for (int_fast32_t c = 0; c < 3; c++) {
        output[(yi + 32 * y) * W + x] = ratio[yi * W + x] * input[(3 + yi + 32 * y) * (W + 6) * 3 + (3 + x) * 3 + c];
      }
    }
  }
  free(ratio);
  free(blur_y);
  free(gray);
}
}

