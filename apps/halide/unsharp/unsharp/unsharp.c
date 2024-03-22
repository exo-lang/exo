#include "unsharp.h"

#include <stdio.h>
#include <stdlib.h>

// unsharp(
//     W : size,
//     H : size,
//     output : f32[H, W] @DRAM,
//     input : f32[H + 6, W + 6, 3] @DRAM
// )
void unsharp(void *ctxt, int_fast32_t W, int_fast32_t H, float *output,
    const float *input) {
  EXO_ASSUME(H % 32 == 0);
  EXO_ASSUME(W % 256 == 0);
  float *gray = (float *)malloc((6 + H) * (6 + W) * sizeof(*gray));
  for (int_fast32_t y = 0; y < 6 + H; y++) {
    for (int_fast32_t x = 0; x < 6 + W; x++) {
      gray[y * (6 + W) + x] = 0.299f * input[y * (W + 6) * 3 + x * 3] +
                              0.587f * input[y * (W + 6) * 3 + x * 3 + 1] +
                              0.114f * input[y * (W + 6) * 3 + x * 3 + 2];
    }
  }
  float *blur_y = (float *)malloc(H * (6 + W) * sizeof(*blur_y));
  for (int_fast32_t y = 0; y < H; y++) {
    for (int_fast32_t x = 0; x < 6 + W; x++) {
      blur_y[y * (6 + W) + x] =
          0.2659615202676218f * gray[(3 + y) * (6 + W) + x] +
          0.2129653370149015f *
              (gray[(2 + y) * (6 + W) + x] + gray[(4 + y) * (6 + W) + x]) +
          0.10934004978399575f *
              (gray[(1 + y) * (6 + W) + x] + gray[(5 + y) * (6 + W) + x]) +
          0.035993977675458706f *
              (gray[y * (6 + W) + x] + gray[(6 + y) * (6 + W) + x]);
    }
  }
  float *blur_x = (float *)malloc(H * W * sizeof(*blur_x));
  for (int_fast32_t y = 0; y < H; y++) {
    for (int_fast32_t x = 0; x < W; x++) {
      blur_x[y * W + x] =
          0.2659615202676218f * blur_y[y * (6 + W) + 3 + x] +
          0.2129653370149015f *
              (blur_y[y * (6 + W) + 2 + x] + blur_y[y * (6 + W) + 4 + x]) +
          0.10934004978399575f *
              (blur_y[y * (6 + W) + 1 + x] + blur_y[y * (6 + W) + 5 + x]) +
          0.035993977675458706f *
              (blur_y[y * (6 + W) + x] + blur_y[y * (6 + W) + 6 + x]);
    }
  }
  free(blur_y);
  for (int_fast32_t y = 0; y < ((H) / (32)); y++) {
    for (int_fast32_t yi = 0; yi < 32; yi++) {
      float *ratio = (float *)malloc(1 * W * sizeof(*ratio));
      for (int_fast32_t yii = 0; yii < 1; yii++) {
        for (int_fast32_t x = 0; x < W; x++) {
          float *sharpen = (float *)malloc(1 * 1 * sizeof(*sharpen));
          for (int_fast32_t xi = 0; xi < 1; xi++) {
            for (int_fast32_t yiii = 0; yiii < 1; yiii++) {
              sharpen[yiii + xi] =
                  2.0f * gray[(3 + yi + yii + yiii + 32 * y) * (6 + W) + 3 + x +
                              xi] -
                  blur_x[(yi + yii + yiii + 32 * y) * W + x + xi];
            }
          }
          ratio[yii * W + x] =
              sharpen[0] / gray[(3 + yi + yii + 32 * y) * (6 + W) + 3 + x];
          free(sharpen);
        }
      }
      for (int_fast32_t x = 0; x < W; x++) {
        for (int_fast32_t c = 0; c < 3; c++) {
          output[(yi + 32 * y) * W + x] =
              ratio[x] *
              input[(3 + yi + 32 * y) * (W + 6) * 3 + (3 + x) * 3 + c];
        }
      }
      free(ratio);
    }
  }
  free(blur_x);
  free(gray);
}
