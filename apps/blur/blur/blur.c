#include "blur.h"

#include <stdio.h>
#include <stdlib.h>

// exo_blur_halide(
//     W : size,
//     H : size,
//     blur_y : ui16[H, W] @DRAM,
//     inp : ui16[H + 2, W + 2] @DRAM
// )
void exo_blur_halide(void *ctxt, int_fast32_t W, int_fast32_t H,
    uint16_t *blur_y, const uint16_t *inp) {
  EXO_ASSUME(H % 32 == 0);
  EXO_ASSUME(W % 16 == 0);
  uint16_t *blur_x = (uint16_t *)malloc(34 * W * sizeof(*blur_x));
  for (int_fast32_t y = 0; y < ((H) / (32)); y++) {
    for (int_fast32_t yi = 0; yi < 32; yi++) {
      for (int_fast32_t x = 0; x < W; x++) {
        for (int_fast32_t yii = 0; yii < 3; yii++) {
          blur_x[(yi + yii) * W + x] =
              (inp[(yi + yii + 32 * y) * (W + 2) + x] +
                  inp[(yi + yii + 32 * y) * (W + 2) + 1 + x] +
                  inp[(yi + yii + 32 * y) * (W + 2) + 2 + x]) /
              3.0;
        }
        blur_y[(yi + 32 * y) * W + x] =
            (blur_x[yi * W + x] + blur_x[(1 + yi) * W + x] +
                blur_x[(2 + yi) * W + x]) /
            3.0;
      }
    }
  }
  free(blur_x);
}

// exo_blur_staged(
//     W : size,
//     H : size,
//     blur_y : ui16[H, W] @DRAM,
//     inp : ui16[H + 2, W + 2] @DRAM
// )
void exo_blur_staged(void *ctxt, int_fast32_t W, int_fast32_t H,
    uint16_t *blur_y, const uint16_t *inp) {
  EXO_ASSUME(H % 32 == 0);
  EXO_ASSUME(W % 16 == 0);
  uint16_t *blur_x = (uint16_t *)malloc((H + 2) * W * sizeof(*blur_x));
  for (int_fast32_t y = 0; y < H + 2; y++) {
    for (int_fast32_t x = 0; x < W; x++) {
      blur_x[y * W + x] = (inp[y * (W + 2) + x] + inp[y * (W + 2) + x + 1] +
                              inp[y * (W + 2) + x + 2]) /
                          3.0;
    }
  }
  for (int_fast32_t y = 0; y < H; y++) {
    for (int_fast32_t x = 0; x < W; x++) {
      blur_y[y * W + x] = (blur_x[y * W + x] + blur_x[(y + 1) * W + x] +
                              blur_x[(y + 2) * W + x]) /
                          3.0;
    }
  }
  free(blur_x);
}
