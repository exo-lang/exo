
#include "naive_sgemm.h"

#include <arm_neon.h>
#include <cstdio>

int test_neon() {
  // vector addition 8x8 example.
  uint8x8_t vec_a, vec_b, vec_dest;  // a vector of 8 8bit ints
  vec_a = vdup_n_u8(9);
  vec_b = vdup_n_u8(10);
  vec_dest = vec_a * vec_b;  // 90
  vec_a = vec_dest * vec_b;  // 90*10 = 900
  vec_dest = vec_a * vec_b;  // 900*10 = 9000
  int i = 0;
  int result;

  result = vget_lane_u8(vec_dest, 0);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 1);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 2);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 3);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 4);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 5);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 6);
  printf("Lane %d: %d\n", i, result);
  i++;
  result = vget_lane_u8(vec_dest, 7);
  printf("Lane %d: %d\n", i, result);

  float32x4_t fv_a, fv_b, fv_c;
}

void naive_sgemm_square(const float *a, const float *b, float *c, long n) {
  // test_neon();

  for (long i = 0; i < n; i++) {
    for (long j = 0; j < n; j++) {
      for (long k = 0; k < n; k++) {
        c[i * n + j] += a[i * n + k] * b[k * n + j];
      }
    }
  }
}
