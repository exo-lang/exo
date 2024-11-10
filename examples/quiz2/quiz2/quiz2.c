#include "quiz2.h"

#include <stdio.h>
#include <stdlib.h>

// scaled_add(
//     N : size,
//     a : f32[N] @DRAM,
//     b : f32[N] @DRAM,
//     c : f32[N] @DRAM
// )
void scaled_add( void *ctxt, int_fast32_t N, const float* a, const float* b, float* c ) {
EXO_ASSUME(N % 8 == 0);
for (int_fast32_t i = 0; i < N; i++) {
  c[i] = 2.0f * a[i] + 3.0f * b[i];
}
}

// scaled_add_scheduled(
//     N : size,
//     a : f32[N] @DRAM,
//     b : f32[N] @DRAM,
//     c : f32[N] @DRAM
// )
void scaled_add_scheduled( void *ctxt, int_fast32_t N, const float* a, const float* b, float* c ) {
EXO_ASSUME(N % 8 == 0);
for (int_fast32_t io = 0; io < ((N) / (8)); io++) {
  float *vec = (float*) malloc(8 * sizeof(*vec));
  float *vec_1 = (float*) malloc(8 * sizeof(*vec_1));
  float *vec_2 = (float*) malloc(8 * sizeof(*vec_2));
  float *vec_3 = (float*) malloc(8 * sizeof(*vec_3));
  float *vec_4 = (float*) malloc(8 * sizeof(*vec_4));
  float *vec_5 = (float*) malloc(8 * sizeof(*vec_5));
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    vec_1[ii] = 2.0f;
  }
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    vec_2[ii] = a[8 * io + ii];
  }
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    vec[ii] = vec_1[ii] * vec_2[ii];
  }
  free(vec_2);
  free(vec_1);
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    vec_4[ii] = 3.0f;
  }
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    vec_5[ii] = b[8 * io + ii];
  }
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    vec_3[ii] = vec_4[ii] * vec_5[ii];
  }
  free(vec_5);
  free(vec_4);
  for (int_fast32_t ii = 0; ii < 8; ii++) {
    c[8 * io + ii] = vec[ii] + vec_3[ii];
  }
  free(vec_3);
  free(vec);
}
}

