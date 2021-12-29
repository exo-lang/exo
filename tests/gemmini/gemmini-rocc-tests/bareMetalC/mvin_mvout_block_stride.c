// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define BLOCK_STRIDE (DIM*2)
#define COLS (DIM*MAX_BLOCK_LEN)

void printMatrixFull(elem_t m[DIM][COLS]) {
  for (size_t i = 0; i < DIM; ++i) {
    for (size_t j = 0; j < COLS; ++j)
#ifndef ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%x ", elem_t_to_elem_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  static elem_t In[DIM][COLS] row_align(1);
  static elem_t Out[DIM][COLS] row_align(1);

  // printf("Flush\n");
  gemmini_flush(0);
  gemmini_extended4_config_ld(COLS * sizeof(elem_t), MVIN_SCALE_IDENTITY, false, BLOCK_STRIDE, 0);
  gemmini_config_st(COLS * sizeof(elem_t));

  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < COLS; ++j)
      In[i][j] = i*COLS + j;

  gemmini_block_mvin(In, 0, MAX_BLOCK_LEN);

  gemmini_fence();

  for (size_t n = 0; n < MAX_BLOCK_LEN; ++n) {
    gemmini_mvout((elem_t*)Out + n * DIM, n*BLOCK_STRIDE);
  }

  // printf("Fence");
  gemmini_fence();

  if (!MAT_IS_EQUAL(DIM, COLS, In, Out)) {
    printf("Matrix:\n");
    printMatrixFull(In);
    printf("\nMatrix output:\n");
    printMatrixFull(Out);
    printf("\n");

    exit(1);
  }

  exit(0);
}

