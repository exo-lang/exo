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

#define MIN(a,b) ((a > b) ? b : a)
#ifdef FAST
#define BIG_DIM (DIM*2)
#define BINIT MIN(MAX_BLOCK_LEN_ACC, BIG_DIM/DIM)
#else
#define BIG_DIM 64
#define BINIT 1
#endif

#if (BIG_DIM % DIM) != 0
#error incorrect dimensions
#endif

#if (BIG_DIM * BIG_DIM / DIM) > (BANK_ROWS * BANK_NUM)
#error not enough rows
#endif

int is_equal_big(elem_t x[BIG_DIM][BIG_DIM], elem_t y[BIG_DIM][BIG_DIM]) {
  for (size_t i = 0; i < BIG_DIM; ++i)
    for (size_t j = 0; j < BIG_DIM; ++j)
      if (x[i][j] != y[i][j])
          return 0;
  return 1;
}

void printMatrix_big(elem_t m[BIG_DIM][BIG_DIM]) {
  for (size_t i = 0; i < BIG_DIM; ++i) {
    for (size_t j = 0; j < BIG_DIM; ++j)
      printf("%d ", m[i][j]);
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

  gemmini_flush(0);

  for (int block_len = BINIT; block_len <= BIG_DIM/DIM && block_len <= MAX_BLOCK_LEN; block_len++) {
    // printf("block_len: %d\n", block_len);

    static elem_t In[BIG_DIM][BIG_DIM] row_align(1);
    static elem_t Out[BIG_DIM][BIG_DIM] row_align(1);

    for (size_t i = 0; i < BIG_DIM; ++i)
      for (size_t j = 0; j < BIG_DIM; ++j) {
#ifdef FAST
#define RAND (j + i)
#else
#define RAND rand()
#endif
        In[i][j] = (RAND % 64) - 32; // i*BIG_DIM + j;
        Out[i][j] = 0;
      }

    gemmini_config_ld(BIG_DIM*sizeof(elem_t));
    gemmini_config_st(BIG_DIM*sizeof(elem_t));

    for (size_t i = 0; i < BIG_DIM; i += DIM) {
      for (size_t j = 0; j < BIG_DIM; j += DIM) {
        // printf("i: %u, j: %u\n", i, j);

        elem_t * dram_addr_in = &In[i][j];
        elem_t * dram_addr_out = &Out[i][j];
        uint32_t sp_addr = i*(BIG_DIM/DIM) + j;

        int already_moved_in = (j/DIM) % block_len != 0;

        if (!already_moved_in) {
          int len = j + block_len*DIM <= BIG_DIM ? block_len : (BIG_DIM-j)/DIM;
          // printf("Moving in with len: %d\n", len);
          gemmini_block_mvin(dram_addr_in, sp_addr, len);
          gemmini_mvout(dram_addr_out, sp_addr);
        } else {
          // printf("Already moved in, so moving out\n");
          gemmini_mvout(dram_addr_out, sp_addr);
        }
      }
    }

    gemmini_fence();

    if (!is_equal_big(In, Out)) {
      printf("fails at block_len: %d\n", block_len);

      // printf("Matrix output:\n");
      // printMatrix_big(Out);
      // printf("Matrix gold:\n");
      // printMatrix_big(In);
      // printf("\n");

      exit(1);
    }
  }

  exit(0);
}

