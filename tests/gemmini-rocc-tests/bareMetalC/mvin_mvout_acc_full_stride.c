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

#if (BIG_DIM * BIG_DIM / DIM) > ACC_ROWS
#error not enough accumulator space
#endif

#define MIN(a,b) ((a > b) ? b : a)




int is_equal_big(acc_t x[BIG_DIM][BIG_DIM], acc_t y[BIG_DIM][BIG_DIM]) {
  for (size_t i = 0; i < BIG_DIM; ++i)
    for (size_t j = 0; j < BIG_DIM; ++j) {
#ifndef ELEM_T_IS_FLOAT
      if (x[i][j] != y[i][j]) {
#else
      bool isnanx = acc_t_isnan(x[i][j]);
      bool isnany = acc_t_isnan(y[i][j]);

      if (x[i][j] != y[i][j] && !(isnanx && isnany)) {
          printf("x[i][j] == %x\n", acc_t_to_acc_t_bits(x[i][j]));
          printf("y[i][j] == %x\n", acc_t_to_acc_t_bits(y[i][j]));

#endif
        printf("Unequal in row %u and column %u\n", i, j);
        return 0;
      }
    }
  return 1;
}

void printMatrix_acc_big(acc_t m[BIG_DIM][BIG_DIM]) {
  for (size_t i = 0; i < BIG_DIM; ++i) {
    for (size_t j = 0; j < BIG_DIM; ++j)
#ifndef ELEM_T_IS_FLOAT
      printf("%d ", m[i][j]);
#else
      printf("%llx ", acc_t_to_acc_t_bits(m[i][j]));
#endif
    printf("\n");
  }
}

int main() {
#ifdef ACC_READ_FULL_WIDTH

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  for (int block_len = BINIT; block_len <= BIG_DIM/DIM && block_len <= MAX_BLOCK_LEN_ACC; block_len++) {
    static acc_t In[BIG_DIM][BIG_DIM] row_align_acc(1);
    static acc_t Out[BIG_DIM][BIG_DIM] row_align(1);

    for (size_t i = 0; i < BIG_DIM; ++i) {
      for (size_t j = 0; j < BIG_DIM; ++j) {
#ifndef ELEM_T_IS_FLOAT
        In[i][j] = 0;
#ifdef FAST
#define RAND (j + i)
#else
#define RAND rand()
#endif
        int bytes = RAND % 2 ? sizeof(acc_t) : sizeof(elem_t);
        for (size_t b = 0; b < bytes; ++b) {
          In[i][j] |= (RAND % 255) << (b*8);
        }
#else
        acc_t_bits data;

        do {
          data = 0;

          int bytes = rand() % 2 ? sizeof(acc_t) : sizeof(elem_t);
          for (size_t b = 0; b < bytes; ++b) {
            data |= (uint64_t)(rand() % 255) << (b*8);
          }

          In[i][j] = acc_t_bits_to_acc_t(data);
        } while (acc_t_isnan(In[i][j]));
#endif
      }
    }

    const uint32_t acc_addr = 5 << (ADDR_LEN-3);

    gemmini_config_ld(BIG_DIM*sizeof(acc_t));
    gemmini_config_ex(0, NO_ACTIVATION, 0, 0);
    gemmini_config_st(BIG_DIM*sizeof(acc_t));

    for (size_t i = 0; i < BIG_DIM; i += DIM) {
      for (size_t j = 0; j < BIG_DIM; j += DIM) {
        // printf("i: %u, j: %u\n", i, j);

        acc_t * dram_addr_in = &In[i][j];
        acc_t * dram_addr_out = &Out[i][j];
        uint32_t sp_addr = acc_addr + i*(BIG_DIM/DIM) + j;

        int already_moved_in = (j/DIM) % block_len != 0;

        if (!already_moved_in) {
          int len = j + block_len*DIM <= BIG_DIM ? block_len : (BIG_DIM-j)/DIM;
          // printf("Moving in with len: %d\n", len);
          gemmini_block_mvin(dram_addr_in, sp_addr, len);
          gemmini_mvout(dram_addr_out, sp_addr);
        } else {
          // printf("Already moved in\n");
          gemmini_mvout(dram_addr_out, sp_addr);
        }
      }
    }

    // printf("Fence\n");
    gemmini_fence();

    // printf("Check\n");
    if (!is_equal_big(Out, In)) {
      /*printf("Out:\n");
      for (size_t i = 0; i < BIG_DIM; i++) {
        for (size_t j = 0; j < BIG_DIM; j++) {
          printf("%d, ", Out[i][j]);
        }
        printf("\n");
      }

      printf("\n");

      printf("Gold:\n");
      for (size_t i = 0; i < BIG_DIM; i++) {
        for (size_t j = 0; j < BIG_DIM; j++) {
          printf("%d, ", Out[i][j]);
        }
        printf("\n");
      }*/

      printf("Matrix:\n");
      printMatrix_acc_big(In);
      printf("Matrix output:\n");
      printMatrix_acc_big(Out);
      printf("\n");

      exit(1);
    }
  }

#endif // #ifdef ACC_READ_FULL_WIDTH

  exit(0);
}

