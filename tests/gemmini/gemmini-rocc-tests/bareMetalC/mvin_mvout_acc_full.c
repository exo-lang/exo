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


#define N 4

#if (N*DIM) > ACC_ROWS
#error not enough accumulator space
#endif

int main() {
#ifdef ACC_READ_FULL_WIDTH

#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  static acc_t In[N][DIM][DIM] row_align_acc(1);
  static acc_t Out[N][DIM][DIM] row_align_acc(1);

  // printf("Initializing matrices\n");
  for (size_t n = 0; n < N; ++n)
    for (size_t i = 0; i < DIM; ++i)
      for (size_t j = 0; j < DIM; ++j) {
#ifndef ELEM_T_IS_FLOAT
        In[n][i][j] = 0;
#ifdef FAST
#define RAND (j + i)
#else
#define RAND rand()
#endif

        int bytes = RAND % 2 ? sizeof(acc_t) : sizeof(elem_t);
        for (size_t b = 0; b < bytes; ++b) {
          In[n][i][j] |= (RAND % 255) << (b*8);
        }
#else
        acc_t_bits data;

        do {
          data = 0;

          int bytes = rand() % 2 ? sizeof(acc_t) : sizeof(elem_t);
          for (size_t b = 0; b < bytes; ++b) {
            data |= (uint64_t)(rand() % 255) << (b*8);
          }

          In[n][i][j] = acc_t_bits_to_acc_t(data);
        } while (acc_t_isnan(In[n][i][j]));
#endif
      }

  const uint32_t acc_addr = 5 << (ADDR_LEN-3);

  // printf("Config\n");
  gemmini_config_ld(DIM*sizeof(acc_t));
  gemmini_config_ex(0, NO_ACTIVATION, 0, 0);
  gemmini_config_st(DIM*sizeof(acc_t));

  // printf("Mvin and mvout\n");
  for (size_t n = 0; n < N; ++n) {
    // printf("Mvin n: %u\n", n);
    gemmini_mvin(In[n], acc_addr + n*DIM);
    // printf("Mvout n: %u\n", n);
    gemmini_mvout(Out[n], acc_addr + n*DIM);
  }

  // printf("Fence\n");
  gemmini_fence();

  // printf("Check\n");
  for (size_t n = 0; n < N; ++n)
    if (!MAT_IS_EQUAL(DIM, DIM, Out[n], In[n])) {
      printf("Matrix %u:\n", n);

      for (size_t i = 0; i < DIM; ++i) {
        for (size_t j = 0; j < DIM; ++j)
#ifndef ELEM_T_IS_FLOAT
          printf("%d ", In[n][i][j]);
#else
          printf("%llx ", acc_t_to_acc_t_bits(In[n][i][j]));
#endif
        printf("\n");
      }

      printf("\nMatrix %u output:\n", n);

      for (size_t i = 0; i < DIM; ++i) {
        for (size_t j = 0; j < DIM; ++j)
#ifndef ELEM_T_IS_FLOAT
          printf("%d ", Out[n][i][j]);
#else
          printf("%llx ", acc_t_to_acc_t_bits(Out[n][i][j]));
#endif
        printf("\n");
      }

      printf("\n");

      exit(1);
    }

#endif // #ifdef ACC_READ_FULL_WIDTH

  exit(0);
}

