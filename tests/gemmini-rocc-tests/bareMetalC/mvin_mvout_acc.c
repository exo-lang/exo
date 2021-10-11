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
#define N 2
#define AINIT 2
#define SINIT 12
#else
#define N 4
#define AINIT 0
#define SINIT 12
#endif

#if (N*DIM) > ACC_ROWS
#error not enough accumulator space
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  for (int activation = AINIT; activation <= 2; ++activation) {
    for (int scale = SINIT; scale <= 12; scale += 4) {
      // printf("activation: %d, scale: %d\n", activation, scale);

      static acc_t In[N][DIM][DIM] row_align_acc(1);
      static full_t In_full[N][DIM][DIM];
      static elem_t Out[N][DIM][DIM] row_align(1);
      static elem_t Out_gold[N][DIM][DIM];

      int relu6_shift = scale+1;

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

            In_full[n][i][j] = In[n][i][j];
          }

      // printf("Shifting and activating matrices\n");
      for (size_t n = 0; n < N; ++n) {
        matscale(In_full[n], Out_gold[n], scale);

        if (activation == RELU)
          matrelu(Out_gold[n], Out_gold[n]);
        else if (activation == RELU6)
          matrelu6(Out_gold[n], Out_gold[n], 1 << relu6_shift);
      }

      const uint32_t acc_addr = 1 << (ADDR_LEN-1);

      // printf("Config\n");
      gemmini_config_ld(DIM*sizeof(acc_t));
      gemmini_config_ex(0, 0, 0, relu6_shift);
      gemmini_extended_config_st(DIM*sizeof(elem_t), activation, scale);

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
        if (!is_equal(Out[n], Out_gold[n])) {
          printf("activation: %d, scale: %d\n", activation, scale);

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
          printf("Matrix %u output:\n", n);
          printMatrix(Out[n]);
          printf("Matrix %u gold output:\n", n);
          printMatrix(Out_gold[n]);
          printf("\n");

          exit(1);
        }
    }
  }

  exit(0);
}

