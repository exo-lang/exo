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

#define PG_SIZE (4*1024)
#define OFFSET 1

/*struct aligned_buffer {
  char garbage[0];
  elem_t data[DIM][DIM];
} __attribute__((__packed__));*/

struct offset_buffer {
  elem_t garbage[OFFSET];
  elem_t data[DIM][DIM];
} __attribute__((__packed__));

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);

  // static struct aligned_buffer In __attribute__((aligned(PG_SIZE)));
  static struct offset_buffer In __attribute__((aligned(PG_SIZE)));
  static struct offset_buffer Out __attribute__((aligned(PG_SIZE)));

  for (size_t i = 0; i < OFFSET; ++i) {
      In.garbage[i] = ~0;
      Out.garbage[i] = 1;
  }

  for (size_t i = 0; i < DIM; ++i)
    for (size_t j = 0; j < DIM; ++j) {
      In.data[i][j] = i*DIM + j;
      Out.data[i][j] = 1;
    }

  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  // printf("Mvin\n");
  gemmini_mvin(In.data, 0);
  // printf("Mvout\n");
  gemmini_mvout(Out.data, 0);

  // printf("Fence\n");
  gemmini_fence();

  if (!is_equal(In.data, Out.data)) {
    printf("Matrix:\n");
    printMatrix(In.data);
    printf("Matrix output:\n");
    printMatrix(Out.data);
    printf("\n");

    exit(1);
  }

  exit(0);
}

