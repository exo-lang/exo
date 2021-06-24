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

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  // printf("Flush\n");
  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  static elem_t Out[DIM][DIM] row_align(1);

  // printf("Mvin %d\n", n);
  gemmini_mvin(NULL, 0);
  // printf("Mvout %d\n", n);
  gemmini_mvout(Out, 0);

  // printf("Fence");
  gemmini_fence();

  bool success = true;

  for (size_t i = 0; i < DIM; i++)
    for (size_t j = 0; j < DIM; j++)
      if (Out[i][j] != 0)
        success = false;

  if (!success) {
    printf("Matrix:\n");
    printMatrix(Out);
    printf("\n");

    exit(1);
  }

  exit(0);
}

