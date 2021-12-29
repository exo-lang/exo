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

#if BANK_NUM*BANK_ROWS < 5*DIM
#error need more memory capacity
#endif

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

  gemmini_flush(0);
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  const int a_additions = 10;
  const int b_additions = 10;
  const int d_additions = 10;

  static elem_t IDENTITY[DIM][DIM] row_align(1);

  static elem_t result_A[DIM][DIM] row_align(1);
  static elem_t result_B[DIM][DIM] row_align(1);
  static elem_t result_D[DIM][DIM] row_align(1);

  static elem_t gold_A[DIM][DIM];
  static elem_t gold_B[DIM][DIM];
  static elem_t gold_D[DIM][DIM];

  for (size_t i = 0; i < DIM; i++) {
    for (size_t j = 0; j < DIM; j++) {
      IDENTITY[i][j] = i == j;
      gold_A[i][j] = i == j ? (a_additions+1) : 0;
      gold_B[i][j] = i == j ? (b_additions+1) : 0;
      gold_D[i][j] = i == j ? (d_additions+1) : 0;
    }
  }

  int IDENTITY1_addr = 0;
  int IDENTITY2_addr = DIM;
  int A_addr = 2*DIM;
  int B_addr = 3*DIM;
  int D_addr = 4*DIM;

  // printf("Moving in\n");
  gemmini_mvin(IDENTITY, IDENTITY1_addr);
  gemmini_mvin(IDENTITY, IDENTITY2_addr);
  gemmini_mvin(IDENTITY, A_addr);
  gemmini_mvin(IDENTITY, B_addr);
  gemmini_mvin(IDENTITY, D_addr);
  
  // printf("Setting mode\n");
  gemmini_config_ex(OUTPUT_STATIONARY, 0, 0, 0);

  // printf("RAW with A\n");
  for (int i = 0; i < a_additions; i++) {
    // printf("  %d\n", i);

    gemmini_preload(IDENTITY1_addr, A_addr);
    gemmini_compute_preloaded(A_addr, IDENTITY2_addr);
  }

  // printf("RAW with B\n");
  for (int i = 0; i < b_additions; i++) {
    gemmini_preload(IDENTITY1_addr, B_addr);
    gemmini_compute_preloaded(IDENTITY2_addr, B_addr);
  }

  // printf("RAW with D\n");
  for (int i = 0; i < d_additions; i++) {
    gemmini_preload(D_addr, D_addr);
    gemmini_compute_preloaded(IDENTITY1_addr, IDENTITY2_addr);
  }

  // printf("Moving out A\n");
  gemmini_mvout(result_A, A_addr);
  // printf("Moving out B\n");
  gemmini_mvout(result_B, B_addr);
  // printf("Moving out D\n");
  gemmini_mvout(result_D, D_addr);

  // printf("Fencing\n");
  gemmini_fence();

  // printf("Checking\n");
  int fail = 0;

  if (!is_equal(result_A, gold_A)) {
    printf("A:\n");
    printMatrix(result_A);
    printf("\n");
    // printMatrix(gold_A);
    // printf("\n");

    fail = 1;
  }
  
  if (!is_equal(result_B, gold_B)) {
    printf("B:\n");
    printMatrix(result_B);
    printf("\n");
    // printMatrix(gold_B);
    // printf("\n");

    fail = 1;
  }
  
  if (!is_equal(result_D, gold_D)) {
    printf("D:\n");
    printMatrix(result_D);
    printf("\n");
    // printMatrix(gold_D);
    // printf("\n");

    fail = 1;
  }

  exit(fail);
}

