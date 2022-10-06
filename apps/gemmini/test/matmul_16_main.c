#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/gemmini_testutils.h"

#include "gemm_acc_malloc.h"
#include "gemm_malloc.h"
#include "gemmini_lib.h"

#include "helpers.h"

static float c_scale[1] = {2.0f};
static int8_t x[3136 * 512];
static int8_t y[512 * 128];
static int8_t z_cpu[3136 * 128] = {0};
static int8_t z_gemmini[3136 * 128] = {0};

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  for (int i = 0; i < 3136; i++) {
    for (int j = 0; j < 512; j++) {
      x[(512) * i + j] = i + j * 2;
    }
  }

  for (int i = 0; i < 512; i++) {
    for (int j = 0; j < 128; j++) {
      y[(128) * i + j] = j * 3 + i;
    }
  }

  gemmini_lib_Context *ctxt;

  unsigned long cpu_start = read_cycles();
  cpu_matmul_16(ctxt, c_scale, false, x, y, z_cpu);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %d\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  matmul_16(ctxt, c_scale, false, x, y, z_gemmini);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %d\n", gemmini_stop - gemmini_start);

  if (check_eq_2i8(3136, 128, z_cpu, z_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (z_cpu):\n");
    print_2i8(3136, 128, z_cpu);
    printf("Computed Roundtrip (z_gemmini):\n");
    print_2i8(3136, 128, z_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
