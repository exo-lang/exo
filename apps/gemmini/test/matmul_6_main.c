#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "include/gemmini_testutils.h"

#include "gemmini_lib.h"
#include "gemm_malloc.h"
#include "gemm_acc_malloc.h"

#include "helpers.h"

static int8_t x[12544 * 256];
static int8_t y[256 * 64];
float c_scale[1];
static int8_t z_cpu[12544 * 64];
static int8_t z_gemmini[12544 * 64];

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  gemmini_lib_Context *ctxt;
  for (int i = 0; i < 12544; i++) {
    for (int j = 0; j < 256; j++) {
      x[(256) * i + j] = i + j * 2;
    }
  }

  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 64; j++) {
      y[(64) * i + j] = j * 3 + i;
    }
  }

  c_scale[0] = 2.0f;

  for (int i = 0; i < 12544; i++) {
    for (int j = 0; j < 64; j++) {
      z_cpu[(64) * i + j] = 0;
    }
  }

  for (int i = 0; i < 12544; i++) {
    for (int j = 0; j < 64; j++) {
      z_gemmini[(64) * i + j] = 0;
    }
  }

  unsigned long cpu_start = read_cycles();
  cpu_matmul_6(ctxt, c_scale, false, x, y, z_cpu);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %d\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  matmul_6(ctxt, c_scale, false, x, y, z_gemmini);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %d\n", gemmini_stop - gemmini_start);

  if (check_eq_2i8(12544, 64, z_cpu, z_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (z_cpu):\n");
    print_2i8(12544, 64, z_cpu);
    printf("Computed Roundtrip (z_gemmini):\n");
    print_2i8(12544, 64, z_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
