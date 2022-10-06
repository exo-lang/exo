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

#if !defined(MM) || !defined(NN) || !defined(KK)
#error "Must define MM, NN, and KK"
#endif

#if !defined(KERNEL_FN)
#error "Must define KERNEL_FN"
#endif

#define CPU_KERNEL_FN CAT(cpu_, KERNEL_FN)

static float c_scale[1] = {2.0f};
static int8_t x[NN * KK];
static int8_t y[KK * MM];
static int8_t z_cpu[NN * MM] = {0};
static int8_t z_gemmini[NN * MM] = {0};

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  for (int i = 0; i < NN; i++) {
    for (int j = 0; j < KK; j++) {
      x[(KK)*i + j] = i + j * 2;
    }
  }

  for (int i = 0; i < KK; i++) {
    for (int j = 0; j < MM; j++) {
      y[(MM)*i + j] = j * 3 + i;
    }
  }

  gemmini_lib_Context *ctxt;

  unsigned long cpu_start = read_cycles();
  CPU_KERNEL_FN(ctxt, c_scale, false, x, y, z_cpu);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %ld\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  KERNEL_FN(ctxt, c_scale, false, x, y, z_gemmini);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %ld\n", gemmini_stop - gemmini_start);

  if (check_eq_2i8(NN, MM, z_cpu, z_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (z_cpu):\n");
    print_2i8(NN, MM, z_cpu);
    printf("Computed Roundtrip (z_gemmini):\n");
    print_2i8(NN, MM, z_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
