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

static float scale[1] = {2.0};
static int32_t bias[1 * 256];
static int8_t output_cpu[4 * 14 * 14 * 256] = {0};
static int8_t output_gemmini[4 * 14 * 14 * 256] = {0};
static int8_t inp[4 * 16 * 16 * 256];
static int8_t weights[256 * 3 * 3 * 256];

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 256; j++) {
      bias[(256) * i + j] = -1 * j;
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 16; j++) {
      for (int k = 0; k < 16; k++) {
        for (int r = 0; r < 256; r++) {
          inp[(16 * 16 * 256) * i + (16 * 256) * j + (256) * k + r] =
              j + i + k * 2 + r * 3;
        }
      }
    }
  }

  for (int i = 0; i < 256; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int r = 0; r < 256; r++) {
          weights[(3 * 3 * 256) * i + (3 * 256) * j + (256) * k + r] =
              i + k * 3 + r;
        }
      }
    }
  }

  gemmini_lib_Context *ctxt;

  unsigned long cpu_start = read_cycles();
  conv_30_cpu(ctxt, output_cpu, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %d\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  conv_30(ctxt, output_gemmini, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %d\n", gemmini_stop - gemmini_start);

  if (check_eq_4i8(4, 14, 14, 256, output_cpu, output_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (output_cpu):\n");
    print_4i8(4, 14, 14, 256, output_cpu);
    printf("Computed Roundtrip (output_gemmini):\n");
    print_4i8(4, 14, 14, 256, output_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
