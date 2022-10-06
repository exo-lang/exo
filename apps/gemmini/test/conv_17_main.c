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

float scale[1];
static int32_t bias[1 * 128];
static int8_t output_cpu[4 * 28 * 28 * 128];
static int8_t output_gemmini[4 * 28 * 28 * 128];
static int8_t inp[4 * 30 * 30 * 128];
static int8_t weights[128 * 3 * 3 * 128];

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  gemmini_lib_Context *ctxt;
  scale[0] = 1.0;

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 128; j++) {
      bias[(128) * i + j] = -1 * j;
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        for (int r = 0; r < 128; r++) {
          output_cpu[(28 * 28 * 128) * i + (28 * 128) * j + (128) * k + r] = 0;
        }
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 28; j++) {
      for (int k = 0; k < 28; k++) {
        for (int r = 0; r < 128; r++) {
          output_gemmini[(28 * 28 * 128) * i + (28 * 128) * j + (128) * k + r] =
              0;
        }
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 30; j++) {
      for (int k = 0; k < 30; k++) {
        for (int r = 0; r < 128; r++) {
          inp[(30 * 30 * 128) * i + (30 * 128) * j + (128) * k + r] =
              j + k + r * 3;
        }
      }
    }
  }

  for (int i = 0; i < 128; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int r = 0; r < 128; r++) {
          weights[(3 * 3 * 128) * i + (3 * 128) * j + (128) * k + r] =
              i + k * 3 + r;
        }
      }
    }
  }

  unsigned long cpu_start = read_cycles();
  conv_17_cpu(ctxt, output_cpu, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %d\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  conv_17(ctxt, output_gemmini, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %d\n", gemmini_stop - gemmini_start);

  if (check_eq_4i8(4, 28, 28, 128, output_cpu, output_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (output_cpu):\n");
    print_4i8(4, 28, 28, 128, output_cpu);
    printf("Computed Roundtrip (output_gemmini):\n");
    print_4i8(4, 28, 28, 128, output_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
