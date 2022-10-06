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

float scale[1];
static int32_t bias[1 * 64];
static int8_t output_cpu[4 * 56 * 56 * 64];
static int8_t output_gemmini[4 * 56 * 56 * 64];
static int8_t inp[4 * 58 * 58 * 64];
static int8_t weights[64 * 3 * 3 * 64];

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  gemmini_lib_Context *ctxt;
  scale[0] = 1.0;

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < 64; j++) {
      bias[(64) * i + j] = -1 * j;
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 56; j++) {
      for (int k = 0; k < 56; k++) {
        for (int r = 0; r < 64; r++) {
          output_cpu[(56 * 56 * 64) * i + (56 * 64) * j + (64) * k + r] = 0;
        }
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 56; j++) {
      for (int k = 0; k < 56; k++) {
        for (int r = 0; r < 64; r++) {
          output_gemmini[(56 * 56 * 64) * i + (56 * 64) * j + (64) * k + r] = 0;
        }
      }
    }
  }

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 58; j++) {
      for (int k = 0; k < 58; k++) {
        for (int r = 0; r < 64; r++) {
          inp[(58 * 58 * 64) * i + (58 * 64) * j + (64) * k + r] =
              j + k + r * 3;
        }
      }
    }
  }

  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 3; j++) {
      for (int k = 0; k < 3; k++) {
        for (int r = 0; r < 64; r++) {
          weights[(3 * 3 * 64) * i + (3 * 64) * j + (64) * k + r] =
              i + k * 3 + r;
        }
      }
    }
  }

  unsigned long cpu_start = read_cycles();
  conv_3_cpu(ctxt, output_cpu, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %d\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  conv_3(ctxt, output_gemmini, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %d\n", gemmini_stop - gemmini_start);

  if (check_eq_4i8(4, 56, 56, 64, output_cpu, output_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (output_cpu):\n");
    print_4i8(4, 56, 56, 64, output_cpu);
    printf("Computed Roundtrip (output_gemmini):\n");
    print_4i8(4, 56, 56, 64, output_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
