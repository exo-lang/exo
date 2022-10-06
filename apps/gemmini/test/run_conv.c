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

#if !defined(BATCH_SIZE) || !defined(IN_CHANNEL) || !defined(IN_DIM) ||        \
    !defined(KERNEL_DIM) || !defined(OUT_CHANNEL) || !defined(OUT_DIM)
#error                                                                         \
    "Must define BATCH_SIZE, IN_CHANNEL, IN_DIM, KERNEL_DIM, OUT_CHANNEL, and OUT_DIM"
#endif

#if !defined(KERNEL_FN)
#error "Must define KERNEL_FN"
#endif

#define CPU_KERNEL_FN CAT(KERNEL_FN, _cpu)

static float scale[1] = {1.0};
static int32_t bias[1 * OUT_CHANNEL];
static int8_t output_cpu[BATCH_SIZE * OUT_DIM * OUT_DIM * OUT_CHANNEL] = {0};
static int8_t output_gemmini[BATCH_SIZE * OUT_DIM * OUT_DIM * OUT_CHANNEL] = {
    0};
static int8_t inp[BATCH_SIZE * IN_DIM * IN_DIM * IN_CHANNEL];
static int8_t weights[OUT_CHANNEL * KERNEL_DIM * KERNEL_DIM * IN_CHANNEL];

int main() {
  gemm_init_mem();
  gemm_acc_init_mem();
  gemmini_flush(0);

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < OUT_CHANNEL; j++) {
      bias[OUT_CHANNEL * i + j] = -1 * j;
    }
  }

  for (int i = 0; i < BATCH_SIZE; i++) {
    for (int j = 0; j < IN_DIM; j++) {
      for (int k = 0; k < IN_DIM; k++) {
        for (int r = 0; r < OUT_CHANNEL; r++) {
          inp[IN_DIM * IN_DIM * OUT_CHANNEL * i + IN_DIM * OUT_CHANNEL * j +
              OUT_CHANNEL * k + r] = j + k + r * 3;
        }
      }
    }
  }

  for (int i = 0; i < OUT_CHANNEL; i++) {
    for (int j = 0; j < KERNEL_DIM; j++) {
      for (int k = 0; k < KERNEL_DIM; k++) {
        for (int r = 0; r < IN_CHANNEL; r++) {
          weights[KERNEL_DIM * KERNEL_DIM * IN_CHANNEL * i +
                  KERNEL_DIM * IN_CHANNEL * j + IN_CHANNEL * k + r] =
              i + k * 3 + r;
        }
      }
    }
  }

  gemmini_lib_Context *ctxt;

  unsigned long cpu_start = read_cycles();
  CPU_KERNEL_FN(ctxt, output_cpu, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long cpu_stop = read_cycles();
  printf("Cycles for CPU version: %ld\n", cpu_stop - cpu_start);

  unsigned long gemmini_start = read_cycles();
  KERNEL_FN(ctxt, output_gemmini, bias, inp, weights, false, scale);
  gemmini_fence();
  unsigned long gemmini_stop = read_cycles();
  printf("Cycles for GEMMINI version: %ld\n", gemmini_stop - gemmini_start);

  if (check_eq_4i8(BATCH_SIZE, OUT_DIM, OUT_DIM, OUT_CHANNEL, output_cpu,
          output_gemmini)) {
    printf("Correct\n");
  } else {
    printf("Results Don't Match\n");
    printf("Correct Result (output_cpu):\n");
    print_4i8(BATCH_SIZE, OUT_DIM, OUT_DIM, OUT_CHANNEL, output_cpu);
    printf("Computed Roundtrip (output_gemmini):\n");
    print_4i8(BATCH_SIZE, OUT_DIM, OUT_DIM, OUT_CHANNEL, output_gemmini);
    exit(1);
  }

  printf("\nDone\n");

  exit(0);
}
