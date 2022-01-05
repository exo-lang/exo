#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#ifndef BAREMETAL

#define BATCHES 4
#define INPUT_DIM 7
#define CHANNELS 2048

#else

#define BATCHES 2
#define INPUT_DIM 3
#define CHANNELS 47

#endif

void init_random(elem_t * buf, int len) {
  for (int i = 0; i < len; i++) {
    buf[i] = rand() % 10;
  }
}

bool is_same(elem_t * x, elem_t * y, int len) {
  for (int i = 0; i < len; i++)
    if (x[i] != y[i])
      return false;
  return true;
}

int main() {
  static elem_t input[BATCHES][INPUT_DIM][INPUT_DIM][CHANNELS];
  static elem_t output[BATCHES][CHANNELS];
  static elem_t gold[BATCHES][CHANNELS];

  init_random((elem_t*)input, BATCHES * INPUT_DIM * INPUT_DIM * CHANNELS);

  printf("CPU average pooling...\n");
  tiled_global_average_auto((elem_t*)input, (elem_t*)gold,
    BATCHES, CHANNELS, INPUT_DIM, CPU);

  printf("Gemmini average pooling...\n");
  tiled_global_average_auto((elem_t*)input, (elem_t*)output,
    BATCHES, CHANNELS, INPUT_DIM, WS);

  if (!is_same((elem_t*)gold, (elem_t*)output, BATCHES * CHANNELS)) {
    printf("Fail\n");

    printf("Input:\n");
    for (int b = 0; b < BATCHES; b++) {
      for (int row = 0; row < INPUT_DIM; row++) {
        printf("{");
        for (int col = 0; col < INPUT_DIM; col++) {
          printf("{");
          for (int ch = 0; ch < CHANNELS; ch++) {
            printf("%d ", input[b][row][col][ch]);
          }
          printf("}");
        }
        printf("}");
      }
      printf("\n");
    }

    printf("Output:\n");
    for (int b = 0; b < BATCHES; b++) {
      for (int ch = 0; ch < CHANNELS; ch++) {
        printf("%d ", output[b][ch]);
      }
      printf("\n");
    }

    printf("Gold:\n");
    for (int b = 0; b < BATCHES; b++) {
      for (int ch = 0; ch < CHANNELS; ch++) {
        printf("%d ", gold[b][ch]);
      }
      printf("\n");
    }

    exit(1);
  }

  exit(0);
}

