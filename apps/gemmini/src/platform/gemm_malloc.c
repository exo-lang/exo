#include "gemm_malloc.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#ifndef GEMM_HEAP_SIZE
#define GEMM_HEAP_SIZE 100000
#endif

#ifndef GEMM_DIM
#define GEMM_DIM 16
#endif

typedef struct __attribute__((__packed__)) NewBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} NewBlock;

NewBlock BLOCKS[GEMM_HEAP_SIZE / sizeof(NewBlock)];
uint32_t gemm_last_ptr;

void gemm_init_mem() {
  for (uint32_t i = 0; i < sizeof(BLOCKS); i++)
    ((uint8_t *)BLOCKS)[i] = 0;
  gemm_last_ptr = 0;
}

uint32_t gemm_malloc(long unsigned int size) {
  if (size == 0)
    return -1;
  size = (size + GEMM_DIM - 1) / GEMM_DIM;
  int i;
  for (i = 0; i < GEMM_HEAP_SIZE / sizeof(NewBlock) && BLOCKS[i].size > 0;
       i++) {
    if (BLOCKS[i].is_used)
      continue;
    if (BLOCKS[i].size < size)
      continue;
    break;
  }
  if (BLOCKS[i].size == 0) {
    BLOCKS[i].loc = gemm_last_ptr;
    BLOCKS[i].size = size;
    BLOCKS[i].is_used = 1;
    gemm_last_ptr += size;
    return BLOCKS[i].loc;
  }

  BLOCKS[i].is_used = 1;
  return BLOCKS[i].loc;
}

void gemm_free(uint32_t addr) {
  for (int i = 0; BLOCKS[i].size > 0; i++) {
    if (BLOCKS[i].is_used && BLOCKS[i].loc == addr) {
      BLOCKS[i].is_used = 0;
      return;
    }
  }
  return;
}
