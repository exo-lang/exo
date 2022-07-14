#include "gemm_malloc.h"

#include "include/gemmini.h"
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

/*
#include <stdio.h>
#include <stdlib.h>
int main(void) {
fprintf(stderr, "calling init_mem\n");
init_mem();
uint32_t zero = gemm_malloc(10);
fprintf(stderr, "zero: %d\n", zero);
uint32_t one = gemm_malloc(20);
fprintf(stderr, "one: %d\n", one);
uint32_t three = gemm_malloc(40);
fprintf(stderr, "three: %d\n", three);
uint32_t six = gemm_malloc(100);
fprintf(stderr, "six: %d\n", six);
uint32_t _13 = gemm_malloc(200);
fprintf(stderr, "_13: %d\n", _13);

gemm_free(one);
uint32_t one2 = gemm_malloc(20);
fprintf(stderr, "one2: %d\n", one2);
gemm_free(_13);
uint32_t _13_2 = gemm_malloc(300);
fprintf(stderr, "_13_2: %d\n", _13_2);

uint32_t t = gemm_malloc(30);
fprintf(stderr, "t: %d\n", t);

for (int i = 0; i < 100; i ++) {
uint32_t ptr = gemm_malloc(rand()%100);
printf("%d\n", ptr);
gemm_free(ptr);
}

return 0;
}
*/
