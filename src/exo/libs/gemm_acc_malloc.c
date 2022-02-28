// clang-format off
// these are filled in by Python's str.format()
#define HEAP_SIZE {heap_size}
#define DIM {dim}
// clang-format on

#include "include/gemmini.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

typedef struct __attribute__((__packed__)) AccBlock {
  uint32_t size;
  uint32_t loc;
  uint8_t is_used;
} AccBlock;

// maintain a stack of blocks corresponding to
// a stack alloc and free strategy
#define N_ACC_BLOCKS (HEAP_SIZE / sizeof(AccBlock))
AccBlock ACC_BLOCKS[N_ACC_BLOCKS];
uint32_t gemm_acc_free_block;

void gemm_acc_init_mem() {
  uint8_t *buf = (uint8_t *)ACC_BLOCKS;
  for (uint32_t i = 0; i < sizeof(ACC_BLOCKS); i++)
    buf[i] = 0;
  gemm_acc_free_block = 0;
}

uint32_t gemm_acc_malloc(long unsigned int size) {
  // must have two free metadata blocks and
  // this allocation must have > 0 size
  if (size == 0)
    return -1;
  if (gemm_acc_free_block >= N_ACC_BLOCKS)
    return -1;

  size = (size + DIM - 1) / DIM;
  uint32_t i = gemm_acc_free_block;

  uint32_t loc = 0;
  if (i > 0) {
    loc = ACC_BLOCKS[i - 1].loc + ACC_BLOCKS[i - 1].size;
  }
  ACC_BLOCKS[i].size = size;
  ACC_BLOCKS[i].loc = 0;
  ACC_BLOCKS[i].is_used = 1;
  gemm_acc_free_block = i + 1;

  return (ACC_BLOCKS[i].loc | ((uint32_t)0x80000000));
}

void gemm_acc_free(uint32_t addr) {
  if (gemm_acc_free_block == 0)
    return;
  addr = addr & (uint32_t)(0x7FFFFFFF);
  // first case: free-ing the top of the block-stack
  if (ACC_BLOCKS[gemm_acc_free_block - 1].loc == addr) {
    ACC_BLOCKS[gemm_acc_free_block - 1].is_used = 0;

    // Then go through and release as many blocks
    // as we can
    for (int i = gemm_acc_free_block - 1; i >= 0; i--) {
      if (ACC_BLOCKS[i].is_used)
        break;  // loop termination
      // otherwise...
      gemm_acc_free_block = i;
    }
    // second case: find the freed block and mark it
  } else {
    for (int i = gemm_acc_free_block - 1; i >= 0; i--) {
      if (ACC_BLOCKS[i].loc == addr) {
        ACC_BLOCKS[i].is_used = 0;
        break;
      }
    }
  }
  return;
}
