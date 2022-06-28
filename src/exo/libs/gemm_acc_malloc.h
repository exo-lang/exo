#define HEAP_SIZE 100000
#define DIM 16

#include "include/gemmini.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

void gemm_acc_init_mem(void);
uint32_t gemm_acc_malloc(long unsigned int size);
void gemm_acc_free(uint32_t addr);
