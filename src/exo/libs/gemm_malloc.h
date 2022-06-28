#include "include/gemmini.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

void gemm_init_mem(void);
uint32_t gemm_malloc(long unsigned int size);
void gemm_free(uint32_t addr);
