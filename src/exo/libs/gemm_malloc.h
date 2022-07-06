#ifndef GEMM_MALLOC_H
#define GEMM_MALLOC_H
void gemm_init_mem(void);
uint32_t gemm_malloc(long unsigned int size);
void gemm_free(uint32_t addr);
#endif
