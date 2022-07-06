#ifndef GEMM_ACC_MALLOC_H
#define GEMM_ACC_MALLOC_H
void gemm_acc_init_mem(void);
uint32_t gemm_acc_malloc(long unsigned int size);
void gemm_acc_free(uint32_t addr);
#endif
