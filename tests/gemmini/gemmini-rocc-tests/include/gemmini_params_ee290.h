#ifndef GEMMINI_PARAMS_H
#define GEMMINI_PARAMS_H

#include <stdint.h>
#include <limits.h>

#define DIM 32
#define ADDR_LEN 32
#define BANK_NUM 4
#define BANK_ROWS 2048
#define ACC_ROWS 512
#define MAX_BYTES 64
#define MAX_BLOCK_LEN (MAX_BYTES/(DIM*1))
#define MAX_BLOCK_LEN_ACC 1

typedef int8_t elem_t;
elem_t elem_t_max = 127;
elem_t elem_t_min = -128;
typedef int32_t acc_t;
typedef int64_t full_t;

#define row_align(blocks) __attribute__((aligned(blocks*DIM*sizeof(elem_t))))
#define row_align_acc(blocks) __attribute__((aligned(blocks*DIM*sizeof(acc_t))))

#endif // GEMMINI_PARAMS_H