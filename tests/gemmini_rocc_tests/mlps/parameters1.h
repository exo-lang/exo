
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 64
// before zeropad: 784x2500x2000x1500x1000x500x10
// after zeropad: 832x2560x2048x1536x1024x512x64
static elem_t input_mat[64][832] row_align(1)= {0};
static elem_t weights0[832][2560] row_align(1)= {0};
static elem_t inter_results0[64][2560] row_align(1)= {0};
static elem_t weights1[2560][2048] row_align(1)= {0};
static elem_t inter_results1[64][2048] row_align(1)= {0};
static elem_t weights2[2048][1536] row_align(1)= {0};
static elem_t inter_results2[64][1536] row_align(1)= {0};
static elem_t weights3[1536][1024] row_align(1)= {0};
static elem_t inter_results3[64][1024] row_align(1)= {0};
static elem_t weights4[1024][512] row_align(1)= {0};
static elem_t inter_results4[64][512] row_align(1)= {0};
static elem_t weights5[512][64] row_align(1)= {0};
static elem_t inter_results5[64][64] row_align(1)= {0};
