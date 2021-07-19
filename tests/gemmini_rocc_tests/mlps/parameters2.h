
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 64
// before zeropad: 784x800x10
// after zeropad: 832x832x64
static elem_t input_mat[64][832] row_align(1)= {0};
static elem_t weights0[832][832] row_align(1)= {0};
static elem_t inter_results0[64][832] row_align(1)= {0};
static elem_t weights1[832][64] row_align(1)= {0};
static elem_t inter_results1[64][64] row_align(1)= {0};
