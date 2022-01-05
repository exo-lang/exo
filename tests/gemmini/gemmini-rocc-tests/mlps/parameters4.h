
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 64
// before zeropad: 3036x4554x3036
// after zeropad: 3072x4608x3072
static elem_t input_mat[64][3072] row_align(1)= {0};
static elem_t weights0[3072][4608] row_align(1)= {0};
static elem_t inter_results0[64][4608] row_align(1)= {0};
static elem_t weights1[4608][3072] row_align(1)= {0};
static elem_t inter_results1[64][3072] row_align(1)= {0};
