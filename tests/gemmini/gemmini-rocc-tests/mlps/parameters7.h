
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 64
// before zeropad: 400x500x440
// after zeropad: 448x512x448
static elem_t input_mat[64][448] row_align(1)= {0};
static elem_t weights0[448][512] row_align(1)= {0};
static elem_t inter_results0[64][512] row_align(1)= {0};
static elem_t weights1[512][448] row_align(1)= {0};
static elem_t inter_results1[64][448] row_align(1)= {0};
