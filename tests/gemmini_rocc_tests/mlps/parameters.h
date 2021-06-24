
#include <stdio.h>
#include "include/gemmini.h"

#define LEN(arr) ((int) (sizeof (arr) / sizeof (arr[0])))

// batch size: 16
// before zeropad: 100x140x20x50x10
// after zeropad: 112x144x32x64x16
elem_t input_mat[16][112] row_align(1)= {0};
elem_t weights0[112][144] row_align(1)= {0};
elem_t inter_results0[16][144] row_align(1)= {0};
elem_t weights1[144][32] row_align(1)= {0};
elem_t inter_results1[16][32] row_align(1)= {0};
elem_t weights2[32][64] row_align(1)= {0};
elem_t inter_results2[16][64] row_align(1)= {0};
elem_t weights3[64][16] row_align(1)= {0};
elem_t inter_results3[16][16] row_align(1)= {0};
