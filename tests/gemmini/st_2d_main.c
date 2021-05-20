#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_window_st_2d.h"

int main() {
    gemmini_flush(0);

    int size_n = 30;
    int size_m = 40;

    float *x = (float*) 0;
    float y[size_n][size_m];

    st_2d(size_n, size_m, x, y);

    printf("\nDone\n");

    exit(0);
}
