#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_ld_2d.h"

int main() {
    gemmini_flush(0);

    int size_n = 20;
    int size_m = 16;

    float x[size_n*size_m];
    for (int i = 0; i < size_n; i++) {
        for (int j = 0; j < size_m; j++) {
            x[size_m*i + j] = (float)1.0*i*j;
        }
    }
    float *y = (float*) 0;

    ld_2d(size_n, size_m, x, y);

    printf("\nDone\n");

    exit(0);
}
