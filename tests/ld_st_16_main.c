#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "tmp/test_ld_st_16.h"

int main() {
    gemmini_flush(0);

    float x[16][16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            x[i][j] = (float)1.0*i*j;
        }
    }
    float *y = (float*) 0;
    float z[16][16];

    ld_st_16(x, y, z);

    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            printf("%d ", (int)z[i][j]);
        }
        printf("\n");
    }

    printf("\nDone\n");

    exit(0);
}
