#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "tmp/test_load_16.h"

int main() {
    pin_all();
    gemmini_flush(0);

    float x[16][16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            x[i][j] = (float)1.0*i*j;
        }
    }
    float *y = (float*) 0;

    ld_16(x, y);

    printf("\nDone\n");

    exit(0);
}
