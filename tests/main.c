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
    float y[16][16];

    ld_16(x, y);
    /*
    unsigned long cpu_start = read_cycles();
    unsigned long cpu_end = read_cycles();
    printf("\nCycles taken for simple_blur: %u\n", cpu_end-cpu_start);

    cpu_start = read_cycles();
    unroll_blur(n, m, 5, image, kernel, res);
    cpu_end = read_cycles();
    printf("Cycles taken for unroll_blur: %u\n", cpu_end-cpu_start);
    */

    printf("\nDone\n");

    exit(0);
}
