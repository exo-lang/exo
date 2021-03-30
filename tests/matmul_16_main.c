#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "tmp/test_matmul_16.h"

int main() {
    gemmini_flush(0);

    int8_t x[16][16];
    int8_t y[16][16];
    int8_t z[16][16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            x[i][j] = (int8_t)i*j;
            y[i][j] = (int8_t)i*j;
        }
    }

    int8_t *xg = (int8_t*) 0;
    int8_t *yg = (int8_t*) 300;
    int8_t *zg = (int8_t*) 600;

    matmul_16(x, xg, y, yg, z, zg);

    bool flag = true;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            if (z[i][j] != (x[i][j] * y[i][j])) {
                flag = false;
            }
        }
    }
    if (flag == false) {
        printf("Test failed!\n");
        printf("Expected output is:\n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++)
                printf("%d ", (int)(x[i][j]*y[i][j]));
            printf("\n");
        }
        printf("The actual output is:\n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++)
                printf("%d ", (int)z[i][j]);
            printf("\n");
        }
    } else {
        printf("Success!\n");
    }

    printf("\nDone\n");

    exit(0);
}
