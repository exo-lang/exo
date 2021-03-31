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

    float x[16][16];
    float y[16][16];
    float z[16][16];
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            x[i][j] = (float)i*j;
            y[i][j] = (float)(i*j*2.5);
        }
    }

    float *xg = (float*) 0;
    float *yg = (float*) 16;
    float *zg = (float*) 32;

    matmul_16(x, xg, y, yg, z, zg);

    bool flag = true;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            float res = 0;
            for (int k = 0; k < 16; k++)
                res += x[i][k] * y[k][j];
            if (z[i][j] != res) {
                flag = false;
            }
        }
    }
    if (flag == false) {
        printf("Test failed!\n");
        printf("Expected output is:\n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++) {
                float res = 0;
                for (int k = 0; k < 16; k++)
                    res += x[i][k] * y[k][j];
                printf("%d ", (int)res);
            }
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
