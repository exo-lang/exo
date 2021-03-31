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
            x[i][j] = (float)i*j;
        }
    }

    float *y = (float*) 300;
    float z[16][16];

    ld_st_16(x, y, z);

    bool flag = true;
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++) {
            if (x[i][j] != z[i][j]) {
                flag = false;
            }
        }
    }
    if (flag == false) {
        printf("Test failed!\n");
        printf("Expected output is:\n");
        for (int i = 0; i < 16; i++) {
            for (int j = 0; j < 16; j++)
                printf("%d ", (int)x[i][j]);
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
