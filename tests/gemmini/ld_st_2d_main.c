#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_ld_st_2d.h"

int main() {
    gemmini_flush(0);

    int size_n = 16;
    int size_m = 31;

    float x[size_n*size_m];
    for (int i = 0; i < size_n; i++) {
        for (int j = 0; j < size_m; j++) {
            x[size_m*i + j] = (float)i*j;
        }
    }

    float *y = (float*) 0;
    float z[size_n*size_m];

    ld_st_2d(size_n, size_m, x, y, z);

    bool flag = true;
    for (int i = 0; i < size_n; i++) {
        for (int j = 0; j < size_m; j++) {
            if ((int)x[size_m*i + j] != (int)z[size_m*i + j]) {
                flag = false;
            }
        }
    }
    if (flag == false) {
        printf("Test failed!\n");
        printf("Expected output is:\n");
        for (int i = 0; i < size_n; i++) {
            for (int j = 0; j < size_m; j++)
                printf("%d ", (int)x[size_m*i + j]);
            printf("\n");
        }
        printf("The actual output is:\n");
        for (int i = 0; i < size_n; i++) {
            for (int j = 0; j < size_m; j++)
                printf("%d ", (int)z[size_m*i + j]);
            printf("\n");
        }
    } else {
        printf("Success!\n");
    }

    printf("\nDone\n");

    exit(0);
}
