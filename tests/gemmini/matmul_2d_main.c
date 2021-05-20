#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_window_matmul_2d.h"

int main() {
    gemmini_flush(0);

    int size_n = 32;
    int size_m = 16;
    int size_k = 16;

    float x[size_n][size_k];
    float y[size_k][size_m];
    float z[size_n][size_m];

    for (int i = 0; i < size_n; i++) {
        for (int j = 0; j < size_m; j++) {
            for (int k = 0; k < size_k; k++) {
                x[i][k] = (float)i*k;
                y[k][j] = (float)(k*j*2.5);
            }
        }
    }

    float *xg = (float*) 0;
    float *yg = (float*) 32;
    float *zg = (float*) 64;

    // CPU
    float tmp[size_n][size_m];
    for (int i = 0; i < size_n; i++)
        for (int j = 0; j < size_m; j++)
            tmp[i][j] = 0;
    unsigned int cpu_start = read_cycles();
    for (int i = 0; i < size_n; i++)
        for (int j = 0; j < size_m; j++)
            for (int k = 0; k < size_k; k++)
                tmp[i][j] += x[i][k] * y[k][j];
    unsigned int cpu_end = read_cycles();
    printf("\nCPU Cycles taken : %u\n", cpu_end-cpu_start);


    // GEMMINI
    cpu_start = read_cycles();
    matmul_2d(size_n, size_m, size_k, x, xg, y, yg, z, zg);
    cpu_end = read_cycles();
    printf("\nGEMMINI Cycles taken : %u\n", cpu_end-cpu_start);


    // Result check
    bool flag = true;
    for (int i = 0; i < size_n; i++) {
        for (int j = 0; j < size_m; j++) {
            float res = 0;
            for (int k = 0; k < size_k; k++)
                res += x[i][k] * y[k][j];
            if (z[i][j] != res) {
                flag = false;
            }
        }
    }

    if (flag == false) {
        printf("Test failed!\n");
        printf("Expected output is:\n");
        for (int i = 0; i < size_n; i++) {
            for (int j = 0; j < size_m; j++) {
                float res = 0;
                for (int k = 0; k < size_k; k++)
                    res += x[i][k] * y[k][j];
                printf("%d ", (int)res);
            }
            printf("\n");
        }
        printf("The actual output is:\n");
        for (int i = 0; i < size_n; i++) {
            for (int j = 0; j < size_m; j++)
                printf("%d ", (int)z[i][j]);
            printf("\n");
        }
    } else {
        printf("Success!\n");
    }

    printf("\nDone\n");

    exit(0);
}
