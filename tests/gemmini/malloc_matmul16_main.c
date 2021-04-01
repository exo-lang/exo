#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_matmul_16.h"

void init_mem();
void gemm_init_mem();

int main() {
    gemmini_flush(0);
    init_mem();
    gemm_init_mem();

    float z[16][16];

    matmul_16_malloc(z);

    printf("The actual output is:\n");
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < 16; j++)
            printf("%d ", (int)z[i][j]);
        printf("\n");
    }
    printf("\nDone\n");

    exit(0);
}
