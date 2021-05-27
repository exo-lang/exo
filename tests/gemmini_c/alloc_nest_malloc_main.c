#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "../tmp/test_alloc_nest_malloc.h"

void init_mem();

int main() {
    gemmini_flush(0);
    init_mem();

    int n = 10;
    int m = 20;
    float x[n*m];
    float y[n*m];
    float res[n*m];

    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++) {
            x[m*i + j] = 1.0;
            y[m*i + j] = 2.0;
        }

    alloc_nest_malloc(n, m, x, y, res);

    printf("The actual output is:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < m; j++)
            printf("%d ", (int)res[m*i + j]);
        printf("\n");
    }
    printf("\nDone\n");

    exit(0);
}
