#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "include/gemmini_testutils.h"
#include "tmp/test_store_16.h"

int main() {
    gemmini_flush(0);

    int8_t *x = (int8_t*) 0;
    int8_t y[16][16];

    st_16(x, y);

    printf("\nDone\n");

    exit(0);
}
