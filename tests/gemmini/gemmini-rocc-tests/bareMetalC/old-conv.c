#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

#define BATCH_SIZE 4
#define IN_DIM 30
#define IN_CHANNELS 128
#define OUT_CHANNELS 128
#define KERNEL_DIM 3
#define PADDING 0
#define STRIDE 1

#define NO_BIAS false

#define OUT_DIM ((IN_DIM + 2*PADDING - KERNEL_DIM) / STRIDE + 1)
#define PATCH_SIZE (KERNEL_DIM * KERNEL_DIM * IN_CHANNELS)
#define N_PATCHES (BATCH_SIZE * OUT_DIM * OUT_DIM)



bool vec_is_equal(elem_t * a, elem_t * b, int len) {
    for (int i = 0; i < len; i++)
        if (a[i] != b[i])
            return false;
    return true;
}

void init_random(elem_t * buf, int len) {
    elem_t i = 0;
    for (elem_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
#ifdef FAST
      *ptr = 1;
#else
      *ptr = (rand() % 5) - 2;
#endif
    }
}

void init_random_acc(acc_t * buf, int len) {
    elem_t i = 0;
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        // *ptr = (rand() % 32) - 16;
#ifdef FAST
      *ptr = 1;
#else
      *ptr = (rand() % 5) - 2;
#endif
    }
}

void init_zeros_acc(acc_t * buf, int len) {
    for (acc_t * ptr = buf; ptr < buf + len; ptr++) {
        *ptr = 0;
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    // assert((in_dim + 2*padding - kernel_dim) % stride == 0);

    printf("Output dimension: %u\n\n", OUT_DIM);

    static elem_t input[BATCH_SIZE][IN_DIM][IN_DIM][IN_CHANNELS] row_align(1);
    static elem_t weights[OUT_CHANNELS][KERNEL_DIM][KERNEL_DIM][IN_CHANNELS] row_align(1);
    static acc_t bias[OUT_CHANNELS] row_align_acc(1);
    static elem_t output[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS] row_align(1);

    printf("Randomize inputs...\n");
    init_random(&input[0][0][0][0], sizeof(input) / sizeof(elem_t));

    printf("Randomize weights...\n");
    init_random(&weights[0][0][0][0], sizeof(weights) / sizeof(elem_t));

    printf("Randomize bias...\n");
    if (NO_BIAS)
        init_zeros_acc(&bias[0], sizeof(bias) / sizeof(acc_t));
    else
        init_random_acc(&bias[0], sizeof(bias) / sizeof(acc_t));

    static elem_t output2[BATCH_SIZE][OUT_DIM][OUT_DIM][OUT_CHANNELS] row_align(1);

    /*
    conv_30_lib_Context *ctxt;
    float scale = 1.0f;
    conv_on_cpu(ctxt, output, bias, input, weights, false, &scale);
    */

    printf("our conv...\n");
    uint64_t start_gemmini = read_cycles();
    tiled_conv_A_stride_auto(
        BATCH_SIZE, IN_DIM, IN_CHANNELS,
        OUT_CHANNELS, OUT_DIM,
        STRIDE, 1, 1, PADDING, KERNEL_DIM,
        false, false, false, false, false,

        (elem_t*)input,
        (elem_t*)weights,
        NO_BIAS ? NULL : (acc_t*)bias,
        (elem_t*)output2,

        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0, 0, 0, 0,

        WS);
    uint64_t end_gemmini = read_cycles();
    printf("our conv took %llu cycles\n", end_gemmini - start_gemmini);

    /*
    bool success = vec_is_equal(&output[0][0][0][0], &output2[0][0][0][0], sizeof(output) / sizeof(elem_t));

    if (success == true) {
      printf("success!\n");
    } else {
      printf("fail!\n");
    }
    */

    return 0;
}
