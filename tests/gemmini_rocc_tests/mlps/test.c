#include <stdio.h>
#include <string.h>
#include <stdbool.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "parameters.h"

int main (int argc, char * argv[]) {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    gemmini_flush(0);

    enum tiled_matmul_type_t tiled_matmul_type;
    if (argc < 2) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "cpu") == 0) {
        tiled_matmul_type = CPU;
    } else if (strcmp(argv[1], "os") == 0) {
        tiled_matmul_type = OS;
    } else if (strcmp(argv[1], "ws") == 0) {
        tiled_matmul_type = WS;
    } else if (strcmp(argv[1], "-h") == 0) {
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(0);
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    bool check;
    if (argc < 3) {
        check = false;
    } else if (strcmp(argv[2], "check") == 0) {
        check = true;
    } else {
        printf("Unknown command-line argument\n");
        printf("usage: %s [-h] matmul_option [check]\n  matmul_option may be 'os', 'ws', or cpu'\n", argv[0]);
        exit(1);
    }

    uint64_t cycles[4]={0};
    uint64_t start, end;

    /* matmul number: 0 */
    start = read_cycles();

    tiled_matmul_nn_auto(16, 112, 144,
        input_mat, weights0, NULL, inter_results0,
        RELU, 0, 0, false,
        tiled_matmul_type, check, "layer_0");

    end = read_cycles();
    cycles[0] = end-start;

    /* matmul number: 1 */
    start = read_cycles();

    tiled_matmul_nn_auto(16, 144, 32,
        inter_results0, weights1, NULL, inter_results1,
        RELU, 0, 0, false,
        tiled_matmul_type, check, "layer_1");

    end = read_cycles();
    cycles[1] = end-start;

    /* matmul number: 2 */
    start = read_cycles();

    tiled_matmul_nn_auto(16, 32, 64,
        inter_results1, weights2, NULL, inter_results2,
        RELU, 0, 0, false,
        tiled_matmul_type, check, "layer_2");

    end = read_cycles();
    cycles[2] = end-start;

    /* matmul number: 3 */
    start = read_cycles();

    tiled_matmul_nn_auto(16, 64, 16,
        inter_results2, weights3, NULL, inter_results3,
        RELU, 0, 0, false,
        tiled_matmul_type, check, "layer_3");

    end = read_cycles();
    cycles[3] = end-start;

    uint64_t overall_cycles = 0;
    for(int cyc = 0; cyc < 4 ; cyc++){
        overall_cycles += cycles[cyc];
        printf("Cycles taken in layer %d: %llu\n", cyc,cycles[cyc]);
    }
    printf("Overall cycles taken: %llu\n",overall_cycles);

    return 0;
}

