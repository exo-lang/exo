// See LICENSE for license details.

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include "include/gemmini_testutils.h"

void print_matrix(size_t rows, size_t cols, elem_t mat[rows][cols]) {
    for (size_t r = 0; r < rows; r++) {
        for (size_t c = 0; c < cols; c++)
#ifndef ELEM_T_IS_FLOAT
            printf("%d ", mat[r][c]);
#else
            printf("%x ", elem_t_to_elem_t_bits(mat[r][c]));
#endif
        printf("\n");
    }
}

int main() {
#ifndef BAREMETAL
    if (mlockall(MCL_CURRENT | MCL_FUTURE) != 0) {
      perror("mlockall failed");
      exit(1);
    }
#endif

    // Test padded mvins
    {
        const size_t rows = rand() % (DIM-1) + 1;
        const size_t cols = rand() % (DIM-1) + 1;
        elem_t input[rows][cols];
        elem_t output[DIM][DIM];

        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
#ifndef ELEM_T_IS_FLOAT
                input[r][c] = rand() % elem_t_max;
#else
                input[r][c] = rand_double();
#endif

        const size_t sp_addr = 0;

        gemmini_config_ld(cols * sizeof(elem_t));
        gemmini_config_st(DIM * sizeof(elem_t));

        gemmini_extended_mvin(input, sp_addr, cols, rows);
        gemmini_mvout(output, sp_addr);
        gemmini_fence();

        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                if (input[r][c] != output[r][c]) {
                    printf("Matrices don't match!\n");

                    printf("input:\n");
                    print_matrix(rows, cols, input);

                    printf("output:\n");
                    printMatrix(output);

                    exit(1);
                }
    }

    // Test padded mvins and padded mvouts
    {
        const size_t rows = rand() % (DIM-1) + 1;
        const size_t cols = rand() % (DIM-1) + 1;
        elem_t input[rows][cols];
        elem_t output[rows][cols];

        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
#ifndef ELEM_T_IS_FLOAT
                input[r][c] = rand() % elem_t_max;
#else
                input[r][c] = rand_double();
#endif

        const size_t sp_addr = 0;

        gemmini_config_ld(cols * sizeof(elem_t));
        gemmini_config_st(cols * sizeof(elem_t));

        gemmini_extended_mvin(input, sp_addr, cols, rows);
        gemmini_extended_mvout(output, sp_addr, cols, rows);
        gemmini_fence();

        for (size_t r = 0; r < rows; r++)
            for (size_t c = 0; c < cols; c++)
                if (input[r][c] != output[r][c]) {
                    printf("Matrices don't match!\n");

                    printf("input:\n");
                    print_matrix(rows, cols, input);

                    printf("output:\n");
                    print_matrix(rows, cols, output);

                    exit(1);
                }
    }

    // Test padded matmuls
    for (int dataflow = 0; dataflow <= 1; dataflow++) {
        const size_t I = rand() % (DIM-1) + 1;
        const size_t J = rand() % (DIM-1) + 1;
        const size_t K = rand() % (DIM-1) + 1;
        elem_t A[I][K];
        elem_t B[K][J];
        elem_t D[I][J];
        elem_t C[I][J];
        elem_t gold[I][J];

        for (size_t i = 0; i < I; i++)
            for (size_t k = 0; k < K; k++)
#ifndef ELEM_T_IS_FLOAT
                A[i][k] = rand() % elem_t_max;
#else
                A[i][k] = rand_double();
#endif

        for (size_t k = 0; k < K; k++)
            for (size_t j = 0; j < J; j++)
#ifndef ELEM_T_IS_FLOAT
                B[k][j] = rand() % elem_t_max;
#else
                B[k][j] = rand_double();
#endif

        for (size_t i = 0; i < I; i++)
            for (size_t j = 0; j < J; j++)
#ifndef ELEM_T_IS_FLOAT
                D[i][j] = rand() % elem_t_max;
#else
                D[i][j] = rand_double();
#endif

        for (size_t i = 0; i < I; i++)
            for (size_t j = 0; j < J; j++) {
                acc_t result = D[i][j];
                for (size_t k = 0; k < K; k++)
                    result += A[i][k] * B[k][j];

                gold[i][j] = result < elem_t_min ? elem_t_min : (result > elem_t_max ? elem_t_max : result);
            }

        const size_t A_sp_addr = 0;
        const size_t B_sp_addr = DIM;
        const size_t D_sp_addr = 2*DIM;
        const size_t C_sp_addr = 3*DIM;

        gemmini_config_ex(dataflow, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 0);
        gemmini_config_st(J * sizeof(elem_t));

        gemmini_config_ld(K * sizeof(elem_t));
        gemmini_extended_mvin(A, A_sp_addr, K, I);

        gemmini_config_ld(J * sizeof(elem_t));
        gemmini_extended_mvin(B, B_sp_addr, J, K);

        gemmini_config_ld(J * sizeof(elem_t));
        gemmini_extended_mvin(D, D_sp_addr, J, I);

        if (dataflow == OUTPUT_STATIONARY) {
            gemmini_extended_preload(D_sp_addr, C_sp_addr, J, I, J, I);
            gemmini_extended_compute_preloaded(A_sp_addr, B_sp_addr, K, I, J, K);
        } else {
            gemmini_extended_preload(B_sp_addr, C_sp_addr, J, K, J, I);
            gemmini_extended_compute_preloaded(A_sp_addr, D_sp_addr, K, I, J, I);
        }

        gemmini_extended_mvout(C, C_sp_addr, J, I);

        gemmini_fence();

        for (size_t r = 0; r < I; r++)
            for (size_t c = 0; c < J; c++)
                if (C[r][c] != gold[r][c]) {
                    printf("Matrices don't match! (dataflow == %d)\n", dataflow);

                    printf("C:\n");
                    print_matrix(I, J, C);

                    printf("gold:\n");
                    print_matrix(I, J, gold);

                    exit(1);
                }
    }

    exit(0);
}

