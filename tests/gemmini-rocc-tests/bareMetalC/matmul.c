// See LICENSE for license details.
// The main point of this test is just to check whether we can switch between output- and weight-stationary dataflows

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <stdlib.h>
#include <stdio.h>
#ifndef BAREMETAL
#include <sys/mman.h>
#endif
#include <time.h>
#include "include/gemmini_testutils.h"

static elem_t ZERO[DIM][DIM];

#ifdef FAST
#define AINIT 2
#define SINIT 4
#define RAND (rand())
#define N 1
#else
#define AINIT 0
#define SINIT 0
#define RAND (rand())
#define N 2
#endif


void operands(int c, int * a, int * b, int * d) {
  *d = c % N;
  *b = (c / N) % N;
  *a = c / (N*N);
}



void test_os (bool A_transpose, bool B_transpose) {
  // Output stationary
  printf("Output-stationary\n");

  void (*matmul_ptr)(elem_t[DIM][DIM], elem_t[DIM][DIM], elem_t[DIM][DIM], full_t[DIM][DIM]);
  void (*matmul_full_ptr)(elem_t[DIM][DIM], elem_t[DIM][DIM], full_t[DIM][DIM], full_t[DIM][DIM]);

  if (!A_transpose && !B_transpose) {
    matmul_ptr = &matmul;
    matmul_full_ptr = &matmul_full;
  } else if (A_transpose && !B_transpose) {
    matmul_ptr = &matmul_A_transposed;
    matmul_full_ptr = &matmul_full_A_transposed;
  } else if (!A_transpose && B_transpose) {
    // Just return immediately because we don't support this
    return;
  } else if (A_transpose && B_transpose) {
    matmul_ptr = &matmul_AB_transposed;
    matmul_full_ptr = &matmul_full_AB_transposed;
  }

  for (int activation = AINIT; activation <= 2; ++activation) {
    for (int shift = SINIT; shift <= 4; shift += 4) {
      // printf("activation: %d, shift: %d\n", activation, shift);

      static elem_t A[N][DIM][DIM] row_align(1);
      static elem_t B[N][DIM][DIM] row_align(1);
      static elem_t D[N][DIM][DIM] row_align(1);

      // We will try out every combination of A, B, D possible
      static elem_t C[N*N*N][DIM][DIM] row_align(1);
      static full_t gold_full[N*N*N][DIM][DIM];
      static elem_t gold[N*N*N][DIM][DIM];

      int relu6_shift = shift+1;

      // ...taking into account the preloads or accumulates
      static int preload[N*N*N] = {1};
      for (int i = 1; i < N*N*N; ++i)
        preload[i] = 1; // rand() % 2;

      // ...and for the actual preloads, do we just preload zeros?
      static int preload_zeros[N*N*N];
      for (int i = 0; i < N*N*N; ++i)
        preload_zeros[i] = 1; // rand() % 2;

      // ...and finally, which results won't produce any output
      static int no_output[N*N*N];
      for (int i = 0; i < N*N*N-1; ++i)
        no_output[i] = !preload[i+1];
      no_output[N*N*N-1] = 0;

      // Print the sequence out
      /*printf("Preloads: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", preload[i]);
      printf("\n");
      printf("Zeros: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", preload_zeros[i]);
      printf("\n");
      printf("No outputs: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", no_output[i]);
      printf("\n");*/

      for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            A[n][i][j] = (RAND % 64) - 32;
            B[n][i][j] = (RAND % 64) - 32;
            D[n][i][j] = (RAND % 64) - 32;
          }
        }
      }
 #ifdef FAST1
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            gold[0][i][j] = 1;
          }
        }
#else
      for (size_t g = 0; g < N*N*N; ++g) {
        int a, b, d; 
        operands(g, &a, &b, &d);

        if (!preload[g])
          (*matmul_full_ptr)(A[a], B[b], gold_full[g-1], gold_full[g]);
        else if (preload_zeros[g]) {
          (*matmul_ptr)(A[a], B[b], ZERO, gold_full[g]);
        } else {
          (*matmul_ptr)(A[a], B[b], D[d], gold_full[g]);
        }
      }

      for (size_t g = 0; g < N*N*N; ++g) {
        matshift(gold_full[g], gold[g], shift);
        if (activation == RELU)
          matrelu(gold[g], gold[g]);
        else if (activation == RELU6)
          matrelu6(gold[g], gold[g], 1 << relu6_shift);
      }
#endif
      int A_addr = 0;
      int B_addr = N*DIM;
      int D_addr = 2*N*DIM;
      int C_addr = 3*N*DIM;

      // printf("Moving in\n");
      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(A[n], A_addr + n*DIM);
      
      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(B[n], B_addr + n*DIM);

      for (size_t n = 0; n < N; ++n) {
        gemmini_mvin(D[n], D_addr + n*DIM);
      }

      // printf("Setting mode\n");
      gemmini_extended_config_ex(OUTPUT_STATIONARY, activation, shift, relu6_shift, 1, A_transpose, B_transpose);

      // printf("Matmulling\n");
      for (size_t c = 0; c < N*N*N; ++c) {
        int a, b, d;
        operands(c, &a, &b, &d);
        
        uint64_t out_addr = C_addr + c*DIM;
        if (no_output[c])
          out_addr = GARBAGE_ADDR;

        if (!preload[c]) {
          gemmini_preload_zeros(out_addr);
          gemmini_compute_accumulated(A_addr + a*DIM, B_addr + b*DIM);
        } else if (preload_zeros[c]) {
          gemmini_preload_zeros(out_addr);
          gemmini_compute_preloaded(A_addr + a*DIM, B_addr + b*DIM);
        } else {
          gemmini_preload(D_addr + d*DIM, out_addr);
          gemmini_compute_preloaded(A_addr + a*DIM, B_addr + b*DIM);
        }
      }

      // printf("Moving out\n");
      for (size_t c = 0; c < N*N*N; ++c)
        if (!no_output[c]) {
          gemmini_mvout(C[c], C_addr + c*DIM);
        }

      gemmini_fence();

      /*
      printf("Moved out\n");

      printf("A[0]:\n");
      printMatrix(A[0]);
      printf("B[0]:\n");
      printMatrix(B[0]);

      for (int n = 0; n < N*N*N; ++n) {
        if (!no_output[n]) {
          printf("C:\n");
          printMatrix(C[n]);
          printf("Gold:\n");
          printMatrix(gold[n]);
          printf("\n");
        }
      }
      */

      for (int n = 0; n < N*N*N; ++n)
        if (!no_output[n] && !is_equal(C[n], gold[n])) {
          printf("activation: %d, shift: %d, n: %d, A_transpose: %d, B_transpose: %d\n", activation, shift, n, A_transpose, B_transpose);

          printf("A:\n");
          printMatrix(A[n]);
          printf("B:\n");
          printMatrix(B[n]);
          printf("C:\n");
          printMatrix(C[n]);
          printf("Gold:\n");
          printMatrix(gold[n]);
          printf("\n");

          exit(1);
        }
    }
  }
}

void test_ws(bool A_transpose, bool B_transpose) {
  // Weight-stationary
  printf("Weight-stationary\n");

  void (*matmul_ptr)(elem_t[DIM][DIM], elem_t[DIM][DIM], elem_t[DIM][DIM], full_t[DIM][DIM]);

  if (!A_transpose && !B_transpose) {
    matmul_ptr = &matmul;
  } else if (A_transpose && !B_transpose) {
    matmul_ptr = &matmul_A_transposed;
  } else if (!A_transpose && B_transpose) {
    matmul_ptr = &matmul_B_transposed;
  } else if (A_transpose && B_transpose) {
    return;
  }

  for (int activation = AINIT; activation <= 2; ++activation) {
    for (int scale = SINIT; scale <= 4; scale += 4) {
      static elem_t A[N][DIM][DIM] row_align(1);
      static elem_t B[N][DIM][DIM] row_align(1);
      static elem_t D[N][DIM][DIM] row_align(1);

      // We will try out every combination of A, B, D possible
      static elem_t C[N*N*N][DIM][DIM] row_align(1);
      static full_t gold_full[N*N*N][DIM][DIM];
      static elem_t gold[N*N*N][DIM][DIM];

      int relu6_shift = scale+1;

      // ...taking into account whether we preload new weights or re-use the old ones
      static int preload[N*N*N] = {1};
      for (int i = 1; i < N*N*N; ++i)
        preload[i] = RAND % 2;

      // ...whether we pass in a D or just use zeros
      static int add_to_zeros[N*N*N];
      for (int i = 0; i < N*N*N; ++i)
        add_to_zeros[i] = RAND % 2;

      // ...and whether we accumulate on top of the previous result
      static int accumulate[N*N*N] = {0};
      for (int i = 1; i < N*N*N; ++i)
        accumulate[i] = RAND % 2;

      static int no_output[N*N*N];
      for (int i = 0; i < N*N*N-1; ++i)
        no_output[i] = accumulate[i+1];
      no_output[N*N*N-1] = 0;

      // Print the sequence out
      /*printf("Preloads: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", preload[i]);
      printf("\n");
      printf("Zeros: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", add_to_zeros[i]);
      printf("\n");
      printf("Accumulates: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", accumulate[i]);
      printf("\n");
      printf("No outputs: ");
      for (int i = 0; i < N*N*N; ++i)
        printf("%d, ", no_output[i]);
      printf("\n");*/

      for (size_t n = 0; n < N; ++n) {
        for (size_t i = 0; i < DIM; ++i) {
          for (size_t j = 0; j < DIM; ++j) {
            A[n][i][j] = (RAND % 64) - 32;
            B[n][i][j] = (RAND % 64) - 32;
            D[n][i][j] = (RAND % 64) - 32;
          }
        }
      }
#ifdef FAST1
      for (size_t i = 0; i < DIM; ++i) {
	for (size_t j = 0; j < DIM; ++j) {
	  gold[0][i][j] = 64;
        }
      }
#else
      for (size_t g = 0; g < N*N*N; ++g) {
        int a, b, d;
        operands(g, &a, &b, &d);

        // We need to find the last B value in case we aren't preloading new weights
        for (int last_g = g; last_g >= 0; --last_g) {
            int tmp_a, tmp_d;
            if (preload[last_g]) {
                operands(last_g, &tmp_a, &b, &tmp_d);
                break;
            }
        }

        if (add_to_zeros[g]) {
          (*matmul_ptr)(A[a], B[b], ZERO, gold_full[g]);
        } else {
          (*matmul_ptr)(A[a], B[b], D[d], gold_full[g]);
        }

        if (accumulate[g])
          matadd(gold_full[g], gold_full[g-1], gold_full[g]);
      }

      for (size_t g = 0; g < N*N*N; ++g) {
        matscale(gold_full[g], gold[g], scale);
        if (activation == RELU)
          matrelu(gold[g], gold[g]);
        else if (activation == RELU6)
          matrelu6(gold[g], gold[g], 1 << relu6_shift);
      }
#endif
      uint32_t A_addr = 0;
      uint32_t B_addr = N*DIM;
      uint32_t D_addr = 2*N*DIM;
      uint32_t C_addr_acc = 1 << (ADDR_LEN-1);

      // Calculate the proper destination addresses of everything
      uint32_t C_addrs[N*N*N];
      for (size_t c = 0; c < N*N*N; ++c)
        C_addrs[c] = C_addr_acc + c*DIM;
      for (size_t c = 0; c < N*N*N; ++c) {
        int last_c;
        for (last_c = c; last_c >= 0; --last_c)
          if (!accumulate[last_c])
            break;
        if (c != last_c)
          C_addrs[c] = C_addrs[last_c] | (1 << (ADDR_LEN-2));
      }

      // printf("Moving in\n");
      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(A[n], A_addr + n*DIM);

      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(B[n], B_addr + n*DIM);

      for (size_t n = 0; n < N; ++n)
        gemmini_mvin(D[n], D_addr + n*DIM);

      // printf("Setting mode\n");
      gemmini_extended_config_ex(WEIGHT_STATIONARY, 0, 0, relu6_shift, 1, A_transpose, B_transpose);
      gemmini_extended_config_st(DIM * sizeof(elem_t), activation, scale);

      // printf("Matmulling\n");
      for (size_t c = 0; c < N*N*N; ++c) {
        int a, b, d;
        operands(c, &a, &b, &d);

        uint64_t d_addr = D_addr + d*DIM;
        if (add_to_zeros[c])
          d_addr = GARBAGE_ADDR;

        if (!preload[c]) {
          gemmini_preload_zeros(C_addrs[c]);
          gemmini_compute_accumulated(A_addr + a*DIM, d_addr);
        } else {
          gemmini_preload(B_addr + b*DIM, C_addrs[c]);
          gemmini_compute_preloaded(A_addr + a*DIM, d_addr);
        }
      }

      // printf("Moving out\n");
      for (size_t c = 0; c < N*N*N; ++c)
        if (!no_output[c]) {
          gemmini_mvout(C[c], C_addrs[c] & ~(1 << (ADDR_LEN-2)));
        }

      gemmini_fence();

      /*printf("Moved out\n");
      for (int n = 0; n < N*N*N; ++n) {
        if (!no_output[n]) {
          printf("C:\n");
          printMatrix(C[n]);
          printf("Gold:\n");
          printMatrix(gold[n]);
          printf("\n");
        }
      }*/

      for (int n = 0; n < N*N*N; ++n)
        if (!no_output[n] && !is_equal(C[n], gold[n])) {
          printf("activation: %d, scale: %d, n: %d, A_transpose: %d, B_transpose: %d\n", activation, scale, n, A_transpose, B_transpose);

          printf("A:\n");
          printMatrix(A[n]);
          printf("B:\n");
          printMatrix(B[n]);
          printf("C:\n");
          printMatrix(C[n]);
          printf("Gold:\n");
          printMatrix(gold[n]);
          printf("\n");

          exit(1);
        }
    }
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
  gemmini_config_ld(DIM * sizeof(elem_t));
  gemmini_config_st(DIM * sizeof(elem_t));

  for (int A_transpose = 0; A_transpose < 2; A_transpose++) {
    for (int B_transpose = 0; B_transpose < 2; B_transpose++) {
      for (int dataflow = OUTPUT_STATIONARY; dataflow <= WEIGHT_STATIONARY; dataflow++) {
        printf("A_transpose: %d, B_transpose: %d, dataflow: %d\n", A_transpose, B_transpose, dataflow);

        if (dataflow == OUTPUT_STATIONARY)
          test_os(A_transpose, B_transpose);
        else
          test_ws(A_transpose, B_transpose);
      }
    }
  }

  exit(0);
}

