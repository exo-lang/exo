#include "avx2_matmul.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>


/* relying on the following instruction..."
mm256_broadcast_ss(out,val)
{out_data} = _mm256_broadcast_ss(&{val_data});
*/

/* relying on the following instruction..."
mm256_fmadd_ps(dst,src1,src2)
{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});
*/

/* relying on the following instruction..."
mm256_loadu_ps(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
mm256_storeu_ps(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
// rank_k_reduce_6x16(
//     K : size,
//     A : f32[6, K] @DRAM,
//     B : f32[K, 16] @DRAM,
//     C : f32[6, 16] @DRAM
// )
void rank_k_reduce_6x16( void *ctxt, int_fast32_t K, const float* A, const float* B, float* C ) {
for (int_fast32_t i = 0; i < 6; i++) {
  for (int_fast32_t j = 0; j < 16; j++) {
    for (int_fast32_t k = 0; k < K; k++) {
      C[i * 16 + j] += A[i * K + k] * B[k * 16 + j];
    }
  }
}
}

// rank_k_reduce_6x16_scheduled(
//     K : size,
//     A : f32[6, K] @DRAM,
//     B : f32[K, 16] @DRAM,
//     C : f32[6, 16] @DRAM
// )
void rank_k_reduce_6x16_scheduled( void *ctxt, int_fast32_t K, const float* A, const float* B, float* C ) {
__m256 C_reg[6][2];
for (int_fast32_t i0 = 0; i0 < 6; i0++) {
  for (int_fast32_t i2 = 0; i2 < 2; i2++) {
    C_reg[i0][i2] = _mm256_loadu_ps(&C[(i0) * (16) + 8 * i2]);
  }
}
for (int_fast32_t k = 0; k < K; k++) {
  __m256 B_reg[2];
  for (int_fast32_t io = 0; io < 2; io++) {
    B_reg[io] = _mm256_loadu_ps(&B[(k) * (16) + 8 * io]);
  }
  for (int_fast32_t i = 0; i < 6; i++) {
    __m256 A_reg;
    A_reg = _mm256_broadcast_ss(&A[(i) * K + k]);
    for (int_fast32_t jo = 0; jo < 2; jo++) {
      C_reg[i][jo] = _mm256_fmadd_ps(A_reg, B_reg[jo], C_reg[i][jo]);
    }
  }
}
for (int_fast32_t i0 = 0; i0 < 6; i0++) {
  for (int_fast32_t i2 = 0; i2 < 2; i2++) {
    _mm256_storeu_ps(&C[(i0) * (16) + 8 * i2], C_reg[i0][i2]);
  }
}
}

