#include "quiz1.h"

#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>

// vec_double(
//     N : size,
//     inp : f32[N] @DRAM,
//     out : f32[N] @DRAM
// )
void vec_double( void *ctxt, int_fast32_t N, const float* inp, float* out ) {
EXO_ASSUME(N % 8 == 0);
for (int_fast32_t i = 0; i < N; i++) {
  out[i] = 2.0f * inp[i];
}
}

// vec_double_optimized(
//     N : size,
//     inp : f32[N] @DRAM,
//     out : f32[N] @DRAM
// )
void vec_double_optimized( void *ctxt, int_fast32_t N, const float* inp, float* out ) {
EXO_ASSUME(N % 8 == 0);
__m256 two_vec;
two_vec = _mm256_broadcast_ss(2.0);
for (int_fast32_t io = 0; io < ((N) / (8)); io++) {
  __m256 out_vec;
  __m256 inp_vec;
  inp_vec = _mm256_loadu_ps(&inp[8 * io]);
  out_vec = _mm256_mul_ps(two_vec, inp_vec);
  _mm256_storeu_ps(&out[8 * io], out_vec);
}
}


/* relying on the following instruction..."
vector_assign_two(out)
{out_data} = _mm256_broadcast_ss(2.0);
*/

/* relying on the following instruction..."
vector_load(dst,src)
{dst_data} = _mm256_loadu_ps(&{src_data});
*/

/* relying on the following instruction..."
vector_multiply(out,x,y)
{out_data} = _mm256_mul_ps({x_data}, {y_data});
*/

/* relying on the following instruction..."
vector_store(dst,src)
_mm256_storeu_ps(&{dst_data}, {src_data});
*/
