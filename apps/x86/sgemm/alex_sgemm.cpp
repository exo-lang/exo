#include "alex_sgemm.h"

#include <immintrin.h>

#include <algorithm>
#include <vector>

typedef __m512 vec_type;

#define I_REG_BLK 6
#define J_REG_BLK (4 * 16)

static_assert(J_REG_BLK % 16 == 0);

#define I_L2_FAC 44
#define J_L2_FAC 1

#define I_L2_BLK (I_REG_BLK * I_L2_FAC)
#define J_L2_BLK (J_REG_BLK * J_L2_FAC)
#define K_L2_BLK 512

static_assert(J_L2_BLK % 16 == 0);
static_assert(K_L2_BLK % 16 == 0);

static float a_l2[I_L2_BLK * K_L2_BLK] __attribute__((aligned(64)));
static float b_l2[K_L2_BLK * J_L2_BLK] __attribute__((aligned(64)));

static void copy_and_pad(float *__restrict b, long ldb,
    const float *__restrict a, long lda, long M, long N) {
  __mmask16 mask = (1 << (N % 16)) - 1;
  constexpr auto streams = 3;

  for (long io = 0; io < M; io += streams) {
    _mm_prefetch(&a[(io + streams) * lda], _MM_HINT_T0);
    _mm_prefetch(&b[(io + streams) * ldb], _MM_HINT_T0);
    for (long j = 0; j < N; j += 16) {
      if (io + streams < M) {
        for (long i = io; i < io + streams; i++) {
          _mm512_storeu_ps(&b[i * ldb + j],
              j + 16 > N ? _mm512_maskz_loadu_ps(mask, &a[i * lda + j])
                         : _mm512_loadu_ps(&a[i * lda + j]));
        }
      } else {
        for (long i = io; i < M; i++) {
          _mm512_storeu_ps(&b[i * ldb + j],
              j + 16 > N ? _mm512_maskz_loadu_ps(mask, &a[i * lda + j])
                         : _mm512_loadu_ps(&a[i * lda + j]));
        }
      }
    }
  }
}

template <int I_REG, int J_REG>
static void sgemm_padded_load(
    vec_type (&inner_c)[I_REG][J_REG], float *c, long ldc, short mask) {
  for (long i = 0; i < I_REG; i++) {
    long jo = 0;
    for (; jo < J_REG - 1; jo++) {
      inner_c[i][jo] = _mm512_loadu_ps(&c[16 * jo]);
    }
    inner_c[i][jo] = _mm512_maskz_loadu_ps(mask, &c[16 * jo]);
    c += ldc;
  }
}

template <int I_REG, int J_REG>
static void sgemm_padded_store(
    vec_type (&inner_c)[I_REG][J_REG], float *c, long ldc, short mask) {
  // Write back to C
  for (long i = 0; i < I_REG; i++) {
    long jo = 0;
    for (; jo < J_REG - 1; jo++) {
      _mm512_storeu_ps(&c[16 * jo], inner_c[i][jo]);
    }
    _mm512_mask_storeu_ps(&c[16 * jo], mask, inner_c[i][jo]);
    c += ldc;
  }
}

template <int I_REG, int J_REG>
static void sgemm_accumulate(
    const float *a, const float *b, vec_type (&inner_c)[I_REG][J_REG], long K) {
  // Accumulate into registers
  long k = -K;
  do {
#pragma GCC unroll 128
    for (long i = 0; i < I_REG; i++) {
#pragma GCC unroll 128
      for (long jo = 0; jo < J_REG; jo++) {
        vec_type _a = _mm512_set1_ps(a[i * K_L2_BLK + k + K]);
        vec_type _b = _mm512_loadu_ps(&b[(k + K) * J_L2_BLK + jo * 16]);
        inner_c[i][jo] = _mm512_fmadd_ps(_a, _b, inner_c[i][jo]);
      }
    }
  } while (++k < 0);
}

template <int I_REG, int J_REG>
void sgemm_micro_kernel_staged_M_N(const float *a, const float *b, float *c,
    long ldc, const long N, const long K) {
  vec_type inner_c[I_REG][J_REG] = {{0}};

  const auto Nb = N - 16 * (J_REG - 1);
  short mask = (1 << Nb) - 1;

  sgemm_padded_load(inner_c, c, ldc, mask);
  sgemm_accumulate(a, b, inner_c, K);
  sgemm_padded_store(inner_c, c, ldc, mask);
}

static void sgemm_micro_kernel(
    const float *a, const float *b, float *c, long ldc, long K) {
  sgemm_micro_kernel_staged_M_N<I_REG_BLK, J_REG_BLK / 16>(
      a, b, c, ldc, J_REG_BLK, K);
}

template <int M>
void sgemm_micro_kernel_staged_inner(const float *a, const float *b, float *c,
    long ldc, const long N, const long K) {
  switch ((N + 15) / 16) {
  case 1:
    sgemm_micro_kernel_staged_M_N<M, 1>(a, b, c, ldc, N, K);
    break;
  case 2:
    sgemm_micro_kernel_staged_M_N<M, 2>(a, b, c, ldc, N, K);
    break;
  case 3:
    sgemm_micro_kernel_staged_M_N<M, 3>(a, b, c, ldc, N, K);
    break;
  case 4:
    sgemm_micro_kernel_staged_M_N<M, 4>(a, b, c, ldc, N, K);
    break;
  default:
    __builtin_unreachable();
  }
}

static void sgemm_micro_kernel_staged(const float *a, const float *b, float *c,
    long ldc, const long M, const long N, const long K) {
  switch (M) {
  case 1:
    sgemm_micro_kernel_staged_inner<1>(a, b, c, ldc, N, K);
    break;
  case 2:
    sgemm_micro_kernel_staged_inner<2>(a, b, c, ldc, N, K);
    break;
  case 3:
    sgemm_micro_kernel_staged_inner<3>(a, b, c, ldc, N, K);
    break;
  case 4:
    sgemm_micro_kernel_staged_inner<4>(a, b, c, ldc, N, K);
    break;
  case 5:
    sgemm_micro_kernel_staged_inner<5>(a, b, c, ldc, N, K);
    break;
  case 6:
    sgemm_micro_kernel_staged_inner<6>(a, b, c, ldc, N, K);
    break;
  default:
    __builtin_unreachable();
  }
}

static void sgemm_l1_blocked(const float *a, const float *b, float *c, long ldc,
    const long M, const long N, const long K) {
  long i;
  long j = 0;
  // In the common case, stream directly from memory
  for (i = 0; i <= M - I_REG_BLK; i += I_REG_BLK) {
    for (j = 0; j <= N - J_REG_BLK; j += J_REG_BLK) {
      sgemm_micro_kernel(&a[i * K_L2_BLK], &b[j], &c[i * ldc + j], ldc, K);
    }
  }
  // Handle the right unaligned panel.
  if (j < N) {
    for (long ii = 0; ii <= M - I_REG_BLK; ii += I_REG_BLK) {
      sgemm_micro_kernel_staged(
          &a[ii * K_L2_BLK], &b[j], &c[ii * ldc + j], ldc, I_REG_BLK, N - j, K);
    }
  }
  // Handle the bottom unaligned panel.
  if (i < M) {
    for (long jj = 0; jj < N; jj += J_REG_BLK) {
      sgemm_micro_kernel_staged(&a[i * K_L2_BLK], &b[jj], &c[i * ldc + jj], ldc,
          M - i, std::min((long)J_REG_BLK, N - jj), K);
    }
  }
}

static void sgemm_l2_blocked(const float *__restrict a, const long lda,
    const float *__restrict b, const long ldb, float *__restrict c,
    const long ldc, const long M, const long N, const long K) {
  for (long k = 0; k < K; k += K_L2_BLK) {
    long Kb = std::min((long)K_L2_BLK, K - k);
    for (long i = 0; i < M; i += I_L2_BLK) {
      long Mb = std::min((long)I_L2_BLK, M - i);
      copy_and_pad(a_l2, K_L2_BLK, &a[i * lda + k], lda, Mb, Kb);
      for (long j = 0; j < N; j += J_L2_BLK) {
        long Nb = std::min((long)J_L2_BLK, N - j);
        copy_and_pad(b_l2, J_L2_BLK, &b[k * ldb + j], ldb, Kb, Nb);
        sgemm_l1_blocked(a_l2, b_l2, &c[i * ldc + j], ldc, Mb, Nb, Kb);
      }
    }
  }
}

void sgemm_square(const float *a, const float *b, float *c, long n) {
  sgemm_l2_blocked(a, n, b, n, c, n, n, n, n);
}
