from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.platforms.x86 import *
from SYS_ATL.syntax import *


# noinspection PyPep8Naming
@proc
def sgemm_masked_kernel_avx512_template(
        M: size,
        N: size,
        K: size,
        A: f32[M, K],
        B: f32[K, 16 * ((N + 15) / 16)],
        C: [f32][M, N],
):
    assert M >= 1
    assert N >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    if K < 1:
        unreachable()

    C_reg: f32[M, ((N + 15) / 16), 16] @ AVX512
    for i in par(0, M):
        for j in par(0, N / 16):
            mm512_loadu_ps(C_reg[i, j, :], C[i, 16 * j:16 * j + 16])
        if N % 16 > 0:
            mm512_maskz_loadu_ps(
                N % 16,
                C_reg[i, N / 16, :],
                C[i, 16 * (N / 16):16 * (N / 16) + N % 16]
            )

    for k in par(0, K):
        for i in par(0, M):
            a_vec: f32[16] @ AVX512
            mm512_set1_ps(a_vec, A[i, k:k + 1])
            for j in par(0, ((N + 15) / 16)):
                b_vec: f32[16] @ AVX512
                mm512_loadu_ps_reg(b_vec, B[k, j * 16:j * 16 + 16])
                mm512_fmadd_ps(a_vec, b_vec, C_reg[i, j, :])

    for i in par(0, M):
        for j in par(0, N / 16):
            mm512_storeu_ps(C[i, 16 * j:16 * j + 16], C_reg[i, j, :])
        if N % 16 > 0:
            mm512_mask_storeu_ps(
                N % 16,
                C[i, 16 * (N / 16):16 * (N / 16) + N % 16],
                C_reg[i, N / 16, :]
            )


sgemm_kernel_avx512 = dict()
for i in range(1, 6 + 1):
    for j in range(1, 4 + 1):
        sgemm_kernel_avx512[(i, j)] = (
            sgemm_masked_kernel_avx512_template
                .partial_eval(i, 16 * j)
                .simplify()
                .unroll('j')
                .unroll('i')
                .simplify()
                .rename(f'sgemm_kernel_avx512_{i}x{j}')
        )

sgemm_kernel_avx512_6x4 = sgemm_kernel_avx512[(6, 4)]


# noinspection PyPep8Naming
@proc
def sgemm_sys_atl(
        M: size,
        N: size,
        K: size,
        A: f32[M, K],
        B: f32[K, N],
        C: f32[M, N],
):
    assert M >= 1
    assert N >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for k in par(0, K):
        for i in par(0, M):
            for j in par(0, N):
                C[i, j] += A[i, k] * B[k, j]


__all__ = ['sgemm_kernel_avx512_6x4', 'sgemm_sys_atl']
