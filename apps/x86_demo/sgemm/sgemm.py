from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.platforms.x86 import *
from SYS_ATL.syntax import *


#
# def new_config_x86():
#     @config
#     class ConfigLoad:
#         masked_N: size
#
#     return ConfigLoad
#
#
# Config = new_config_x86()

#
# # TODO: We need a better way of stack-allocating masks / config
# @instr('__mmask16 dumb_mask = (1 << {N}) - 1;')
# def mk_mask(N: size):
#     Config.masked_N = N
#
#
# @instr('{dst_data} = _mm512_maskz_loadu_ps(dumb_mask, &{src_data});')
# def dumb_mm512_maskz_loadu_ps(
#         masked_N: size,
#         dst: [f32][16] @ AVX512,
#         src: [f32][masked_N] @ DRAM,
# ):
#     assert stride(src, 0) == 1
#     assert stride(dst, 0) == 1
#     assert masked_N > 0
#     assert masked_N <= 16
#
#     for i in par(0, 16):
#         if i < masked_N:
#             dst[i] = src[i]
#         else:
#             dst[i] = 0.0
#
#
# @instr('_mm512_mask_storeu_ps(&{dst_data}, dumb_mask, {src_data});')
# def dumb_mm512_mask_storeu_ps(
#         masked_N: size,
#         dst: [f32][masked_N] @ DRAM,
#         src: [f32][16] @ AVX512
# ):
#     assert stride(src, 0) == 1
#     assert stride(dst, 0) == 1
#     assert masked_N > 0
#     assert masked_N <= 16
#
#     for i in par(0, 16):
#         if i < masked_N:
#             dst[i] = src[i]

#
# # noinspection PyPep8Naming
# @proc
# def sgemm_masked_kernel_avx512_template(
#         M: size,
#         N: size,
#         K: size,
#         A: f32[M, K],
#         B: f32[K, 16 * ((N + 15) / 16)],
#         C: [f32][M, N],
# ):
#     assert M >= 1
#     assert N >= 1
#     assert K >= 1
#     assert stride(A, 1) == 1
#     assert stride(B, 1) == 1
#     assert stride(C, 1) == 1
#
#     mk_mask(N % 16)
#
#     C_reg: f32[M, ((N + 15) / 16), 16] @ AVX512
#     for i in par(0, M):
#         for j in par(0, N / 16):
#             mm512_loadu_ps(C_reg[i, j, :], C[i, 16 * j:16 * j + 16])
#         if N % 16 > 0:
#             dumb_mm512_maskz_loadu_ps(
#                 Config.masked_N,
#                 C_reg[i, N / 16, :],
#                 C[i, 16 * (N / 16):16 * (N / 16) + N % 16]
#             )
#
#     for k in par(0, K):
#         for i in par(0, M):
#             a_vec: f32[16] @ AVX512
#             mm512_set1_ps(a_vec, A[i, k:k + 1])
#             for j in par(0, ((N + 15) / 16)):
#                 b_vec: f32[16] @ AVX512
#                 mm512_loadu_ps(b_vec, B[k, j * 16:j * 16 + 16])
#                 mm512_fmadd_ps(a_vec, b_vec, C_reg[i, j, :])
#
#     for i in par(0, M):
#         for j in par(0, N / 16):
#             mm512_storeu_ps(C[i, 16 * j:16 * j + 16], C_reg[i, j, :])
#         if N % 16 > 0:
#             dumb_mm512_mask_storeu_ps(
#                 Config.masked_N,
#                 C[i, 16 * (N / 16):16 * (N / 16) + N % 16],
#                 C_reg[i, N / 16, :]
#             )
#
#
# # The primary kernel with no masked loads
# sgemm_kernel_avx512_6x4 = (
#     sgemm_masked_kernel_avx512_template
#         .partial_eval(6, 64)
#         .simplify()
#         .unroll('j')
#         .unroll('i')
#         .simplify()
#         .rename(f'sgemm_kernel_avx512_6x4')
# )


# TODO: the 6*4 tail kernels with masked loads on right edge

def trace(message):
    @instr(f'puts("{message}");')
    def trace_impl():
        pass

    return trace_impl.rename(f'trace_{message}')


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

    for i in par(0, M):
        for j in par(0, N):
            for k in par(0, K):
                C[i, j] += A[i, k] * B[k, j]


# Constants for scheduling
VEC_W = 16

I_REG_BLK = 6
J_REG_BLK = (4 * VEC_W)

I_L2_FAC = 44
J_L2_FAC = 1

I_L2_BLK = I_REG_BLK * I_L2_FAC
J_L2_BLK = J_REG_BLK * J_L2_FAC
K_L2_BLK = 512

COPY_STREAMS = 3

basic_kernel = (
    sgemm_sys_atl
        .set_window('A', True)
        .set_window('B', True)
        .set_window('C', True)
        # Specialize to kernel size
        .partial_eval(I_REG_BLK, J_REG_BLK)
        # Make reduction dimension outermost
        .reorder('j', 'k')
        .reorder('i', 'k')
        .rename('basic_kernel')
)

sgemm_kernel_avx512_6x4 = (
    basic_kernel
        # Vectorize columns
        .split('j', VEC_W, ['jo', 'ji'], perfect=True)
        .par_to_seq('for k in _: _')
        # Stage C for reduction
        .stage_assn('C_reg', 'C[_] += _')
        .set_memory('C_reg', AVX512)
        .lift_alloc('C_reg: _', n_lifts=4)
        .double_fission('C_reg[_] = C[_]', 'C_reg[_] += _', n_lifts=4)
        # Stage A
        .bind_expr('A_vec', 'A[_, _]')
        .set_memory('A_vec', AVX512)
        .lift_alloc('A_vec: _', keep_dims=True)
        .fission_after('A_vec[_] = _')
        # Stage B
        .bind_expr('B_vec', 'B[_, _]')
        .set_memory('B_vec', AVX512)
        .lift_alloc('B_vec: _', keep_dims=True)
        .fission_after('B_vec[_] = _')
        # Schedule ops
        .replace(mm512_loadu_ps, 'for ji in _: _ #0')
        .replace(mm512_storeu_ps, 'for ji in _: _ #3')
        .replace_all(mm512_set1_ps)
        .replace_all(mm512_loadu_ps)
        .replace_all(mm512_fmadd_ps)
        # LICM
        .lift_alloc('A_vec: _')
        .fission_after('mm512_set1_ps(_)')
        # # Tracing
        # .insert_pass('C_reg: _')
        # .replace(trace("inner"), 'pass')
        # Clean up
        .simplify()
        .rename('sgemm_kernel_avx512_6x4')
)

sgemm_sys_atl = (
    sgemm_sys_atl
        .split('j', J_REG_BLK, ['jo', 'ji'], tail='cut')
        .split('i', I_REG_BLK, ['io', 'ii'], tail='cut')
        .fission_after('for jo in _: _ #0', n_lifts=2)
        .reorder('ii #0', 'jo')
        .reorder('ji #0', 'k')
        .reorder('ii #0', 'k')
        .replace_all(basic_kernel)
        #
        .call_eqv(sgemm_kernel_avx512_6x4, 'basic_kernel(_)')
        # #
        # .insert_pass('for io in _: _')
        # .replace(trace("outer"), 'pass')
        # #
        # .insert_pass('sgemm_kernel_avx512_6x4(_, _, _, _)')
        # .replace(trace("in_loop"), 'pass')
        .simplify()
)

if __name__ == '__main__':
    print(basic_kernel)
    print(sgemm_sys_atl)

__all__ = ['sgemm_kernel_avx512_6x4', 'sgemm_sys_atl']
