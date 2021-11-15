from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.platforms.x86 import *
from SYS_ATL.syntax import *


def trace(message):
    @instr(f'puts("{message}");')
    def trace_impl():
        pass

    return trace_impl.rename(f'trace_{message}')


# noinspection PyPep8Naming
@proc
def SGEMM(
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

SGEMM_WINDOW = (
    SGEMM
        .rename('SGEMM_WINDOW')
        .set_window('A', True)
        .set_window('B', True)
        .set_window('C', True)
        # Make reduction dimension outermost
        .reorder('j', 'k')
        .reorder('i', 'k')
)

basic_kernel = (
    SGEMM_WINDOW
        .rename('basic_kernel')
        # Specialize to kernel size
        .partial_eval(I_REG_BLK, J_REG_BLK)
)

sgemm_kernel_avx512_6x4 = (
    basic_kernel
        .rename('sgemm_kernel_avx512_6x4')
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
)

bottom_panel_kernel = (
    SGEMM_WINDOW
        .rename('bottom_panel_kernel')
        .partial_eval(N=64)
        .simplify()
)

sgemm_sys_atl = (
    SGEMM
        .rename('sgemm_sys_atl')
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
        .fission_after('for jo in _: _ #1')
        # Right panel
        .reorder('ji', 'k')
        .reorder('ii', 'k')
        # Bottom panel
        .reorder('ii', 'jo')
        .reorder('ii', 'k')
        # Bottom-right tile
        .replace_all(SGEMM_WINDOW)
        .simplify()
)

if __name__ == '__main__':
    print(sgemm_sys_atl)
    print(bottom_panel_kernel)

__all__ = ['sgemm_sys_atl']
