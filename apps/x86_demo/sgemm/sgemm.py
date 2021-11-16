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

basic_kernel_Mx4 = {}
sgemm_kernel_avx512_Mx4 = {}
for M in range(1, I_REG_BLK + 1):
    basic_kernel_Mx4[M] = (
        SGEMM_WINDOW
            .rename(f'basic_kernel_{M}x4')
            .partial_eval(M, J_REG_BLK)
            .simplify()
    )
    sgemm_kernel_avx512_Mx4[M] = (
        basic_kernel_Mx4[M]
            .rename(f'sgemm_kernel_avx512_{M}x4')
            # Vectorize columns
            .split('j', VEC_W, ['jo', 'ji'], perfect=True)
            # Mark k as a reduction loop
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
            # Clean up
            .simplify()
    )

bottom_panel_kernel = (
    SGEMM_WINDOW
        .rename('bottom_panel_kernel')
        .partial_eval(N=J_REG_BLK)
        .simplify()
)

bottom_panel_kernel_scheduled = (
    bottom_panel_kernel
        .rename('bottom_panel_kernel_scheduled')
        # Specialize branches (simplify needed to unify with basic kernels)
        .add_ifelse('for k in _: _ #0', 'M == 1')
        .add_ifelse('for k in _: _ #1', 'M == 2')
        .add_ifelse('for k in _: _ #2', 'M == 3')
        .add_ifelse('for k in _: _ #3', 'M == 4')
        .add_ifelse('for k in _: _ #4', 'M == 5')
        .simplify()
        #
        .replace_all(basic_kernel_Mx4[1])
        .replace_all(basic_kernel_Mx4[2])
        .replace_all(basic_kernel_Mx4[3])
        .replace_all(basic_kernel_Mx4[4])
        .replace_all(basic_kernel_Mx4[5])
        #
        .call_eqv(sgemm_kernel_avx512_Mx4[1], 'basic_kernel_1x4(_)')
        .call_eqv(sgemm_kernel_avx512_Mx4[2], 'basic_kernel_2x4(_)')
        .call_eqv(sgemm_kernel_avx512_Mx4[3], 'basic_kernel_3x4(_)')
        .call_eqv(sgemm_kernel_avx512_Mx4[4], 'basic_kernel_4x4(_)')
        .call_eqv(sgemm_kernel_avx512_Mx4[5], 'basic_kernel_5x4(_)')
        #
        .simplify()
)

right_panel_kernel = (
    SGEMM_WINDOW
        .rename('right_panel_kernel')
        .partial_eval(M=I_REG_BLK)
        .simplify()
)

right_panel_kernel_opt = (
    right_panel_kernel
        .rename('right_panel_kernel_opt')
        #
        .stage_assn('C_reg', 'C[_] += _')
        .split('j', VEC_W, ['jo', 'ji'], tail='cut')
        .bound_and_guard('for ji in _: _ #1')
        .fission_after('for jo in _: _', n_lifts=2)
        #
        .par_to_seq('for k in _: _')
        #
        .lift_alloc('C_reg: _', n_lifts=4)
        .fission_after('C_reg[_] = _', n_lifts=4)
        .fission_after('C_reg[_] += _', n_lifts=4)
        #
        .reorder_before('C_reg: _ #1')
        .reorder_before('C_reg: _ #1')
        .reorder_before('C_reg: _ #1')
        #
        .set_memory('C_reg', AVX512)
        #
        .bind_expr('A_reg', 'A[_]')
        .lift_alloc('A_reg: _', keep_dims=True)
        .set_memory('A_reg', AVX512)
        .fission_after('A_reg[_] = _')
        #
        .bind_expr('B_reg', 'B[_]')
        .lift_alloc('B_reg: _', keep_dims=True)
        .set_memory('B_reg', AVX512)
        .fission_after('B_reg[_] = _')
        #
        .replace_all(mm512_set1_ps)
        .replace_all(mm512_fmadd_ps)
        .replace(mm512_loadu_ps, 'for ji in _: _ #0')
        .replace(mm512_loadu_ps, 'for ji in _: _ #0')
        .replace(mm512_storeu_ps, 'for ji in _: _ #0')
        .replace(mm512_maskz_loadu_ps, 'for ji in _: _ #0')
        .replace(mm512_mask_storeu_ps, 'for ji in _: _ #1')
        #
        .bind_expr('A_reg2', 'A[_] #1')
        .lift_alloc('A_reg2: _', keep_dims=True, n_lifts=2)
        .set_memory('A_reg2', AVX512)
        .fission_after('A_reg2[_] = _', n_lifts=2)
        #
        .bind_expr('B_reg2', 'B[_] #1')
        .lift_alloc('B_reg2: _', keep_dims=True, n_lifts=2)
        .set_memory('B_reg2', AVX512)
        .fission_after('B_reg2[_] = _', n_lifts=2)
        #
        .replace_all(mm512_dumb_mask_set1_ps)
        .replace_all(mm512_dumb_mask_fmadd_ps)
        .replace_all(mm512_maskz_loadu_ps)
        #
        .simplify()
)

right_panel_kernel_scheduled = (
    right_panel_kernel
        .rename('right_panel_kernel_scheduled')
        #
        .replace_all(right_panel_kernel)
        #
        .add_ifelse('right_panel_kernel(_) #0', '(N / 16) == 0')
        .add_ifelse('right_panel_kernel(_) #1', '(N / 16) == 1')
        .add_ifelse('right_panel_kernel(_) #2', '(N / 16) == 2')
        .add_ifelse('right_panel_kernel(_) #3', '(N / 16) == 3')
        #
        .call_eqv(right_panel_kernel_opt, 'right_panel_kernel(_)')
        .inline('right_panel_kernel_opt(_)')
        .inline_window('A = _')
        .simplify()
)

sgemm_sys_atl = (
    SGEMM
        .rename('sgemm_sys_atl')
        # Split up into cases
        .split('j', J_REG_BLK, ['jo', 'ji'], tail='cut_and_guard')
        .split('i', I_REG_BLK, ['io', 'ii'], tail='cut_and_guard')
        .fission_after('for jo in _: _ #0', n_lifts=2)
        .reorder('ii #0', 'jo')
        .reorder('ji #0', 'k')
        .reorder('ii #0', 'k')
        .fission_after('for jo in _: _ #1')
        .lift_if('if N % 64 > 0: _', n_lifts=2)
        .fission_after('for ii in _: _ #2')
        # Main block
        .replace_all(basic_kernel_Mx4[6])
        .call_eqv(sgemm_kernel_avx512_Mx4[6], 'basic_kernel_6x4(_)')
        # Right panel
        .reorder('ji', 'k')
        .reorder('ii', 'k')
        .replace_all(right_panel_kernel)
        .call_eqv(right_panel_kernel_scheduled, 'right_panel_kernel(_)')
        # Bottom panel
        .reorder('ii', 'jo')
        .reorder('ii', 'k')
        .replace_all(bottom_panel_kernel)
        .call_eqv(bottom_panel_kernel_scheduled, 'bottom_panel_kernel(_)')
        # TODO: bottom-right tile
        .simplify()
)

if __name__ == '__main__':
    print(right_panel_kernel_opt)
    # print(right_panel_kernel_scheduled)
    # print(right_panel_kernel_scheduled.c_code_str())

__all__ = ['sgemm_sys_atl']
