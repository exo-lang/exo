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

    for k in par(0, K):
        for i in par(0, M):
            for j in par(0, N):
                C[i, j] += A[i, k] * B[k, j]


# Constants for scheduling
VEC_W = 16

M_REG_BLK = 6
N_REG_BLK = (4 * VEC_W)

M_L1_FAC = 44
N_L1_FAC = 1

M_L1_BLK = M_REG_BLK * M_L1_FAC
N_L1_BLK = N_REG_BLK * N_L1_FAC
K_L1_BLK = 512

COPY_STREAMS = 3

SGEMM_WINDOW = (
    SGEMM
        .rename('SGEMM_WINDOW')
        .set_window('A', True)
        .set_window('B', True)
        .set_window('C', True)
)

basic_kernel_Mx4 = {}
sgemm_kernel_avx512_Mx4 = {}
for M in range(1, M_REG_BLK + 1):
    basic_kernel_Mx4[M] = (
        SGEMM_WINDOW
            .rename(f'basic_kernel_{M}x4')
            .partial_eval(M, N_REG_BLK)
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
        .partial_eval(N=N_REG_BLK)
        .add_assertion('M < 6')
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
        .partial_eval(M=M_REG_BLK)
        .add_assertion('N / 16 < 4')
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
        .reorder_before('C_reg: _ #1')
        #
        .fission_after('C_reg[_] = _', n_lifts=4)
        .fission_after('C_reg[_] += _', n_lifts=4)
        #
        .reorder_before('for i in _: _ #3')
        .reorder_before('for i in _: _ #2')
        #
        .reorder_before('for k in _: _ #1')
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
        .replace(mm512_loadu_ps, 'for ji in _: _ #1')
        .replace(mm512_storeu_ps, 'for ji in _: _ #2')
        #
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
        .replace_all(mm512_mask_set1_ps)
        .replace_all(mm512_mask_fmadd_ps)
        .replace_all(mm512_maskz_loadu_ps)
        #
        .fuse_loop('for i in _: _ #0', 'for i in _: _ #1')
        .fuse_loop('for k in _: _ #0', 'for k in _: _ #1')
        .fuse_loop('for i in _: _ #1', 'for i in _: _ #2')
        .fuse_loop('for i in _: _ #2', 'for i in _: _ #3')
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
        .call_eqv(right_panel_kernel_opt, 'right_panel_kernel(_)')
        .call_eqv(right_panel_kernel_opt, 'right_panel_kernel(_)')
        .call_eqv(right_panel_kernel_opt, 'right_panel_kernel(_)')
        .call_eqv(right_panel_kernel_opt, 'right_panel_kernel(_)')
        .inline('right_panel_kernel_opt(_)')
        .inline('right_panel_kernel_opt(_)')
        .inline('right_panel_kernel_opt(_)')
        .inline('right_panel_kernel_opt(_)')
        .inline('right_panel_kernel_opt(_)')
        #
        .simplify()
        #
        .inline_window('A = _')
        .inline_window('A = _')
        .inline_window('A = _')
        .inline_window('A = _')
        .inline_window('A = _')
        #
        .inline_window('B = _')
        .inline_window('B = _')
        .inline_window('B = _')
        .inline_window('B = _')
        .inline_window('B = _')
        #
        .inline_window('C = _')
        .inline_window('C = _')
        .inline_window('C = _')
        .inline_window('C = _')
        .inline_window('C = _')
        #
        .simplify()
)

sgemm_above_kernel = (
    SGEMM_WINDOW
        .rename('sgemm_above_kernel')
        # Split up into cases
        .split('j', N_REG_BLK, ['jo', 'ji'], tail='cut_and_guard')
        .split('i', M_REG_BLK, ['io', 'ii'], tail='cut_and_guard')
        .fission_after('for jo in _: _ #0', n_lifts=2)
        .reorder('ii #0', 'jo')
        .fission_after('for io in _: _')
        .reorder('k #0', 'io')
        .reorder('k #0', 'jo')
        .lift_if('if N % _ > 0: _ #0', n_lifts=3)
        .reorder('k', 'io')
        .lift_if('if M % _ > 0: _ #0')
        .fission_after('for jo in _: _ #1', n_lifts=2)
        .reorder('ii', 'jo')
        .reorder('k', 'jo')
        .lift_if('if N % _ > 0: _ #1', n_lifts=2)
        # Main block
        .replace_all(basic_kernel_Mx4[6])
        .call_eqv(sgemm_kernel_avx512_Mx4[6], 'basic_kernel_6x4(_)')
        # Right panel
        .replace_all(right_panel_kernel)
        .call_eqv(right_panel_kernel_scheduled, 'right_panel_kernel(_)')
        # Bottom panel
        .replace_all(bottom_panel_kernel)
        .call_eqv(bottom_panel_kernel_scheduled, 'bottom_panel_kernel(_)')
        # TODO: bottom-right tile
        .simplify()
)


class DRAM_STATIC(DRAM):
    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        return f'static {prim_type} {new_name}[{" * ".join(shape)}];'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ''


sgemm_sys_atl = (
    SGEMM
        .rename('sgemm_sys_atl')
        # Split all loops
        .split('k', K_L1_BLK, ['ko', 'ki'], tail='cut_and_guard')
        .split('i', M_L1_BLK, ['io', 'ii'], tail='cut_and_guard')
        .split('j', N_L1_BLK, ['jo', 'ji'], tail='cut_and_guard')
        # Explode into 8 cases
        .fission_after('for io in _: _', n_lifts=2)
        .fission_after('for jo in _: _', n_lifts=4)
        # Case 1:
        .reorder('ki', 'io')
        .reorder('ii', 'jo')
        .reorder('ki', 'jo')
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 2:
        .lift_if('if N % _ > 0: _ #0', n_lifts=4)
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 3:
        .lift_if('if M % _ > 0: _ #0', n_lifts=2)
        .reorder('ki', 'jo')
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 4:
        .lift_if('if M % _ > 0: _ #1', n_lifts=2)
        .lift_if('if N % _ > 0: _ #1', n_lifts=3)
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 5:
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 6:
        .lift_if('if N % _ > 0: _ #2', n_lifts=3)
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 7:
        .lift_if('if M % _ > 0: _ #2')
        .reorder('ki', 'jo')
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        # Case 8:
        .lift_if('if M % _ > 0: _ #3')
        .lift_if('if N % _ > 0: _ #3', n_lifts=2)
        .replace(SGEMM_WINDOW, 'for ki in _: _ #0')
        ## Merge K ifs
        # .fuse_if('if K % _ > 0: _ #0', 'if K % _ > 0: _ #1')
        # .fuse_if('if K % _ > 0: _ #0', 'if K % _ > 0: _ #1')
        # .fuse_if('if K % _ > 0: _ #0', 'if K % _ > 0: _ #1')
        ## Merge M ifs
        # .fuse_if('if M % _ > 0: _ #0', 'if M % _ > 0: _ #1')
        ## Case 1 memory staging
        # Stage A
        .stage_window('A1_cache', 'A[_] #0', DRAM_STATIC)
        .par_to_seq('for ko in _: _ #0')
        .par_to_seq('for io in _: _ #0')
        .par_to_seq('for jo in _: _ #0')
        .lift_alloc('A1_cache: _', n_lifts=3)
        .fission_after('for i0 in _: _')
        # Stage B
        .stage_window('B1_cache', 'B[_] #0', DRAM_STATIC)
        .par_to_seq('for ko in _: _ #0')
        .par_to_seq('for io in _: _ #0')
        .par_to_seq('for jo in _: _ #0')
        .lift_alloc('B1_cache: _', n_lifts=3)
        ## Case 2 memory staging
        .stage_window('B2_cache', 'B[_] #1', DRAM_STATIC)
        .bound_alloc('B2_cache: _', [None, '64'])
        .par_to_seq('for io in _: _ #1')
        .lift_alloc('B2_cache: _', )
        .fission_after('for i0 in _: _ #2')
        # This does not seem to be helpful here:
        # .stage_window('A2_cache', 'A[_] #1', DRAM_STATIC)
        ## Case 3 memory staging
        .stage_window('B3_cache', 'B[_] #2', DRAM_STATIC)
        ## Case 4 memory staging
        .stage_window('B4_cache', 'B[_] #3', DRAM_STATIC)
        .bound_alloc('B4_cache: _', [None, '64'])
        ## Case 5 memory staging
        .stage_window('B5_cache', 'B[_] #4', DRAM_STATIC)
        .bound_alloc('B5_cache: _', ['512', None])
        ## Case 6 memory staging
        .stage_window('B6_cache', 'B[_] #5', DRAM_STATIC)
        .bound_alloc('B6_cache: _', ['512', '64'])
        ## Case 7 memory staging
        .stage_window('B7_cache', 'B[_] #6', DRAM_STATIC)
        .bound_alloc('B7_cache: _', ['512', None])
        ## Case 8 memory staging
        .stage_window('B8_cache', 'B[_] #7', DRAM_STATIC)
        .bound_alloc('B8_cache: _', ['512', '64'])
        ## Replace SGEMM_WINDOW with optimized form
        # These must come AFTER bound_alloc since the internal check-effects
        # is a whole program analysis that is VERY expensive
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 1
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 2
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 3
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 4
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 5
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 6
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 7
        .call_eqv(sgemm_above_kernel, 'SGEMM_WINDOW(_)')  # 8
        # Clean up
        .simplify()
)

if __name__ == '__main__':
    # print(sgemm_above_kernel)
    print(sgemm_sys_atl)

__all__ = ['sgemm_sys_atl']
