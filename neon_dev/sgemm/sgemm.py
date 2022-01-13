from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.platforms.neon import *


def stage_C(kernel):
    return (kernel.par_to_seq('for k in _: _')
        .stage_assn('C_reg', 'C[_] += _')
        .lift_alloc('C_reg : _', n_lifts=4)
        .double_fission('C_reg[_] = C[_]', 'C_reg[_] += _', n_lifts=4)
    )

def stage_A_and_B(kernel):
    return (kernel
        .stage_expr('A_vec', 'A[_,_]', memory=Neon4f)
        .stage_expr('B_vec', 'B[_,_]', memory=Neon4f)
    )


# Compute Matrix-Matrix Multiplication C += A * B
@proc
def SGEMM(
    M: size,
    N: size,
    K: size,
    A: f32[M, K],
    B: f32[K, N],
    C: f32[M, N]
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

sgemm_win = (
    SGEMM.rename('sgemm_win')
        .set_window('A', True)
        .set_window('B', True)
        .set_window('C', True)
)

kernel_6x32 = (
    sgemm_win.rename('kernel_6x32')
        .partial_eval(6,32)
        .simplify()
)


sgemm_tiled = (
    SGEMM.rename('sgemm_tiled')
        # tile i & j for the kernel
        .split('i', 6, ['io', 'ii'], tail='cut_and_guard')
        .split('j #0', 32, ['jo', 'ji'], tail='cut_and_guard')
        # isolate the main chunk of work
        .fission_after('for jo in _: _', n_lifts=2)
        .reorder('ii','jo')
        .replace_all(kernel_6x32)
        .simplify()
)

neon_kernel_6x32 = (
    kernel_6x32.rename('neon_kernel_6x32')
        # Move k to the outermost loop
        .reorder('j','k')
        .reorder('i','k')
        # expose inner-loop for 4-wide vectorization
        .split('j', 4, ['jo','ji'], perfect=True)
)

neon_kernel_6x32 = ( stage_C(neon_kernel_6x32)
    .replace(neon_vld_4xf32, 'for ji in _: _ #0')
    .replace(neon_vst_4xf32, 'for ji in _: _ #1')
    .set_memory('C_reg', Neon4f)
)

neon_kernel_6x32 = ( stage_A_and_B(neon_kernel_6x32)
    .replace_all(neon_vld_4xf32)
    .replace_all(neon_broadcast_4xf32)
    .replace_all(neon_vfmadd_4xf32)
    .lift_alloc('A_vec : _', n_lifts=2)
    .fission_after('neon_broadcast_4xf32(_)', n_lifts=2)
    .lift_alloc('B_vec : _', n_lifts=2)
    .fission_after('neon_vld_4xf32(_) #1', n_lifts=2)
)

neon_kernel_6x32 = neon_kernel_6x32.simplify()
print(neon_kernel_6x32)

sgemm_tiled = (sgemm_tiled
    .call_eqv(neon_kernel_6x32, 'kernel_6x32(_)')
)

sgemm_systl = sgemm_tiled.rename('sgemm_systl')
print(sgemm_systl)

__all__ = ['sgemm_systl']
