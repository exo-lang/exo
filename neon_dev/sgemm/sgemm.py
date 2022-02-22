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


micro_N = 2
micro_M = 16
assert micro_M % 4 == 0

L1_N = 16
L1_M = 2
L1_K = 64

microkernel = (
    sgemm_win.rename('microkernel')
        .partial_eval(micro_N,micro_M)
        .simplify()
)


sgemm_tiled = (
    SGEMM.rename('sgemm_tiled')
        # tile i & j for the kernel
        .split('i', micro_N, ['io', 'ii'], tail='cut_and_guard')
        .split('j #0', micro_M, ['jo', 'ji'], tail='cut_and_guard')
        # isolate the main chunk of work
        .fission_after('for jo in _: _', n_lifts=2)
        .reorder('ii','jo')
        # tile k now, before we do the microkernel replacement
        .split('k #0', L1_K, ['ko', 'ki'], tail='cut_and_guard')
        .fission_after('for ko in _: _', n_lifts=2)
        .reorder('ji','ko')
        .reorder('ii','ko')
        .replace_all(microkernel)
        .simplify()
)
print(sgemm_tiled)

neon_microkernel = (
    microkernel.rename('neon_microkernel')
        # Move k to the outermost loop
        .reorder('j','k')
        .reorder('i','k')
        # expose inner-loop for 4-wide vectorization
        .split('j', 4, ['jo','ji'], perfect=True)
)

neon_microkernel = ( stage_C(neon_microkernel)
    .replace(neon_vld_4xf32, 'for ji in _: _ #0')
    .replace(neon_vst_4xf32, 'for ji in _: _ #1')
    .set_memory('C_reg', Neon4f)
)

neon_microkernel = ( stage_A_and_B(neon_microkernel)
    .replace_all(neon_vld_4xf32)
    .replace_all(neon_broadcast_4xf32)
    .replace_all(neon_vfmadd_4xf32_4xf32)
    #.replace_all(neon_vfmadd_1xf32_4xf32)
    .lift_alloc('A_vec : _', n_lifts=2)
    .fission_after('neon_broadcast_4xf32(_)', n_lifts=2)
    .lift_alloc('B_vec : _', n_lifts=2)
    .fission_after('neon_vld_4xf32(_) #1', n_lifts=2)
)

neon_microkernel = neon_microkernel.simplify()
print(neon_microkernel)

sgemm_tiled = (sgemm_tiled
    .call_eqv(neon_microkernel, 'microkernel(_)')
    #.call_eqv(neon_microkernel, 'microkernel(_)')
    # actually tile for L1 cache
    .split('io #0', L1_N, ['io', 'im'], tail='cut')
    .split('jo #0', L1_M, ['jo', 'jm'], tail='cut')
    .fission_after('for jo in _: _ #0', n_lifts=1)
    .reorder('im','jm')
    .reorder('im','jo')
    .simplify()
)

sgemm_systl = sgemm_tiled.rename('sgemm_systl')
print(sgemm_systl)

__all__ = ['sgemm_systl']
