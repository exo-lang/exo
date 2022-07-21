from __future__ import annotations

from exo import *
from exo.platforms.neon import *


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


micro_N = 4
micro_M = 16
assert micro_M % 4 == 0

L1_N = 64
L1_M = 64
L1_K = 64

assert L1_N % micro_N == 0
assert L1_M % micro_M == 0
mid_N = L1_N // micro_N
mid_M = L1_M // micro_M
mid_K = L1_K

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
        .split('k #0', mid_K, ['ko', 'ki'], tail='cut_and_guard')
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
    # clean up tail case from earlier
    .fission_after('for ko in _: _ #0', n_lifts=2)
    # actually tile for L1 cache
    .reorder('jo #0','ko')
    .reorder('io #0','ko')
    .split('io #0', mid_N, ['io', 'im'], tail='cut')
    .split('jo #0', mid_M, ['jo', 'jm'], tail='cut')
    .fission_after('for jo in _: _ #0', n_lifts=3)
    .reorder('im','jm')
    .reorder('im','jo')
    #.reorder('ko #0', 'io')
    #.reorder('ko #0', 'jo')
    .simplify()
    # stage per-tile memory at appropriate levels
    .stage_mem(f'A[{L1_N}*io : {L1_N}*io + {L1_N},'
               f'  {L1_K}*ko : {L1_K}*ko + {L1_K}]',
               'Atile', 'for jo in _: _ #0')
    .lift_alloc_simple('Atile : _', n_lifts=2)
    .stage_mem(f'B[{L1_K}*ko : {L1_K}*ko + {L1_K},'
               f'  {L1_M}*jo : {L1_M}*jo + {L1_M}]',
               'Btile', 'for im in _: _ #0')
    .lift_alloc_simple('Btile : _', n_lifts=3)
    # cleanup
    .simplify()
)

sgemm_exo = sgemm_tiled.rename('sgemm_exo')
print(sgemm_exo)

__all__ = ['sgemm_exo']
