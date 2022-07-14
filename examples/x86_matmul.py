from __future__ import annotations

import os
import sys

from exo import proc
from exo.platforms.x86 import *

# Hide output when running through exocc.
if __name__ != '__main__' and hasattr(os, 'devnull'):
    sys.stdout = open(os.devnull, 'w')


# Algorithm definition
@proc
def rank_k_reduce_6x16(
    K: size,
    C: f32[6, 16] @ DRAM,
    A: f32[6, K] @ DRAM,
    B: f32[K, 16] @ DRAM,
):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


print("Original algorithm:")
print(rank_k_reduce_6x16)

# Schedule start here

# First block
"""
avx = rank_k_reduce_6x16.rename("rank_k_reduce_6x16_scheduled")
avx = avx.stage_assn('C_reg', 'C[_] += _')
avx = avx.set_memory('C_reg', AVX2)
print("First block:")
print(avx)
"""

# Second block
"""
avx = avx.split('j', 8, ['jo', 'ji'], perfect=True)
avx = avx.reorder('ji', 'k')
avx = avx.reorder('jo', 'k')
avx = avx.reorder('i', 'k')
print("Second block:")
print(avx)
"""

# Third block
"""
avx = avx.lift_alloc('C_reg:_', n_lifts=3)
avx = avx.fission_after('C_reg = _ #0', n_lifts=3)
avx = avx.fission_after('C_reg[_] += _ #0', n_lifts=3)
avx = avx.lift_alloc('C_reg:_', n_lifts=1)
avx = avx.fission_after('for i in _:_#0', n_lifts=1)
avx = avx.fission_after('for i in _:_#1', n_lifts=1)
avx = avx.simplify()
print("Third block:")
print(avx)
"""

# Fourth block
"""
avx = avx.bind_expr('a_vec', 'A[i, k]')
avx = avx.set_memory('a_vec', AVX2)
avx = avx.lift_alloc('a_vec:_', keep_dims=True)
avx = avx.fission_after('a_vec[_] = _')
print("Fourth block:")
print(avx)
"""

# Fifth block
"""
avx = avx.bind_expr('b_vec', 'B[k, _]')
avx = avx.set_memory('b_vec', AVX2)
avx = avx.lift_alloc('b_vec:_', keep_dims=True)
avx = avx.fission_after('b_vec[_] = _')
print("Fifth block:")
print(avx)
"""

# Sixth block
"""
avx = avx.replace_all(avx2_set0_ps)
avx = avx.replace_all(mm256_broadcast_ss)
avx = avx.replace_all(mm256_fmadd_ps)
avx = avx.replace_all(avx2_fmadd_memu_ps)
avx = avx.replace(mm256_loadu_ps, 'for ji in _:_ #0')
avx = avx.replace(mm256_loadu_ps, 'for ji in _:_ #0')
avx = avx.replace(mm256_storeu_ps, 'for ji in _:_ #0')
print("Sixth block:")
print(avx)
"""
