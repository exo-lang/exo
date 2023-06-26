from __future__ import annotations

import os
import sys

from exo import proc
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *

# Hide output when running through exocc.
if __name__ != "__main__" and hasattr(os, "devnull"):
    sys.stdout = open(os.devnull, "w")


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
avx = rename(rank_k_reduce_6x16, "rank_k_reduce_6x16_scheduled")
avx = stage_mem(avx, 'C[_] += _', 'C[i, j]', 'C_reg')
print("First block:")
print(avx)
"""

# Second block
"""
avx = divide_loop(avx, 'j', 8, ['jo', 'ji'], perfect=True)
avx = reorder_loops(avx, 'ji k')
avx = reorder_loops(avx, 'jo k')
avx = reorder_loops(avx, 'i k')
print("Second block:")
print(avx)
"""

# Third block
"""
avx = autolift_alloc(avx, 'C_reg:_', n_lifts=4, keep_dims=True)
avx = autofission(avx, avx.find('C_reg = _ #0').after(), n_lifts=3)
avx = autofission(avx, avx.find('C_reg[_] += _ #0').after(), n_lifts=3)
avx = autofission(avx, avx.find('for i in _:_#0').after(), n_lifts=1)
avx = autofission(avx, avx.find('for i in _:_#1').after(), n_lifts=1)
avx = simplify(avx)
print("Third block:")
print(avx)
"""

# Fourth block
"""
avx = bind_expr(avx, 'A[i, k]', 'a_vec')
avx = expand_dim(avx, 'a_vec:_', '8', 'ji')
avx = autolift_alloc(avx, 'a_vec:_')
avx = autofission(avx, avx.find('a_vec[_] = _').after())
print("Fourth block:")
print(avx)
"""

# Fifth block
"""
avx = bind_expr(avx, 'B[k, _]', 'b_vec')
avx = autolift_alloc(avx, 'b_vec:_', keep_dims=True)
avx = autofission(avx, avx.find('b_vec[_] = _').after())
print("Fifth block:")
print(avx)
"""

# Sixth block
"""
avx = set_memory(avx, 'C_reg', AVX2)
avx = set_memory(avx, 'a_vec', AVX2)
avx = set_memory(avx, 'b_vec', AVX2)
avx = replace_all(avx, mm256_loadu_ps)
avx = replace_all(avx, mm256_broadcast_ss)
avx = replace_all(avx, mm256_fmadd_ps)
avx = replace_all(avx, mm256_storeu_ps)
print("Sixth block:")
print(avx)
"""
