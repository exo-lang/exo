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
    A: f32[6, K] @ DRAM,
    C: f32[6, 16] @ DRAM,
    B: f32[K, 16] @ DRAM,
):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


# print("=============Original algorithm==============")
# print(rank_k_reduce_6x16)

# The first step is thinking about the output memory.
# In this ex, we want the computation to be "output stationary", which means,
# we want to preallocate all the output registers at the start.
avx = rename(rank_k_reduce_6x16, "rank_k_reduce_6x16_scheduled")
avx = reorder_loops(avx, "j k")
avx = reorder_loops(avx, "i k")

# The staging of C will cause us to consume 12 out of the 16 vector registers
avx = divide_loop(avx, "for j in _: _", 8, ["jo", "ji"], perfect=True)
avx = stage_mem(avx, "for k in _:_", "C[0:6, 0:16]", "C_reg")
avx = simplify(avx)

# Reshape C_reg so we can map it into vector registers
avx = divide_dim(avx, "C_reg:_", 1, 8)
avx = repeat(divide_loop)(avx, "for i1 in _: _", 8, ["i2", "i3"], perfect=True)
avx = simplify(avx)

# Map C_reg operations to vector instructions
avx = set_memory(avx, "C_reg:_", AVX2)
avx = replace_all(avx, mm256_loadu_ps)
avx = replace_all(avx, mm256_storeu_ps)
avx = simplify(avx)

# Now, the rest of the compute needs to work with the constraint that the
# we only have 4 more registers to work with here.

# B is easy, it is just two vector loads
avx = stage_mem(avx, "for i in _:_", "B[k, 0:16]", "B_reg")
avx = simplify(avx)
avx = divide_loop(avx, "for i0 in _: _ #1", 8, ["io", "ii"], perfect=True)
avx = divide_dim(avx, "B_reg:_", 0, 8)
avx = set_memory(avx, "B_reg:_", AVX2)
avx = simplify(avx)
avx = replace_all(avx, mm256_loadu_ps)
avx = simplify(avx)

# Now we've used up two more vector registers.
# The final part is staging A
# avx = stage_mem(avx, 'for jo in _:_', 'A[i, k]', 'A_reg')
avx = bind_expr(avx, "A[i, k]", "A_reg")
avx = expand_dim(avx, "A_reg", 8, "ji")
avx = lift_alloc(avx, "A_reg", n_lifts=2)
avx = fission(avx, avx.find("A_reg[ji] = _").after(), n_lifts=2)
avx = remove_loop(avx, "for jo in _: _")
avx = set_memory(avx, "A_reg:_", AVX2)
avx = replace_all(avx, mm256_broadcast_ss)

# DO THE COMPUTE!!!
avx = replace_all(avx, mm256_fmadd_ps)
avx = simplify(avx)

print("============= Rewritten ==============")
print(avx)
