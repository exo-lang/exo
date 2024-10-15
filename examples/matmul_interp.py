from __future__ import annotations

import os
import sys
import numpy as np

from exo import proc
from exo.platforms.neon import *
from exo.stdlib.scheduling import *

# Hide output when running through exocc.
if __name__ != "__main__" and hasattr(os, "devnull"):
    sys.stdout = open(os.devnull, "w"
)

# Algorithm definition
@proc
def rank_k_reduce_6x16(
    K: size, A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM, C: f32[6, 16] @ DRAM
):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]

K = 4
A = np.arange(K*6, dtype=float).reshape((6,K))
B = np.arange(K*16, dtype=float).reshape((K,16))
C = np.zeros(6*16, dtype=float).reshape((6,16))
rank_k_reduce_6x16.interpret(K=K, A=A, B=B, C=C)