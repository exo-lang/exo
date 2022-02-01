from __future__ import annotations

from SYS_ATL import *


# row-major, upper-triangular, C := C + A @ A^T.
# noinspection PyPep8Naming
@proc
def SSYRK(M: size, K: size, A: f32[M, K], C: f32[M, M]):
    assert M >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):  # row i
        for j in seq(0, M):  # column j
            for k in seq(0, K):
                if j >= i:
                    C[i, j] += A[i, k] * A[j, k]


systl_ssyrk = (
    SSYRK
        .rename('systl_ssyrk')
        .partial_eval(16)
        .split('k', 16, ['ko', 'ki'])
        .reorder('j', 'ko')
        .reorder('i', 'ko')
    # .lift_if('if j >= i: _')  # WOAH
)

if __name__ == '__main__':
    print(systl_ssyrk)

__all__ = ['SSYRK']
