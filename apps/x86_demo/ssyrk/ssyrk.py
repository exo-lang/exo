from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.syntax import *


# row-major, upper-triangular, C := C + A @ A^T.
# noinspection PyPep8Naming
@proc
def systl_ssyrk(M: size, K: size, A: f32[M, K], C: f32[M, M]):
    assert M >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(C, 1) == 1

    for i in par(0, M):  # row i
        for j in par(0, M):  # column j
            if j >= i:
                for k in par(0, K):
                    C[i, j] += A[i, k] * A[j, k]


SSYRK_WINDOW = (systl_ssyrk.rename('systl_ssyrk_win')
                .set_window('A', True)
                .set_window('C', True))

if __name__ == '__main__':
    print(systl_ssyrk)

__all__ = ['systl_ssyrk']
