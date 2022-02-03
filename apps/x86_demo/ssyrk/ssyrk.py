from __future__ import annotations

from SYS_ATL import *
from SYS_ATL.platforms.x86 import *


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
            c_acc: R[16]
            for k in seq(0, 16):
                c_acc[k] = 0.0
            if j >= i:
                for ko in seq(0, K / 16):
                    for ki in seq(0, 16):
                        c_acc[ki] += A[i, 16 * ko + ki] * A[j, 16 * ko + ki]
                if K % 16 > 0:
                    for ki in seq(0, K % 16):
                        c_acc[ki] += (A[i, 16 * (K / 16) + ki] *
                                      A[j, 16 * (K / 16) + ki])
            for k in seq(0, 16):
                C[i, j] += c_acc[k]


systl_ssyrk = (
    SSYRK
        .rename('systl_ssyrk')
        .lift_alloc('c_acc: _', n_lifts=2)
        .expand_dim('c_acc: _', 'M', 'j')
        .expand_dim('c_acc: _', 'M', 'i')
        .set_memory('c_acc', AVX512)

        .fission_after('for k in _: _ #0', n_lifts=2)
        .fission_after('if j >= i: _', n_lifts=2)

        .bind_expr('A_vec', 'A[_] #0')
        .bind_expr('At_vec', 'A[_] #1')
        .lift_alloc('A_vec: _')
        .lift_alloc('At_vec: _')
        .fission_after('A_vec[_] = _')
        .fission_after('At_vec[_] = _')
        .replace_all(mm512_loadu_ps)
        .replace_all(mm512_fmadd_ps)
        .simplify()

        .bound_and_guard('for ki in _: _')
        .bind_expr('A_vec_mask', 'A[_] #2')
        .bind_expr('At_vec_mask', 'A[_] #3')
        .lift_alloc('A_vec_mask: _', n_lifts=2)
        .lift_alloc('At_vec_mask: _', n_lifts=2)
        .fission_after('A_vec_mask[_] = _', n_lifts=2)
        .fission_after('At_vec_mask[_] = _', n_lifts=2)
        .replace_all(mm512_maskz_loadu_ps)
        .replace_all(mm512_mask_fmadd_ps)
        .simplify()

        .fission_after('for ko in _: _', n_lifts=3)
        .lift_if('if K % _ > 0: _', n_lifts=3)
)

if __name__ == '__main__':
    print(systl_ssyrk)

__all__ = ['SSYRK']
