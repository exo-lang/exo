from __future__ import annotations

import pytest

from SYS_ATL import proc, DRAM
from SYS_ATL.libs.memories import GEMM_SCRATCH
from .helper import TMP_DIR, generate_lib


def test_inline_window():

    @proc
    def foo(n : size, m : size, x : R[n,m]):
        assert n > 4
        assert m > 4
        y = x[2:n-2,1:m-3]

        for i in seq(0,n-4):
            for j in seq(0,m-4):
                a : R
                a = x[i,j] * y[i,j]
                y[i,j] = a + x[i+1,j+1]

    foo = foo.inline_window('y = _')
    print(foo)
    assert 'y' not in str(foo)
