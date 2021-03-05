from __future__ import annotations
import sys
from SYS_ATL import proc, Procedure
sys.path.append(sys.path[0]+"/..")


def test_conv1d():
    @proc
    def conv1d(n : size, m : size, r: size,
               x : R[n], w : R[m], res : R[r] ):
        for i in par(0,r):
            res[i] = 0.0
        for i in par(0,r):
            for j in par(0,n):
                if i <= j < i + m:
                    res[i] += x[j]*w[i-j+m-1]

    assert type(conv1d) is Procedure
    print(conv1d)
