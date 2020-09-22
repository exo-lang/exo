from __future__ import annotations
import sys
sys.path.append(sys.path[0]+"/..")

from SYS_ATL import proc, Procedure

def test_conv1d():
  @proc
  def conv1d(n : size, m : size, r: size,
             x : R[n] @ IN, w : R[m] @ IN, res : R[r] @ OUT ):
    for i in range(0,r):
      res[i] = 0.0
    for i in range(0,r):
      for j in range(0,n):
        if i <= j < i + m:
          res[i] += x[j]*w[i-j+m-1]

  assert type(conv1d) is Procedure
  print(conv1d._TESTING_UAST())