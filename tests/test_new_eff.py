from __future__ import annotations

import pytest

from SYS_ATL.new_eff import *



print()
print("Dev Tests for new_eff.py")


def debug_let_and_mod():
  N = AInt(Sym('N'))
  j = AInt(Sym('j'))
  i = AInt(Sym('i'))
  x = AInt(Sym('x'))

  F =  A.ForAll(i.name,
          A.Let([x.name],
                [A.Let([j.name],[AInt(64) * i],
                          N + j, T.index, j.srcinfo)],
                AEq(x % AInt(64), AInt(0)), T.bool, x.srcinfo),
          T.bool, i.srcinfo)

  print(F)

  slv = SMTSolver()

  slv.assume(F)
  print(slv.debug_str(smt=True))


