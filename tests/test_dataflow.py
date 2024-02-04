from __future__ import annotations
import pytest
from exo import proc, DRAM, Procedure, config


def test_print():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        z = 4.2
        z = 2.0

    print(foo.dataflow())
