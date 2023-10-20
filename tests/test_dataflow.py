from __future__ import annotations
import pytest
from exo import proc, DRAM, Procedure, config
from exo.stdlib.scheduling import *


def test_print():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        z = 4.2
        z = 2.0

    print()
    print(foo.dataflow())
    print()


def test_print_1():
    @proc
    def foo(x: R[3], y: R[3], z: R[3]):
        z[0] = x[0] * y[2]
        for i in seq(0, 3):
            z[i] = 3.0
        z[2] = 2.0

    print()
    print(foo.dataflow())
    print()


def test_print_2():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    print()
    print(foo.dataflow())
    print()


def test_print_3():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 4.0
        z = 0.0

    print()
    print(foo.dataflow())
    print()


def test_print_4():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 3.0
        z = 0.0

    print()
    print(foo.dataflow())
    print()


def test_print_5():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = 3.0
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    print()
    print(foo.dataflow())
    print()
