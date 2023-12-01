from __future__ import annotations

import pytest

from exo import proc, Procedure, DRAM, compile_procs_to_strings
from exo.stdlib.scheduling import *


def test_pragma_parallel_loop(golden):
    @proc
    def foo(x: i8[10]):
        for i in par(0, 10):
            x[i] = 1.0

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_parallel_fail():
    @proc
    def foo(A: i8[10]):
        total: i8
        for i in par(0, 10):
            total += A[i]

    with pytest.raises(
        TypeError,
        match=r"parallel loop\'s body is not parallelizable because of potential data races",
    ):
        c_file, _ = compile_procs_to_strings([foo], "test.h")


def test_parallel_fail_2():
    @proc
    def foo(A: i8[10]):
        total: i8
        for i in par(0, 10):
            total = A[i]

    with pytest.raises(
        TypeError,
        match=r"parallel loop\'s body is not parallelizable because of potential data races",
    ):
        c_file, _ = compile_procs_to_strings([foo], "test.h")
