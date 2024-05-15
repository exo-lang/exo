from __future__ import annotations
from exo import proc, compile_procs_to_strings
from exo.API_scheduling import rename


def test_unrolling(golden):
    @proc
    def foo(a: i8):
        b: i8
        b = 0
        with meta:
            for _ in range(10):
                with ~meta:
                    b += a

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert c_file == golden


def test_conditional(golden):
    def foo(cond: bool):
        @proc
        def bar(a: i8):
            b: i8
            with meta:
                if cond:
                    with ~meta:
                        b = 0
                else:
                    with ~meta:
                        b += 1

        return bar

    c_file, _ = compile_procs_to_strings(
        [rename(foo(False), "bar1"), rename(foo(True), "bar2")], "test.h"
    )
    assert c_file == golden


def test_scoping(golden):
    a = 3

    @proc
    def foo(a: i8):
        a = {a}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_scope_nesting(golden):
    x = 3

    @proc
    def foo(a: i8, b: i8):
        with meta:
            y = 2
            with ~meta:
                a = {~{b} if x == 3 and y == 2 else ~{a}}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_global_scope():
    cell = [0]

    @proc
    def foo(a: i8):
        a = 0
        with meta:
            with ~meta:
                with meta:
                    global dict
                    cell[0] = dict
            dict = None

    assert cell[0] == dict


def test_constant_lifting(golden):
    x = 1.3

    @proc
    def foo(a: f64):
        a = {(x**x + x) / x}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_type_params(golden):
    def foo(T: str, U: str):
        @proc
        def bar(a: {T}, b: {U}):
            c: {T}[4]
            for i in seq(0, 3):
                d: {T}
                d = b
                c[i + 1] = a + c[i] * d
            a = c[3]

        return bar

    c_file, _ = compile_procs_to_strings(
        [rename(foo("i32", "i8"), "bar1"), rename(foo("f64", "f64"), "bar2")], "test.h"
    )
    assert c_file == golden


def test_captured_closure(golden):
    cell = [0]

    def foo():
        cell[0] += 1

    @proc
    def bar(a: i32):
        with meta:
            for _ in range(10):
                foo()
                with ~meta:
                    a += {cell[0]}

    c_file, _ = compile_procs_to_strings([bar], "test.h")
    assert c_file == golden


def test_capture_nested_quote(golden):
    a = 2

    @proc
    def foo(a: i32):
        with meta:
            for _ in range(3):
                with ~meta:
                    a = {a}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden
