from __future__ import annotations
from exo import proc, compile_procs_to_strings
from exo.API_scheduling import rename
from exo.pyparser import ParseError
import pytest


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


def test_quote_elision(golden):
    @proc
    def foo(a: i32, b: i32):
        with meta:

            def bar():
                return a

            with ~meta:
                b = {bar()}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_unquote_elision(golden):
    @proc
    def foo(a: i32):
        with meta:
            x = 2
            with ~meta:
                a = a * x

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_scope_collision1(golden):
    @proc
    def foo(a: i32):
        with meta:
            b = 1
            with ~meta:
                b: i32
                b = 2
                with meta:
                    c = b
                    with ~meta:
                        a = c

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_scope_collision2(golden):
    @proc
    def foo(a: i32, b: i32):
        with meta:
            a = 1
            with ~meta:
                b = a

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_scope_collision3():
    with pytest.raises(
        NameError,
        match="free variable 'x' referenced before assignment in enclosing scope",
    ):

        @proc
        def foo(a: i32, b: i32):
            with meta:
                with ~meta:
                    a = b * x
                x = 1


def test_type_quote_elision(golden):
    T = "i8"

    @proc
    def foo(a: T, x: T[2]):
        a += x[0]
        a += x[1]

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_unquote_in_slice(golden):
    @proc
    def foo(a: [i8][2]):
        a[0] += a[1]

    @proc
    def bar(a: i8[10, 10]):
        with meta:
            x = 2
            with ~meta:
                for i in seq(0, 5):
                    foo(a[i, {x} : {2 * x}])

    c_file, _ = compile_procs_to_strings([foo, bar], "test.h")
    assert c_file == golden


def test_unquote_slice_object1(golden):
    @proc
    def foo(a: [i8][2]):
        a[0] += a[1]

    @proc
    def bar(a: i8[10, 10]):
        with meta:
            for s in [slice(1, 3), slice(5, 7), slice(2, 4)]:
                with ~meta:
                    for i in seq(0, 10):
                        foo(a[i, s])

    c_file, _ = compile_procs_to_strings([foo, bar], "test.h")
    assert c_file == golden


def test_unquote_slice_object2():
    with pytest.raises(
        ParseError, match="cannot perform windowing on left-hand-side of an assignment"
    ):

        @proc
        def foo(a: i8[10, 10]):
            with meta:
                for s in [slice(1, 3), slice(5, 7), slice(2, 4)]:
                    with ~meta:
                        for i in seq(0, 10):
                            a[i, s] = 2


def test_unquote_index_tuple(golden):
    @proc
    def foo(a: [i8][2, 2]):
        a[0, 0] += a[0, 1]
        a[1, 0] += a[1, 1]

    @proc
    def bar(a: i8[10, 10, 10]):
        with meta:

            def get_index(i):
                return slice(i, ~{i + 2}), slice(~{i + 1}, ~{i + 3})

            with ~meta:
                for i in seq(0, 7):
                    foo(a[i, {get_index(i)}])

    c_file, _ = compile_procs_to_strings([foo, bar], "test.h")
    assert c_file == golden


def test_unquote_err():
    with pytest.raises(
        ParseError, match="Unquote computation did not yield valid type"
    ):
        T = 1

        @proc
        def foo(a: T):
            a += 1


def test_quote_complex_expr(golden):
    @proc
    def foo(a: i32):
        with meta:

            def bar(x):
                return ~{x + 1}

            with ~meta:
                a = {bar(~{a + 1})}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_statement_assignment(golden):
    @proc
    def foo(a: i32):
        with meta:
            with ~meta as s1:
                a += 1
                a += 2
            with ~meta as s2:
                a += 3
                a += 4
            s = s1 if True else s2
            with ~meta:
                {s}
                {s}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden
