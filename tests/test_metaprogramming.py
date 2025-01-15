from __future__ import annotations
from exo import proc, compile_procs_to_strings
from exo.API_scheduling import rename
from exo.frontend.pyparser import ParseError
import pytest
import warnings
from exo.libs.externs import *
from exo.platforms.x86 import DRAM


def test_unrolling(golden):
    @proc
    def foo(a: i8):
        b: i8
        b = 0
        with python:
            for _ in range(10):
                with exo:
                    b += a

    c_file, _ = compile_procs_to_strings([foo], "test.h")

    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_conditional(golden):
    def foo(cond: bool):
        @proc
        def bar(a: i8):
            b: i8
            with python:
                if cond:
                    with exo:
                        b = 0
                else:
                    with exo:
                        b += 1

        return bar

    bar1 = rename(foo(False), "bar1")
    bar2 = rename(foo(True), "bar2")

    c_file, _ = compile_procs_to_strings([bar1, bar2], "test.h")
    assert f"EXO IR:\n{str(bar1)}\n{str(bar2)}\nC:\n{c_file}" == golden


def test_scoping(golden):
    a = 3

    @proc
    def foo(a: i8):
        a = {a}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_scope_nesting(golden):
    x = 3

    @proc
    def foo(a: i8, b: i8):
        with python:
            y = 2
            with exo:
                a = {~{b} if x == 3 and y == 2 else ~{a}}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_global_scope():
    cell = [0]

    @proc
    def foo(a: i8):
        a = 0
        with python:
            with exo:
                with python:
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
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


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

    bar1 = rename(foo("i32", "i8"), "bar1")
    bar2 = rename(foo("f64", "f64"), "bar2")

    c_file, _ = compile_procs_to_strings([bar1, bar2], "test.h")
    assert f"EXO IR:\n{str(bar1)}\n{str(bar2)}\nC:\n{c_file}" == golden


def test_captured_closure(golden):
    cell = [0]

    def foo():
        cell[0] += 1

    @proc
    def bar(a: i32):
        with python:
            for _ in range(10):
                foo()
                with exo:
                    a += {cell[0]}

    c_file, _ = compile_procs_to_strings([bar], "test.h")
    assert f"EXO IR:\n{str(bar)}\nC:\n{c_file}" == golden


def test_capture_nested_quote(golden):
    a = 2

    @proc
    def foo(a: i32):
        with python:
            for _ in range(3):
                with exo:
                    a = {a}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_quote_elision(golden):
    @proc
    def foo(a: i32, b: i32):
        with python:

            def bar():
                return a

            with exo:
                b = {bar()}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_unquote_elision(golden):
    @proc
    def foo(a: i32):
        with python:
            x = 2
            with exo:
                a = a * x

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_scope_collision1(golden):
    @proc
    def foo(a: i32):
        with python:
            b = 1
            with exo:
                b: i32
                b = 2
                with python:
                    c = b
                    with exo:
                        a = c

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_scope_collision2(golden):
    @proc
    def foo(a: i32, b: i32):
        with python:
            a = 1
            with exo:
                b = a

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_scope_collision3():
    with pytest.raises(
        NameError,
        match="free variable 'x' referenced before assignment in enclosing scope",
    ):

        @proc
        def foo(a: i32, b: i32):
            with python:
                with exo:
                    a = b * x
                x = 1


def test_type_quote_elision(golden):
    T = "i8"

    @proc
    def foo(a: T, x: T[2]):
        a += x[0]
        a += x[1]

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_unquote_in_slice(golden):
    @proc
    def foo(a: [i8][2]):
        a[0] += a[1]

    @proc
    def bar(a: i8[10, 10]):
        with python:
            x = 2
            with exo:
                for i in seq(0, 5):
                    foo(a[i, {x} : {2 * x}])

    c_file, _ = compile_procs_to_strings([foo, bar], "test.h")
    assert f"EXO IR:\n{str(foo)}\n{str(bar)}\nC:\n{c_file}" == golden


def test_unquote_slice_object1(golden):
    @proc
    def foo(a: [i8][2]):
        a[0] += a[1]

    @proc
    def bar(a: i8[10, 10]):
        with python:
            for s in [slice(1, 3), slice(5, 7), slice(2, 4)]:
                with exo:
                    for i in seq(0, 10):
                        foo(a[i, s])

    c_file, _ = compile_procs_to_strings([foo, bar], "test.h")
    assert f"EXO IR:\n{str(foo)}\n{str(bar)}\nC:\n{c_file}" == golden


def test_unquote_slice_object2():
    with pytest.raises(
        ParseError, match="cannot perform windowing on left-hand-side of an assignment"
    ):

        @proc
        def foo(a: i8[10, 10]):
            with python:
                for s in [slice(1, 3), slice(5, 7), slice(2, 4)]:
                    with exo:
                        for i in seq(0, 10):
                            a[i, s] = 2


def test_unquote_index_tuple(golden):
    @proc
    def foo(a: [i8][2, 2]):
        a[0, 0] += a[0, 1]
        a[1, 0] += a[1, 1]

    @proc
    def bar(a: i8[10, 10, 10]):
        with python:

            def get_index(i):
                return slice(i, ~{i + 2}), slice(~{i + 1}, ~{i + 3})

            with exo:
                for i in seq(0, 7):
                    foo(a[i, {get_index(i)}])

    c_file, _ = compile_procs_to_strings([foo, bar], "test.h")
    assert f"EXO IR:\n{str(foo)}\n{str(bar)}\nC:\n{c_file}" == golden


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
        with python:

            def bar(x):
                return ~{x + 1}

            with exo:
                a = {bar(~{a + 1})}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_statement_assignment(golden):
    @proc
    def foo(a: i32):
        with python:
            with exo as s1:
                a += 1
                a += 2
            with exo as s2:
                a += 3
                a += 4
            s = s1 if True else s2
            with exo:
                {s}
                {s}

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_statement_in_expr():
    with pytest.raises(
        TypeError, match="Cannot unquote Exo statements in this context."
    ):

        @proc
        def foo(a: i32):
            with python:

                def bar():
                    with exo:
                        a += 1
                    return 2

                with exo:
                    a += {bar()}
                    a += {bar()}


def test_nonlocal_disallowed():
    with pytest.raises(ParseError, match="nonlocal is not supported"):
        x = 0

        @proc
        def foo(a: i32):
            with python:
                nonlocal x


def test_outer_return_disallowed():
    with pytest.raises(ParseError, match="cannot return from metalanguage fragment"):

        @proc
        def foo(a: i32):
            with python:
                return


def test_with_block():
    @proc
    def foo(a: i32):
        with python:

            def issue_warning():
                warnings.warn("deprecated", DeprecationWarning)

            with warnings.catch_warnings(record=True) as recorded_warnings:
                issue_warning()
                assert len(recorded_warnings) == 1
        pass


def test_unary_ops(golden):
    @proc
    def foo(a: i32):
        with python:
            x = ~1
            with exo:
                a = x

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_return_in_async():
    @proc
    def foo(a: i32):
        with python:

            async def bar():
                return 1

        pass


def test_local_externs(golden):
    my_sin = sin

    @proc
    def foo(a: f64):
        a = my_sin(a)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_unquote_multiple_exprs():
    with pytest.raises(ParseError, match="Unquote must take 1 argument"):
        x = 0

        @proc
        def foo(a: i32):
            a = {x, x}


def test_disallow_with_in_exo():
    with pytest.raises(ParseError, match="Expected unquote"):

        @proc
        def foo(a: i32):
            with a:
                pass


def test_unquote_multiple_stmts():
    with pytest.raises(ParseError, match="Unquote must take 1 argument"):

        @proc
        def foo(a: i32):
            with python:
                with exo as s:
                    a += 1
                with exo:
                    {s, s}


def test_unquote_non_statement():
    with pytest.raises(
        ParseError,
        match="Statement-level unquote expression must return Exo statements",
    ):

        @proc
        def foo(a: i32):
            with python:
                x = ~{a}
                with exo:
                    {x}


def test_unquote_slice_with_step():
    with pytest.raises(ParseError, match="Unquote returned slice index with step"):

        @proc
        def bar(a: [i32][10]):
            a[0] = 0

        @proc
        def foo(a: i32[20]):
            with python:
                x = slice(0, 20, 2)
                with exo:
                    bar(a[x])


def test_typecheck_unquote_index():
    with pytest.raises(
        ParseError, match="Unquote received input that couldn't be unquoted"
    ):

        @proc
        def foo(a: i32[20]):
            with python:
                x = "0"
                with exo:
                    a[x] = 0


def test_proc_shadowing(golden):
    @proc
    def sin(a: f32):
        a = 0

    @proc
    def foo(a: f32):
        sin(a)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden


def test_eval_expr_in_mem(golden):
    mems = [DRAM]

    @proc
    def foo(a: f32 @ mems[0]):
        pass

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert f"EXO IR:\n{str(foo)}\nC:\n{c_file}" == golden
