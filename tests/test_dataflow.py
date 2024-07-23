from __future__ import annotations
import pytest
from exo import proc, DRAM, Procedure, config
from exo.stdlib.scheduling import *


def test_simple(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_simple2(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        x[0] = 2.0
        x[1] = 3.0
        x[2] = 5.0
        x[0] = 12.0

    assert str(foo.dataflow()[0]) == golden


def test_simple_stmts(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        z = 2.0

    d_ir, stmts = foo.dataflow(foo.find("z = _ ; z = _"))

    assert str(d_ir) + "".join([str(s) for s in stmts]) == golden


def test_simple_stmts2(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        pass

    d_ir, stmts = foo.dataflow(foo.find("if n < 3: _"))

    assert str(d_ir) + "".join([str(s) for s in stmts]) == golden


def test_simple3(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        x[2] = 5.0
        x[0] = 12.0

    assert str(foo.dataflow()[0]) == golden


def test_print(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        z = 4.2
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_print_1(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R[3]):
        z[0] = x[0] * y[2]
        for i in seq(0, 3):
            z[i] = 3.0
        z[2] = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_print_2(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_print_3(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 4.0
        z = 0.0

    assert str(foo.dataflow()[0]) == golden


def test_print_4(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 3.0
        z = 0.0

    assert str(foo.dataflow()[0]) == golden


def test_print_5(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = 3.0
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


# TODO: Currently add_unsafe_guard lacks analysis, but we should be able to analyze this
def test_sliding_window(golden):
    @proc
    def foo(n: size, m: size, dst: i8[n + m], src: i8[n + m]):
        for i in seq(0, n):
            for j in seq(0, m):
                dst[i + j] = src[i + j]

    foo = add_unsafe_guard(foo, "dst[_] = src[_]", "i == 0 or j == m - 1")

    assert str(foo.dataflow()[0]) == golden


# TODO: fission should be able to handle this
def test_fission_fail():
    @proc
    def foo(n: size, dst: i8[n + 1], src: i8[n + 1]):
        for i in seq(0, n):
            dst[i] = src[i]
            dst[i + 1] = src[i + 1]

    with pytest.raises(SchedulingError, match="Cannot fission"):
        foo = fission(foo, foo.find("dst[i] = _").after())
        print(foo)


# TODO: This is unsafe, lift_alloc should give an error
def test_lift_alloc_unsafe(golden):
    @proc
    def foo():
        for i in seq(0, 10):
            a: i8[11] @ DRAM
            a[i] = 1.0
            a[i + 1] += 1.0

    foo = lift_alloc(foo, "a : _")

    assert str(foo.dataflow()[0]) == golden


# TODO: We are not supporting this AFAIK but should keep this example in mind
def test_reduc(golden):
    @proc
    def foo(n: size, a: f32, c: f32):
        tmp: f32[n]
        for i in seq(0, n):
            for j in seq(0, 4):
                tmp[i] = a
                a = tmp[i] + 1.0
        for i in seq(0, n):
            c += tmp[i]  # some use of tmp

    assert str(foo.dataflow()[0]) == golden


def test_absval_init(golden):
    @proc
    def foo1(n: size, dst: f32[n]):
        for i in seq(0, n):
            dst[i] = 0.0

    @proc
    def foo2(n: size, dst: f32[n], src: f32[n]):
        for i in seq(0, n):
            dst[i] = src[i]

    assert str(foo1.dataflow()[0]) + str(foo2.dataflow()[0]) == golden


# Below are Configuration sanity checking tests


def new_config_f32():
    @config
    class ConfigAB:
        a: f32
        b: f32

    return ConfigAB


def new_control_config():
    @config
    class ConfigControl:
        i: index
        s: stride
        b: bool

    return ConfigControl


def test_config_1(golden):
    ConfigAB = new_config_f32()

    @proc
    def foo(x: f32, y: f32):
        ConfigAB.a = 1.0
        ConfigAB.b = 3.0
        x = ConfigAB.a
        ConfigAB.b = ConfigAB.a

    assert str(foo.dataflow()[0]) == golden


def test_config_2(golden):
    ConfigAB = new_config_f32()

    @proc
    def foo(x: f32, y: f32):
        ConfigAB.a = 1.0
        ConfigAB.b = 3.0
        for i in seq(0, 10):
            x = ConfigAB.a
            ConfigAB.b = ConfigAB.a
        ConfigAB.a = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_config_3(golden):
    CTRL = new_control_config()

    @proc
    def foo(n: size):
        for i in seq(0, n):
            if CTRL.i == 2:
                CTRL.i = 4
            if n == n - 1:
                CTRL.i = 3

    assert str(foo.dataflow()[0]) == golden


def test_config_4(golden):
    CTRL = new_control_config()

    @proc
    def foo(n: size, src: [i8][n]):
        assert stride(src, 0) == CTRL.s
        pass

    assert str(foo.dataflow()[0]) == golden


# Below are function inlining tests


def test_function(golden):
    @proc
    def bar():
        for i in seq(0, 10):
            A: i8
            A = 0.3

    @proc
    def foo(n: size, src: [i8][n]):
        bar()

    assert str(foo.dataflow()[0]) == golden


def test_window_stmt(golden):
    @proc
    def foo(n: size, src: [i8][20]):
        tmp = src[0:10]
        for i in seq(0, 10):
            tmp[i] = 1.0

    assert str(foo.dataflow()[0]) == golden


def test_config_function(golden):
    ConfigAB = new_config_f32()

    @proc
    def bar(z: f32):
        z = 3.0
        ConfigAB.a = 2.0

    @proc
    def foo(x: f32):
        ConfigAB.a = 1.0
        bar(x)
        ConfigAB.b = x

    assert str(foo.dataflow()[0]) == golden
