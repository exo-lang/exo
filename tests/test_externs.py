from __future__ import annotations

import pytest

from exo import proc, DRAM, Procedure, config, compile_procs_to_strings
from exo.libs.externs import *
from exo.stdlib.scheduling import SchedulingError


def test_relu(golden, compiler):
    @proc
    def foo(x: f32[16]):
        for i in seq(0, 16):
            x[i] = relu(3.0)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_relu2(golden, compiler):
    @proc
    def foo(x: f32[16]):
        for i in seq(0, 16):
            x[i] = relu(x[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_relu3(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16], z: f32[16]):
        for i in seq(0, 16):
            z[i] = relu(x[i] + y[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_relu4(golden, compiler):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = relu(3.0)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_relu5():
    with pytest.raises(TypeError, match="expected 1 argument, got 2"):

        @proc
        def foo(x: i8[16]):
            for i in seq(0, 16):
                x[i] = relu(3.0, 2.0)

    with pytest.raises(
        TypeError, match="expected argument 1 to be a real scalar value,"
    ):

        @proc
        def foo(x: i8[16]):
            for i in seq(0, 16):
                x[i] = relu(i)


def test_sin(golden, compiler):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = sin(x[i] * 2)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_sin2(golden, compiler):
    with pytest.raises(TypeError, match="expected 1 argument, got 2"):

        @proc
        def foo(x: i8[16]):
            for i in seq(0, 16):
                x[i] = sin(x[i] * 2, 3)

    with pytest.raises(
        TypeError, match="expected argument 1 to be a real scalar value,"
    ):

        @proc
        def foo(x: i8[16]):
            for i in seq(0, 16):
                x[i] = sin(i)


def test_select(golden, compiler):
    @proc
    def foo(x: i8[16], y: i8[16], z: i8[16]):
        for i in seq(0, 16):
            z[i] = select(x[i] * 2, y[i], z[i] + y[i], -x[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_expf(golden, compiler):
    @proc
    def foo(x: i8[16], y: i8[16]):
        for i in seq(0, 16):
            y[i] = expf(x[i] + y[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_expf2():
    with pytest.raises(TypeError, match="expected 1 argument, got"):

        @proc
        def foo(x: i8[16], y: i8[16]):
            y[0] = expf(x[0], x[0])

    with pytest.raises(
        TypeError, match="expected argument 1 to be a real scalar value,"
    ):

        @proc
        def foo(x: i8[16], y: i8[16]):
            y[0] = expf(True)


def test_fmaxf(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = fmaxf(x[i], y[i] * 2)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_fmaxf2():
    with pytest.raises(TypeError, match="expected 2 argument, got 1"):

        @proc
        def foo(x: f32[16], y: f32[16]):
            for i in seq(0, 16):
                y[i] = fmaxf(x[i])

    with pytest.raises(
        TypeError, match="expected argument 1 to be a real scalar value,"
    ):

        @proc
        def foo(x: f32[16], y: f32[16]):
            for i in seq(0, 16):
                y[i] = fmaxf(i, x[i])

    with pytest.raises(
        TypeError, match="expected argument 2 to be a real scalar value,"
    ):

        @proc
        def foo(x: f32[16], y: f32[16]):
            for i in seq(0, 16):
                y[i] = fmaxf(x[i], i)


def test_sigmoid(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = sigmoid(x[i] + y[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_sigmoid2():
    with pytest.raises(TypeError, match="expected 1 argument, got"):

        @proc
        def foo(x: i8[16], y: i8[16]):
            y[0] = sigmoid(x[0], x[0])

    with pytest.raises(
        TypeError, match="expected argument 1 to be a real scalar value,"
    ):

        @proc
        def foo(x: i8[16], y: i8[16]):
            y[0] = sigmoid(True)


def test_sqrt(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = sqrt(x[i] + y[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_sqrt2():
    with pytest.raises(TypeError, match="expected 1 argument, got"):

        @proc
        def foo(x: i8[16], y: i8[16]):
            y[0] = sqrt(x[0], x[0])

    with pytest.raises(
        TypeError, match="expected argument 1 to be a real scalar value,"
    ):

        @proc
        def foo(x: i8[16], y: i8[16]):
            y[0] = sqrt(True)


def test_select_error():
    @proc
    def foo(x: i8[16], y: f32[16], z: f64[16]):
        for i in seq(0, 16):
            z[i] = select(x[i] * 2, y[i], z[i], -x[i])

    with pytest.raises(TypeError, match="all extern arguments must have a same type"):
        c_file, h_file = compile_procs_to_strings([foo], "test.h")


def test_type_error():
    with pytest.raises(TypeError, match="expected scalar type"):

        @proc
        def foo(x: i8[16], y: f32[16], z: f64[16]):
            for i in seq(0, 16):
                z[i] = select(i * 2, y[i], z[i], -x[i])

    with pytest.raises(TypeError, match="expected 4 arguments, got 3"):

        @proc
        def foo(x: i8[16], y: f32[16], z: f64[16]):
            for i in seq(0, 16):
                z[i] = select(i * 2, y[i], z[i])


def test_select_fine():
    @proc
    def foo(x: i8[16], y: i8[16], z: i8[16]):
        for i in seq(0, 16):
            z[i] = select(0.0, y[i], z[i], -x[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")


def test_two():
    c = 2

    @proc
    def foo(a: f32):
        a = a + c

    with pytest.raises(SchedulingError, match="find: failed to find matches"):
        foo.find("a + c").parent()


def test_extern_find(golden):
    @proc
    def foo(a: f32):
        a = sin(a)

    assert golden == str(foo.find("sin(a)").parent())
