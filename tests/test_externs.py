from __future__ import annotations

import pytest

from exo import proc, DRAM, Procedure, config, compile_procs_to_strings
from exo.libs.externs import *


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


def test_sin(golden, compiler):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = sin(x[i] * 2)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_sin(golden, compiler):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = sin(x[i] * 2)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


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


def test_fmaxf(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = fmaxf(x[i], y[i] * 2)

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_sigmoid(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = sigmoid(x[i] + y[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)


def test_sqrt(golden, compiler):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = sqrt(x[i] + y[i])

    c_file, h_file = compile_procs_to_strings([foo], "test.h")
    assert c_file + h_file == golden

    compiler.compile(foo)
