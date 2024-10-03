from __future__ import annotations

import pytest

from exo import proc, DRAM, Procedure, config, compile_procs_to_strings
from exo.libs.externs import *


def test_relu(golden):
    @proc
    def foo(x: f32[16]):
        for i in seq(0, 16):
            x[i] = relu(3.0)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_relu2(golden):
    @proc
    def foo(x: f32[16]):
        for i in seq(0, 16):
            x[i] = relu(x[i])

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_relu3(golden):
    @proc
    def foo(x: f32[16], y: f32[16], z: f32[16]):
        for i in seq(0, 16):
            z[i] = relu(x[i] + y[i])

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_relu4(golden):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = relu(3.0)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_sin(golden):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = sin(x[i] * 2)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_sin(golden):
    @proc
    def foo(x: i8[16]):
        for i in seq(0, 16):
            x[i] = sin(x[i] * 2)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_select(golden):
    @proc
    def foo(x: i8[16], y: i8[16], z: i8[16]):
        for i in seq(0, 16):
            z[i] = select(x[i] * 2, y[i], z[i] + y[i], -x[i])

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_expf(golden):
    @proc
    def foo(x: i8[16], y: i8[16]):
        for i in seq(0, 16):
            y[i] = expf(x[i] + y[i])

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_fmaxf(golden):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = fmaxf(x[i], y[i] * 2)

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_sigmoid(golden):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = sigmoid(x[i] + y[i])

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden


def test_sqrt(golden):
    @proc
    def foo(x: f32[16], y: f32[16]):
        for i in seq(0, 16):
            y[i] = sqrt(x[i] + y[i])

    c_file, _ = compile_procs_to_strings([foo], "test.h")
    assert c_file == golden
