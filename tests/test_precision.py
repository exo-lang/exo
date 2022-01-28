from __future__ import annotations

import numpy as np
import pytest

from SYS_ATL import proc, Procedure
from .helper import nprand, nparray


# ------- Precision casting tests ------

def gen_good_prec1():
    @proc
    def good_prec1(n: size, m: size,
                   x: f32[n, m], y: f32[n, m], res: f64[n, m]):
        for i in par(0, n):
            rloc: f64[m]
            xloc: f32[m]
            yloc: f32[m]
            for j in par(0, m):
                xloc[j] = x[i, j]
            for j in par(0, m):
                yloc[j] = y[i, j]
            for j in par(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0, m):
                res[i, j] = rloc[j]

    return good_prec1


# Binop on different precision
def gen_bad_prec1():
    @proc
    def bad_prec1(n: size, m: size,
                  x: f32[n, m], y: i8[n, m], res: f64[n, m]):
        for i in par(0, n):
            rloc: f64[m]
            xloc: f32[m]
            yloc: i8[m]
            for j in par(0, m):
                xloc[j] = x[i, j]
            for j in par(0, m):
                yloc[j] = y[i, j]
            for j in par(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0, m):
                res[i, j] = rloc[j]

    return bad_prec1


def test_good_prec1(compiler):
    good_prec1 = gen_good_prec1()
    assert isinstance(good_prec1, Procedure)

    fn = compiler.compile(good_prec1)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    res = nprand(size=x.shape, typ=np.float64)

    fn(None, *x.shape, x, y, res)

    np.testing.assert_almost_equal(res, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], typ=np.float64), decimal=6)


def gen_dot():
    @proc
    def dot(m: size, x: f32[m], y: f32[m], r: f32):
        r = 0.0
        for i in par(0, m):
            r += x[i] * y[i]

    return dot


def gen_good_prec2(dot):
    @proc
    def hoge(n: size, x: f32[n], y: f32[n]):
        xy: f32
        dot(n, x, y, xy)

    return hoge


def gen_bad_prec2(dot):
    @proc
    def hoge(n: size, x: i8[n], y: i8[n]):
        xy: f32
        dot(n, x, y, xy)

    return hoge


def test_bad_prec1():
    bad_prec1 = gen_bad_prec1()

    with pytest.raises(TypeError,
                       match='Errors occurred during precision checking'):
        bad_prec1.c_code_str()


def test_good_prec2(golden):
    dot = gen_dot()
    good_prec2 = gen_good_prec2(dot)
    assert isinstance(good_prec2, Procedure)

    assert good_prec2.c_code_str() == golden


def test_bad_prec2():
    dot = gen_dot()
    bad_prec2 = gen_bad_prec2(dot)

    with pytest.raises(TypeError,
                       match='Errors occurred during precision checking'):
        bad_prec2.c_code_str()
