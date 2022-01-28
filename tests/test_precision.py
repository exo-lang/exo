from __future__ import annotations

import os
from ctypes import POINTER, c_int

import numpy as np
import pytest

from SYS_ATL import proc, Procedure
from .helper import TMP_DIR, nprand, generate_lib, nparray, cvt_c


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


def test_good_prec1():
    good_prec1 = gen_good_prec1()
    assert isinstance(good_prec1, Procedure)

    filename = "test_good_prec1"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(good_prec1))
    f_pretty.close()

    good_prec1.compile_c(TMP_DIR, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size), typ=np.float64)
    res_c = cvt_c(res, typ=np.float64)

    test_lib = generate_lib(filename)
    test_lib.good_prec1(POINTER(c_int)(), c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    good_prec1.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], typ=np.float64), decimal=4)


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
    with pytest.raises(TypeError,
                       match='Errors occurred during precision checking'):
        bad_prec1 = gen_bad_prec1()
        filename = "test_bad_prec1"
        bad_prec1.compile_c(TMP_DIR, filename)


def test_good_prec2():
    dot = gen_dot()
    good_prec2 = gen_good_prec2(dot)
    assert isinstance(good_prec2, Procedure)

    filename = "test_good_prec2"

    good_prec2.compile_c(TMP_DIR, filename)


def test_bad_prec2():
    with pytest.raises(TypeError,
                       match='Errors occurred during precision checking'):
        dot = gen_dot()
        bad_prec2 = gen_bad_prec2(dot)
        filename = "test_bad_prec2"
        bad_prec2.compile_c(TMP_DIR, filename)
