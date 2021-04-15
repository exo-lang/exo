from __future__ import annotations
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
import scipy.stats as st
import pytest
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure
sys.path.append(sys.path[0]+"/.")
from .helper import *

def gen_dot():
    @proc
    def dot(m: size, x : R[m] , y : R[m] , r : R ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]

    return dot

def gen_proj(dot):
    @proc
    def proj(n : size, x : R[10, n], y : R[10, n]):
        xy : R
        y2 : R
        dot(n, x[0], y[4], xy)
        dot(n, y[0], y[2], y2)
        s : R
        s = xy / y2
        for i in par(0,n):
            x[0,i] = s * y[0,i]

    return proj

def test_window():
    dot  = gen_dot()
    proj = gen_proj(dot)

    assert type(dot) is Procedure
    assert type(proj) is Procedure

    filename = "test_window_proj"

    proj.compile_c(directory, filename)

def gen_alloc_nest():
    @proc
    def alloc_nest(n : size, m : size,
                   x : R[n,m], y: R[n,m], res : R[1,n,m]):
        assert n > 1
        for i in par(0,n):
            rloc : R[n,m]
            xloc : R[n,m]
            yloc : R[n,m]
            for j in par(0,m):
                xloc[i,j] = x[i,j]
            for j in par(0,m):
                yloc[i-i,j] = y[i,j+0]
            for j in par(0,m):
                rloc[4-3,j] = xloc[i,j] + yloc[0,j]
            for j in par(0,m):
                res[0+0,i,j] = rloc[4-3,j]

    return alloc_nest

def test_alloc_nest():
    alloc_nest = gen_alloc_nest()
    assert type(alloc_nest) is Procedure

    filename = "test_window_alloc_nest"

    alloc_nest.compile_c(directory, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(1,n_size, m_size))
    res_c = cvt_c(res)

    test_lib = generate_lib(directory, filename)
    test_lib.alloc_nest(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(1,n_size, m_size))
    alloc_nest.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]]))

def gen_bad_alloc_nest():
    @proc
    def alloc_nest(n : size, m : size,
                   x : R[n,m], y: R[n,m], res : R[1,n,m]):
        for i in par(0,n):
            rloc : R[n,m]
            xloc : R[n,m]
            yloc : R[n,m]
            for j in par(0,m):
                xloc[i,j] = x[i,j]
            for j in par(0,m):
                yloc[i-i,j] = y[i,j+0]
            for j in par(0,m):
                rloc[4-5,j] = xloc[i,j] + yloc[0,j]
            for j in par(0,m):
                res[21,i,j] = rloc[4-3,j]

    return alloc_nest

def test_bad_alloc_nest():
    alloc_nest = gen_alloc_nest()
    assert type(alloc_nest) is Procedure

    filename = "test_window_bad_alloc_nest"

    alloc_nest.compile_c(directory, filename)
