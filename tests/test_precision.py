from __future__ import annotations
import subprocess
import os
import ctypes
from ctypes import *
import numpy as np
import sys
from PIL import Image
import scipy.stats as st
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, instr, Procedure, DRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

def gen_f32():
    @instr("kurage")
    def f32(n : size, m : size,
               x : F32[n,m], y: F32[n,m] @ DRAM, res : F32[n,m] @ DRAM):
        for i in par(0,n):
            rloc : F32[m] @DRAM
            xloc : F32[m] @DRAM
            yloc : F32[m] @DRAM
            for j in par(0,m):
                xloc[j] = x[i,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return f32

def test_f32():
    f32 = gen_f32()
    assert type(f32) is Procedure

    filename = "test_f32"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(f32))
    f_pretty.close()

    f32.compile_c(directory, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)

    test_lib = generate_lib(directory, filename)
    test_lib.f32(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    f32.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]))

def gen_f64():
    @instr("kurage")
    def f64(n : size, m : size,
            x : F64[n,m], y: F64[n,m] @ DRAM, res : F64[n,m] @ DRAM):
        for i in par(0,n):
            rloc : F64[m] @DRAM
            xloc : F64[m] @DRAM
            yloc : F64[m] @DRAM
            for j in par(0,m):
                xloc[j] = x[i,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return f64

def test_f64():
    f64 = gen_f64()
    assert type(f64) is Procedure

    filename = "test_f64"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(f64))
    f_pretty.close()

    f64.compile_c(directory, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]], typ=np.float64)
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]], typ=np.float64)
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size), typ=np.float64)
    res_c = cvt_c(res, typ=np.float64)

    test_lib = generate_lib(directory, filename)
    test_lib.f64(c_int(n_size), c_int(
        m_size), cvt_c(x, np.float64), cvt_c(y, np.float64), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    f64.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], typ=np.float64))
