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
from SYS_ATL import proc, Procedure
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

def gen_alloc_nest():
    @proc
    def alloc_nest(n : size, m : size,
                   x : R[n,m] @ IN, y: R[n,m] @ IN, res : R[n,m] @ OUT):
        rloc : R[m]
        for i in par(0,n):
            xloc : R[m]
            yloc : R[m]
            for j in par(0,m):
                xloc[j] = x[i,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return alloc_nest

def test_alloc_nest():
    alloc_nest = gen_alloc_nest()
    assert type(alloc_nest) is Procedure

    filename = "test_alloc_nest"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc_nest))
    f_pretty.close()

    alloc_nest.compile_c(directory, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)

    test_lib = generate_lib(directory, filename)
    test_lib.alloc_nest(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    alloc_nest.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]))


"""
@proc
GEMM_Load(y : R[...], i : index, x : R[...], j : index, n : size):
    for j in par(0,n):
        y[i + k] = x[j + k]
    => gemmini_extended_mvin(x + (i0_26)*DIM, y, DIM, DIM);
    GEMM_Load(y, i0, x, i0, 16)
    alloc1 = alloc1.inline('GEMM_Load')
"""
def gen_alloc1():
    @proc
    def alloc1( n : size, x : R[n] @ IN, y : R[n] @ OUT @ GEMM ):
        for i0 in par(0,n/16):
            if i0 == n/16-1:
                instr(GEMM_Load)
                for i1 in par(0,n%16):
                    y[i0] = x[i0*16+i1]
            else:
                instr(GEMM_Load)
                for i1 in par(0,16):
                    y[i0] = x[i0*16+i1]

    return alloc1

@pytest.mark.skip(reason="old instruction annotation deprecated")
def test_alloc1():
    alloc1 = gen_alloc1()
    assert type(alloc1) is Procedure

    filename = "test_alloc1"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc1))
    f_pretty.close()

    alloc1.compile_c(directory, filename)


def gen_alloc2():
    @proc
    def alloc2( n : size, x : R[n] @ IN, y : R[n] @ OUT @ GEMM ):
        for i0 in par(0,n/16-1):
            instr(GEMM_Load)
            for i1 in par(0,16):
                y[i0] = x[i0*16+i1]
        instr(GEMM_Load)
        for i1 in par(0,n%16):
            y[n/16-1] = x[(n/16-1)*16+i1]

    return alloc2

@pytest.mark.skip(reason="old instruction annotation deprecated")
def test_alloc2():
    alloc2 = gen_alloc2()
    assert type(alloc2) is Procedure

    filename = "test_alloc2"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc2))
    f_pretty.close()

    alloc2.compile_c(directory, filename)
