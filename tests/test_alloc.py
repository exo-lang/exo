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
from SYS_ATL import proc, instr, Procedure, DRAM, MDRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest



# ------- Nested alloc test for normal DRAM ------

def gen_alloc_nest():
    @instr("kurage")
    def alloc_nest(n : size, m : size,
                   x : R[n,m], y: R[n,m] @ DRAM, res : R[n,m] @ DRAM):
        for i in par(0,n):
            rloc : R[m] @DRAM
            xloc : R[m] @DRAM
            yloc : R[m] @DRAM
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



# ------- Nested alloc test for custom malloc DRAM ------

def gen_alloc_nest_malloc():
    @instr("kurage")
    def alloc_nest_malloc(n : size, m : size,
                   x : R[n,m] @ MDRAM, y: R[n,m] @ MDRAM, res : R[n,m] @ MDRAM):
        for i in par(0,n):
            rloc : R[m] @MDRAM
            xloc : R[m] @MDRAM
            yloc : R[m] @MDRAM
            for j in par(0,m):
                xloc[j] = x[i,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return alloc_nest_malloc

def test_alloc_nest_malloc():
    alloc_nest = gen_alloc_nest_malloc()
    assert type(alloc_nest) is Procedure

    filename = "test_alloc_nest_malloc"

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
    # Initialize custom malloc here
    test_lib.init_mem()
    test_lib.alloc_nest_malloc(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    alloc_nest.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]))




# ------- Effect check tests ---------

def gen_bad_access1():
    @proc
    def bad_access1(n : size, m : size,
                    x : R[n,m], y: R[n,m], res : R[n,m]):
        rloc : R[m]
        for i in par(0,m):
            xloc : R[m]
            yloc : R[m]
            for j in par(0,n):
                xloc[j] = x[i,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return bad_access1

def gen_bad_access2():
    @proc
    def bad_access2(n : size, m : size,
                   x : R[n,m], y: R[n,m] @ DRAM, res : R[n,m] @ DRAM):
        rloc : R[m]
        for i in par(0,n):
            xloc : R[m]
            yloc : R[m]
            for j in par(0,m):
                xloc[j] = x[i+1,j]
            for j in par(0,m):
                yloc[j] = y[i,j]
            for j in par(0,m):
                rloc[j] = xloc[j] + yloc[j-1]
            for j in par(0,m):
                res[i,j] = rloc[j]

    return bad_access2

def test_bad_access1():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        gen_bad_access1()

def test_bad_access2():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
        gen_bad_access2()
