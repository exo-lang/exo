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
from SYS_ATL.libs.memories import MDRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

# --- Start Blur Test ---

def gen_blur():
    @proc
    def blur(n: size, m: size, k_size: size,
             image: R[n, m], kernel: R[k_size, k_size], res: R[n, m]):
        for i in par(0, n):
            for j in par(0, m):
                res[i, j] = 0.0
        for i in par(0, n):
            for j in par(0, m):
                for k in par(0, k_size):
                    for l in par(0, k_size):
                        if i+k >= 1 and i+k-n < 1 and j+l >= 1 and j+l-m < 1:
                            res[i, j] += kernel[k, l] * image[i+k-1, j+l-1]

    return blur

def test_simple_blur():
    blur = gen_blur()
    assert type(blur) is Procedure

    filename = "test_simple_blur"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(blur))
    f_pretty.close()

    # Compile blur to C file
    blur.compile_c(TMP_DIR, filename)

    # Execute
    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.blur(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(os.path.join(TMP_DIR, filename + '_out.png'))


def test_simple_blur_split():
    @proc
    def simple_blur_split(n: size, m: size, k_size: size,
             image: R[n, m], kernel: R[k_size, k_size], res: R[n, m]):
        for i in par(0, n):
            for j1 in par(0, m/2):
                for j2 in par(0,2):
                    res[i, j1*2+j2] = 0.0
        for i in par(0, n):
            for j1 in par(0, m/2):
                for j2 in par(0, 2):
                    for k in par(0, k_size):
                        for l in par(0, k_size):
                            if i+k >= 1 and i+k-n < 1 and j1*2+j2+l >= 1 and j1*2+j2+l-m < 1:
                                res[i, j1*2+j2] += kernel[k, l] * image[i+k-1, j1*2+j2+l-1]

    assert type(simple_blur_split) is Procedure
    filename = "test_simple_blur_split"
    simple_blur_split.compile_c(TMP_DIR, filename)

    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.simple_blur_split(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(os.path.join(TMP_DIR, filename + '_out.png'))

def test_split_blur():
    blur        = gen_blur()
    orig_blur   = blur

    blur = blur.split('j',4,['j1','j2'])
    blur = blur.split('i#1',4,['i1','i2'])

    assert type(blur) is Procedure
    filename = "test_split_blur"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(blur))
    f_pretty.close()

    blur.compile_c(TMP_DIR, filename)
    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.blur(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(os.path.join(TMP_DIR, filename + '_out.png'))

def test_reorder_blur():
    blur        = gen_blur()
    orig_blur   = blur

    blur = blur.reorder('k','l')
    blur = blur.reorder('i','j')

    assert type(blur) is Procedure
    filename = "test_reorder_blur"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(blur))
    f_pretty.close()

    blur.compile_c(TMP_DIR, filename)

    #Execute
    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.blur(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(os.path.join(TMP_DIR, filename + '_out.png'))

def test_unroll_blur():
    blur        = gen_blur()
    orig_blur   = blur

#    blur = blur.split('i',6,['i','iunroll']).simpler_unroll('iunroll')
    blur = blur.split('j',4,['j1','j2'])
    blur = blur.unroll('j2')

    assert type(blur) is Procedure
    filename = "test_unroll_blur"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(blur))
    f_pretty.close()

    blur.compile_c(TMP_DIR, filename)
    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.blur(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(os.path.join(TMP_DIR, filename + '_out.png'))

# --- End Blur Test ---


# --- conv1d test ---
def test_conv1d():
    @proc
    def conv1d(n: size, m: size, r: size,
               x: R[n], w: R[m], res: R[r]):
        for i in par(0, r):
            res[i] = 0.0
        for i in par(0, r):
            for j in par(0, n):
                if j < i+1 and j >= i-m+1:
                    res[i] += x[j]*w[i-j]

    assert type(conv1d) is Procedure
    filename = "test_conv1d"
    conv1d.compile_c(TMP_DIR, filename)

    n_size = 5
    m_size = 3
    r_size = n_size + m_size - 1
    x = nparray([0.2, 0.5, -0.4, 1.0, 0.0])
    w = nparray([0.6, 1.9, -2.2])
    res = nprand(size=(r_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.conv1d(c_int(n_size), c_int(m_size),
                    c_int(r_size), cvt_c(x), cvt_c(w), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(r_size,))
    np.testing.assert_almost_equal(res_c, nparray(
        [0.12, 0.68, 0.27, -1.26, 2.78, -2.2, 0]))

# ------- Nested alloc test for normal DRAM ------

def test_alloc_nest():
    @proc
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

    assert type(alloc_nest) is Procedure

    filename = "test_alloc_nest"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc_nest))
    f_pretty.close()

    # Write effect printing to a file
    f_effect = open(os.path.join(TMP_DIR, filename + "_effect.atl"), "w")
    f_effect.write(str(alloc_nest.show_effects()))
    f_effect.close()

    alloc_nest.compile_c(TMP_DIR, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)

    test_lib = generate_lib(filename)
    test_lib.alloc_nest(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    alloc_nest.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]))



# ------- Nested alloc test for custom malloc DRAM ------

def test_alloc_nest_malloc():
    @proc
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

    assert type(alloc_nest_malloc) is Procedure

    filename = "test_alloc_nest_malloc"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc_nest_malloc))
    f_pretty.close()

    alloc_nest_malloc.compile_c(TMP_DIR, filename)

    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)

    test_lib = generate_lib(filename)
    # Initialize custom malloc here
    test_lib.init_mem()
    test_lib.alloc_nest_malloc(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    alloc_nest_malloc.interpret(n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]))

