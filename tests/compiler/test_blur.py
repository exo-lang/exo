from __future__ import annotations
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
from PIL import Image
import scipy.stats as st
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure
from .helper import *

def gen_blur():
    @proc
    def blur(n: size, m: size, k_size: size,
             image: R[n, m] @ IN, kernel: R[k_size, k_size] @ IN, res: R[n, m] @ OUT):
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

def test_blur():
    blur = gen_blur()
    assert type(blur) is Procedure

    filename = "compiler_test_simple_blur"

    # Write pretty printing to a file
    f_header = open(os.path.join(directory, filename + "_pretty.py"), "w")
    f_header.write(str(blur))
    f_header.close()

    blur.compile_c(directory, filename)

    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(directory, filename)
    test_lib.blur(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(directory + filename + '_out.png')


def test_blur_split():
    @proc
    def blur_split(n: size, m: size, k_size: size,
             image: R[n, m] @ IN, kernel: R[k_size, k_size] @ IN, res: R[n, m] @ OUT):
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

    assert type(blur_split) is Procedure
    filename = "test_compiler_blur_split"
    blur_split.compile_c(directory, filename)

    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(directory, filename)
    test_lib.blur_split(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(directory + filename + '_out.png')

def test_sched_blur():
    blur        = gen_blur()
    orig_blur   = blur

    blur = blur.split('j',4,['j1','j2']) # This should just be equivalent to test_blur_split
    blur = blur.split('i[2]',4,['i1','i2'])
    blur = blur.reorder('i','j1[1]')
    blur = blur.reorder('k','l')

    # TODO: Should compare new and original IR
    assert type(blur) is Procedure
    filename = "test_compiler_sched_blur"

    # Write pretty printing to a file
    f_header = open(os.path.join(directory, filename + "_pretty.py"), "w")
    f_header.write(str(blur))
    f_header.close()
    
    blur.compile_c(directory, filename)

    n_size = image.shape[0]
    m_size = image.shape[1]
    k_size = 5
    kernel = gkern(k_size,1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(directory, filename)
    test_lib.blur(c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(image), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(directory + filename + '_out.png')
