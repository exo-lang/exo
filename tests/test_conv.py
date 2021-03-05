from __future__ import annotations
import pytest
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
import scipy.stats as st
from PIL import Image
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure
sys.path.append(sys.path[0]+"/.")
from .helper import *

def gen_conv1d():
    @proc
    def conv1d(n: size, m: size, r: size,
               x: R[n], w: R[m], res: R[r]):
        for i in par(0, r):
            res[i] = 0.0
        for i in par(0, r):
            for j in par(0, n):
                if j < i+1 and j >= i-m+1:
                    res[i] += x[j]*w[i-j]

    return conv1d

def test_conv1d():
    conv1d = gen_conv1d()

    assert type(conv1d) is Procedure
    filename = "test_conv1d"
    conv1d.compile_c(directory, filename)

    n_size = 5
    m_size = 3
    r_size = n_size + m_size - 1
    x = nparray([0.2, 0.5, -0.4, 1.0, 0.0])
    w = nparray([0.6, 1.9, -2.2])
    res = nprand(size=(r_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(directory, filename)
    test_lib.conv1d(c_int(n_size), c_int(m_size),
                    c_int(r_size), cvt_c(x), cvt_c(w), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(r_size,))
    np.testing.assert_almost_equal(res_c, nparray(
        [0.12, 0.68, 0.27, -1.26, 2.78, -2.2, 0]))
