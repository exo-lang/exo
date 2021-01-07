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
from helper import *

def gen_dot():
    @proc
    def dot(m: size, x : R[m] @ IN, y : R[m] @ IN, r : R @ OUT):
        z = 0.0
        for i in par(0, m):
            z += x[i]*y[i]

    return dot

def gen_proj(dot):
    # implement proj x onto y
    # i.e. vector math (x . y) / (y . y) * y
    # because...
    #   (x.y)/|y|  * (y /|y|) = (x.y)/(|y|^2) * y = (x.y)/(y.y) * y
    @proc
    def proj(n : size, x : R[n] @ INOUT, y : R[n] @ IN):
        xy : R
        y2 : R
        dot(n, x, y, xy)
        dot(n, y, y, y2)
        s : R
        s = xy / y2
        for i in par(0,n):
            x[i] = s * y[i]

    return proj

#@pytest.mark.skip(reason="working on function call!")
def test_normalize():
    dot  = gen_dot()
    proj = gen_proj(dot)

    assert type(dot) is Procedure
    assert type(proj) is Procedure

    filename = "test_proj"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(dot))
    f_pretty.close()

    proj.compile_c(directory, filename)

"""
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
"""
