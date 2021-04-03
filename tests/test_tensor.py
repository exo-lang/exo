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


# ------- Tensor expression test ------

def gen_tensor():
    @proc
    def tensor(n : size, m : size,
               x : R[n+m], y: R[n+m], res : R[n+m]):
        for i in par(0,n):
            res[i] = x[i] + y[i]
        for j in par(0,m):
            res[n+j] = x[n+j] + y[n+j]

    return tensor
@pytest.mark.skip
def test_tensor():
    tensor = gen_tensor()
    assert type(tensor) is Procedure

    filename = "test_tensor"

    tensor.compile_c(directory, filename)

# -------- Support partial windowing call ------
def gen_dot():
    @proc
    def dot(m: size, x : F32[m] , y : F32[m] , r : F32 ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]

    return dot

def gen_proj(dot):
    @proc
    def proj(n : size, m : size, x : F32[n,m], y : F32[m,n]):
        xy : F32
        y2 : F32
        dot(n, x[1,:], y[:,2], xy)
        dot(n, y[:,3], y[:,3], y2)
        s : F32
        s = xy / y2
        for i in par(0,n):
            x[i] = s * y[i]

    return proj

@pytest.mark.skip
def test_normalize():
    dot  = gen_dot()
    proj = gen_proj(dot)

    assert type(dot) is Procedure
    assert type(proj) is Procedure

    filename = "test_proj_partial"

    proj.compile_c(directory, filename)
