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

    tensor.compile_c(TMP_DIR, filename)

# -------- Support partial windowing call ------
def gen_dot():
    @proc
    def dot(m: size, x : [f32][m] , y : [f32][m] , r : f32 ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]

    return dot


def gen_proj(dot):
    @proc
    def proj(n : size, m : size, x : f32[n,m], y : f32[m,n]):
        xy : f32
        y2 : f32
        dot(m, x[1,:], y[:,2], xy)
        dot(m, y[:,3], y[:,3], y2)
        mv_gemmini(x[i:i+16,j:j+16])
        # WindowExpr( sym base, w_access *idx ) -- oh wait, this is broken
        # UAST
        # w_access = Interval( expr? lo, expr? hi )
        #          | Point( expr pt )
        # LoopIR
        # w_access = Interval( expr lo, expr hi )
        #          | Point( expr pt )
        #
        s : f32
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

    proj.compile_c(TMP_DIR, filename)

def gen_assign():
    @proc
    def assign(n : size, m : size, x : f32[n, m]):
        #y : f32[m]
        # WindowStmt( sym lhs, WindowExpr window_e )
        y = x[:, 0]
        # y
        # y --> (float *)
        # y : R[n] windowed from (x : R[n,m]) by x[0:n,0]
        # y : R[n,1] windowed from (x : R[n,m]) by x[0:n,0:1]
        #
        # y : R[n]    ----    i.e. y is a tensor of R-values and dim (n)
        #               --> <R>*
        # y : [R][n,m] ----- i.e. y is a window of R-values and dim (n)
        #               --> struct win_R_2 { <R>* data; int strides[2]; }
        # [R][n,m]
        # i8 f32
        #  R  f32
        #
        # ____ = z : R[n,m] windowed from (x : R[n,m,p]) by x[:,:,3]
        # y : R[n] windowed from (_____) by z[0:n,0]
        #
        #   (x[ilo:ihi, jlo:jhi, 3])[iilo:iihi, 5]
        #
        #   x[ ilo+iilo:ilo+iihi, jlo+5, 3 ]
        #
        # WindowType = ( type base_type, expr* dims,
        #                TensorType orig_tensor,
        #                WindowExpr window )
        #
        #   hi - lo : index
        #
        #   (d0, d1, d2)     --> strides = (d1*d2, d2, 1)
        #   [:,1,:]
        #                    --> strides = (d1*d2,1)
        #
        # --> float *y = x;
        # --> int y_stride = m;
        # --> what about strides?
        y[2] = 3.0
        # --> y[ (2)*m ] = 3.0;
        z : f32[3]
        z = y[0 : 3]
    return assign

@pytest.mark.skip
def test_assign():
    assign = gen_assign()
    filename = "test_assign"
    assign.compile_c(TMP_DIR, filename)
