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

def gen_conv1d():
    # K is the # of output channels,
    # C is the # of input channels,
    # W is the "image" width,
    # R is the filter width
    @proc
    def conv1d(K: size, C : size, W: size, R: size,
               w : R[K,C,R] @ IN, x: R[C,W] @ IN, res: R[K,W] @ OUT):
        # zero-out the result tensor
        for k in par(0,K):
            for i in par(0, W):
                res[k,i] = 0.0

        # do the convolution
        for k in par(0,K):
            for c in par(0,C):
                for i in par(0,W):
                    for r in par(0,R):
                        if 0 <= i-r:
                            res[k,i] += x[c,i-r] * w[k,c,r]

    return conv1d

"""
Ideas for transforming this to use im2col...


# V0
for k in par(0,K):
    for c in par(0,C):
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    res[k,i] += x[c,i-r] * w[k,c,r]

# V1 (factor out sub-expression and name it)
for k in par(0,K):
    for c in par(0,C):
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    y : R  |--->   y : R[C,W,R]  && [ y |-> y[c,i,r] ]
                    y = x[c,i-r]
                    res[k,i] += y * w[k,c,r]

# break-out intermediate expression as its own temporary
# over-allocate memory and subst indexing exprs
# hoist memory allocation

# fission statements inside if
# fission statements inside loop (conditions?)

# V2
for k in par(0,K):
    for c in par(0,C):
        for i in par(0,W):
            y : R
            for r in par(0,R):
                if 0 <= i-r:
                    y = x[c,i-r]
                    res[k,i] += y * w[k,c,r]

# V3
for k in par(0,K):
    for c in par(0,C):
        y : R[W,R]
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    y[i,r] = x[c,i-r]
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    res[k,i] += y[i,r] * w[k,c,r]

# V4
for k in par(0,K):
    y : R[C,W,R]
    for c in par(0,C):
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    y[c,i,r] = x[c,i-r]
    for c in par(0,C):
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    res[k,i] += y[c,i,r] * w[k,c,r]

# V5 (note no dependence on the loop variable)
y : R[C,W,R]
for c in par(0,C):
    for i in par(0,W):
        for r in par(0,R):
            if 0 <= i-r:
                y[c,i,r] = x[c,i-r]
for k in par(0,K):
    for c in par(0,C):
        for i in par(0,W):
            for r in par(0,R):
                if 0 <= i-r:
                    res[k,i] += y[c,i,r] * w[k,c,r]

# V6 (re-orderings)
y : R[C,R,W]
for c in par(0,C):
    for r in par(0,R):
        for i in par(0,W):
            if 0 <= i-r:
                y[c,r,i] = x[c,i-r]
for k in par(0,K):
    for c in par(0,C):
        for r in par(0,R):
            for i in par(0,W):
                if 0 <= i-r:
                    res[k,i] += y[c,r,i] * w[k,c,r]



im2col, conv1d = conv1d.factor_out("for c in _: _")

conv1d = conv1d.abstract(im2col, "for c in _: _")


# V6 (re-orderings)
def 1dconv(...):
    ...

    y : R[C,R,W]

    im2col( ... )

    matmul( ... )

def im2col( ... )
    for c in par(0,C):
        for r in par(0,R):
            for i in par(0,W):
                if 0 <= i-r:
                    y[c,r,i] = x[c,i-r]

def matmul(K,C,R,W, w, y, res):
    for k in par(0,K):
        for c in par(0,C):
            for r in par(0,R):
                for i in par(0,W):
                    if 0 <= i-r:
                        res[k,i] += y[c,r,i] * w[k,c,r]


matmul.make_instruction(" GEMM_BLAS(K,C,R,W, w + 32, );")



@proc
def GEMM_mvin(
    N : size, M : size,
    src : R[N][M] @ DRAM,
    dst : R[16][16] @ GEMM,
    src_i : index,
    src_j : index
):
    for i in par(0,16):
        for j in par(0,16):
            dst[i,j] = src[ src_i + i, src_j + j ]

GEMM_mvin = GEMM_mvin.make_instruction(
    "gemmini_mvin(src + src_i * M, );"
)

@proc
def GEMM_mvout(...):
    ...

@proc
def GEMM_matmul(...):
    ...


def tile_loop_for_GEMMINI(p, loop_pattern, ...):
    ...



try:
    tile_loop_for_GEMMINI(...)
catch SchedulingError:
    # try other scheduling method here

























"""
