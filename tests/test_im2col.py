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

"""
