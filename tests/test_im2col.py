from __future__ import annotations        # make Python behave
import numpy as np                        # standard array library
import time                               # timers
import sys                                # add DSL library to the Python path
from ctypes import *
import os
import subprocess
import scipy.stats as st
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure
sys.path.append(sys.path[0]+"/.")
from .helper import *

# I'm going to define a 1-d version of a standard convolutional layer, like in CuDNN
# K - # of output channels
# C - # of input channels
# W - length of the input signal/tensor
# R - width of the filter kernel
def gen_conv1d():
    @proc
    def conv1d(K : size, C : size, W : size, R : size,
               w : R[K,C,R],
               x : R[C,W],
               res : R[K,W],
              ):
        # zero out the result memory
        for k_init in par(0,K):
            for i_init in par(0,W):
                res[k_init, i_init] = 0.0

        # do the convolution
        for k in par(0,K):
            for c in par(0,C):
                for i in par(0,W):
                    for r in par(0,R):
                        if 0 <= i-r:
                            res[k,i] += w[k,c,r] * x[c,i-r]
    return conv1d

def test_im2col():
    conv1d = gen_conv1d()

    # Let's start applying scheduling
    im2col_conv = conv1d.rename('im2col_conv')
    im2col_conv = im2col_conv.reorder('i','r')
    im2col_conv = im2col_conv.bind_expr('y','x[c, i-r]')

    # next, we can start to lift that allocation
    # up and out of the loop
    im2col_conv = im2col_conv.lift_alloc('y:R', 5)

    # Then, we can fission the loop correspondingly,
    # separating what is now a data-marshalling statement from
    # the actual compute statement in two subsequent
    # loop nests via fissioning
    im2col_conv = im2col_conv.fission_after('y[c,r,i] = _',5)

    # Now, in order to expose these two parts of the computation as
    # re-usable sub-procedures, we want a way to factor them out.
    im2col_conv, im2col = im2col_conv.factor_out_stmt('im2col', 'for c in _: _')
    im2col_conv, matmul = im2col_conv.factor_out_stmt('matmul', 'for k in _: _')

    # Given this factoring, we can then proceed
    # to schedule these sub-procedures themselves.
    tiled_matmul =      (matmul.rename('tiled_matmul')
                         # split the loops we want to tile together
                         .reorder('r','i')
                         .split('k',8,['khi','klo'], cut_tail=True)
                         .reorder('klo #1','c').reorder('klo #1','i')
                         .split('c #1',8,['chi','clo'], cut_tail=True)
                         .reorder('clo #1','i').reorder('clo #1','klo')
                         .split('i #1', 8, ['ihi','ilo'], cut_tail=True)
                         .reorder('ilo #1','klo').reorder('ilo #1','clo'))

    # We can invoke another scheduling directive
    # to change which version of the matmul gets scheduled
    im2col_conv = im2col_conv.call_eqv(tiled_matmul, 'matmul(_,_,_,_,_,_,_)')
"""
    filename = "test_im2col"

    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(im2col_conv))
    f_pretty.close()

    im2col.compile_c(directory, filename)
"""
