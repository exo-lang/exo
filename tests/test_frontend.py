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
from SYS_ATL import proc, Procedure, DRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *

# ------- Effect check tests ---------
def test_bad_access1():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
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

def test_bad_access2():
    with pytest.raises(TypeError,
                       match='Errors occurred during effect checking'):
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

def test_bad_access3():
    with pytest.raises(TypeError,
                       match='x2 is read out-of-bounds'):
        @proc
        def foo():
            x2 : R[1]
            huga : R
            huga = x2[100]

# do sizes match

# something that works but is on the edge of working

# we added complex stuff about division and modulo
# is that working?

# are we testing a case of an else branch?

# what if effset.pred is None?

# size positivity checks

# making sure asserts are checked
# making sure asserts are used to prove other things

# make sure that effects are getting translated through windowing
# correctly

# check that windowing is always in-bounds
#   note checking above translation is maybe better done for data-races

# Data Race? Yes
# for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = x[i+1,i+1]

# Data Race? No
# for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = x[i+1,i]

# Data Race? No
# y = x[1:,:]
# for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = y[i,i]

# one big issue is aliasing in sub-procedure arguments
# def foo(n : size, x : [R][n,n], y : [R][n,n]):
#   for i in par(0,n):
#     if i+1 < n:
#         x[i,i] = y[i,i]

# stride assert

# test basic commutativity properties exhaustively in combinations
# R,W,+  R,W,+
#
# def foo():
#   x : R
#   for i in par(0,2):
#       x = 3
#       y = x

# for i in par(0, n):
#   x[2*i] = x[2*i+1]

# https://en.wikipedia.org/wiki/Jacobi_method
# red black gauss seidel
# https://www.cs.cornell.edu/~bindel/class/cs5220-s10/slides/lec14.pdf
# wavefront parallel

# ------- Window related tests ---------

def test_window():
    @proc
    def window(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ DRAM,
    ):
        assert n <= 16
        assert m <= 16

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    assert type(window) is Procedure

    filename = "test_window_window"
    window.compile_c(TMP_DIR, filename)

def test_stride_assert():
    @proc
    def stride_assert(
        n   : size,
        m   : size,
        src : [i8][n, m] @ DRAM,
        dst : [i8][n, 16] @ DRAM,
    ):
        assert n <= 16
        assert m <= 16
        assert stride(src, 1) == 1
        assert stride(dst, 0) == 16
        assert stride(dst, 1) == 1

        for i in par(0, n):
            for j in par(0, m):
                dst[i,j] = src[i,j]

    assert type(stride_assert) is Procedure

    filename = "test_window_stride_assert"
    stride_assert.compile_c(TMP_DIR, filename)

def test_window_stmt():
    @proc
    def window_stmt(n : size, m : size, x : f32[n, m]):
        y = x[:, 0]
        z : f32[n]
        for i in par(0, n):
            z[i] = y[i]

    assert type(window_stmt) is Procedure

    filename = "test_window_stmt"
    window_stmt.compile_c(TMP_DIR, filename)

def test_normalize():
    @proc
    def dot(m: size, x : [f32][m] , y : [f32][m] , r : f32 ):
        r = 0.0
        for i in par(0, m):
            r += x[i]*y[i]

    @proc
    def proj(n : size, m : size, x : f32[n,m], y : f32[m,n]):
        assert n > 4
        assert m > 4
        xy : f32
        y2 : f32
        dot(m, x[1,:], y[:,2], xy)
        dot(m, y[:,3], y[:,3], y2)

    assert type(dot) is Procedure
    assert type(proj) is Procedure
    filename = "test_window_proj"
    proj.compile_c(TMP_DIR, filename)



# --- Typechecking tests ---

def test_badpred():
    with pytest.raises(TypeError,
                       match='Errors occurred during typechecking'):
        @proc
        def badpred(m: size):
            assert m + 1
            assert 10
            tmp : R[m]

def test_badaccess():
    with pytest.raises(TypeError,
                       match='expected access of variable'):
        @proc
        def badaccess(m : size, x : R[m]):
            res : R[m]
            for i in par(0, m):
                res[i] = x[i, 3]

def test_badaccess2():
    with pytest.raises(TypeError,
                       match='expected lvalue access of variable'):
        @proc
        def badaccess2(m : size, x : R[m]):
            res : R[m,m]
            for i in par(0, m):
                res[i] = x[i]

def test_badaccess3():
    with pytest.raises(TypeError,
                       match='cannot assign/reduce to'):
        @proc
        def badaccess3(m : size, n : index, x : R):
            n = x

def test_badaccess4():
    with pytest.raises(TypeError,
                       match='cannot assign/reduce a'):
        @proc
        def badaccess4():
            x : R
            for i in par(0, 10):
                x = i

def test_pass():
    @proc
    def p(x : R[10]):
        pass
    return p

    @proc
    def p():
        pass
    return p

def test_if1():
    with pytest.raises(TypeError,
                       match='expected a bool expression'):
        @proc
        def hoge():
            if 4:
                pass

def test_if2():
    @proc
    def hoge():
        if (1 == 0):
            pass
        else:
            x : R
            pass

def test_par1():
    with pytest.raises(TypeError,
                       match='currently only supporting for-loops of the form'):
        @proc
        def hoge():
            for i in par(1, 2):
                pass

def test_par2():
    with pytest.raises(TypeError,
                       match='expected loop bound of type \'int\' or type \'size\''):
        @proc
        def hoge(x : R):
            for i in par(0, x):
                pass

def test_call_pass1():
    with pytest.raises(TypeError,
                       match='expected scalar type'):
        @proc
        def hoge(y : R):
            pass

        @proc
        def huga():
            pass
            x : R
            hoge(3 + x)

def test_call_read_size1():
    with pytest.raises(TypeError,
                       match='expected size arguments to have'):
        @proc
        def hoge(y : size):
            pass
        @proc
        def foo(x : R):
            hoge(x)

def test_call_index_read1():
    with pytest.raises(TypeError,
                       match='expected index-type expression, but got type'):
        @proc
        def hoge(y : index):
            pass

        @proc
        def foo(x : R):
            hoge(x)

def test_call_tensor1_read1():
    with pytest.raises(TypeError,
                       match='expected argument of type '):
        @proc
        def hoge(n : size, m : size, y : f64[n,m]):
            pass
        @proc
        def foo(n : size, m : size, x : R[m, n, 10]):
            hoge(n, m, x)

def test_call_tensor2_read1():
    with pytest.raises(TypeError,
                       match='expected scalar arguments to be simply variable names for now'):
        @proc
        def hoge(y : f64):
            pass
        @proc
        def foo():
            y : R
            x : R
            hoge(x + y)

def test_const_bool():
    with pytest.raises(TypeError,
                       match='literal of unexpected type'):
        @proc
        def hoge(x : R):
            x = True

def test_usub():
    @proc
    def hoge(x : R):
        x = -x

def test_usub2():
    with pytest.raises(TypeError,
                       match='cannot negate expression of type '):
        @proc
        def hoge(x : R[1]):
            x = -x

def test_binop1():
    with pytest.raises(TypeError,
                       match='cannot negate expression of type '):
        @proc
        def hoge(x : R[1]):
            x = -x + 3.0

def test_binop2():
    with pytest.raises(TypeError,
                       match='expected \'bool\' argument to logical op'):
        @proc
        def hoge():
            if (1 == 1) and 3:
                pass

def test_binop3():
    with pytest.raises(TypeError,
                       match='expected \'bool\' argument to logical op'):
        @proc
        def hoge():
            if 3 and (1 == 1):
                pass

def test_binop4():
    with pytest.raises(TypeError,
                       match='using \"==\" for boolean not supported.'):
        @proc
        def hoge():
            if ((0 == 1) == (1 == 1)):
                pass

def test_binop5():
    with pytest.raises(TypeError,
                       match='expected \'index\' or \'size\' argument to comparison op'):
        @proc
        def hoge():
            if (1 < (1 == 1)):
                pass

def test_binop6():
    with pytest.raises(TypeError,
                       match='expected \'index\' or \'size\' argument to comparison op'):
        @proc
        def hoge():
            if ((1 == 1) < 0):
                pass

def test_binop7():
    with pytest.raises(TypeError,
                       match='expected scalar type'):
        @proc
        def hoge(x : R):
            x = x + 8

def test_binop8():
    with pytest.raises(TypeError,
                       match='cannot compute modulus of'):
        @proc
        def hoge(x : R):
            x = x % 8.0

def test_binop9():
    @proc
    def hoge(x : f64):
        x = x + 8.0

def test_binop10():
    @proc
    def hoge(x : i8):
        x = x + 8.0

def test_binop11():
    with pytest.raises(TypeError,
                       match='cannot perform arithmetic on \'bool\' values'):
        @proc
        def hoge(x : i8):
            x = (1 == 0) + (0 == 1)

def test_binop12():
    with pytest.raises(TypeError,
                       match='cannot divide or modulo by a non-constant value'):
        @proc
        def hoge(x : size, y : size):
            if ((x / y) > 0):
                pass

def test_binop13():
    @proc
    def hoge(x : size, y : size):
        if ((x / -3) > 0):
            pass

def test_binop14():
    @proc
    def hoge(x : size, y : size):
        if ((4 * x) > 0):
            pass

def test_binop15():
    with pytest.raises(TypeError,
                       match='cannot multiply two non-constant indexing/sizing expressions'):
        @proc
        def hoge(x : size, y : size):
            if ((y * x) > 0):
                pass

def test_proj_bad():
    with pytest.raises(TypeError,
                       match='type-shape of calling argument may not equal the required type-shape'):
        @proc
        def dot(m: size, x : R[1,1] , y : R[m] ):
            huga : R
            pass
        @proc
        def proj(n : size, x : R[100, 10, 1], y : R[10, n]):
            dot(n, x[1], y[0])
