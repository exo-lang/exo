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

# --- Typechecking tests ---

def test_sin1():
    @proc
    def sin(x : f32):
        y : f32
        y = sin(x)

def test_sin2():
    @proc
    def sin(x : f32):
        y : f32
        if False:
            y = sin(x)

def test_bool1():
    @proc
    def bool(b : bool):
        assert b == True

        x : f32
        if b == True:
            x = 0.0

def test_bool2():
    @proc
    def bool(a : bool, b : bool):

        x : f32
        if a == b:
            x = 0.0

def test_bool3():
    @proc
    def bool(a : bool, b : bool):

        x : f32
        if False:
            x = 0.0


def test_bool4():
    @proc
    def bool(a : bool, b : bool):

        x : f32
        if a:
            x = 0.0

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
                       match='cannot assign/reduce a \'bool\' type value'):
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

@pytest.mark.skip()
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
    with pytest.raises(TypeError,
                       match='cannot divide or modulo by zero or a negative value'):
        @proc
        def hoge(x : size, y : size):
            if ((x / -3) > 0):
                pass
def test_binop13_2():
    with pytest.raises(TypeError,
                       match='cannot divide or modulo by zero or a negative value'):
        @proc
        def hoge(x : size, y : size):
            if ((x / 0) > 0):
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
