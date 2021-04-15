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

def gen_badpred():
    @proc
    def badpred(m: size):
        assert m + 1
        assert 10
        tmp : R[m]

    return badpred

def test_badpred():
    with pytest.raises(TypeError,
                       match='Errors occurred during typechecking'):
        badpred = gen_badpred()

def gen_badaccess():
    @proc
    def badaccess(m : size, x : R[m]):
        res : R[m]
        for i in par(0, m):
            res[i] = x[i, 3]

    return badaccess

def test_badaccess():
    with pytest.raises(TypeError,
                       match='expected access of variable'):
        badaccess = gen_badaccess()

def gen_badaccess2():
    @proc
    def badaccess2(m : size, x : R[m]):
        res : R[m,m]
        for i in par(0, m):
            res[i] = x[i]

    return badaccess2

def test_badaccess2():
    with pytest.raises(TypeError,
                       match='expected lvalue access of variable'):
        badaccess2 = gen_badaccess2()

def gen_badaccess3():
    @proc
    def badaccess3(m : size, n : index, x : R):
        n = x

    return badaccess3

def test_badaccess3():
    with pytest.raises(TypeError,
                       match='cannot assign/reduce to'):
        badaccess3 = gen_badaccess3()

def gen_badaccess4():
    @proc
    def badaccess4():
        x : R
        for i in par(0, 10):
            x = i

    return badaccess4

def test_badaccess4():
    with pytest.raises(TypeError,
                       match='cannot assign/reduce a'):
        badaccess4 = gen_badaccess4()

def gen_pass1():
    @proc
    def p(x : R[10]):
        pass
    return p
def gen_pass2():
    @proc
    def p():
        pass
    return p
def test_pass():
    p = gen_pass1()
    p = gen_pass2()

def gen_if1():
    @proc
    def hoge():
        if 4:
            pass
    return hoge
def test_if1():
    with pytest.raises(TypeError,
                       match='expected a bool expression'):
        if1 = gen_if1()

def gen_if2():
    @proc
    def hoge():
        if (1 == 0):
            pass
        else:
            x : R
            pass
    return hoge
def test_if2():
    if2 = gen_if2()

def gen_par1():
    @proc
    def hoge():
        for i in par(1, 2):
            pass
    return hoge
def test_par1():
    with pytest.raises(TypeError,
                       match='currently only supporting for-loops of the form'):
        par1 = gen_par1()

def gen_par2():
    @proc
    def hoge(x : R):
        for i in par(0, x):
            pass
    return hoge
def test_par2():
    with pytest.raises(TypeError,
                       match='expected loop bound of type \'int\' or type \'size\''):
        par2 = gen_par2()


def gen_call_pass1():
    @proc
    def hoge(y : R):
        pass
    return hoge
def gen_call_pass2(call_pass1):
    @proc
    def hoge():
        pass
        x : R
        call_pass1(3 + x)
    return hoge
def test_call_pass1():
    with pytest.raises(TypeError,
                       match='expected scalar type'):
        call_pass1 = gen_call_pass1()
        call_pass2 = gen_call_pass2(call_pass1)


def gen_call_read1():
    @proc
    def hoge(y : size):
        pass
    return hoge
def gen_call_read2(call_read1):
    @proc
    def hoge(x : size, r : size):
        call_read1(r + x)
    return hoge
def test_call_read1():
    with pytest.raises(TypeError,
                       match='expected size arguments to be simply variables or constants for now'):
        call_read1 = gen_call_read1()
        call_read2 = gen_call_read2(call_read1)


def gen_call_int_read1():
    @proc
    def hoge(y : size):
        pass
    return hoge
def gen_call_int_read2(call_int_read1):
    @proc
    def hoge():
        call_int_read1(3 + 2)
    return hoge
def test_call_int_read1():
    with pytest.raises(TypeError,
                       match='expected size arguments to be simply variables or constants for now'):
        call_int_read1 = gen_call_int_read1()
        call_int_read2 = gen_call_int_read2(call_int_read1)


def gen_call_read_size1():
    @proc
    def hoge(y : size):
        pass
    return hoge
def gen_call_read_size2(call_read_size1):
    @proc
    def hoge(x : R):
        call_read_size1(x)
    return hoge
def test_call_read_size1():
    with pytest.raises(TypeError,
                       match='expected argument of '):
        call_read_size1 = gen_call_read_size1()
        call_read_size2 = gen_call_read_size2(call_read_size1)


def gen_call_index_read1():
    @proc
    def hoge(y : index):
        pass
    return hoge
def gen_call_index_read2(call_index_read1):
    @proc
    def hoge(x : R):
        call_index_read1(x)
    return hoge
def test_call_index_read1():
    with pytest.raises(TypeError,
                       match='expected index-type expression, but got type'):
        call_index_read1 = gen_call_index_read1()
        call_index_read2 = gen_call_index_read2(call_index_read1)


def gen_call_tensor1_read1():
    @proc
    def hoge(n : size, m : size, y : F64[n,m]):
        pass
    return hoge
def gen_call_tensor1_read2(call_tensor1_read1):
    @proc
    def hoge(n : size, m : size, x : R[m, n, 10]):
        call_tensor1_read1(n, m, x)
    return hoge
def test_call_tensor1_read1():
    with pytest.raises(TypeError,
                       match='expected argument of type '):
        call_tensor1_read1 = gen_call_tensor1_read1()
        call_tensor1_read2 = gen_call_tensor1_read2(call_tensor1_read1)


def gen_call_tensor2_read1():
    @proc
    def hoge(y : F64):
        pass
    return hoge
def gen_call_tensor2_read2(call_tensor2_read1):
    @proc
    def hoge():
        y : R
        x : R
        call_tensor2_read1(x + y)
    return hoge
def test_call_tensor2_read1():
    with pytest.raises(TypeError,
                       match='expected scalar arguments to be simply variable names for now'):
        call_tensor2_read1 = gen_call_tensor2_read1()
        call_tensor2_read2 = gen_call_tensor2_read2(call_tensor2_read1)


def gen_const_bool():
    @proc
    def hoge(x : R):
        x = True
    return hoge
def test_const_bool():
    with pytest.raises(TypeError,
                       match='literal of unexpected type'):
        const_bool = gen_const_bool()


def gen_usub():
    @proc
    def hoge(x : R):
        x = -x
    return hoge
def test_usub():
    usub = gen_usub()


def gen_usub2():
    @proc
    def hoge(x : R[1]):
        x = -x
    return hoge
def test_usub2():
    with pytest.raises(TypeError,
                       match='cannot negate expression of type '):
        usub2 = gen_usub2()


def gen_binop1():
    @proc
    def hoge(x : R[1]):
        x = -x + 3.0
    return hoge
def test_binop1():
    with pytest.raises(TypeError,
                       match='cannot negate expression of type '):
        binop1 = gen_binop1()


def gen_binop2():
    @proc
    def hoge():
        if (1 == 1) and 3:
            pass
    return hoge
def test_binop2():
    with pytest.raises(TypeError,
                       match='expected \'bool\' argument to logical op'):
        binop2 = gen_binop2()


def gen_binop3():
    @proc
    def hoge():
        if 3 and (1 == 1):
            pass
    return hoge
def test_binop3():
    with pytest.raises(TypeError,
                       match='expected \'bool\' argument to logical op'):
        binop3 = gen_binop3()


def gen_binop4():
    @proc
    def hoge():
        if ((0 == 1) == (1 == 1)):
            pass
    return hoge
def test_binop4():
    with pytest.raises(TypeError,
                       match='using \"==\" for boolean not supported. Use \"and\" instead'):
        binop4 = gen_binop4()


def gen_binop5():
    @proc
    def hoge():
        if (1 < (1 == 1)):
            pass
    return hoge
def test_binop5():
    with pytest.raises(TypeError,
                       match='expected \'index\' or \'size\' argument to comparison op'):
        binop5 = gen_binop5()


def gen_binop6():
    @proc
    def hoge():
        if ((1 == 1) < 0):
            pass
    return hoge
def test_binop6():
    with pytest.raises(TypeError,
                       match='expected \'index\' or \'size\' argument to comparison op'):
        binop6 = gen_binop6()


def gen_binop7():
    @proc
    def hoge(x : R):
        x = x + 8
    return hoge
def test_binop7():
    with pytest.raises(TypeError,
                       match='expected scalar type'):
        binop7 = gen_binop7()


def gen_binop8():
    @proc
    def hoge(x : R):
        x = x % 8.0
    return hoge
def test_binop8():
    with pytest.raises(TypeError,
                       match='cannot compute modulus of'):
        binop8 = gen_binop8()


def gen_binop9():
    @proc
    def hoge(x : F64):
        x = x + 8.0
    return hoge
def test_binop9():
    binop9 = gen_binop9()


def gen_binop10():
    @proc
    def hoge(x : INT8):
        x = x + 8.0
    return hoge
def test_binop10():
    binop10 = gen_binop10()


def gen_binop11():
    @proc
    def hoge(x : INT8):
        x = (1 == 0) + (0 == 1)
    return hoge
def test_binop11():
    with pytest.raises(TypeError,
                       match='cannot perform arithmetic on \'bool\' values'):
        binop11 = gen_binop11()


def gen_binop12():
    @proc
    def hoge(x : size, y : size):
        if ((x / y) > 0):
            pass
    return hoge
def test_binop12():
    with pytest.raises(TypeError,
                       match='cannot divide or modulo by a non-constant value'):
        binop12 = gen_binop12()


def gen_binop13():
    @proc
    def hoge(x : size, y : size):
        if ((x / -3) > 0):
            pass
    return hoge
def test_binop13():
    with pytest.raises(TypeError,
                       match='cannot divide or modulo by a non-constant value'):
        binop13 = gen_binop13()


def gen_binop14():
    @proc
    def hoge(x : size, y : size):
        if ((4 * x) > 0):
            pass
    return hoge
def test_binop14():
    binop14 = gen_binop14()


def gen_binop15():
    @proc
    def hoge(x : size, y : size):
        if ((y * x) > 0):
            pass
    return hoge
def test_binop15():
    with pytest.raises(TypeError,
                       match='cannot multiply two non-constant indexing/sizing expressions'):
        binop15 = gen_binop15()

def gen_dot_bad():
    @proc
    def dot(m: size, x : R[1,1] , y : R[m] ):
        huga : R
        pass

    return dot

def gen_proj_bad(dot):
    @proc
    def proj(n : size, x : R[100, 10, 1], y : R[10, n]):
        dot(n, x[1], y[0])

    return proj

def test_proj_bad():
    with pytest.raises(TypeError,
                       match='type-shape of calling argument may not equal the required type-shape'):
        dot  = gen_dot_bad()
        proj = gen_proj_bad(dot)
