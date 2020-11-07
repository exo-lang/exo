from __future__ import annotations
import subprocess
import os
import ctypes
from ctypes import *
import numpy as np
import sys
import pytest
from PIL import Image
import scipy.stats as st
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_compiler import Compiler, run_compile
from SYS_ATL.LoopIR_interpreter import Interpreter
from SYS_ATL import proc, Procedure
from .helper import *

# Test 1 is add vector
#
#   add_vec( n : size, x : R[n], y : R[n], res : R[n]):
#       forall i = 0,n:
#           res[i] = x[i] + y[i]
#

def gen_add_vec_ir():
    n = Sym('n')
    x = Sym('x')
    y = Sym('y')
    res = Sym('res')
    i = Sym('i')

    src0 = null_srcinfo()

    ai = IR.AVar(i, src0)
    rhs = IR.BinOp('+', IR.Read(x, [ai], src0),
                   IR.Read(y, [ai], src0),
                   src0)
    s_a = IR.Assign(res, [ai], rhs, src0)
    an = IR.ASize(n, src0)
    loop = IR.ForAll(i, an, s_a, src0)

    return Proc('add_vec',
                [n],
                [(x, R[n], 'IN'),
                 (y, R[n], 'IN'),
                 (res, R[n], 'OUT')],
                [
                    loop
                ])


def test_add_vec_ir():
    TEST_1 = gen_add_vec_ir()
    filename = "test1"
    run_compile([TEST_1], directory, (filename + ".c"), (filename + ".h"))
    compile_so_cmd = ("clang -Wall -Werror -fPIC -O3 -shared " +
                      "-o " + directory + filename + ".so " +
                      directory + filename + ".c")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    abspath = os.path.dirname(os.path.abspath(filename))
    test_lib = ctypes.CDLL(abspath + '/' + directory + filename + ".so")
    x = nparray([3.0, 6.0, 9.0])
    y = nparray([1.0, 2.0, 3.0])
    a_size = 3
    res = nprand(size=a_size)
    res_c = cvt_c(res)
    test_lib.add_vec(c_int(a_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(a_size,))
    Interpreter(TEST_1, n=3, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res, [4, 8, 12])

# TEST 2 is alloc
#   alloc( n : size, x : R[n]):
#       float *ptr = (float*) malloc (n * sizeof(float));
#       forall i = 0,n:
#           ptr[i] = x[i];
#       free(ptr);


def gen_alloc_ir():
    n = Sym('n')
    x = Sym('x')
    ptr = Sym('ptr')
    i = Sym('i')

    src0 = null_srcinfo()

    # How to pass n to alloc?
    ma = IR.Alloc(ptr, R[n].typ, None, src0)
    ai = IR.AVar(i, src0)
    rhs = IR.Read(x, [ai], src0)
    s_a = IR.Assign(ptr, [ai], rhs, src0)
    an = IR.ASize(n, src0)
    loop = IR.ForAll(i, an, s_a, src0)
    seq = IR.Seq(ma, loop, src0)

    return Proc('alloc',
                [n],
                [(x, R[n], 'IN')],
                [
                    seq
                ])


def test_alloc_ir():
    TEST_2 = gen_alloc_ir()
    run_compile([TEST_2], directory, "test_alloc.c", "test_alloc.h")

# TEST 3 is nested alloc
#   alloc_nest( n : size, m : size, x : R[n,m], y: R[n,m], res : R[n,m] ):
#       rloc : R[m]
#       forall i = 0,n:
#           xloc : R[m]
#           yloc : R[m]
#           forall j = 0,m:
#               xloc[j] = x[i,j]
#           forall j = 0,m:
#               yloc[j] = y[i,j]
#           forall j = 0,m:
#               rloc[j] = xloc[j] + yloc[j]
#           forall j = 0,m:
#               res[i,j] = rloc[j]


def gen_alloc_nest_ir():
    n = Sym('n')
    m = Sym('m')
    x = Sym('x')
    y = Sym('y')
    res = Sym('res')
    i = Sym('i')
    j1 = Sym('j1')
    j2 = Sym('j2')
    j3 = Sym('j3')
    j4 = Sym('j4')

    rloc = Sym('rloc')
    xloc = Sym('xloc')
    yloc = Sym('yloc')

    src0 = null_srcinfo()

    rloc_a = IR.Alloc(rloc, R[m].typ, None, src0)

    ai = IR.AVar(i, src0)
    aj1 = IR.AVar(j1, src0)
    aj2 = IR.AVar(j2, src0)
    aj3 = IR.AVar(j3, src0)
    aj4 = IR.AVar(j4, src0)

    xloc_a = IR.Alloc(xloc, R[m].typ, None, src0)
    yloc_a = IR.Alloc(yloc, R[m].typ, None, src0)
    seq_alloc = IR.Seq(xloc_a, yloc_a, src0)

#           forall j = 0,m:
#               xloc[j] = x[i,j]
    rhs_1 = IR.Read(x, [ai, aj1], src0)
    body_1 = IR.Assign(xloc, [aj1], rhs_1, src0)
    am = IR.ASize(m, src0)
    loop_1 = IR.ForAll(j1, am, body_1, src0)
    seq_1 = IR.Seq(seq_alloc, loop_1, src0)

#           forall j = 0,m:
#               yloc[j] = y[i,j]
    rhs_2 = IR.Read(y, [ai, aj2], src0)
    body_2 = IR.Assign(yloc, [aj2], rhs_2, src0)
    loop_2 = IR.ForAll(j2, am, body_2, src0)
    seq_2 = IR.Seq(seq_1, loop_2, src0)

#           forall j = 0,m:
#               rloc[j] = xloc[j] + yloc[j]
    rhs_3 = IR.BinOp('+', IR.Read(xloc, [aj3], src0),
                     IR.Read(yloc, [aj3], src0),
                     src0)
    body_3 = IR.Assign(rloc, [aj3], rhs_3, src0)
    loop_3 = IR.ForAll(j3, am, body_3, src0)
    seq_3 = IR.Seq(seq_2, loop_3, src0)

#           forall j = 0,m:
#               res[i,j] = rloc[j]
    rhs_4 = IR.Read(rloc, [aj4], src0)
    body_4 = IR.Assign(res, [ai, aj4], rhs_4, src0)
    loop_4 = IR.ForAll(j4, am, body_4, src0)
    seq_4 = IR.Seq(seq_3, loop_4, src0)

    an = IR.ASize(n, src0)
    loop = IR.ForAll(i, an, seq_4, src0)
    seq_top = IR.Seq(rloc_a, loop, src0)

    return Proc('alloc_nest',
                [n, m],
                [
                    (x, R[n, m], 'IN'),
                    (y, R[n, m], 'IN'),
                    (res, R[n, m], 'OUT')
                ],
                [
                    seq_top
                ])



def test_alloc_nest_ir():
    TEST_3 = gen_alloc_nest_ir()
    filename = "test_alloc_nest"
    run_compile([TEST_3], directory, (filename + ".c"), (filename + ".h"))
    compile_so_cmd = ("clang -Wall -Werror -fPIC -O3 -shared " +
                      "-o " + directory + filename + ".so " +
                      directory + filename + ".c")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    abspath = os.path.dirname(os.path.abspath(filename))
    test_lib = ctypes.CDLL(abspath + '/' + directory + filename + ".so")
    x = nparray([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]])
    y = nparray([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]])
    n_size = 2
    m_size = 3
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib.alloc_nest(c_int(n_size), c_int(
        m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    Interpreter(TEST_3, n=n_size, m=m_size, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res_c, nparray(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]]))


def gen_alloc():
    @proc
    def alloc(n: size, x: R[n] @ IN @ HEAP, res: R[n] @ OUT @ HEAP):
        ptr : R[n] @ HEAP
        for i in par(0,n/16):
            instr(GEMM_Load)
            for i2 in par(0,16):
                if i*16+i2 < n:
                    ptr[i] = x[i*16+i2]

    return alloc

def test_alloc():
    alloc = gen_alloc()
    assert type(alloc) is Procedure

    filename = "compiler_test_simple_alloc"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc))
    f_pretty.close()

    alloc.compile_c(directory, filename)


def gen_alloc_nest():
    @proc
    def alloc_nest(n: size, m: size, x: R[n, m] @ IN @ HEAP):
        ptr : R[n, m] @ GEMM
        for i in par(0,n/16):
            for j in par(0,m/16):
                instr(GEMM_Load)
                for i2 in par(0,16):
                    for j2 in par(0,16):
                        if i*16+i2 < n and j*16+j2 < n:
                            ptr[i, j] = x[i*16+i2, j*16+j2]

    return alloc_nest

"""
# step 1: explore tail strategies w/o GEMMINI
# step 2: extend exploration of tail strategies to consider GEMMINI
@proc
def foo( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i in par(0,n):
        y[i] = x[i] + 1

foo = foo.split(i,16,[i0,i1],"split_case0")
         .instr(i1, GEMMINI)

    foo.split()
       .peel_off_last_iter(i0)
       .simplify(i1[0])

@proc
def foo_split_guard( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        for i1 in par(0,16):
            if i0*16 + i1 < n:
                y[i0*16+i1] = x[i0*16+i1] + 1

#peel_off_last_iter ->>
    for i0 in par(0,ceil(n/16)-1):
        for i1 in par(0,16):
            if i0*16 + i1 < n:
                y[i0*16+i1] = x[i0*16+i1] + 1
    i0' = ceil(n/16)-1
    for i1 in par(0,16):
        if i0'*16 + i1 < n:
            y[i0'*16+i1] = x[i0'*16+i1] + 1

#simplify(i1[0]) ->>
    for i0 in par(0,ceil(n/16)-1):   i0 < ceil(n/16) - 1
        for i1 in par(0,16):
            y[i0*16+i1] = x[i0*16+i1] + 1
    i0' = ceil(n/16)-1
    for i1 in par(0,16):
        if i0'*16 + i1 < n:
            y[i0'*16+i1] = x[i0'*16+i1] + 1


@Proc
Foo( n : size, m : size, p : size )
    RESERVE SPACE globmem in DRAM

    buf : R[n]
        --> elem* buf = globmem

    A   : R[m]
        --> elem* A   = globmem + n

    for k in ...:
        B : R[p]
        --> elem* B   = globmem + n + m
        for j
            B = A lbuf
        for j
            x += B
        free B
        --> noop;

    free A
    free buf

    FREE globmem

required_bytes = Query_Foo_memory(n,m,p)
globmem = my_globmem + offset;
Foo_manual_mem(globmem, n,m,p, ...)

Foo(n,m,p, ...)


A : R[n,m]

buf : R[n,m] @ GEMMINI_SCRATCHPAD

buf : R[n,16] @ GEMM
instr(GEMM_LD)
for i0 in par(0,16):
    for i1 in par(0,16):
        buf[i0,i1] = input[i0][i1]


=>
gemmini_extended_config_ld(0,1)
gemmini_extended_mvin(input, sp_start_addr, 3, 16)

LOAD ... into buf

@proc
def foo_split_case0( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        if i0 == ceil(n/16)-1:
            instr(GEMMINI)
            for i1 in par(0,n%16):
                y[i0*16+i1] = x[i0*16+i1] + 1
        else:
            instr(GEMMINI)
            for i1 in par(0,16):
                y[i0*16+i1] = x[i0*16+i1] + 1

@proc
def foo_split_case1( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        for i1 in par(0,16):
            y[i0*16+i1] = x[i0*16+i1] + 1
    for i1 in par(0,n%16):
        y[(ceil(n/16)-1)*16+i1] = x[(ceil(n/16)-1)*16+i1] + 1

@proc
def foo_split_shift_in( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,ceil(n/16)):
        i_base = i0*16
        if i0 == ceil(n/16)-1:
            i_base = n - 16
        for i1 in par(0,16):
            i = ibase + i1
            y[i] = x[i] + 1

@proc
def foo_split_shift_in_case1( n : size, x : R[n] @ IN, y : R[n] @ OUT):
    for i0 in par(0,floor((n+1)/16)):
        @instr(HWACHA)
        for i1 in par(0,16):
            y[i0*16 + i1] = x[i0*16 + i1] + 1
    @instr(HWACHA)
    for i1 in par(0,16):
        y[n-16+i1] = x[n-16+i1] + 1

@proc
def foo_split_overcompute( n : size, x : R[16*ceil(n/16)] @ IN, y : R[16*ceil(n/16)] @ OUT):
    for i0 in par(0,ceil(n/16)):
        for i1 in par(0,16):
            y[i0*16+i1] = x[i0*16+i1] + 1
"""

def test_alloc_nest():
    alloc_nest = gen_alloc_nest()
    assert type(alloc_nest) is Procedure

    filename = "compiler_test_alloc_nest"

    # Write pretty printing to a file
    f_pretty = open(os.path.join(directory, filename + "_pretty.atl"), "w")
    f_pretty.write(str(alloc_nest))
    f_pretty.close()

    alloc_nest.compile_c(directory, filename)
