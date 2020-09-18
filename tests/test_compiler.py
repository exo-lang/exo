import numpy as np
import sys
import pytest
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_compiler import Compiler, run_compile
from SYS_ATL.LoopIR_interpreter import Interpreter
from ctypes import *
import ctypes
import os
import sys
import subprocess

def cvt_c(n_array):
    c_float_p = ctypes.POINTER(ctypes.c_float)
    return n_array.astype(np.float32).ctypes.data_as(c_float_p)

# Test 1 is add vector
#
#   add_vec( n : size, x : R[n], y : R[n], res : R[n]):
#       forall i = 0,n:
#           res[i] = x[i] + y[i]
#
def gen_add_vec():
    n   = Sym('n')
    x   = Sym('x')
    y   = Sym('y')
    res = Sym('res')
    i   = Sym('i')

    src0= null_srcinfo()

    ai  = IR.AVar(i,src0)
    rhs = IR.BinOp('+', IR.Read(x, [ai], src0),
                        IR.Read(y, [ai], src0),
                   src0)
    s_a = IR.Assign(res, [ai], rhs, src0)
    loop = IR.ForAll( i, n, s_a, src0 )

    return Proc('add_vec',
                [n],
                [ (x,R[n],'IN'),
                  (y,R[n],'IN'),
                  (res,R[n],'OUT') ],
                [
                    loop
                ])

def test_add_vec():
    TEST_1 = gen_add_vec()
    filename = "test1"
    run_compile([TEST_1],(filename + ".c"), (filename + ".h"))
    compile_so_cmd = ("clang -Wall -Werror -fPIC -O3 -shared "+
                       "-o " + filename + ".so " + filename + ".c")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    abspath  = os.path.dirname(os.path.abspath(filename))
    test_lib = ctypes.CDLL(abspath + '/' + filename + ".so")
    x = np.array([3.0,6.0,9.0])
    y = np.array([1.0,2.0,3.0])
    a_size = 3
    res = np.random.uniform(size=a_size)
    res_c = cvt_c(res)
    test_lib.add_vec(c_int(a_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(a_size,))
    Interpreter(TEST_1, n=3, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, res_c)
    np.testing.assert_almost_equal(res,[4,8,12])

# TEST 2 is alloc
#   alloc( n : size, x : R[n]):
#       float *ptr = (float*) malloc (n * sizeof(float));
#       forall i = 0,n:
#           ptr[i] = x[i];
#       free(ptr);
def gen_alloc():
    n   = Sym('n')
    x   = Sym('x')
    ptr = Sym('ptr')
    i   = Sym('i')

    src0= null_srcinfo()

    # How to pass n to alloc?
    ma  = IR.Alloc(ptr, R[n].typ, src0)
    ai  = IR.AVar(i, src0)
    rhs = IR.Read(x, [ai], src0)
    s_a = IR.Assign(ptr, [ai], rhs, src0)
    loop = IR.ForAll(i, n, s_a, src0)
    seq = IR.Seq(ma, loop, src0)

    return Proc('alloc',
                [n],
                [ (x,R[n],'IN')],
                [
                    seq
                ])

#@pytest.mark.skip(reason="WIP test")
def test_alloc():
    TEST_2 = gen_alloc()
    run_compile([TEST_2],"test_alloc.c", "test_alloc.h")

# TEST 3 is nested alloc
#   alloc_nest( n : size, m : size, x : R[n,m], y: R[n,m], res : R[n,m] ):
#       rloc : R[m]
#       forall i = 0,n:
#           xloc : R[m] @ GEMM_scratchpad
#           yloc : R[m]
#           forall j = 0,m:
#               xloc[i,j] = x[i,j]
#           forall j = 0,m:
#               yloc[i,j] = y[i,j]
#           forall j = 0,m:
#               rloc[i,j] = xloc[i,j] + yloc[i,j]
#           forall j = 0,m:
#               res[i,j] = rloc[i,j]
#           free(xloc)
#           free(yloc);
#       free(rloc);

def gen_alloc_nest():
    n   = Sym('n')
    m   = Sym('m')
    x   = Sym('x')
    y   = Sym('y')
    res = Sym('res')
    i   = Sym('i')
    j   = Sym('j')

    rloc= Sym('rloc')
    xloc= Sym('xloc')
    yloc= Sym('yloc')

    src0= null_srcinfo()

    rloc_a = IR.Alloc(rloc, R[m].typ, src0)

    ai  = IR.AVar(i, src0)
    aj  = IR.AVar(j, src0)

    xloc_a = IR.Alloc(xloc, R[m].typ, src0)
    yloc_a = IR.Alloc(yloc, R[m].typ, src0)
    seq_alloc = IR.Seq(xloc_a, yloc_a, src0)

#           forall j = 0,m:
#               xloc[i,j] = x[i,j]
    rhs_1  = IR.Read(x, [ai,aj], src0)
    body_1 = IR.Assign(xloc, [ai,aj], rhs_1, src0)
    loop_1 = IR.ForAll(j, m, body_1, src0)
    seq_1  = IR.Seq(seq_alloc, loop_1, src0)

#           forall j = 0,m:
#               yloc[i,j] = y[i,j]
    rhs_2  = IR.Read(y, [ai,aj], src0)
    body_2 = IR.Assign(yloc, [ai,aj], rhs_2, src0)
    loop_2 = IR.ForAll(j, m, body_2, src0)
    seq_2  = IR.Seq(seq_1, loop_2, src0)

#           forall j = 0,m:
#               rloc[i,j] = xloc[i,j] + yloc[i,j]
    rhs_3  = IR.BinOp('+', IR.Read(xloc, [ai,aj], src0),
                        IR.Read(yloc, [ai,aj], src0),
                   src0)
    body_3 = IR.Assign(rloc, [ai,aj], rhs_3, src0)
    loop_3 = IR.ForAll(j, m, body_3, src0)
    seq_3  = IR.Seq(seq_2, loop_3, src0)

#           forall j = 0,m:
#               res[i,j] = rloc[i,j]
    rhs_4  = IR.Read(rloc, [ai,aj], src0)
    body_4 = IR.Assign(res, [ai,aj], rhs_4, src0)
    loop_4 = IR.ForAll(j, m, body_4, src0)
    seq_4  = IR.Seq(seq_3, loop_4, src0)

    loop = IR.ForAll(i, n, seq_4, src0)
    seq_top = IR.Seq(rloc_a, loop, src0)

    return Proc('alloc_nest',
                [n, m],
                [
                    (x,R[n,m],'IN'),
                    (y,R[n,m],'IN'),
                    (res,R[n,m],'OUT')
                ],
                [
                    seq_top
                ])

#@pytest.mark.skip(reason="WIP test")
def test_alloc_nest():
    TEST_3 = gen_alloc_nest()
    filename = "test_alloc_nest"
    run_compile([TEST_3],(filename + ".c"), (filename + ".h"))
    compile_so_cmd = ("clang -Wall -Werror -fPIC -O3 -shared "+
                       "-o " + filename + ".so " + filename + ".c")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    abspath  = os.path.dirname(os.path.abspath(filename))
    test_lib = ctypes.CDLL(abspath + '/' + filename + ".so")
    x = np.array([[1.0,2.0,3.0], [3.2,4.0,5.3]])
    y = np.array([[2.6,3.7,8.9], [1.3,2.3,6.7]])
    n_size = 2
    m_size = 3
    res = [np.random.uniform(size=m_size), np.random.uniform(size=m_size)] 
    res_c = cvt_c(res)
    test_lib.add_vec(c_int(n_size), c_int(m_size), cvt_c(x), cvt_c(y), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size,m_size))
    Interpreter(TEST_1, n=n_size, m=m_size, x=x, y=y, res=res)
    print (res_c)
    np.testing.assert_almost_equal(res, res_c)
    #np.testing.assert_almost_equal(res,[4,8,12])
