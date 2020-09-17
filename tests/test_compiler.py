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
    aptr = IR.AVar(ptr, src0)
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
    x   = Sym('x')
    ptr = Sym('ptr')
    i   = Sym('i')

    src0= null_srcinfo()

    # How to pass n to alloc?
    aptr = IR.AVar(ptr, src0)
    ma  = IR.Alloc(ptr, R[n].typ, src0)
    ai  = IR.AVar(i, src0)
    rhs = IR.Read(x, [ai], src0)
    s_a = IR.Assign(ptr, [ai], rhs, src0)
    loop = IR.ForAll(i, n, s_a, src0)
    seq = IR.Seq(ma, loop, src0)

    return Proc('alloc_nest',
                [n],
                [ (x,R[n],'IN')],
                [
                    seq
                ])

#@pytest.mark.skip(reason="WIP test")
def test_alloc_nest():
    TEST_3 = gen_alloc_nest()
    run_compile([TEST_3],"test_alloc_nest.c", "test_alloc_nest.h")
