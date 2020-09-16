import numpy as np
import sys
import pytest
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_compiler import Compiler, run_compile
from ctypes import *
import ctypes
import os
import sys
import subprocess

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
    FloatArray = c_float * 3
    x = FloatArray(3.0,6.0,9.0)
    y = FloatArray(1.0,2.0,3.0)
    res = FloatArray(0.0,0.0,0.0)
    test_lib.add_vec(c_int(3), x, y, res)
    np.testing.assert_almost_equal(res,[4,8,12])

# TEST 2 is alloc
#   alloc( n : size, x : R[n]):
#       int *ptr = (int*) malloc (n * sizeof(int));
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
