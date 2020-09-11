import numpy as np
import sys
import pytest
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_compiler import Compiler, run_compile

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

@pytest.mark.skip(reason="trying to implement C compiler")
def test_add_vec():
    TEST_1 = gen_add_vec()
    #x = np.array([3.0,6.0,9.0])
    #y = np.array([1.0,2.0,3.0])
    #res = np.random.uniform(size=3)
    #c = Compiler(TEST_1, n=3, x=x, y=y, res=res)
    #c = Compiler(TEST_1)
    run_compile([TEST_1],"test.c", "test.h")
