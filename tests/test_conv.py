import numpy as np
import sys\
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_interpreter import Interpreter

# Test 1 is Conv1d
#
#   conv1d( C : size, W : size, K : size, R : size,
#           x : R[C, W], w : R[K, R, C], res : R[K,W]):
#       forall i = 0,n:
#           res[i] = x[i] + y[i]
#
#
def gen_conv1d():
    n   = Sym('n')
    x   = Sym('x')
    w   = Sym('w')
    res = Sym('res')
    i   = Sym('i')
    c   = Sym('c')
    k   = Sym('k')

    src0= null_srcinfo()

    ai  = IR.AVar(i,src0)
    rhs = IR.BinOp('+', IR.Read(x, [ai], src0),
                        IR.Read(y, [ai], src0),
                   src0)
    s_a = IR.Assign(res, [ai], rhs, src0)
    loop = IR.ForAll( i, n, s_a, src0 )

    return Proc('conv1d',
                [n],
                [ (x,R[n],'IN'),
                  (w,R[n],'IN'),
                  (res,R[n],'OUT') ],
                [
                    loop
                ])

def test_conv1d():
    TEST_1 = gen_conv1d()
    x = np.array([3.0,6.0,9.0])
    y = np.array([1.0,2.0,3.0])
    res = np.random.uniform(size=3)
    Interpreter(TEST_1, n=3, x=x, y=y, res=res)
    print(res)
    np.testing.assert_almost_equal(res,[4,8,12])
