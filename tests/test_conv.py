import numpy as np
import sys
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_interpreter import Interpreter

# Test 1 is Full 1D convolution
#
#   conv1d(n : size, m : size, r: size, x : R[n], w : R[m],
#                                       res : R[r] ):
#       forall i = 0,r:
#           res[i] = 0.0
#       forall i = 0,r:
#           forall j = 0,n:
#               if (j < i+1 and j >= i-(m-1)) then
#                    res[i] += x[j]*w[i-j]
#
def gen_conv1d():
    n   = Sym('n')
    m   = Sym('m')
    r   = Sym('r')
    x   = Sym('x')
    w   = Sym('w')
    res = Sym('res')
    i   = Sym('i')
    j   = Sym('j')

    src0= null_srcinfo()

    ai  = IR.AVar(i,src0)
    aj  = IR.AVar(j,src0)
    am  = IR.ASize(m,src0)

    loop_cond  = IR.And(IR.Cmp('<=', aj, ai, src0),
                        IR.Cmp('>', aj, IR.ASub(ai,am,src0), src0),
                        src0)
    statement  = IR.Reduce(res, [ai], IR.BinOp('*', IR.Read(x, [aj], src0),
                                                    IR.Read(w, [
                                                    IR.ASub(ai,aj, src0)
                                                    ], src0),
                                                    src0), src0)
    loop_nest  = IR.ForAll(i, r,
                    IR.ForAll(j, n,
                        IR.If(loop_cond,
                            statement,
                        src0), src0), src0)

    zero_res   = IR.ForAll(i, r,
                    IR.Assign(res, [ai], IR.Const(0.0,src0), src0), src0)

    return Proc('conv1d',
                [n, m, r],
                [ (x,R[n],'IN'),
                  (w,R[m],'IN'),
                  (res,R[r],'OUT') ],
                [
                    zero_res,
                    loop_nest
                ])

def test_conv1d():
    TEST_1 = gen_conv1d()
    n = 5
    m = 3
    r = n + m - 1
    x = np.array([0.2, 0.5, -0.4, 1.0, 0.0])
    w = np.array([0.6, 1.9, -2.2])
    res = np.random.uniform(size=r)
    Interpreter(TEST_1, n=n, m=m, r=r, x=x, w=w, res=res)
    print(res)
    np.testing.assert_almost_equal(res,[0.12,0.68,0.27,-1.26,2.78,-2.2,0])
