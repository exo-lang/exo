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

# Test 2 is Full 2D convolution
#
#   conv2d(w : size, h : size, kw : size, kh : size, rw : size, rh : size, 
#          x : R[h,w], k : R[kh, kw], res : R[rh, rw] ):
#       forall i = 0,rh:
#         forall j = 0,rw:
#           res[i,j] = 0.0
#       forall i = 0,rh: //padding? kw//2?
#         forall j = 0,rw:
#           forall ki = 0,h:
#             forall kj = 0,w:     
#               if (ki < i+1 and ki >= i-(kh-1) and kj < j+1 and kj >= j-(kw-1)) then
#                    res[i,j] += x[ki,kj]*k[i-ki,j-kj]
#
def gen_conv2d():
    w = Sym('w')
    h = Sym('h')
    kw = Sym('kw')
    kh = Sym('kh')
    rw = Sym('rw')
    rh = Sym('rh')

    x   = Sym('x')
    k   = Sym('k')
    res = Sym('res')
    i   = Sym('i')
    j   = Sym('j')
    ki   = Sym('ki')
    kj   = Sym('kj')

    src0 = null_srcinfo()

    ai  = IR.AVar(i,src0)
    aj  = IR.AVar(j,src0)
    aki  = IR.AVar(ki,src0)
    akj  = IR.AVar(kj,src0)

    akw  = IR.ASize(kw,src0)
    akh  = IR.ASize(kh,src0)

    loop_cond  = IR.And(
                    IR.And(IR.Cmp('<=', aki, ai, src0),
                           IR.Cmp('>', aki, IR.ASub(ai,akh,src0), src0),
                        src0),
                    IR.And(IR.Cmp('<=', akj, aj, src0),
                           IR.Cmp('>', akj, IR.ASub(aj,akw,src0), src0),
                        src0),
                    src0)

    statement  = IR.Reduce(res, [ai,aj], IR.BinOp('*', IR.Read(x, [ai,aj], src0),
                                                    IR.Read(k, [
                                                    IR.ASub(ai,aki, src0),
                                                    IR.ASub(aj,akj, src0)
                                                    ], src0),
                                                    src0), src0)
    loop_nest  = IR.ForAll(i, rh,
                   IR.ForAll(j, rw,
                     IR.ForAll(ki, h,
                       IR.ForAll(kj, w,
                         IR.If(loop_cond,
                            statement,
                         src0), src0), src0), src0), src0)

    zero_res   = IR.ForAll(i, rh,
                   IR.ForAll(j, rw,
                     IR.Assign(res, [ai, aj], IR.Const(0.0,src0), src0), src0), src0)

    return Proc('conv2d',
                [w,h,kw,kh,rw,rh],
                [ (x,R[h,w],'IN'),
                  (w,R[kh,kw],'IN'),
                  (res,R[rh,rw],'OUT') ],
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
