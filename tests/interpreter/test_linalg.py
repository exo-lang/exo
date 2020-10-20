import numpy as np
import sys
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.LoopIR_interpreter import Interpreter
from SYS_ATL.prelude import *
from SYS_ATL.debug_frontend_LoopIR import *

# Test 1 is add vector
#
#   add_vec( n : size, x : R[n], y : R[n], res : R[n]):
#       forall i = 0,n:
#           res[i] = x[i] + y[i]
#


def gen_add_vec():
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


def test_add_vec():
    TEST_1 = gen_add_vec()
    x = np.array([3.0, 6.0, 9.0])
    y = np.array([1.0, 2.0, 3.0])
    res = np.random.uniform(size=3)
    Interpreter(TEST_1, n=3, x=x, y=y, res=res)
    print(res)
    np.testing.assert_almost_equal(res, [4, 8, 12])


# Test 2 is multiply matrix
#
#   C = A * B
#   gemm( n : size, m : size, p : size,
#         C : R[n,m] : OUT,
#         A : R[n,p] : IN,
#         B : R[p,m] : IN,
#       ):
#           forall i = 0,n:
#               forall j = 0,m:
#                   C[i,j] = 0.0
#                   forall k = 0,p:
#                       C[i,j] += A[i,k] * B[k,j]
#
def gen_gemm():
    n = Sym('n')
    m = Sym('m')
    p = Sym('p')
    C = Sym('C')
    A = Sym('A')
    B = Sym('B')
    i = Sym('i')
    j = Sym('j')
    k = Sym('k')

    ns = null_srcinfo()
    ai = IR.AVar(i, ns)
    aj = IR.AVar(j, ns)
    ak = IR.AVar(k, ns)

    an = IR.ASize(n, ns)
    am = IR.ASize(m, ns)
    ap = IR.ASize(p, ns)

    zeroC = IR.Assign(C, [ai, aj], IR.Const(0.0, ns), ns)
    accC = IR.Reduce(C, [ai, aj], IR.BinOp('*', IR.Read(A, [ai, ak], ns),
                                           IR.Read(B, [ak, aj], ns),
                                           ns), ns)

    loop = IR.ForAll(i, an,
                     IR.ForAll(j, am,
                               IR.Seq(zeroC,
                                      IR.ForAll(k, ap, accC, ns),
                                      ns),
                               ns),
                     ns)

    return Proc('mat_mul',
                [n, m, p],
                [(A, R[n, p], 'IN'),
                 (B, R[p, m], 'IN'),
                 (C, R[n, m], 'OUT')],
                [
                    loop
                ])


def test_gemm():
    A = np.array([[-1.0, 4.0],
                  [-2.0, 5.0],
                  [6.0, -3.0],
                  [7.0, 8.0]])

    B = np.array([[9.0, 0.0,  2.0],
                  [3.0, 1.0, 10.0]])

    C_answer = [[3.0, 4.0, 38.0],
                [-3.0, 5.0, 46.0],
                [45.0, -3.0, -18.0],
                [87.0, 8.0, 94.0]]

    TEST_GEMM = gen_gemm()
    C = np.random.uniform(size=(4, 3))
    Interpreter(TEST_GEMM, n=4, m=3, p=2, A=A, B=B, C=C)
    print(C)
    np.testing.assert_almost_equal(C, C_answer)
