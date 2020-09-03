import numpy as np
import sys
sys.path.append(sys.path[0]+"/..")
from SYS_ATL.debug_frontend_LoopIR import *
from SYS_ATL.prelude import *
from SYS_ATL.LoopIR_interpreter import Interpreter

# Test 1 is Full 1D convolution
#
#   conv1d(n : size, m : size, r: size, x : R[n], k : R[m],
#                                       res : R[r] ):
#       forall i = 0,r:
#           jmin = 0
#           if (i >= m-1) then
#               jmin = i-(m-1)
#           jmax = n-1
#           if (i < n-1) then
#               jmax = i
#           forall j = 0,jmax:
#               if (j < jmin) then
#                   continue
#               res[i] += x[j]*k[i-j]
#
def gen_conv1d():
    n   = Sym('n')
    m   = Sym('m')
    r   = Sym('r')
    x   = Sym('x')
    k   = Sym('k')
    res = Sym('res')
    i   = Sym('i')
    jmin = Sym('jmin')
    jmax = Sym('jmax')
    j   = Sym('j')

    src0= null_srcinfo()

    ai  = IR.AVar(i,src0)
    ajmin = IR.AVar(jmin,src0)
    ajmax = IR.AVar(jmax,src0)
    aj  = IR.AVar(j,src0)

    jmin_a = IR.Assign(ajmin, [], IR.Const(0.0))
    jmin_if = IR.If( IR.Cmp( ">=", ai, IR.ASub(m,1)),
            IR.Assign(ajmin, [], IR.ASub(i,IR.ASub(m,1))))
    jminseq = IR.Seq(jmin_a, jmin_if)

    jmax_a = IR.Assign(ajmax, [], IR.ASub(n, 1))
    jmax_if = IR.If( IR.Cmp("<", ai, IR.ASub(n,1)),
            IR.Assign(ajmax, [], ai))
    jmaxseq = IR.Seq(jmax_a, jmax_if)

    jseq = IR.Seq(jminseq, jmaxseq)

    inner_statement = # TODO: How to implement continue???
    inner_loop = IR.ForAll(j, jmax, inner_statement, src0)

    statement = IR.Seq(jseq, inner_loop)

    outer_loop = IR.ForAll(i, r, statement, src0)

#    rhs = IR.BinOp('+', IR.Read(x, [ai], src0),
#                        IR.Read(y, [ai], src0),
#                   src0)
#    s_a = IR.Assign(res, [ai], rhs, src0)
#    loop = IR.ForAll( i, n, s_a, src0 )

    return Proc('conv1d',
                [n, m, r],
                [ (x,R[n],'IN'),
                  (k,R[m],'IN'),
                  (res,R[r],'OUT') ],
                [
                    outer_loop
                ])

def test_conv1d():
    TEST_1 = gen_conv1d()
    n = 5
    m = 3
    r = n + m -1
    x = np.array([0.2, 0.5, -0.4, 1.0, 0.0])
    k = np.array([0.6, 1.9, -2.2])
    res = np.random.uniform(size=r)
    Interpreter(TEST_1, n=n, m=m, r=r, x=x, k=k, res=res)
    print(res)
    np.testing.assert_almost_equal(res,[0.12,0.68,0.27,-1.26,2.78,-2.2,0])
