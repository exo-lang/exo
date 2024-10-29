from __future__ import annotations

import pytest

import numpy as np

from exo import proc, DRAM
from exo.libs.memories import GEMM_SCRATCH
from exo.stdlib.scheduling import SchedulingError

# ------- Interpreter tests ---------

def test_reduce_add(compiler):
    @proc
    def acc(N: size, A: f32[N], acc: f32):
        acc = 0
        for i in seq(0, N):
            acc += A[i]

    fn = compiler.compile(acc)

    n = 3
    A = np.arange(n, dtype=np.float32)
    a1 = np.zeros(1, dtype=np.float32)
    a2 = np.zeros(1, dtype=np.float32)    
    fn(None, n, A, a1)
    acc.interpret(N=n, A=A, acc=a2)

    assert(a1 == a2)

def test_scope1(compiler):
    @proc
    def foo(res: f32):
        a: f32
        a = 1
        for i in seq(0,4):
            a: f32
            a = 2
        res = a

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    fn(None, x)
    foo.interpret(res=y)

    assert(x == y)

def test_scope2(compiler):
    @proc
    def foo(res: f32):
        a: f32
        a = 1
        for i in seq(0,4):
            a = 2
        res = a

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    fn(None, x)
    foo.interpret(res=y)

    assert(x == y)

def test_empty_seq(compiler):
    @proc
    def foo(res: f32):
        for i in seq(0, 0):
            res = 1

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    fn(None, x)
    foo.interpret(res=y)

    assert(x == y)
    
def test_cond(compiler):
    @proc
    def foo(res: f32, p: bool):
        if p:
            res = 1
        else:
            res = 2

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    fn(None, x, False)
    foo.interpret(res=y, p=False)

    assert(x == y)

def test_call(compiler):
    @proc
    def bar(res: f32):
        res = 3

    @proc
    def foo(res: f32):
        res = 2
        bar(res)
        res += 1

    fn = compiler.compile(foo)

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)
    fn(None, x)
    foo.interpret(res=y)

    assert(x == y)

def test_window(compiler):
    @proc
    def bar(B: [f32][4], res: f32):
        res = B[2]
    @proc
    def foo(A: f32[8], res: f32):
        bar(A[0:4], res)

    fn = compiler.compile(foo)

    A = np.arange(8, dtype=np.float32)
    res1 = np.zeros(1, dtype=np.float32)
    res2 = np.zeros(1, dtype=np.float32)
    fn(None, A, res1)
    foo.interpret(A=A, res=res2)

    assert(res1 == res2)

def test_stride_simple1(compiler):
    @proc
    def bar(s0: stride, s1: stride, B: [i8][3,4]):
        assert stride(B,0) == s0
        assert stride(B,1) == s1
        pass
    @proc
    def foo(A: i8[3,4]):
        bar(stride(A, 0), stride(A, 1), A[:,:])
    
    fn = compiler.compile(foo)

    A = np.arange(3*4, dtype=float).reshape((3,4))
    fn(None, A)
    foo.interpret(A=A)

def test_stride_simple2(compiler):
    @proc
    def bar(s0: stride, s1: stride, B: [i8][1,1]):
        assert stride(B,0) == s0
        assert stride(B,1) == s1
        pass
    @proc
    def foo(A: [i8][3,4]):
        bar(stride(A, 0), stride(A, 1), A[0:1,1:2])
    
    fn = compiler.compile(foo)

    A = np.arange(6*8, dtype=float).reshape((6,8))
    fn(None, A[::2,::2])
    foo.interpret(A=A[::2,::2])

def test_stride1(compiler):
    @proc
    def foo(A: [i8][3,2,3]):
        assert stride(A,0) == 20
        assert stride(A,1) == 5 * 2
        assert stride(A,2) == 1 * 2
        pass
    
    fn = compiler.compile(foo)

    A = np.arange(3*4*5, dtype=float).reshape((3,4,5))
    fn(None, A[::1,::2,::2])
    foo.interpret(A=A[::1,::2,::2])

def test_stride2(compiler):
    @proc
    def foo(A: [i8][2,4,2]):
        assert stride(A,0) == 20 * 2
        assert stride(A,1) == 5 * 1
        assert stride(A,2) == 1 * 3
        pass

    fn = compiler.compile(foo)

    A = np.arange(3*4*5, dtype=float).reshape((3,4,5))
    fn(None, A[::2,::1,::3])
    foo.interpret(A=A[::2,::1,::3])

def test_branch_stride1(compiler):
    @proc
    def bar(B: [i8][3,4], res: f32):
        a: f32
        if (stride(B, 0) == 8):
            a = 1
        res = a
    @proc
    def foo(A: i8[3,4], res: f32):
        bar(A[:,:], res)
    
    fn = compiler.compile(foo)

    A = np.arange(3*4, dtype=float).reshape((3,4))
    x1 = np.zeros(1)
    x2 = np.zeros(1)
    fn(None, A, x1)
    foo.interpret(A=A, res=x2)

# TODO: discuss
# updating param within stride conditional triggers validation error
def test_branch_stride2(compiler):
    @proc
    def bar(B: [i8][3,4], res: f32):
        if (stride(B, 0) == 8):
            res = 1
    @proc
    def foo(A: i8[3,4], res: f32):
        bar(A[:,:], res)
    
    fn = compiler.compile(foo)

    A = np.arange(3*4, dtype=float).reshape((3,4))
    x1 = np.zeros(1)
    x2 = np.zeros(1)
    fn(None, A, x1)
    foo.interpret(A=A, res=x2)

def test_bounds_err_interp():
    with pytest.raises(TypeError):

        @proc
        def foo(N: size, A:f32[N], res: f32):
            a: f32
            res = A[3]

        N = 2
        A = np.arange(N, dtype=np.float32)
        x = np.zeros(1, dtype=np.float32)

        foo.interpret(N=N, A=A, res=x)

def test_precond_interp_simple():
    with pytest.raises(AssertionError):

        @proc
        def foo(N: size, A:f32[N], res: f32):
            assert N == 4
            res = A[3]

        N = 2
        A = np.arange(N, dtype=np.float32)
        x = np.zeros(1, dtype=np.float32)

        foo.interpret(N=N, A=A, res=x)

# TODO: discuss
# shouldn't this raise a runtime error? 
def test_precond_comp_simple(compiler):
    @proc
    def foo(N: size, A:f32[N], res: f32):
        assert N == 4
        res = A[3]

    N = 2
    A = np.arange(N, dtype=np.float32)
    x = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
    fn(None, N, A, x)

def test_precond_interp_stride():
    with pytest.raises(AssertionError):

        @proc
        def foo(A:f32[1,8]):
            assert stride(A, 0) == 8
            pass

        A = np.arange(16, dtype=np.float32).reshape((1,16))
        foo.interpret(A=A[:,::2])

# TODO: discuss
# does not raise an error, but (incorrectly) informs the compiler about the stride
def test_precond_comp_stride(compiler):
    @proc
    def foo(A:f32[1,8]):
        assert stride(A, 0) == 8
        pass

    fn = compiler.compile(foo)

    A = np.arange(16, dtype=np.float32).reshape((1,16))
    fn(None, A[:,::2])