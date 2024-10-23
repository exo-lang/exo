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

    n = 3
    A = np.arange(n, dtype=np.float32)
    a1 = np.zeros(1, dtype=np.float32)
    a2 = np.zeros(1, dtype=np.float32)
    
    fn = compiler.compile(acc)
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

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
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

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
    fn(None, x)
    foo.interpret(res=y)

    assert(x == y)

def test_empty_seq(compiler):
    @proc
    def foo(res: f32):
        for i in seq(0, 0):
            res = 1

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
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

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
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

    x = np.zeros(1, dtype=np.float32)
    y = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
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

    A = np.arange(8, dtype=np.float32)
    res1 = np.zeros(1, dtype=np.float32)
    res2 = np.zeros(1, dtype=np.float32)

    fn = compiler.compile(foo)
    fn(None, A, res1)
    foo.interpret(A=A, res=res2)

    assert(res1 == res2)

def test_stride1(compiler):
    @proc
    def bar(s0: stride, s1: stride, s2: stride, B: [i8][2,2,2]):
        assert stride(B,0) == s0
        assert stride(B,1) == s1
        assert stride(B,2) == s2
        pass

    @proc
    def foo(M: size, N: size, O: size, A: i8[M,N,O]):
        assert M > 1 and N > 1 and O > 1
        bar(stride(A, 0), stride(A, 1), stride(A, 2), A[0:2,0:2,0:2])
    
    M = 3
    N = 4
    O = 5
    A = np.arange(M*N*O, dtype=float).reshape((M,N,O))
    
    fn = compiler.compile(foo)
    fn(None, M, N, O, A)
    foo.interpret(M=M, N=N, O=O, A=A)

def test_branch_stride1(compiler):
    @proc
    def bar(B: [i8][2,2], res: f32):
        a: f32
        if (stride(B,0) == 1):
            a = 1
    @proc
    def foo(M: size, N: size, A: i8[M,N], res: f32):
        assert M > 1 and N > 1
        bar(A[0:2,0:2], res)

def test_branch_stride2(compiler):
    @proc
    def bar(B: [i8][2,2], res: f32):
        a: f32
        if (stride(B,0) == 1):
            res = 1
    @proc
    def foo(M: size, N: size, A: i8[M,N], res: f32):
        assert M > 1 and N > 1
        bar(A[0:2,0:2], res)

    # M = 2
    # N = 8
    # A = np.arange(M*N, dtype=float).reshape((M,N))
    # res1 = np.zeros(1, dtype=float)
    # res2 = np.zeros(1, dtype=float)
    
    # fn = compiler.compile(foo)
    # fn(None, M, N, A, res1)
    # foo.interpret(M=M, N=N, A=A, res=res2)

    # assert(res1 == res2)

def test_bounds_err():
    with pytest.raises(TypeError):

        @proc
        def foo(N: size, A:f32[N], res: f32):
            a: f32
            res = A[3]

        N = 2
        A = np.arange(N, dtype=np.float32)
        x = np.zeros(1, dtype=np.float32)

        foo.interpret(N=N, A=A, res=x)

def test_precond():
    with pytest.raises(AssertionError):

        @proc
        def foo(N: size, A:f32[N], res: f32):
            assert N == 4
            a: f32
            res = A[3]

        N = 2
        A = np.arange(N, dtype=np.float32)
        x = np.zeros(1, dtype=np.float32)

        foo.interpret(N=N, A=A, res=x)

