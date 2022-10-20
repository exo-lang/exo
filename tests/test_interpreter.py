from __future__ import annotations

import numpy as np

from exo import proc


# Test 1 is Full 1D convolution
def gen_conv1d():
    @proc
    def conv1d(n: size, m: size, r: size, x: R[n], w: R[m], res: R[r]):
        for i in seq(0, r):
            res[i] = 0.0
        for i in seq(0, r):
            for j in seq(0, n):
                if j < i + 1 and j >= i - (m - 1):
                    res[i] += x[j] * w[i - j]

    return conv1d


def test_conv1d():
    conv1d = gen_conv1d()
    n = 5
    m = 3
    r = n + m - 1
    x = np.array([0.2, 0.5, -0.4, 1.0, 0.0])
    w = np.array([0.6, 1.9, -2.2])
    res = np.random.uniform(size=r)
    conv1d.interpret(n=n, m=m, r=r, x=x, w=w, res=res)
    np.testing.assert_almost_equal(res, [0.12, 0.68, 0.27, -1.26, 2.78, -2.2, 0])


# add vector test
def gen_add_vec():
    @proc
    def add_vec(n: size, x: R[n], y: R[n], res: R[n]):
        for i in seq(0, n):
            res[i] = x[i] + y[i]

    return add_vec


def test_add_vec():
    add_vec = gen_add_vec()
    x = np.array([3.0, 6.0, 9.0])
    y = np.array([1.0, 2.0, 3.0])
    res = np.random.uniform(size=3)
    add_vec.interpret(n=3, x=x, y=y, res=res)
    np.testing.assert_almost_equal(res, [4, 8, 12])


# multiply matrix test
#   C = A * B
def gen_gemm():
    @proc
    def gemm(
        n: size,
        m: size,
        p: size,
        C: R[n, m],
        A: R[n, p],
        B: R[p, m],
    ):
        for i in seq(0, n):
            for j in seq(0, m):
                C[i, j] = 0.0
                for k in seq(0, p):
                    C[i, j] += A[i, k] * B[k, j]

    return gemm


def test_gemm():
    A = np.array([[-1.0, 4.0], [-2.0, 5.0], [6.0, -3.0], [7.0, 8.0]])

    B = np.array([[9.0, 0.0, 2.0], [3.0, 1.0, 10.0]])

    C_answer = [
        [3.0, 4.0, 38.0],
        [-3.0, 5.0, 46.0],
        [45.0, -3.0, -18.0],
        [87.0, 8.0, 94.0],
    ]

    gemm = gen_gemm()
    C = np.random.uniform(size=(4, 3))
    gemm.interpret(n=4, m=3, p=2, A=A, B=B, C=C)
    np.testing.assert_almost_equal(C, C_answer)
