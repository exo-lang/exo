from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    pytest.skip("pytorch is not available, skipping winograd", allow_module_level=True)

from exo import proc


def wconv_3x3():
    # res <- A @ B @ A^T
    @proc
    def multiply(n: size, m: size, A: f32[n, m], B: f32[m, m], res: f32[n, n]):

        tmp: f32[n, m]
        for i in seq(0, n):
            for j in seq(0, m):
                tmp[i, j] = 0.0
                for k in seq(0, m):
                    tmp[i, j] += A[i, k] * B[k, j]

        for i in seq(0, n):
            for j in seq(0, n):
                res[i, j] = 0.0
                for k in seq(0, m):
                    res[i, j] += tmp[i, k] * A[j, k]

                    # C <- A @ B (element wise)

    @proc
    def element(n: size, A: f32[n, n], B: f32[n, n], C: f32[n, n]):

        for i in seq(0, n):
            for j in seq(0, n):
                C[i, j] = A[i, j] * B[i, j]

    @proc
    def wconv(
        inp: f32[4, 4],
        kernel: f32[3, 3],
        res: f32[2, 2],
        B_T: f32[4, 4],
        G: f32[4, 3],
        A_T: f32[2, 4],
    ):

        # Cannot do this
        # B_T = [1,	0,	-1,	0, \
        #       0,	1,	1,	0, \
        #       0,	-1,	1,	0, \
        #       0,	1,	0,	-1]

        U: f32[4, 4]
        # U <- G @ kernel @ G^T
        multiply(4, 3, G, kernel, U)

        V: f32[4, 4]
        # V <- B^T @ inp @ B
        multiply(4, 4, B_T, inp, V)

        M: f32[4, 4]
        # M <- U @ V (element wise)
        element(4, U, V, M)

        # res <- A^T @ M @ A
        multiply(2, 4, A_T, M, res)

    return wconv


def test_winograd(compiler):
    conv = wconv_3x3()

    wconv = compiler.compile(conv)

    B_T = np.array([(1, 0, -1, 0), (0, 1, 1, 0), (0, -1, 1, 0), (0, 1, 0, -1)])

    G = np.array([(1, 0, 0), (1 / 2, 1 / 2, 1 / 2), (1 / 2, -1 / 2, 1 / 2), (0, 0, 1)])

    A_T = np.array([(1, 1, 1, 0), (0, 1, -1, -1)])

    n = 4
    m = 3
    # batch, channel = 1
    kernel = torch.randint(high=100, size=(1, 1, m, m)).float()
    inp = torch.randint(high=100, size=(1, 1, n, n)).float()

    ref = F.conv2d(inp, kernel)

    kernel = kernel.detach().numpy()
    inp = inp.detach().numpy()
    res = np.zeros(shape=(2, 2), dtype=np.float32)

    wconv(None, inp, kernel, res, B_T, G, A_T)
    print(res)

    # TODO: finish test, validate output (returns NaN right now)


"""
    n_size = IMAGE.shape[0]
    m_size = IMAGE.shape[1]
    k_size = 5
    kernel = gkern(k_size, 1)
    res = nprand(size=(n_size, m_size))
    res_c = cvt_c(res)
    test_lib = generate_lib(filename)
    test_lib.blur(POINTER(c_int)(), c_int(n_size), c_int(m_size), c_int(
        k_size), cvt_c(IMAGE), cvt_c(kernel), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(n_size, m_size))
    res_c = res_c.astype(np.uint8)
    out = Image.fromarray(res_c)
    out.save(os.path.join(TMP_DIR, filename + '_out.png'))
"""
