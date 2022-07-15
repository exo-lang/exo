from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from exo import proc, Procedure, DRAM
from exo.libs.memories import MDRAM


# --- Start Blur Test ---

def gen_blur() -> Procedure:
    @proc
    def blur(n: size, m: size, k_size: size,
             image: R[n, m], kernel: R[k_size, k_size], res: R[n, m]):
        for i in par(0, n):
            for j in par(0, m):
                res[i, j] = 0.0
        for i in par(0, n):
            for j in par(0, m):
                for k in par(0, k_size):
                    for l in par(0, k_size):
                        if i + k >= 1 and i + k - n < 1 and j + l >= 1 and j + l - m < 1:
                            res[i, j] += kernel[k, l] * image[
                                i + k - 1, j + l - 1]

    return blur


def _test_blur(compiler, tmp_path, blur):
    fn = compiler.compile(blur)

    k_size = 5

    image = np.asarray(Image.open(Path(__file__).parent / 'input.png'),
                       dtype="float32")

    x = np.linspace(-1, 1, k_size + 1)
    kern1d = np.diff(np.random.normal(x))
    kern2d = np.outer(kern1d, kern1d)
    kern = np.asarray(kern2d / kern2d.sum(), dtype=np.float32)

    res = np.zeros_like(image)

    fn(None, *image.shape, k_size, image, kern, res)

    out = Image.fromarray(res.astype(np.uint8))
    out.save(tmp_path / 'out.png')


def test_simple_blur(compiler, tmp_path):
    blur = gen_blur()
    _test_blur(compiler, tmp_path, blur)


def test_simple_blur_split(compiler, tmp_path):
    @proc
    def simple_blur_split(n: size, m: size, k_size: size,
                          image: R[n, m], kernel: R[k_size, k_size],
                          res: R[n, m]):
        for i in par(0, n):
            for j1 in par(0, m / 2):
                for j2 in par(0, 2):
                    res[i, j1 * 2 + j2] = 0.0
        for i in par(0, n):
            for j1 in par(0, m / 2):
                for j2 in par(0, 2):
                    for k in par(0, k_size):
                        for l in par(0, k_size):
                            if i + k >= 1 and i + k - n < 1 and j1 * 2 + j2 + l >= 1 and j1 * 2 + j2 + l - m < 1:
                                res[i, j1 * 2 + j2] += kernel[k, l] * image[
                                    i + k - 1, j1 * 2 + j2 + l - 1]

    _test_blur(compiler, tmp_path, simple_blur_split)


def test_split_blur(compiler, tmp_path):
    blur = gen_blur()

    blur = blur.split('j', 4, ['j1', 'j2'])
    blur = blur.split('i#1', 4, ['i1', 'i2'])

    _test_blur(compiler, tmp_path, blur)


def test_reorder_blur(compiler, tmp_path):
    blur = gen_blur()

    blur = blur.reorder('k', 'l')
    blur = blur.reorder('i', 'j')

    _test_blur(compiler, tmp_path, blur)


def test_unroll_blur(compiler, tmp_path):
    blur = gen_blur()

    #    blur = blur.split('i',6,['i','iunroll']).simpler_unroll('iunroll')
    blur = blur.split('j', 4, ['j1', 'j2'])
    blur = blur.unroll('j2')

    _test_blur(compiler, tmp_path, blur)


# --- End Blur Test ---


# --- conv1d test ---
def test_conv1d(compiler):
    @proc
    def conv1d(n: size, m: size, r: size,
               x: R[n], w: R[m], res: R[r]):
        for i in par(0, r):
            res[i] = 0.0
        for i in par(0, r):
            for j in par(0, n):
                if j < i + 1 and j >= i - m + 1:
                    res[i] += x[j] * w[i - j]

    n_size = 5
    m_size = 3
    r_size = n_size + m_size - 1
    x = np.array([0.2, 0.5, -0.4, 1.0, 0.0], dtype=np.float32)
    w = np.array([0.6, 1.9, -2.2], dtype=np.float32)
    res = np.zeros(shape=r_size, dtype=np.float32)

    fn = compiler.compile(conv1d)
    fn(None, n_size, m_size, r_size, x, w, res)

    np.testing.assert_almost_equal(res, np.array(
        [0.12, 0.68, 0.27, -1.26, 2.78, -2.2, 0], dtype=np.float32))


# ------- Nested alloc test for normal DRAM ------

def test_alloc_nest(compiler, tmp_path):
    @proc
    def alloc_nest(n: size, m: size,
                   x: R[n, m], y: R[n, m] @ DRAM, res: R[n, m] @ DRAM):
        for i in par(0, n):
            rloc: R[m] @ DRAM
            xloc: R[m] @ DRAM
            yloc: R[m] @ DRAM
            for j in par(0, m):
                xloc[j] = x[i, j]
            for j in par(0, m):
                yloc[j] = y[i, j]
            for j in par(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0, m):
                res[i, j] = rloc[j]

    # Write effect printing to a file
    (tmp_path / f'{alloc_nest.name()}_effect.atl').write_text(
        str(alloc_nest.show_effects())
    )

    x = np.array([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]], dtype=np.float32)
    y = np.array([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]], dtype=np.float32)
    res = np.zeros_like(x)

    fn = compiler.compile(alloc_nest)
    fn(None, *x.shape, x, y, res)

    np.testing.assert_almost_equal(res, np.array(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], dtype=np.float32))


# ------- Nested alloc test for custom malloc DRAM ------

def test_alloc_nest_malloc(compiler):
    @proc
    def alloc_nest_malloc(n: size, m: size,
                          x: R[n, m] @ MDRAM, y: R[n, m] @ MDRAM,
                          res: R[n, m] @ MDRAM):
        for i in par(0, n):
            rloc: R[m] @ MDRAM
            xloc: R[m] @ MDRAM
            yloc: R[m] @ MDRAM
            for j in par(0, m):
                xloc[j] = x[i, j]
            for j in par(0, m):
                yloc[j] = y[i, j]
            for j in par(0, m):
                rloc[j] = xloc[j] + yloc[j]
            for j in par(0, m):
                res[i, j] = rloc[j]

    x = np.array([[1.0, 2.0, 3.0], [3.2, 4.0, 5.3]], dtype=np.float32)
    y = np.array([[2.6, 3.7, 8.9], [1.3, 2.3, 6.7]], dtype=np.float32)
    res = np.zeros_like(x)

    root_dir = Path(__file__).parent.parent
    lib = compiler.compile(alloc_nest_malloc,
                           include_dir=str(root_dir / "src/exo/libs"),
                           additional_file=str(root_dir / "src/exo/libs/custom_malloc.c"))

    # Initialize custom malloc here
    lib.init_mem()
    lib(None, *x.shape, x, y, res)

    np.testing.assert_almost_equal(res, np.array(
        [[3.6, 5.7, 11.9], [4.5, 6.3, 12.0]], dtype=np.float32))


def test_unary_neg(compiler):
    @proc
    def negate_array(n: size, x: R[n], res: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, n):
            res[i] = -x[i] + -(x[i]) - -(x[i] + 0.0)

    x = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    res = np.zeros_like(x)

    fn = compiler.compile(negate_array)
    fn(None, res.shape[0], x, res)

    np.testing.assert_almost_equal(res, -x)


def test_bool1(compiler):
    @proc
    def foo(x: bool):
        pass

    @proc
    def bool1(a: bool, b: bool):
        assert b == True

        foo(False)
        x: f32
        if b == True:
            x = 0.0
        else:
            x = 1.0

        if a == b:
            x = 0.0

        if False:
            x = 0.0

        x: f32
        if a:
            x = 0.0

    compiler.compile(bool1, compile_only=True)


def test_sin1(compiler):
    @proc
    def sin1(x: f32):
        x = sin(x)

    buf = np.array([4.0], dtype=np.float32)
    out = np.sin(buf)

    fn = compiler.compile(sin1)
    fn(None, buf)

    np.testing.assert_almost_equal(buf, out)


def test_relu1(compiler):
    @proc
    def relu1(x: f32):
        x = relu(x)

    actual = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    expected = actual * (actual > 0)

    fn = compiler.compile(relu1)
    fn(None, actual)

    np.testing.assert_almost_equal(actual, expected)


def test_select1(compiler):
    @proc
    def select1(x: f32):
        zero: f32
        zero = 0.0
        two: f32
        two = 2.0
        # x < zero ? 2.0 : x
        x = select(x, zero, two, x)

    actual = np.array([-4.0, 0.0, 4.0], dtype=np.float32)
    expected = (actual < 0) * 2.0 + (actual >= 0) * actual

    fn = compiler.compile(select1)
    fn(None, actual)

    np.testing.assert_almost_equal(actual, expected)
