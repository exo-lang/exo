from __future__ import annotations

import itertools
import os
import platform

import pytest

from exo import proc
from exo.platforms.neon import *

from ctypes import POINTER, c_int
import numpy as np

def check_platform():
    return platform.system() != 'Darwin'
    #if platform.system() == 'Darwin':
    #    pytest.skip("skipping Neon tests on non-Apple machines for now",
    #                allow_module_level=True)

@pytest.mark.skip()
def test_neon_memcpy():
    """
    Compute dst = src
    """

    @proc
    def memcpy_neon(n: size, dst: R[n] @ DRAM,
                    src: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, (n + 3) / 4):
            if n - 4 * i >= 4:
                tmp: f32[4] @ Neon4f
                neon_vld_4xf32(tmp, src[4 * i:4 * i + 4])
                neon_vst_4xf32(dst[4 * i:4 * i + 4], tmp)
            else:
                for j in par(0, n - 4 * i):
                    dst[4 * i + j] = src[4 * i + j]

    basename = test_neon_memcpy.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(memcpy_neon))

    memcpy_neon.compile_c(TMP_DIR, basename)

    if check_platform():
        return
    # TODO: -mcpu=apple-a14 here is a hack. Such flags should be
    #   somehow handled automatically
    library = generate_lib(basename, extra_flags="-mcpu=apple-a14")

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = nparray([float(i) for i in range(n)])
        out = nparray([float(0) for _ in range(n)])
        library.memcpy_neon(POINTER(c_int)(), n, cvt_c(out), cvt_c(inp))

        assert np.array_equal(inp, out)

@pytest.mark.skip()
def test_neon_simple_math():
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_neon(n: size,
                         x: R[n] @ DRAM,
                         y: R[n] @ DRAM):  # pragma: no cover
        assert n % 4 == 0
        for i in par(0, n / 4):
            xVec: f32[4] @ Neon4f
            yVec: f32[4] @ Neon4f
            neon_vld_4xf32(xVec, x[4 * i:4 * i + 4])
            neon_vld_4xf32(yVec, y[4 * i:4 * i + 4])
            neon_vmul_4xf32(xVec, xVec, yVec)
            neon_vmul_4xf32(xVec, xVec, yVec)
            neon_vst_4xf32(x[4 * i:4 * i + 4], xVec)

    basename = test_neon_simple_math.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(simple_math_neon))

    simple_math_neon.compile_c(TMP_DIR, basename)
    if check_platform():
        return
    library = generate_lib(basename, extra_flags="-mcpu=apple-a14")

    for n in (4, 8, 12, 16, 24, 32, 64, 128):
        x = nparray([float(i) for i in range(n)])
        y = nparray([float(3 * i) for i in range(n)])
        expected = x * y * y

        library.simple_math_neon(POINTER(c_int)(), n, cvt_c(x), cvt_c(y))
        assert np.allclose(x, expected)

@pytest.mark.skip()
def test_neon_simple_math_scheduling():
    """
    Compute x = x * y^2
    """

    @proc
    def simple_math_neon_sched(n: size,
                               x: R[n] @ DRAM,
                               y: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, n):
            x[i] = x[i] * y[i] * y[i]

    print()
    print()

    simple_math_neon_sched = (
        simple_math_neon_sched
            .split('i', 4, ['io', 'ii'], tail='cut_and_guard')
            .stage_assn('xyy', 'x[_] = _ #0')
            .lift_alloc('xyy: _')
            .fission_after('xyy[_] = _')

            .bind_expr('xVec', 'x[_]')
            .lift_alloc('xVec: _')
            .fission_after('xVec[_] = _')

            .bind_expr('yVec', 'y[_]', cse=True)
            .lift_alloc('yVec: _')
            .fission_after('yVec[_] = _')

            .bind_expr('xy', 'xVec[_] * yVec[_]')
            .lift_alloc('xy: _')
            .fission_after('xy[_] = _')

            .set_memory('xVec', Neon4f)
            .set_memory('yVec', Neon4f)
            .set_memory('xy', Neon4f)
            .set_memory('xyy', Neon4f)
            .replace(neon_vst_4xf32, 'for ii in _: _ #4')
            .replace_all(neon_vld_4xf32)
            .replace_all(neon_vmul_4xf32)
    )

    print(simple_math_neon_sched)

    basename = test_neon_simple_math_scheduling.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(simple_math_neon_sched))

    simple_math_neon_sched.compile_c(TMP_DIR, basename)
    if check_platform():
        return
    library = generate_lib(basename, extra_flags="-mcpu=apple-a14")

    for n in (4, 8, 12, 16, 24, 32, 64, 128):
        x = nparray([float(i) for i in range(n)])
        y = nparray([float(3 * i) for i in range(n)])
        expected = x * y * y

        int_ptr = POINTER(c_int)()
        library.simple_math_neon_sched(int_ptr, n, cvt_c(x), cvt_c(y))
        assert np.allclose(x, expected)


