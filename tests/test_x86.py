from __future__ import annotations

import sys

sys.path.append(sys.path[0] + "/..")
from SYS_ATL import DRAM
from SYS_ATL.libs.memories import AVX2
from .x86 import loadu, storeu

sys.path.append(sys.path[0] + "/.")
from .helper import *


def test_avx2_memcpy():
    @proc
    def memcpy_avx2(n: size, dst: R[n] @ DRAM, src: R[n] @ DRAM):  # pragma: no cover
        for i in par(0, (n + 7) / 8):
            if n - 8 * i >= 8:
                tmp: f32[8] @ AVX2
                loadu(tmp, src[8 * i:8 * i + 8])
                storeu(dst[8 * i:8 * i + 8], tmp)
            else:
                for j in par(0, n - 8 * i):
                    dst[8 * i + j] = src[8 * i + j]

    assert type(memcpy_avx2) is Procedure
    basename = test_avx2_memcpy.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(memcpy_avx2))

    memcpy_avx2.compile_c(TMP_DIR, basename)

    # TODO: -march=native here is a hack. Such flags should be somehow handled automatically.
    #       Maybe this should be inferred by the use of AVX2, but "native" isn't right anyway.
    #       We might need a first-class notion of a Target, which has certain memories available.
    #       Then we can say that e.g. Skylake-X has AVX2, AVX512, etc.
    library = generate_lib(basename, extra_flags="-march=native")

    for n in (7, 8, 9, 31, 32, 33, 127, 128, 129):
        inp = nparray([float(i) for i in range(n)])
        out = nparray([float(0) for _ in range(n)])
        library.memcpy_avx2(n, cvt_c(out), cvt_c(inp))

        assert np.array_equal(inp, out)
