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
        for i in par(0, n / 8):
            tmp: f32[8] @ AVX2
            loadu(tmp, src[8 * i:8 * i + 8])
            storeu(dst[8 * i:8 * i + 8], tmp)

    assert type(memcpy_avx2) is Procedure
    basename = test_avx2_memcpy.__name__

    with open(os.path.join(TMP_DIR, f'{basename}_pretty.atl'), 'w') as f:
        f.write(str(memcpy_avx2))

    memcpy_avx2.compile_c(TMP_DIR, basename)
