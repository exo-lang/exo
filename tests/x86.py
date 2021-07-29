import sys

sys.path.append(sys.path[0] + "/..")
sys.path.append(sys.path[0] + "/.")

from SYS_ATL import instr
from SYS_ATL.libs.memories import AVX2


# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

@instr('{dst} = _mm256_fmadd_ps({src1}, {src2}, {dst});')
def fma(
    dst: f32[8] @ AVX2,
    src1: f32[8] @ AVX2,
    src2: f32[8] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] += src1[i] * src2[i]


@instr('{dst} = _mm256_broadcast_ss({value});')
def broadcast(
    value: f32,
    dst: f32[8] @ AVX2,
):
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = value


@instr('{dst} = _mm256_loadu_ps({src}.data);')
def loadu(
    dst: f32[8] @ AVX2,
    src: [f32][8] @ DRAM
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src[i]


@instr('_mm256_storeu_ps({dst}.data, {src});')
def storeu(
    dst: [f32][8] @ DRAM,
    src: f32[8] @ AVX2
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src[i]
