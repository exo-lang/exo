from __future__ import annotations

import sys

sys.path.append(sys.path[0] + "/..")
sys.path.append(sys.path[0] + "/.")

from SYS_ATL import instr, DRAM
from SYS_ATL.libs.memories import AVX2


# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

@instr('*(__m256*){dst}.data = '
       '_mm256_fmadd_ps({src1}, {src2}, *(__m256*){dst}.data);')
def fma(
    dst: [f32][8] @ AVX2,
    src1: f32[8] @ AVX2,
    src2: f32[8] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] += src1[i] * src2[i]

@instr('''
{{
  __m256 ones = {{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }};
  __m256 dst = _mm256_loadu_ps({dst}.data);
  *(__m256*){val}.data = 
    _mm256_fmadd_ps(ones, dst, *(__m256*){val}.data);
  _mm256_storeu_ps({dst}.data, *(__m256*){val}.data);
}}
''')
def mem_accum(
    dst: [f32][8] @ DRAM,
    val: [f32][8] @ AVX2
):
    assert stride(dst, 0) == 1
    assert stride(val, 0) == 1

    for i in par(0, 8):
        dst[i] += val[i]


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


@instr('{out} = _mm256_mul_ps({x}, {y});')
def mul(
    out: f32[8] @ AVX2,
    x: f32[8] @ AVX2,
    y: f32[8] @ AVX2
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in par(0, 8):
        out[i] = x[i] * y[i]


@instr('{out} = _mm256_broadcast_ss({val}.data);')
def broadcast(
    out: f32[8] @ AVX2,
    val: [f32][1],
):
    assert stride(out, 0) == 1

    for i in par(0, 8):
        out[i] = val[0]


@instr('*(__m256*){out}.data = _mm256_xor_ps(*(__m256*){out}.data, '
       '*(__m256*){out}.data);')
def clear_reg(
    out: [f32][8] @ AVX2
):
    assert stride(out, 0) == 1

    for i in par(0, 8):
        out[i] = 0.0
