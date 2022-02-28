from __future__ import annotations

from .. import instr, DRAM
from ..libs.memories import AVX2, AVX512


# --------------------------------------------------------------------------- #
#   AVX2 intrinsics
# --------------------------------------------------------------------------- #

@instr('{dst_data} = _mm256_loadu_ps(&{src_data});')
def mm256_loadu_ps(
        dst: [f32][8] @ AVX2,
        src: [f32][8] @ DRAM
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src[i]


@instr('_mm256_storeu_ps(&{dst_data}, {src_data});')
def mm256_storeu_ps(
        dst: [f32][8] @ DRAM,
        src: [f32][8] @ AVX2
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] = src[i]


@instr('{dst_data} = _mm256_fmadd_ps({src1}, {src2}, {dst_data});')
def mm256_fmadd_ps(
        dst: [f32][8] @ AVX2,
        src1: f32[8] @ AVX2,
        src2: f32[8] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 8):
        dst[i] += src1[i] * src2[i]


@instr('{out} = _mm256_broadcast_ss(&{val_data});')
def mm256_broadcast_ss(
        out: f32[8] @ AVX2,
        val: [f32][1],
):
    assert stride(out, 0) == 1

    for i in par(0, 8):
        out[i] = val[0]


@instr('{out} = _mm256_mul_ps({x}, {y});')
def mm256_mul_ps(
        out: f32[8] @ AVX2,
        x: f32[8] @ AVX2,
        y: f32[8] @ AVX2
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in par(0, 8):
        out[i] = x[i] * y[i]


# --------------------------------------------------------------------------- #
#   AVX512 intrinsics
# --------------------------------------------------------------------------- #

@instr('{dst_data} = _mm512_loadu_ps(&{src_data});')
def mm512_loadu_ps(
        dst: [f32][16] @ AVX512,
        src: [f32][16] @ DRAM
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in par(0, 16):
        dst[i] = src[i]


@instr('_mm512_storeu_ps(&{dst_data}, {src_data});')
def mm512_storeu_ps(
        dst: [f32][16] @ DRAM,
        src: [f32][16] @ AVX512
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in par(0, 16):
        dst[i] = src[i]


@instr('{dst_data} = _mm512_maskz_loadu_ps(((1 << {N}) - 1), &{src_data});')
def mm512_maskz_loadu_ps(
        N: size,
        dst: [f32][16] @ AVX512,
        src: [f32][N] @ DRAM,
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert N <= 16

    for i in par(0, 16):
        if i < N:
            dst[i] = src[i]


@instr('_mm512_mask_storeu_ps(&{dst_data}, ((1 << {N}) - 1), {src_data});')
def mm512_mask_storeu_ps(
        N: size,
        dst: [f32][N] @ DRAM,
        src: [f32][16] @ AVX512
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert N <= 16

    for i in par(0, 16):
        if i < N:
            dst[i] = src[i]


@instr('{C_data} = _mm512_fmadd_ps({A}, {B}, {C_data});')
def mm512_fmadd_ps(
        A: f32[16] @ AVX512,
        B: f32[16] @ AVX512,
        C: [f32][16] @ AVX512,
):
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in par(0, 16):
        C[i] += A[i] * B[i]


@instr('{C_data} = _mm512_mask_fmadd_ps({A}, ((1 << {N}) - 1), {B}, {C_data});')
def mm512_mask_fmadd_ps(
        N: size,
        A: f32[16] @ AVX512,
        B: f32[16] @ AVX512,
        C: [f32][16] @ AVX512,
):
    assert N >= 1
    assert N < 16
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in par(0, 16):
        if i < N:
            C[i] += A[i] * B[i]


@instr('{dst_data} = _mm512_max_ps({src_data}, (__m512){{0}});')
def mm512_relu_ps(
        dst: [f32][16] @ AVX512,
        src: [f32][16] @ AVX512
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in par(0, 16):
        dst[i] = relu(src[i])


# ---------------------------------------------------------------------------- #


# TODO: this "dumb" instruction exists as a hack around our current lack of
#  overcompute. We'll revisit the proper way of doing this post-deadline.

@instr('{dst} = _mm512_set1_ps({src_data});')
def mm512_mask_set1_ps(
        N: size,
        dst: f32[16] @ AVX512,
        src: [f32][1],
):
    assert N >= 1
    assert N < 16
    assert stride(dst, 0) == 1

    for i in par(0, 16):
        if i < N:
            dst[i] = src[0]


# ---------------------------------------------------------------------------- #

@instr('{dst} = _mm512_set1_ps({src_data});')
def mm512_set1_ps(
        dst: f32[16] @ AVX512,
        src: [f32][1],
):
    assert stride(dst, 0) == 1

    for i in par(0, 16):
        dst[i] = src[0]


# --------------------------------------------------------------------------- #
#   Complex AVX2 operations
# --------------------------------------------------------------------------- #

@instr('{out_data} = _mm256_xor_ps({out_data}, {out_data});')
def avx2_set0_ps(
        out: [f32][8] @ AVX2
):
    assert stride(out, 0) == 1

    for i in par(0, 8):
        out[i] = 0.0


@instr('''
{{
  __m256 ones = {{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }};
  __m256 dst = _mm256_loadu_ps(&{dst_data});
  _mm256_storeu_ps(&{dst_data}, _mm256_fmadd_ps(ones, dst, {val_data}));
}}
''')
def avx2_fmadd_memu_ps(
        dst: [f32][8] @ DRAM,
        val: [f32][8] @ AVX2
):
    assert stride(dst, 0) == 1
    assert stride(val, 0) == 1

    for i in par(0, 8):
        dst[i] += val[i]
