from __future__ import annotations

from .. import instr, DRAM
from ..libs.memories import AVX2, AVX512


# --------------------------------------------------------------------------- #
#   AVX2 intrinsics
# --------------------------------------------------------------------------- #


@instr("{dst_data} = _mm256_setzero_ps();")
def mm256_setzero_ps(dst: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = 0.0


@instr("{dst_data} = _mm256_setzero_pd();")
def mm256_setzero_pd(dst: [f64][4] @ AVX2):
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = 0.0


@instr("{dst_data} = _mm256_loadu_ps(&{src_data});")
def mm256_loadu_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("{dst_data} = _mm256_loadu_pd(&{src_data});")
def mm256_loadu_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("_mm256_storeu_ps(&{dst_data}, {src_data});")
def mm256_storeu_ps(dst: [f32][8] @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("_mm256_storeu_pd(&{dst_data}, {src_data});")
def mm256_storeu_pd(dst: [f64][4] @ DRAM, src: [f64][4] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = _mm256_fmadd_ps({src1_data}, {src2_data}, {dst_data});")
def mm256_fmadd_ps(
    dst: [f32][8] @ AVX2,
    src1: [f32][8] @ AVX2,
    src2: [f32][8] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] += src1[i] * src2[i]


@instr("{dst_data} = _mm256_fmadd_pd({src1_data}, {src2_data}, {dst_data});")
def mm256_fmadd_pd(
    dst: [f64][4] @ AVX2,
    src1: [f64][4] @ AVX2,
    src2: [f64][4] @ AVX2,
):
    assert stride(src1, 0) == 1
    assert stride(src2, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] += src1[i] * src2[i]


@instr("{out_data} = _mm256_broadcast_ss(&{val_data});")
def mm256_broadcast_ss(
    out: [f32][8] @ AVX2,
    val: [f32][1],
):
    assert stride(out, 0) == 1

    for i in seq(0, 8):
        out[i] = val[0]


@instr("{out_data} = _mm256_broadcast_sd(&{val_data});")
def mm256_broadcast_sd(
    out: [f64][4] @ AVX2,
    val: [f64][1],
):
    assert stride(out, 0) == 1

    for i in seq(0, 4):
        out[i] = val[0]


@instr("{out_data} = _mm256_broadcast_ss({val_data});")
def mm256_broadcast_ss_scalar(out: [f32][8] @ AVX2, val: f32):
    assert stride(out, 0) == 1

    for i in seq(0, 8):
        out[i] = val


@instr("{out_data} = _mm256_broadcast_sd({val_data});")
def mm256_broadcast_sd_scalar(out: [f64][4] @ AVX2, val: f64):
    assert stride(out, 0) == 1

    for i in seq(0, 4):
        out[i] = val


@instr("{dst_data} = _mm512_fmadd_ps({dst_data}, {lhs_data}, {rhs_data});")
def mm256_fmadd_ps_broadcast(
    dst: [f32][8] @ AVX2, lhs: [f32][8] @ AVX2, rhs: [f32][1] @ DRAM
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 8):
        dst[i] += lhs[i] * rhs[0]


@instr("{out_data} = _mm256_mul_ps({x_data}, {y_data});")
def mm256_mul_ps(out: [f32][8] @ AVX2, x: [f32][8] @ AVX2, y: [f32][8] @ AVX2):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in seq(0, 8):
        out[i] = x[i] * y[i]


@instr("{out_data} = _mm256_mul_pd({x_data}, {y_data});")
def mm256_mul_pd(out: [f64][4] @ AVX2, x: [f64][4] @ AVX2, y: [f64][4] @ AVX2):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in seq(0, 4):
        out[i] = x[i] * y[i]


@instr("{out_data} = _mm256_add_ps({x_data}, {y_data});")
def mm256_add_ps(out: [f32][8] @ AVX2, x: [f32][8] @ AVX2, y: [f32][8] @ AVX2):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in seq(0, 8):
        out[i] = x[i] + y[i]


@instr("{out_data} = _mm256_add_pd({x_data}, {y_data});")
def mm256_add_pd(out: [f64][4] @ AVX2, x: [f64][4] @ AVX2, y: [f64][4] @ AVX2):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in seq(0, 4):
        out[i] = x[i] + y[i]


# --------------------------------------------------------------------------- #
#   AVX512 intrinsics
# --------------------------------------------------------------------------- #


@instr("{dst_data} = _mm512_loadu_ps(&{src_data});")
def mm512_loadu_ps(dst: [f32][16] @ AVX512, src: [f32][16] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 16):
        dst[i] = src[i]


@instr("_mm512_storeu_ps(&{dst_data}, {src_data});")
def mm512_storeu_ps(dst: [f32][16] @ DRAM, src: [f32][16] @ AVX512):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 16):
        dst[i] = src[i]


@instr("{dst_data} = _mm512_maskz_loadu_ps(((1 << {N}) - 1), &{src_data});")
def mm512_maskz_loadu_ps(
    N: size,
    dst: [f32][16] @ AVX512,
    src: [f32][N] @ DRAM,
):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert N <= 16

    for i in seq(0, 16):
        if i < N:
            dst[i] = src[i]


@instr("_mm512_mask_storeu_ps(&{dst_data}, ((1 << {N}) - 1), {src_data});")
def mm512_mask_storeu_ps(N: size, dst: [f32][N] @ DRAM, src: [f32][16] @ AVX512):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert N <= 16

    for i in seq(0, 16):
        if i < N:
            dst[i] = src[i]


@instr("{C_data} = _mm512_fmadd_ps({A_data}, {B_data}, {C_data});")
def mm512_fmadd_ps(
    A: [f32][16] @ AVX512,
    B: [f32][16] @ AVX512,
    C: [f32][16] @ AVX512,
):
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in seq(0, 16):
        C[i] += A[i] * B[i]


@instr(
    "{C_data} = _mm512_mask_fmadd_ps({A_data}, ((1 << {N}) - 1), {B_data}, {C_data});"
)
def mm512_mask_fmadd_ps(
    N: size,
    A: [f32][16] @ AVX512,
    B: [f32][16] @ AVX512,
    C: [f32][16] @ AVX512,
):
    assert N >= 1
    assert N < 16
    assert stride(A, 0) == 1
    assert stride(B, 0) == 1
    assert stride(C, 0) == 1

    for i in seq(0, 16):
        if i < N:
            C[i] += A[i] * B[i]


@instr("{dst_data} = _mm512_max_ps({src_data}, (__m512){{0}});")
def mm512_relu_ps(dst: [f32][16] @ AVX512, src: [f32][16] @ AVX512):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 16):
        dst[i] = relu(src[i])


# ---------------------------------------------------------------------------- #


# TODO: this "dumb" instruction exists as a hack around our current lack of
#  overcompute. We'll revisit the proper way of doing this post-deadline.


@instr("{dst_data} = _mm512_set1_ps({src_data});")
def mm512_mask_set1_ps(
    N: size,
    dst: [f32][16] @ AVX512,
    src: [f32][1],
):
    assert N >= 1
    assert N < 16
    assert stride(dst, 0) == 1

    for i in seq(0, 16):
        if i < N:
            dst[i] = src[0]


# ---------------------------------------------------------------------------- #


@instr("{dst_data} = _mm512_set1_ps({src_data});")
def mm512_set1_ps(
    dst: [f32][16] @ AVX512,
    src: [f32][1],
):
    assert stride(dst, 0) == 1

    for i in seq(0, 16):
        dst[i] = src[0]


# --------------------------------------------------------------------------- #
#   Complex AVX2 operations
# --------------------------------------------------------------------------- #


@instr("{out_data} = _mm256_xor_ps({out_data}, {out_data});")
def avx2_set0_ps(out: [f32][8] @ AVX2):
    assert stride(out, 0) == 1

    for i in seq(0, 8):
        out[i] = 0.0


@instr(
    """
{{
  __m256 ones = {{ 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f }};
  __m256 dst = _mm256_loadu_ps(&{dst_data});
  _mm256_storeu_ps(&{dst_data}, _mm256_fmadd_ps(ones, dst, {val_data}));
}}
"""
)
def avx2_fmadd_memu_ps(dst: [f32][8] @ DRAM, val: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(val, 0) == 1

    for i in seq(0, 8):
        dst[i] += val[i]


@instr(
    """
{out_data} = _mm256_blendv_ps ({z_data}, {y_data}, 
_mm256_cmp_ps ({x_data}, {v_data}, _CMP_LT_OQ));
"""
)
def avx2_select_ps(
    out: [f32][8] @ AVX2,
    x: [f32][8] @ AVX2,
    v: [f32][8] @ AVX2,
    y: [f32][8] @ AVX2,
    z: [f32][8] @ AVX2,
):
    # WARNING: This instruction above use a lower precision
    #    float32 (C float) than the implementation of
    #    the builtin which uses float64 (C double)
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(v, 0) == 1
    assert stride(y, 0) == 1
    assert stride(z, 0) == 1

    for i in seq(0, 8):
        out[i] = select(x[i], v[i], y[i], z[i])


@instr(
    """
{out_data} = _mm256_blendv_pd ({z_data}, {y_data},
_mm256_cmp_pd ({x_data}, {v_data}, _CMP_LT_OQ));
"""
)
def avx2_select_pd(
    out: [f64][4] @ AVX2,
    x: [f64][4] @ AVX2,
    v: [f64][4] @ AVX2,
    y: [f64][4] @ AVX2,
    z: [f64][4] @ AVX2,
):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(v, 0) == 1
    assert stride(y, 0) == 1
    assert stride(z, 0) == 1

    for i in seq(0, 4):
        out[i] = select(x[i], v[i], y[i], z[i])


@instr(
    """
    {{
        __m256 tmp = _mm256_hadd_ps({x_data}, {x_data});
        tmp = _mm256_hadd_ps(tmp, tmp);
        __m256 upper_bits = _mm256_castps128_ps256(_mm256_extractf128_ps(tmp, 1));
        tmp = _mm256_add_ps(tmp, upper_bits);
        *{result} += _mm256_cvtss_f32(tmp);
    }}
    """
)
def avx2_assoc_reduce_add_ps(x: [f32][8] @ AVX2, result: f32):
    # WARNING: This instruction assumes float addition associativity
    assert stride(x, 0) == 1
    for i in seq(0, 8):
        result += x[i]


@instr(
    """
    {{
        __m256d tmp = _mm256_hadd_pd({x_data}, {x_data});
        __m256d upper_bits = _mm256_castpd128_pd256(_mm256_extractf128_pd (tmp, 1));
        tmp = _mm256_add_pd(tmp, upper_bits);
        *{result} += _mm256_cvtsd_f64(tmp);
    }}
    """
)
def avx2_assoc_reduce_add_pd(x: [f64][4] @ AVX2, result: f64):
    # WARNING: This instruction assumes float addition associativity
    assert stride(x, 0) == 1
    for i in seq(0, 4):
        result += x[i]


@instr("{dst_data} = _mm256_mul_ps({src_data}, _mm256_set1_ps(-1.0f));")
def avx2_sign_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 8):
        dst[i] = -src[i]


@instr("{dst_data} = _mm256_mul_pd({src_data}, _mm256_set1_pd(-1.0f));")
def avx2_sign_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = -src[i]


@instr("{dst_data} = _mm256_add_ps({src_data}, {dst_data});")
def avx2_reduce_add_wide_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 8):
        dst[i] += src[i]


@instr("{dst_data} = _mm256_add_pd({src_data}, {dst_data});")
def avx2_reduce_add_wide_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] += src[i]


# TODO: Hack for procedure aliasing issue, can be deleted once we have
#      better way of handling aliasing
@instr("{dst_data} = {src_data};")
def avx2_reg_copy_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("{dst_data} = {src_data};")
def avx2_reg_copy_pd(dst: [f64][4] @ AVX2, src: [f64][4] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr(
    "__m256i opaque = _mm256_set1_epi8((1<<{N}) - 1);\n"
    + "_mm256_maskstore_ps(&{dst_data}, opaque, {src_data});"
)
def avx2_mask_storeu_ps(N: size, dst: [f32][N] @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert N <= 8

    for i in seq(0, 8):
        if i < N:
            dst[i] = src[i]


# --------------------------------------------------------------------------- #
#   f32 to f64 conversion
# --------------------------------------------------------------------------- #


@instr("{dst_data} = _mm256_cvtps_pd(_mm256_extractf128_ps({src_data}, 0));")
def avx2_convert_f32_lower_to_f64(dst: [f64][4] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = _mm256_cvtps_pd(_mm256_extractf128_ps({src_data}, 1));")
def avx2_convert_f32_upper_to_f64(dst: [f64][4] @ AVX2, src: [f32][8] @ AVX2):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[4 + i]
