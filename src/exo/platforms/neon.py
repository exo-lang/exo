from __future__ import annotations

from exo import Memory, DRAM, instr


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


def _is_some_const_size(sz):
    return sz.isdecimal() and int(sz) > 0


# --------------------------------------------------------------------------- #
#   Neon registers
# --------------------------------------------------------------------------- #


class Neon4f(Memory):
    @classmethod
    def global_(cls):
        return "#include <arm_neon.h>"

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: Neon vectors are not scalar values")
        if not prim_type == "float":
            raise MemGenError(f"{srcinfo}: Neon4f vectors must be f32")
        if not _is_const_size(shape[-1], 4):
            raise MemGenError(f"{srcinfo}: Neon4f vectors must be 4-wide")
        shape = shape[:-1]
        if shape:
            if not all(_is_some_const_size(s) for s in shape):
                raise MemGenError(
                    f"{srcinfo}: Cannot allocate variable " f"numbers of Neon4f vectors"
                )
            result = f'float32x4_t {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"float32x4_t {new_name};"
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"

class Neon8f(Memory):
    @classmethod
    def global_(cls):
        return "#include <arm_neon.h>"

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: Neon vectors are not scalar values")
        #if not prim_type == "float16_t" or not prim_type == "__fp16":
        if not prim_type == "__fp16":
            raise MemGenError(f"{srcinfo}: Neon8f vectors must be f16")
        if not _is_const_size(shape[-1], 8):
            raise MemGenError(f"{srcinfo}: Neon8f vectors must be 8-wide")
        shape = shape[:-1]
        if shape:
            if not all(_is_some_const_size(s) for s in shape):
                raise MemGenError(
                    f"{srcinfo}: Cannot allocate variable " f"numbers of Neon8f vectors"
                )
            result = f'float16x8_t {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"float16x8_t {new_name};"
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"

# --------------------------------------------------------------------------- #
#   Neon intrinsics
# --------------------------------------------------------------------------- #

#
# Load, Store, Broadcast, FMAdd, Mul, Add?
#
# float32

@instr("{dst_data} = vld1q_f32(&{src_data});")
def neon_vld_4xf32(dst: [f32][4] @ Neon4f, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("vst1q_f32(&{dst_data}, {src_data});")
def neon_vst_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ Neon4f):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = vld1q_dup_f32(&{src_data});")
def neon_broadcast_4xf32(dst: [f32][4] @ Neon4f, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[0]


@instr("{dst_data} = vmovq_n_f32(0.0f);")
def neon_zero_4xf32(dst: [f32][4] @ Neon4f):
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = 0.0


@instr("{dst_data} = vaddq_f32({lhs_data}, {rhs_data});")
def neon_vadd_4xf32(
    dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = lhs[i] + rhs[i]


@instr("{dst_data} = vmulq_f32({lhs_data}, {rhs_data});")
def neon_vmul_4xf32(
    dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = lhs[i] * rhs[i]

@instr("{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});")
def neon_vfmla_4xf32_4xf32(
        dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f, lane: index
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert lane >= 0
    assert lane < 4
    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[lane]
        

@instr("{dst_data} = vmlaq_f32({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_4xf32_4xf32(
    dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[i]

@instr("{dst_data} = vmlaq_f32({res_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_ex_4xf32_4xf32(
        dst: [f32][4] @ Neon4f, res: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4f, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(res, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = res[i] + lhs[i] * rhs[i]


@instr("{dst_data} = vmlaq_n_f32({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_4xf32_1xf32(
    dst: [f32][4] @ Neon4f, lhs: [f32][4] @ Neon4f, rhs: [f32][1] @ DRAM
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[0]


@instr("{dst_data} = vmlaq_n_f32({dst_data}, {rhs_data}, {lhs_data});")
def neon_vfmadd_1xf32_4xf32(
    dst: [f32][4] @ Neon4f, lhs: [f32][1] @ DRAM, rhs: [f32][4] @ Neon4f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[0] * rhs[i]


#-----------------------------------------------
# Load, Store, Broadcast, FMAdd, Mul, Add?
#
# float16

@instr("{dst_data} = vld1q_f16(&{src_data});")
def neon_vld_8xf16(dst: [f16][8] @ Neon8f, src: [f16][8] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("vst1q_f16(&{dst_data}, {src_data});")
def neon_vst_8xf16(dst: [f16][8] @ DRAM, src: [f16][8] @ Neon8f):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("{dst_data} = vld1q_dup_f16(&{src_data});")
def neon_broadcast_8xf16(dst: [f16][8] @ Neon8f, src: [f16][1] @ DRAM):
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[0]


@instr("{dst_data} = vmovq_n_f16(0.0f);")
def neon_zero_8xf16(dst: [f16][8] @ Neon8f):
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = 0.0


@instr("{dst_data} = vaddq_f16({lhs_data}, {rhs_data});")
def neon_vadd_8xf16(
    dst: [f16][8] @ Neon8f, lhs: [f16][8] @ Neon8f, rhs: [f16][8] @ Neon8f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 8):
        dst[i] = lhs[i] + rhs[i]


@instr("{dst_data} = vmulq_f16({lhs_data}, {rhs_data});")
def neon_vmul_8xf16(
    dst: [f16][8] @ Neon8f, lhs: [f16][8] @ Neon8f, rhs: [f16][8] @ Neon8f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 8):
        dst[i] = lhs[i] * rhs[i]

@instr("{dst_data} = vfmaq_laneq_f16({dst_data}, {lhs_data}, {rhs_data}, {lane});")
def neon_vfmla_8xf16_8xf16(
        dst: [f16][8] @ Neon8f, lhs: [f16][8] @ Neon8f, rhs: [f16][8] @ Neon8f, lane: index
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert lane >= 0
    assert lane < 8
    for i in seq(0, 8):
        dst[i] += lhs[i] * rhs[lane]
        

@instr("{dst_data} = vfmaq_f16({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_8xf16_8xf16(
    dst: [f16][8] @ Neon8f, lhs: [f16][8] @ Neon8f, rhs: [f16][8] @ Neon8f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 8):
        dst[i] += lhs[i] * rhs[i]

@instr("{dst_data} = vfmaq_f16({res_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_ex_8xf16_8xf16(
        dst: [f16][8] @ Neon8f, res: [f16][8] @ Neon8f, lhs: [f16][8] @ Neon8f, rhs: [f16][8] @ Neon8f
):
    assert stride(dst, 0) == 1
    assert stride(res, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 8):
        dst[i] = res[i] + lhs[i] * rhs[i]


@instr("{dst_data} = vfmaq_n_f16({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_8xf16_1xf16(
    dst: [f16][8] @ Neon8f, lhs: [f16][8] @ Neon8f, rhs: [f16][1] @ DRAM
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 8):
        dst[i] += lhs[i] * rhs[0]


@instr("{dst_data} = vfmaq_n_f16({dst_data}, {rhs_data}, {lhs_data});")
def neon_vfmadd_1xf16_8xf16(
    dst: [f16][8] @ Neon8f, lhs: [f16][1] @ DRAM, rhs: [f16][8] @ Neon8f
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 8):
        dst[i] += lhs[0] * rhs[i]


