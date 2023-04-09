from __future__ import annotations

from exo import Memory, DRAM, instr


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


def _is_some_const_size(sz):
    return sz.isdecimal() and int(sz) > 0


# --------------------------------------------------------------------------- #
#   Neon registers
# --------------------------------------------------------------------------- #


class Neon(Memory):
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

        vec_types = {"float": (4, "float32x4_t"), "double": (4, "float64x2_t")}

        if not prim_type in vec_types.keys():
            raise MemGenError(f"{srcinfo}: Neon vectors must be f32/f64 (for now)")

        reg_width, C_reg_type_name = vec_types[prim_type]

        if not _is_const_size(shape[-1], reg_width):
            raise MemGenError(
                f"{srcinfo}: Neon vectors of type {prim_type} must be {reg_width}-wide, got {shape}"
            )
        shape = shape[:-1]
        if shape:
            if not all(_is_some_const_size(s) for s in shape):
                raise MemGenError(
                    f"{srcinfo}: Cannot allocate variable numbers of Neon vectors"
                )
            result = f'{C_reg_type_name} {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"{C_reg_type_name} {new_name};"

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


@instr("*{result} += vaddvq_f32({x_data});")
def neon_assoc_reduce_add_instr_4xf32(result: f32 @ DRAM, x: [f32][4] @ Neon):
    for i in seq(0, 4):
        result += x[i]


@instr("{dst_data} = vld1q_f32(&{src_data});")
def neon_vld_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("vst1q_f32(&{dst_data}, {src_data});")
def neon_vst_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ Neon):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = vld1q_dup_f32(&{src_data});")
def neon_broadcast_4xf32(dst: [f32][4] @ Neon, src: [f32][1] @ DRAM):
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[0]


@instr("{dst_data} = vld1q_dup_f32({src_data});")
def neon_broadcast_4xf32_scalar(dst: [f32][4] @ Neon, src: f32 @ DRAM):
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src


@instr("{dst_data} = vmovq_n_f32(0.0f);")
def neon_zero_4xf32(dst: [f32][4] @ Neon):
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = 0.0


@instr("{dst_data} = vaddq_f32({lhs_data}, {rhs_data});")
def neon_vadd_4xf32(dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = lhs[i] + rhs[i]


@instr("{dst_data} = vaddq_f32({src_data}, {dst_data});")
def neon_reduce_vadd_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] += src[i]


@instr("{dst_data} = vmulq_f32({lhs_data}, {rhs_data});")
def neon_vmul_4xf32(dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = lhs[i] * rhs[i]


@instr("{dst_data} = vfmaq_laneq_f32({dst_data}, {lhs_data}, {rhs_data}, {lane});")
def neon_vfmla_4xf32_4xf32(
    dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon, lane: index
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
    dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[i]


@instr("{dst_data} = vmlaq_n_f32({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_4xf32_1xf32(
    dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][1] @ DRAM
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[0]


@instr("{dst_data} = vmlaq_n_f32({dst_data}, {rhs_data}, {lhs_data});")
def neon_vfmadd_1xf32_4xf32(
    dst: [f32][4] @ Neon, lhs: [f32][1] @ DRAM, rhs: [f32][4] @ Neon
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[0] * rhs[i]


# TODO: Hack for procedure aliasing issue, can be deleted once we have
#      better way of handling aliasing
@instr("{dst_data} = {src_data};")
def neon_reg_copy_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]


@instr("{dst_data} = vnegq_f32({src_data});")
def neon_vneg_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ Neon):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, 4):
        dst[i] = -src[i]
