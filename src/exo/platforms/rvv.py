from __future__ import annotations

from exo import Memory, DRAM, instr


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


def _is_some_const_size(sz):
    return sz.isdecimal() and int(sz) > 0


# --------------------------------------------------------------------------- #
#   Neon registers
# --------------------------------------------------------------------------- #


class RVV(Memory):
    @classmethod
    def global_(cls):
        return "#include <riscv_vector.h>"

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: RVV vectors are not scalar values")

        vec_types = {
            "float": (4, "vfloat32m1_t")
        }  # , "double": (2, "float64x2_t"), "_Float16" : (8, "float16x8_t")}

        if not prim_type in vec_types.keys():
            raise MemGenError(f"{srcinfo}: RVV vectors must be f32 (for now)")

        reg_width, C_reg_type_name = vec_types[prim_type]

        if not _is_const_size(shape[-1], reg_width):

            # This will help with dynamic lengths (I hope)
            if int(shape[-1]) > reg_width:
                raise MemGenError(
                    f"{srcinfo}: RVV vectors of type {prim_type} must be {reg_width}-wide, got {shape}"
                )
        shape = shape[:-1]
        if shape:
            if not all(_is_some_const_size(s) for s in shape):
                raise MemGenError(
                    f"{srcinfo}: Cannot allocate variable numbers of RVV vectors"
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
#   f32 RVV intrinsics
# --------------------------------------------------------------------------- #

#
# Load, Store, Broadcast, FMAdd, Mul, Add?
#
# float32


@instr("{dst_data} = __riscv_vle32_v_f32m1(&{src_data},{vl});")
def rvv_vld_4xf32(dst: [f32][4] @ RVV, src: [f32][4] @ DRAM, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = src[i]


@instr("__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});")
def rvv_vst_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ RVV, vl: size):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = src[i]


@instr("{dst_data} = __riscv_vfmv_v_f_f32m1({src_data},{vl});")
def rvv_broadcast_4xf32(dst: [f32][4] @ RVV, src: [f32][1] @ DRAM, vl: size):
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = src[0]


@instr("{dst_data} = __riscv_vfmv_v_f_f32m1({src_data},{vl});")
def rvv_broadcast_4xf32_scalar(dst: [f32][4] @ RVV, src: f32 @ DRAM, vl: size):
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = src


@instr("{dst_data} = __riscv_vfmv_v_f_f32m1(0.0f,{vl});")
def rvv_broadcast_4xf32_0(dst: [f32][4] @ RVV, vl: size):
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = 0.0


@instr("{dst_data} = __riscv_vfmacc_vv_f32m1({dst_data}, {lhs_data}, {rhs_data},{vl});")
def rvv_vfmacc_4xf32_4xf32(
    dst: [f32][4] @ RVV, lhs: [f32][4] @ RVV, rhs: [f32][4] @ RVV, vl: size
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] += lhs[i] * rhs[i]


@instr("{dst_data} = __riscv_vfmacc_vf_f32m1{dst_data}, {rhs_data}, {lhs_data},{vl});")
def rvv_vfmacc_4xf32_1xf32(
    dst: [f32][4] @ RVV, lhs: [f32][4] @ RVV, rhs: [f32][1] @ DRAM, vl: size
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] += lhs[i] * rhs[0]


@instr("{dst_data} = __riscv_vfmacc_vf_f32m1{dst_data}, {lhs_data}, {rhs_data},{vl});")
def rvv_vfmacc_1xf32_4xf32(
    dst: [f32][4] @ RVV, lhs: [f32][1] @ DRAM, rhs: [f32][4] @ RVV, vl: size
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] += lhs[0] * rhs[i]
