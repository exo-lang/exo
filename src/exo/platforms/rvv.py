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

        vec_types = {"float": (4, "vfloat32m1_t")} #, "double": (2, "float64x2_t"), "_Float16" : (8, "float16x8_t")}

        if not prim_type in vec_types.keys():
            raise MemGenError(f"{srcinfo}: RVV vectors must be f32 (for now)")

        reg_width, C_reg_type_name = vec_types[prim_type]

        if not _is_const_size(shape[-1], reg_width):
            #This will help with dynamic lengths (I hope)
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
def rvv_vld_4xf32(dst: [f32][4] @ RVV, src: [f32][4] @ DRAM, vl: index):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = src[i]


@instr("__riscv_vse32_v_f32m1(&{dst_data}, {src_data},{vl});")
def rvv_vst_4xf32(dst: [f32][4] @ DRAM, src: [f32][4] @ RVV, vl: index):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] = src[i]

#vfloat32m1_t __riscv_vfmacc_vv_f32m1 (vfloat32m1_t vd, vfloat32m1_t vs1, vfloat32m1_t vs2, size_t vl);

@instr("{dst_data} = __riscv_vfmacc_vv_f32m1({dst_data}, {lhs_data}, {rhs_data},{vl});")
def rvv_vfmadd_4xf32_4xf32(
    dst: [f32][4] @ RVV, lhs: [f32][4] @ RVV, rhs: [f32][4] @ RVV, vl: index):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert vl >= 0
    assert vl <= 4

    for i in seq(0, vl):
        dst[i] += lhs[i] * rhs[i]


"""
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

@instr("{dst_data} = vmulq_f32({dst_data}, {rhs_data});")
def neon_vmul2_4xf32(dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon):
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

#This function uses an extra buffer for a beta=0 approach
@instr("{dst_data} = vfmaq_laneq_f32({b_data}, {lhs_data}, {rhs_data}, {lane});")
def neon_vfmla2_4xf32_4xf32(
        dst: [f32][4] @ Neon, b: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon, lane: index
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1
    assert stride(b, 0) == 1
    assert lane >= 0
    assert lane < 4
    for i in seq(0, 4):
        dst[i] = b[i] + lhs[i] * rhs[lane]

@instr("{dst_data} = vmlaq_f32({dst_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_4xf32_4xf32(
    dst: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon
):
    assert stride(dst, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] += lhs[i] * rhs[i]

@instr("{dst_data} = vmlaq_f32({res_data}, {lhs_data}, {rhs_data});")
def neon_vfmadd_ex_4xf32_4xf32(
        dst: [f32][4] @ Neon, res: [f32][4] @ Neon, lhs: [f32][4] @ Neon, rhs: [f32][4] @ Neon
):
    assert stride(dst, 0) == 1
    assert stride(res, 0) == 1
    assert stride(lhs, 0) == 1
    assert stride(rhs, 0) == 1

    for i in seq(0, 4):
        dst[i] = res[i] + lhs[i] * rhs[i]


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
"""