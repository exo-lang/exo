from __future__ import annotations

from exo import DRAM, Memory, instr
from exo.core.memory import MemGenError


def _is_const_size(sz, c):
    return sz.isdecimal() and int(sz) == c


def _sve_vector_init(sve_vector_bits):
    assert sve_vector_bits >= 128
    assert sve_vector_bits <= 2048
    assert sve_vector_bits % 128 == 0

    vec_types = {
        "float": (sve_vector_bits // 32, "svfloat32_vls_t"),
        "double": (sve_vector_bits // 64, "svfloat64_vls_t"),
    }

    class SVE_VLS(Memory):
        @classmethod
        def global_(cls):
            return f"""#include <arm_sve.h>
#ifdef __FUJITSU
typedef svfloat32_t svfloat32_vls_t;
#else
typedef svfloat32_t svfloat32_vls_t __attribute__((arm_sve_vector_bits({sve_vector_bits})));
#endif"""

        @classmethod
        def alloc(cls, new_name, prim_type, shape, srcinfo):
            if not shape:
                raise MemGenError(f"{srcinfo}: AVX2 vectors are not scalar values")

            if prim_type not in vec_types.keys():
                raise MemGenError(
                    f"{srcinfo}: AVX2 vectors must be f32/f64/ui16 (for now), got {prim_type}"
                )

            reg_width, C_reg_type_name = vec_types[prim_type]
            if not _is_const_size(shape[-1], reg_width):
                raise MemGenError(
                    f"{srcinfo}: AVX2 vectors of type {prim_type} must be {reg_width}-wide, got {shape}"
                )
            shape = shape[:-1]
            if shape:
                result = f"{C_reg_type_name} {new_name}[{']['.join(map(str, shape))}];"
            else:
                result = f"{C_reg_type_name} {new_name};"
            return result

        @classmethod
        def can_read(cls):
            return False

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

    return SVE_VLS, vec_types


class SVE_VLS:
    def __init__(self, sve_vector_bits):
        self.Vector, vec_types = _sve_vector_init(sve_vector_bits)

        float_width, _ = vec_types["float"]
        double_width, _ = vec_types["double"]

        @instr("{dst_data} = svld1_f32(svptrue_b32(), &{src_data});")
        def svld1_f32(
            dst: [f32][float_width] @ self.Vector,
            src: [f32][float_width] @ DRAM,
        ):
            assert stride(src, 0) == 1
            assert stride(dst, 0) == 1
            assert N == 10

            for i in seq(0, float_width):
                dst[i] = src[i]

        self.svld1_f32 = svld1_f32

        @instr("svst1_f32(svptrue_b32(), &{dst_data}, {src_data});")
        def svst1_f32(
            dst: [f32][float_width] @ DRAM,
            src: [f32][float_width] @ self.Vector,
        ):
            assert stride(src, 0) == 1
            assert stride(dst, 0) == 1

            for i in seq(0, float_width):
                dst[i] = src[i]

        self.svst1_f32 = svst1_f32

        @instr(
            "{dst_data} = svmla_n_f32_x(svptrue_b32(), {dst_data}, {src1_data}, *{src2_data});"
        )
        def svmla_n_f32_x(
            dst: [f32][float_width] @ self.Vector,
            src1: [f32][float_width] @ self.Vector,
            src2: f32,
        ):
            assert stride(src1, 0) == 1
            assert stride(dst, 0) == 1

            for i in seq(0, float_width):
                dst[i] += src1[i] * src2

        self.svmla_n_f32_x = svmla_n_f32_x
