from __future__ import annotations

from pathlib import Path

from exo import (
    proc,
    instr,
    Procedure,
    DRAM,
    compile_procs_to_strings,
    SpecialWindow,
    memwin_template,
)
from exo.libs.memories import MDRAM, MemGenError, StaticMemory, DRAM_STACK
from exo.libs.externs import *
from exo.stdlib.scheduling import *


@memwin_template
def TestTensorMap(swizzle, *box):
    assert len(box) == 2
    smem_outer, smem_inner = box
    assert isinstance(smem_outer, int)
    assert isinstance(smem_inner, int)

    if swizzle == 0:
        cu_swizzle = "CU_TENSOR_MAP_SWIZZLE_NONE"
    else:
        assert swizzle in (32, 64, 128)
        cu_swizzle = f"CU_TENSOR_MAP_SWIZZLE_{swizzle}B"

    class Impl(SpecialWindow):
        @classmethod
        def global_(cls):
            return f"""\
#include <cuda.h>
#include <cassert>
#include <stdlib.h>
inline CUtensorMap exo_make_tensor_map_SW{swizzle}_{smem_outer}_{smem_inner}(
        const void* ptr, unsigned gmem_inner, unsigned gmem_outer)
{{
    CUtensorMap tensorMap;
    const CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
    const CUtensorMapDataType tensorDataType = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    const uint32_t tensorRank = 2;
    const cuuint64_t globalDim[2] = {{gmem_inner, gmem_outer}};
    const cuuint64_t globalStrides[1] = {{4*gmem_inner}};
    const cuuint32_t boxDim[2] = {{ {smem_inner}, {smem_outer} }};
    const cuuint32_t elementStrides[2] = {{1, 1}};
    const CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
    const CUtensorMapL2promotion l2Promotion = CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    const CUtensorMapFloatOOBfill oobFill = CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;

    const CUresult result = cuTensorMapEncodeTiled(
            tensorMap,
            tensorDataType,
            tensorRank,
            const_cast<float*>(globalAddress),
            globalDim,
            globalStrides,
            boxDim,
            elementStrides,
            interleave,
            swizzle,
            l2Promotion,
            oobFill);
    if (result != 0) {{
        fprintf(stderr, "cuTensorMapEncodeTiled: %i {{%u, %u}}\n", (int)result, {smem_inner}, {smem_outer});
        assert(0);
    }}
    return tensorMap;
}}"""

        @classmethod
        def window_definition(cls, ctx: WindowStructCtx):
            sname = ctx.struct_name("CUtensorMap", cls.memwin_template_parameters)
            s_def = f"""\
    struct {sname} {{
        unsigned inner_offset, outer_offset;
    }};"""
            return "CUtensorMap", s_def

        @classmethod
        def separate_dataptr(cls):
            return True

        @classmethod
        def window(cls, basetyp, in_expr, indices, strides, srcinfo):
            assert len(indices) == 2
            dataptr, in_layout = in_expr
            out_layout = f"{{ {in_layout}.inner_offset + {indices[1]}, {in_layout}.outer_offset + {indices[0]} }}"
            return dataptr, out_layout

        @classmethod
        def can_read(cls):
            return False

        @classmethod
        def memory_type(cls):
            return DRAM  # WRONG

        @classmethod
        def window_from_dense(cls, ctx: WindowFromDenseCtx):
            shape_strs = ctx.shape_strs()
            d_def = f"exo_make_tensor_map_SW{swizzle}_{smem_outer}_{smem_inner}((void*) {ctx.baseptr()}, {shape_strs[1]}, {shape_strs[0]})"
            w_def = "{}"
            return d_def, w_def

    return Impl


def test_tensor_map():
    @proc
    def test_proc(
        tensor: f32[1024, 2048],
        input_tensor_map: [f32][128, 128] @ TestTensorMap(0, 128, 128),
    ):
        basic_window = tensor[14, :]
        tensor_map_0 = tensor[:, :] @ TestTensorMap(0, 128, 128)
        tensor_map_1 = tensor_map_0[14:, :]
        tensor_map_C = tensor[14:, :] @ TestTensorMap(128, 196, 128)
        tensor_map_D = tensor_map_C[10:, 200:]

    c = test_proc.find("tensor_map_C = _")
    assert c.special_window() is TestTensorMap(128, 196, 128)
    assert c.special_window() is not TestTensorMap(0, 128, 128)

    cc, hh = compile_procs_to_strings([test_proc], "test.h")
    print(cc, hh)
    print(test_proc)
