from __future__ import annotations

from exo import DRAM, instr


@instr(
    "svmla_n_f32_x_vla({N_data}, &{dst_data}, &{src1_data}, *{src2_data});",
    c_global='#include "exo_arm_sve.h"',
)
def svmla_n_f32_x_vla(
    N: size,
    dst: [f32][N] @ DRAM,
    src1: [f32][N] @ DRAM,
    src2: f32,
):
    assert stride(src1, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, N):
        dst[i] += src1[i] * src2
