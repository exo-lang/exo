from __future__ import annotations

import sys


from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs, config
from SYS_ATL.libs.memories import AMX_TILE

# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

_amx_config = ("""
unsigned char config[] = {{
    0x01, // ID
    0x00, // start row
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
    64, 0x00, // bytes per row tile 0
    64, 0x00, // bytes per row tile 1
    64, 0x00, // bytes per row tile 2
    0x00, 0x00, // bytes per row tile 3
    0x00, 0x00, // bytes per row tile 4
    0x00, 0x00, // bytes per row tile 5
    0x00, 0x00, // bytes per row tile 6
    0x00, 0x00, // bytes per row tile 7
    0x00, 0x00, // bytes per row tile 8
    0x00, 0x00, // bytes per row tile 9
    0x00, 0x00, // bytes per row tile 10
    0x00, 0x00, // bytes per row tile 11
    0x00, 0x00, // bytes per row tile 12
    0x00, 0x00, // bytes per row tile 13
    0x00, 0x00, // bytes per row tile 14
    0x00, 0x00, // bytes per row tile 15
    16, // rows tile 0
    16, // rows tile 1
    16, // rows tile 2
    0x00, // rows tile 3
    0x00, // rows tile 4
    0x00, // rows tile 5
    0x00, // rows tile 6
    0x00, // rows tile 7
    0x00, // rows tile 8
    0x00, // rows tile 9
    0x00, // rows tile 10
    0x00, // rows tile 11
    0x00, // rows tile 12
    0x00, // rows tile 13
    0x00, // rows tile 14
    0x00 // rows tile 15
}};
_tile_loadconfig(config);
""")


@instr(_amx_config)
def config():
    pass  # TODO: implement actually configuring tile size


# TODO: Handle custom read_stride
_amx_ld_i8 = ("_tile_loadd({dst_int}, {src}.data, {src}.strides[0]);")
@instr(_amx_ld_i8)
def ld_i8(
    n: size,
    m: size,
    src: [i8][n, m] @ DRAM,
    dst: [i8][n, m] @ AMX_TILE,
):
    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i, j]


# TODO: Handle custom write_stride
_amx_st_i8 = ("_tile_stored({src_int}, {dst}.data, {dst}.strides[0]);")
@instr(_amx_st_i8)
def st_i8(
    n: size,
    m: size,
    src: [i8][n, m] @ AMX_TILE,
    dst: [i8][n, m] @ DRAM,
):
    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i, j]


_amx_st_i32 = ("_tile_stored({src_int}, {dst}.data, 4*{dst}.strides[0]);")
@instr(_amx_st_i32)
def st_i32(
    n: size,
    m: size,
    src: [i32][n, m] @ AMX_TILE,
    dst: [i32][n, m] @ DRAM,
):
    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i, j]

_amx_zero_i32 = ("_tile_zero({tile_int});")
@instr(_amx_zero_i32)
def zero_i32(
    n: size,
    m: size,
    tile: [i32][n, m] @ AMX_TILE,
):
    for i in par(0, n):
        for j in par(0, m):
            tile[i, j] = 0.0

"""
dpbuud(2, 0, 1) // tile2 = tile0*tile1
// tile2 [i32][m,n]
st_i32(2, dram)
ld_i8(dram, 2)
dpbuud(3, 2, 2) // tile3 = tile2*tile2
"""

_amx_dpbuud = "_tile_dpbuud({dst_int}, {src1_int}, {src2_int});"
@instr(_amx_dpbuud)
def dpbuud(
    M: size,
    K: size,
    N: size,
    src1: [i8][M, 4*K] @ AMX_TILE,
    src2: [i8][K, 4*N] @ AMX_TILE,
    dst: [i32][M, N] @ AMX_TILE,
):
    for m in par(0, M):
        for n in par(0, N):
            for k in par(0, K):
                for byte in par(0, 4):
                    a: i32
                    b: i32

                    a = src1[m, 4*k + byte]
                    b = src2[k, 4*n + byte]

                    dst[m, n] += a * b
