from __future__ import annotations

from exo import instr, DRAM
from exo.libs.memories import AMX_TILE

# ---------------------------------------------------------------------------- #
# Config                                                                       #
# ---------------------------------------------------------------------------- #

_amx_config = """
unsigned char config[] = {{
    0x01,                                     // ID
    0x00,                                     // start row
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // reserved
    64, 0x00,                                 // bytes per row tile 0
    64, 0x00,                                 // bytes per row tile 1
    64, 0x00,                                 // bytes per row tile 2
    64, 0x00,                                 // bytes per row tile 3
    64, 0x00,                                 // bytes per row tile 4
    64, 0x00,                                 // bytes per row tile 5
    64, 0x00,                                 // bytes per row tile 6
    64, 0x00,                                 // bytes per row tile 7
    0x00, 0x00,                               // bytes per row tile 8
    0x00, 0x00,                               // bytes per row tile 9
    0x00, 0x00,                               // bytes per row tile 10
    0x00, 0x00,                               // bytes per row tile 11
    0x00, 0x00,                               // bytes per row tile 12
    0x00, 0x00,                               // bytes per row tile 13
    0x00, 0x00,                               // bytes per row tile 14
    0x00, 0x00,                               // bytes per row tile 15
    16,                                       // rows tile 0
    16,                                       // rows tile 1
    16,                                       // rows tile 2
    16,                                       // rows tile 3
    16,                                       // rows tile 4
    16,                                       // rows tile 5
    16,                                       // rows tile 6
    16,                                       // rows tile 7
    0x00,                                     // rows tile 8
    0x00,                                     // rows tile 9
    0x00,                                     // rows tile 10
    0x00,                                     // rows tile 11
    0x00,                                     // rows tile 12
    0x00,                                     // rows tile 13
    0x00,                                     // rows tile 14
    0x00                                      // rows tile 15
}};
_tile_loadconfig(config);
"""


@instr(_amx_config)
def config():
    pass


# ---------------------------------------------------------------------------- #
# ld_i8, ld_i32, ld_i8_3d                                                              #
# ---------------------------------------------------------------------------- #


# TODO: Handle custom read_stride
_amx_ld = "_tile_loadd({dst_int}, {src}.data, {src}.strides[0]);"


@instr(_amx_ld)
def ld_i8(
    m: size,
    n: size,
    src: [i8][m, n] @ DRAM,
    dst: [i8][m, n] @ AMX_TILE,
):
    assert m <= 16
    assert n <= 64
    for i in seq(0, m):
        for j in seq(0, n):
            dst[i, j] = src[i, j]


@instr("_tile_loadd({dst_int}, {src}.data, 4*{src}.strides[0]);")
def ld_i32(
    m: size,
    n: size,
    src: [i32][m, n] @ DRAM,
    dst: [i32][m, n] @ AMX_TILE,
):
    assert m <= 16
    assert n <= 16
    for i in seq(0, m):
        for j in seq(0, n):
            dst[i, j] = src[i, j]


"""
Need this because idk how to rearrange memory
using Exo commands when scheduling and lift_allocing
"""


@instr(_amx_ld)
def ld_i8_3d(
    n: size,
    m: size,
    src: [i8][m, 4 * n] @ DRAM,
    dst: [i8][m, n, 4] @ AMX_TILE,
):
    assert n <= 16
    assert m <= 16
    for i in seq(0, m):
        for j in seq(0, n):
            for k in seq(0, 4):
                dst[i, j, k] = src[i, 4 * j + k]


# ---------------------------------------------------------------------------- #
# st_i8, st_i32, zero_i32                                                      #
# ---------------------------------------------------------------------------- #

# TODO: Handle custom write_stride
@instr("_tile_stored({src_int}, {dst}.data, {dst}.strides[0]);")
def st_i8(
    m: size,
    n: size,
    src: [i8][m, n] @ AMX_TILE,
    dst: [i8][m, n] @ DRAM,
):
    assert m <= 16
    assert n <= 64
    for i in seq(0, m):
        for j in seq(0, n):
            dst[i, j] = src[i, j]


@instr("_tile_stored({src_int}, {dst}.data, 4*{dst}.strides[0]);")
def st_i32(
    m: size,
    n: size,
    src: [i32][m, n] @ AMX_TILE,
    dst: [i32][m, n] @ DRAM,
):
    assert m <= 16
    assert n <= 16
    for i in seq(0, m):
        for j in seq(0, n):
            dst[i, j] = src[i, j]


@instr("_tile_zero({tile_int});")
def zero_i32(
    m: size,
    n: size,
    tile: [i32][m, n] @ AMX_TILE,
):
    assert m <= 16
    assert n <= 16
    for i in seq(0, m):
        for j in seq(0, n):
            tile[i, j] = 0.0


# ---------------------------------------------------------------------------- #
# dpbssd, dpbssd_3d                                                            #
# ---------------------------------------------------------------------------- #

"""
dpbssd(2, 0, 1) // tile2 = tile0*tile1
// tile2 [i32][m,n]
st_i32(2, dram)
ld_i8(dram, 2)
dpbssd(3, 2, 2) // tile3 = tile2*tile2
"""

_amx_dpbssd = "_tile_dpbssd({dst_int}, {src1_int}, {src2_int});"


@instr(_amx_dpbssd)
def dpbssd(
    M: size,
    K: size,
    N: size,
    src1: [i8][M, 4 * K] @ AMX_TILE,
    src2: [i8][K, 4 * N] @ AMX_TILE,
    dst: [i32][M, N] @ AMX_TILE,
):
    assert M <= 16
    assert K <= 16
    assert M <= 16
    for m in seq(0, M):
        for n in seq(0, N):
            for k in seq(0, K):
                for byte in seq(0, 4):
                    a: i32
                    b: i32

                    a = src1[m, 4 * k + byte]
                    b = src2[k, 4 * n + byte]

                    dst[m, n] += a * b


@instr(_amx_dpbssd)
def dpbssd_3d(
    M: size,
    K: size,
    N: size,
    src1: [i8][M, K, 4] @ AMX_TILE,
    src2: [i8][K, N, 4] @ AMX_TILE,
    dst: [i32][M, N] @ AMX_TILE,
):
    assert M <= 16
    assert K <= 16
    assert M <= 16
    for m in seq(0, M):
        for n in seq(0, N):
            for k in seq(0, K):
                for byte in seq(0, 4):
                    a: i32
                    b: i32

                    a = src1[m, k, byte]
                    b = src2[k, n, byte]

                    dst[m, n] += a * b
