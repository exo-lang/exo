from __future__ import annotations

import sys


from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs, config
from SYS_ATL.libs.memories import AMX_TILE

# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

# TODO: Handle read_stride
_amx_ld_i8   = ("_tile_loadd(0, {src}.data, {src}.strides[0]);")
@instr(_amx_ld_i8)
def ld_i8(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, m] @ AMX_TILE,
):

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

# TODO: Handle write_stride
_amx_st_i8   = ("_tile_stored(0, {dst}.data, {src}.strides[0]);")
@instr(_amx_st_i8)
def st_i8(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ AMX_TILE,
    dst   : [i8][n, m] @ DRAM,
):

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_amx_config = (
"""
  unsigned char config[] = {{
        0x01,
        0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        {tile0_bytes}, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        0x00, 0x00,
        {tile0_rows},
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00,
        0x00
    }};\n \n
    _tile_loadconfig(config);
""")

@instr(_amx_config)
def config(
    tile0_bytes : size,
    tile0_rows : size,
        ):
    pass

# --------------------------------------------------------------------------- #
#
# --------------------------------------------------------------------------- #
