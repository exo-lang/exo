from __future__ import annotations

import sys
sys.path.append(sys.path[0]+"/..")
sys.path.append(sys.path[0]+"/.")


from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM

# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

_gemm_ld_i8   = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                 "{scale}[0], 0, 0);\n"+
                 "gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_i8)
def ld_i8(
    n     : size,
    m     : size,
    scale : f32,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            tmp : f32
            tmp      = src[i,j]
            tmp      = tmp * scale
            dst[i,j] = tmp

# in order to load i8 values into the i32 accumulator memory,
# we must specify `shrunk=1` (3rd param of ..._config_ld)
_gemm_ld_acc_i8 = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                   "{scale}[0], 1, 0);\n"+
                   "gemmini_extended_mvin( {src}.data, "+
                                "((uint32_t) {dst}.data), {m}, {n} );")
ld_acc_i8 = (ld_i8.rename('ld_acc_i8')
                  .set_precision('dst', 'i32')
                  .set_memory('dst', GEMM_ACCUM)
                  .make_instr(_gemm_ld_acc_i8))


_gemm_ld_acc_i32   = ("gemmini_extended3_config_ld({src}.strides[0]*4, "+
                      "{scale}[0], 0, 0);\n"+
                      "gemmini_extended_mvin( ((uint64_t) {src}.data), "+
                               "((uint32_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_acc_i32)
def ld_acc_i32(
    n     : size,
    m     : size,
    scale : f32,
    src   : [i32][n, m] @ DRAM,
    dst   : [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            tmp : f32
            tmp      = src[i,j]
            tmp      = tmp * scale
            dst[i,j] = tmp


_gemm_st_i8   = ("gemmini_config_st({dst}.strides[0]*1);\n"+
                 "gemmini_extended_mvout( "+
                      "((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_i8)
def st_i8(
    n     : size,
    m     : size,
    scale : f32,
    src   : [i8][n, 16] @ GEMM_SCRATCH,
    dst   : [i8][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            tmp : f32
            tmp = src[i,j]
            tmp = tmp * scale
            dst[i, j] = tmp

_gemm_st_acc_i8 = ("gemmini_config_ex(WS, NO_ACTIVATION, 0, "+
                   "{scale}[0], 0);\n"+_gemm_st_i8)
st_acc_i8 = (st_i8.rename('st_acc_i8')
                  .set_precision('src', 'i32')
                  .set_memory('src', GEMM_ACCUM)
                  .make_instr(_gemm_st_acc_i8))


_gemm_st_acc_i32 = ("gemmini_config_ex(WS, NO_ACTIVATION, 0, 1.0f, 0);\n"+
                    "gemmini_config_st({dst}.strides[0]*4);\n"+
                    "gemmini_extended_mvout( ((uint64_t) {dst}.data), "+
                    "((uint32_t) {src}.data | 0x20000000), {m}, {n} );")
@instr(_gemm_st_acc_i32)
def st_acc_i32(
    n     : size,
    m     : size,
    src   : [i32][n, 16] @ GEMM_ACCUM,
    dst   : [i32][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i, j]



_gemm_zero_i8 = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"+
                 "gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                       "{m}, {n} );")
@instr(_gemm_zero_i8)
def zero_i8(
    n   : size,
    m   : size,
    dst : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = 0.0

zero_acc_i32 = (zero_i8.rename('zero_acc_i32')
                       .set_precision('dst', 'i32')
                       .set_memory('dst', GEMM_ACCUM)
                       .make_instr(_gemm_zero_i8))

@instr("gemmini_config_ex(WS, NO_ACTIVATION, 0, 1.0f, 0);\n"+
       "gemmini_extended_preload("+
            "(uint32_t)({B}.data), (uint32_t)({C}.data), "+
            "{M}, {K}, "+
            "{M}, {N}"+
       ");\n"+
       "gemmini_extended_compute_preloaded("+
            "(uint32_t)({A}.data), ~((uint32_t)0), "+
            "{K}, {N}, "+
            "16, 16"+
       ");")
def matmul_i8(
    N : size,
    M : size,
    K : size,
    A : [i8][N, 16] @ GEMM_SCRATCH,
    B : [i8][K, 16] @ GEMM_SCRATCH,
    C : [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in par(0,N):
        for j in par(0,M):
            C[i,j] = 0.0
            for k in par(0,K):
                a : i32
                b : i32
                a = A[i,k]
                b = B[k,j]
                C[i, j] += a * b

@instr("gemmini_config_ex(WS, NO_ACTIVATION, 0, 1.0f, 0);\n"+
       "gemmini_extended_preload("+
            "(uint32_t)({B}.data), (uint32_t)({C}.data) | 0x40000000, "+
            "{M}, {K}, "+
            "{M}, {N}"+
       ");\n"+
       "gemmini_extended_compute_preloaded("+
            "(uint32_t)({A}.data), ~((uint32_t)0), "+
            "{K}, {N}, "+
            "16, 16"+
       ");")
def matmul_acc_i8(
    N : size,
    M : size,
    K : size,
    A : [i8][N, 16] @ GEMM_SCRATCH,
    B : [i8][K, 16] @ GEMM_SCRATCH,
    C : [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in par(0,N):
        for j in par(0,M):
            for k in par(0,K):
                a : i32
                b : i32
                a = A[i,k]
                b = B[k,j]
                C[i, j] += a * b

# --------------------------------------------------------------------------- #
#   
# --------------------------------------------------------------------------- #




