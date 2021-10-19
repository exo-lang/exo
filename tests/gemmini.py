from __future__ import annotations

import sys


from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs, config
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM

# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

def new_config_ld():
    @config
    class ConfigLoad:
        scale : f32
        src_stride : stride

    return ConfigLoad

ConfigLoad = new_config_ld()

_gemm_config_ld_i8   = ("gemmini_extended3_config_ld({src_stride}, "+
                        "{scale}[0], 0, 0);\n")
@instr(_gemm_config_ld_i8)
def config_ld_i8(
    scale : f32,
    src_stride : stride
):
    ConfigLoad.scale = scale
    ConfigLoad.src_stride = src_stride


_gemm_do_ld_i8   = ("gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_do_ld_i8)
def do_ld_i8(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1
    # TODO: We need to think about how to handle non-local config in
    # effectcheck.
    assert stride(src, 0) == ConfigLoad.src_stride

    for i in par(0, n):
        for j in par(0, m):
            tmp : f32
            tmp      = src[i,j]
            tmp      = tmp * ConfigLoad.scale
            dst[i,j] = tmp


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

ld_i8_v2 = ld_i8.rename("ld_i8_v2").bind_config('scale', ConfigLoad, 'scale')
ld_i8_v2 = ld_i8_v2.reorder_stmts('tmp = src[_]', 'ConfigLoad.scale = _')
ld_i8_v2 = ld_i8_v2.reorder_stmts('tmp : _', 'ConfigLoad.scale = _')
ld_i8_v2 = ld_i8_v2.fission_after('ConfigLoad.scale = _', n_lifts=3)
ld_i8_v2 = ld_i8_v2.configwrite_after('ConfigLoad.scale = _', ConfigLoad, 'src_stride', 'stride(src, 0)')
ld_i8_v2 = ld_i8_v2.replace(do_ld_i8, 'for i in _:_')
ld_i8_v2 = ld_i8_v2.replace(config_ld_i8, 'ConfigLoad.scale = scale')




_gemm_ld_i8_stride_2 = ("gemmini_extended3_config_ld({src}.strides[0]*2, "+
                        "{scale}[0], 0, 0);\n"+
                        "gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_i8_stride_2)
def ld_i8_s2(
    n     : size,
    m     : size,
    scale : f32,
    src   : [i8][n*2, m] @ DRAM,
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
            tmp      = src[i*2,j]
            tmp      = tmp * scale
            dst[i,j] = tmp #no clamping


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









_gemm_st_i8   = ("gemmini_extended_config_st({dst}.strides[0]*1, 0, 1.0f);\n"+
                 "gemmini_extended_mvout( "+
                      "((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_i8)
def st_i8(
    n     : size,
    m     : size,
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
            dst[i, j] = src[i, j]


@proc
def clamp(src : f32, dst : i8):
    l : f32
    h : f32
    l = -128.0
    h = 127.0
    dst = select(h, src, h, src)
    dst = select(src, l, l, dst)


# TODO: Add act!!!!
def new_config_st():
    @config
    class ConfigStore:
        scale : f32
        dst_stride : stride

    return ConfigStore

ConfigStore = new_config_st()

_gemm_st_acc_i8   = ("gemmini_extended_config_st({dst}.strides[0]*1, {act}, {scale}[0]);\n"+
                     "gemmini_extended_mvout( ((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_acc_i8)
def st_acc_i8(
    n     : size,
    m     : size,
    scale : f32,
    act   : bool,
    src   : [i32][n, 16] @ GEMM_ACCUM,
    dst   : [i8][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            tmp : i8
            if act == True:
                tmp = relu(src[i,j])
            else:
                tmp = src[i,j]
            tmp2 : f32
            tmp2 = tmp
            tmp2  = tmp2 * scale
            clamp(tmp2, tmp)
            dst[i, j] = tmp

# TODO: Add act!!!
_gemm_config_st_acc_i8   = ("gemmini_extended_config_st({dst_stride}, 0, {scale}[0]);\n")
@instr(_gemm_config_st_acc_i8)
def config_st_acc_i8(
    scale : f32,
    dst_stride : stride
):
    ConfigStore.scale = scale
    ConfigStore.dst_stride = dst_stride

_gemm_st_acc_i8   = ("gemmini_extended_mvout( ((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_acc_i8)
def do_st_acc_i8(
    n     : size,
    m     : size,
    act   : bool,
    src   : [i32][n, 16] @ GEMM_ACCUM,
    dst   : [i8][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            tmp : i8
            if act == True:
                tmp = relu(src[i,j])
            else:
                tmp = src[i,j]
            tmp2 : f32
            tmp2 = tmp
            tmp2  = tmp2 * ConfigStore.scale
            clamp(tmp2, tmp)
            dst[i, j] = tmp

st_acc_i8_v2 = st_acc_i8.rename("st_acc_i8_v2").bind_config('scale', ConfigStore, 'scale')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('tmp2 = tmp', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('tmp2 : _', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('if act == _:_', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('tmp : _', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.fission_after('ConfigStore.scale = _', n_lifts=2)
st_acc_i8_v2 = st_acc_i8_v2.configwrite_after('ConfigStore.scale = _', ConfigStore, 'dst_stride', 'stride(dst, 0)')
st_acc_i8_v2 = st_acc_i8_v2.replace(do_st_acc_i8, 'for i in _:_')
st_acc_i8_v2 = st_acc_i8_v2.replace(config_st_acc_i8, 'ConfigStore.scale = scale')



_gemm_st_acc_i32 = ("gemmini_extended_config_st({dst}.strides[0]*4, 0, 1.0f);\n"+
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





_gemm_config_zero   = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n")
@instr(_gemm_config_zero)
def config_zero():
    ConfigLoad.scale = 1.0
    ConfigLoad.src_stride = 0

_gemm_do_zero = ("gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                       "{m}, {n} );")
@instr(_gemm_do_zero)
def do_zero_i8(
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

_gemm_zero = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"+
                 "gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                       "{m}, {n} );")
@instr(_gemm_zero)
def zero_i8(
    n   : size,
    m   : size,
    dst : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    pass

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = 0.0

zero_i8_v2 = zero_i8.rename("zero_i8_v2")
zero_i8_v2 = zero_i8_v2.configwrite_after('pass', ConfigLoad, 'scale', '1.0')
zero_i8_v2 = zero_i8_v2.configwrite_after('ConfigLoad.scale = _', ConfigLoad, 'src_stride', '0')
zero_i8_v2 = zero_i8_v2.replace(do_zero_i8, 'for i in _:_')
zero_i8_v2 = zero_i8_v2.replace(config_zero, 'ConfigLoad.scale = 1.0')

do_zero_acc_i32 = (do_zero_i8.rename('do_zero_acc_i32')
                             .set_precision('dst', 'i32')
                             .set_memory('dst', GEMM_ACCUM)
                             .make_instr(_gemm_do_zero))
zero_acc_i32 = (zero_i8.rename('zero_acc_i32')
                          .set_precision('dst', 'i32')
                          .set_memory('dst', GEMM_ACCUM)
                          .make_instr(_gemm_zero))
zero_acc_i32_v2 = zero_acc_i32.rename("zero_acc_i32_v2")
zero_acc_i32_v2 = zero_acc_i32_v2.configwrite_after('pass', ConfigLoad, 'scale', '1.0')
zero_acc_i32_v2 = zero_acc_i32_v2.configwrite_after('ConfigLoad.scale = _', ConfigLoad, 'src_stride', '0')
zero_acc_i32_v2 = zero_acc_i32_v2.replace(do_zero_acc_i32, 'for i in _:_')
zero_acc_i32_v2 = zero_acc_i32_v2.replace(config_zero, 'ConfigLoad.scale = 1.0')

zero_i8 = zero_i8.delete_pass().make_instr(_gemm_zero)
zero_i8_v2 = zero_i8_v2.delete_pass().make_instr(_gemm_zero)
zero_acc_i32    = zero_acc_i32.delete_pass().make_instr(_gemm_zero)
zero_acc_i32_v2 = zero_acc_i32_v2.delete_pass().make_instr(_gemm_zero)








def new_config_matmul():
    @config
    class ConfigMatmul:
        done : bool

    return ConfigMatmul

ConfigMatmul = new_config_matmul()

_gemm_config_matmul = "gemmini_extended_config_ex(WS, 0, 0, 0, 1, 0, 0);\n"
@instr(_gemm_config_matmul)
def config_matmul():
    ConfigMatmul.done = True

_gemm_matmul = (
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

@instr(_gemm_config_matmul + _gemm_matmul)
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

    pass
    for i in par(0,N):
        for j in par(0,M):
            C[i,j] = 0.0
            for k in par(0,K):
                a : i32
                b : i32

                a = A[i,k]
                b = B[k,j]

                C[i, j] += a * b

@instr(_gemm_matmul)
def do_matmul_i8(
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

matmul_i8_v2 = matmul_i8.rename("matmul_i8_v2")
matmul_i8_v2 = matmul_i8_v2.configwrite_after('pass', ConfigMatmul, 'done', 'True')
matmul_i8_v2 = matmul_i8_v2.replace(do_matmul_i8, 'for i in _:_')
matmul_i8_v2 = matmul_i8_v2.replace(config_matmul, 'ConfigMatmul.done = True')
matmul_i8_v2 = matmul_i8_v2.delete_pass().make_instr(_gemm_matmul)
matmul_i8    = matmul_i8.delete_pass().make_instr(_gemm_config_matmul + _gemm_matmul)




_gemm_matmul_acc = (
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

@instr(_gemm_matmul_acc)
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

    pass
    for i in par(0,N):
        for j in par(0,M):
            for k in par(0,K):
                a : i32
                b : i32

                a = A[i,k]
                b = B[k,j]

                C[i, j] += a * b

@instr(_gemm_matmul_acc)
def do_matmul_acc_i8(
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
matmul_acc_i8_v2 = matmul_acc_i8.rename("matmul_acc_i8_v2")
matmul_acc_i8_v2 = matmul_acc_i8_v2.configwrite_after('pass', ConfigMatmul, 'done', 'True')
matmul_acc_i8_v2 = matmul_acc_i8_v2.replace(do_matmul_acc_i8, 'for i in _:_')
matmul_acc_i8_v2 = matmul_acc_i8_v2.replace(config_matmul, 'ConfigMatmul.done = True')
matmul_acc_i8_v2 = matmul_acc_i8_v2.delete_pass().make_instr(_gemm_matmul_acc)
matmul_acc_i8    = matmul_acc_i8.delete_pass().make_instr(_gemm_config_matmul + _gemm_matmul_acc)

# --------------------------------------------------------------------------- #
#
# --------------------------------------------------------------------------- #
