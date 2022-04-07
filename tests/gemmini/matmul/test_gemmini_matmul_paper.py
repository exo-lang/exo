from __future__ import annotations

import pytest

from exo.platforms.gemmini import *
from ..harness_gemmini import GemmTestBuilder


def new_config_ld():
    @config
    class ConfigLoad:
        src_stride : stride
    return ConfigLoad

ConfigLoad = new_config_ld()

_gemm_config_ld   = ("gemmini_extended3_config_ld({src_stride}, 1.0f, 0, 0);\n")
@instr(_gemm_config_ld)
def config_ld(
    src_stride : stride
):
    ConfigLoad.src_stride = src_stride

_gemm_do_ld_data = ("gemmini_extended_mvin( &{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );")
@instr(_gemm_do_ld_data)
def do_ld_data(
    n     : size,
    m     : size,
    src   : [R][n, m] @ DRAM,
    dst   : [R][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_ld_data   = ("gemmini_extended3_config_ld({src}.strides[0], 1.0f, 0, 0);\n"+
                   "gemmini_extended_mvin( &{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );")
@instr(_gemm_ld_data)
def ld_data(
    n     : size,
    m     : size,
    src   : [R][n, m] @ DRAM,
    dst   : [R][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

ld_data_v2 = ld_data.rename("ld_data_v2")
ld_data_v2 = ld_data_v2.configwrite_root(ConfigLoad, 'src_stride', 'stride(src, 0)')
ld_data_v2 = ld_data_v2.replace(do_ld_data, 'for i in _:_')
ld_data_v2 = ld_data_v2.replace(config_ld, 'ConfigLoad.src_stride = _')




def new_config_ld_acc():
    @config
    class ConfigLoadACC:
        src_stride : stride
    return ConfigLoadACC

ConfigLoadACC = new_config_ld_acc()

_gemm_config_ld_acc   = ("gemmini_extended3_config_ld({src_stride}*4, 1.0f, 0, 0);\n")
@instr(_gemm_config_ld_acc)
def config_ld_acc(
    src_stride : stride
):
    ConfigLoadACC.src_stride = src_stride

_gemm_do_ld_acc = ("gemmini_extended_mvin( &{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );")
@instr(_gemm_do_ld_acc)
def do_ld_acc(
    n     : size,
    m     : size,
    src   : [R][n, m] @ DRAM,
    dst   : [R][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_ld_acc   = ("gemmini_extended3_config_ld({src}.strides[0]*4, 1.0f, 0, 0);\n"+
                  "gemmini_extended_mvin( &{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );")
@instr(_gemm_ld_acc)
def ld_acc(
    n     : size,
    m     : size,
    src   : [R][n, m] @ DRAM,
    dst   : [R][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

ld_acc_v2 = ld_acc.rename("ld_acc_v2")
ld_acc_v2 = ld_acc_v2.configwrite_root(ConfigLoad, 'src_stride', 'stride(src, 0)')
ld_acc_v2 = ld_acc_v2.replace(do_ld_acc, 'for i in _:_')
ld_acc_v2 = ld_acc_v2.replace(config_ld, 'ConfigLoad.src_stride = _')




def new_config_matmul():
    @config
    class ConfigMatmul:
        set : bool
    return ConfigMatmul

ConfigMatmul = new_config_matmul()

_gemm_config_matmul = "gemmini_extended_config_ex(WS, 0, 0, 0, 1, 0, 0);\n"
@instr(_gemm_config_matmul)
def config_matmul():
    ConfigMatmul.set = True

_gemm_matmul = (
       "gemmini_extended_preload("+
            "(uint32_t)(&{B_data}), (uint32_t)(&{C_data}), "+
            "{M}, {K}, "+
            "{M}, {N}"+
       ");\n"+
       "gemmini_extended_compute_preloaded("+
            "(uint32_t)(&{A_data}), ~((uint32_t)0), "+
            "{K}, {N}, "+
            "16, 16"+
       ");")

@instr(_gemm_config_matmul + _gemm_matmul)
def matmul(
    N : size,
    M : size,
    K : size,
    A : [R][N, 16] @ GEMM_SCRATCH,
    B : [R][K, 16] @ GEMM_SCRATCH,
    C : [R][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in par(0,N):
        for j in par(0,M):
            for k in par(0,K):
                C[i, j] += A[i,k] * B[k,j]

@instr(_gemm_matmul)
def do_matmul(
    N : size,
    M : size,
    K : size,
    A : [R][N, 16] @ GEMM_SCRATCH,
    B : [R][K, 16] @ GEMM_SCRATCH,
    C : [R][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in par(0,N):
        for j in par(0,M):
            for k in par(0,K):
                C[i, j] += A[i,k] * B[k,j]

matmul_v2 = matmul.rename("matmul_v2")
matmul_v2 = matmul_v2.configwrite_root(ConfigMatmul, 'set', 'True')
matmul_v2 = matmul_v2.replace(do_matmul, 'for i in _:_')
matmul_v2 = matmul_v2.replace(config_matmul, 'ConfigMatmul.set = True')


def new_config_st():
    @config
    class ConfigStore:
        dst_stride : stride

    return ConfigStore

ConfigStore = new_config_st()

_gemm_st_acc   = ("gemmini_extended_config_st({dst}.strides[0]*1, 0, 1.0f);\n"+
                  "gemmini_extended_mvout( ((uint64_t) &{dst}), (uint32_t) &{src}, {m}, {n} );")
@instr(_gemm_st_acc)
def st_acc(
    n     : size,
    m     : size,
    src   : [R][n, 16] @ GEMM_ACCUM,
    dst   : [R][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i,j]

_gemm_config_st_acc   = ("gemmini_extended_config_st({dst_stride}, 0, 1.0f);\n")
@instr(_gemm_config_st_acc)
def config_st_acc(
    dst_stride : stride
):
    ConfigStore.dst_stride = dst_stride

_gemm_st_acc   = ("gemmini_extended_mvout( ((uint64_t) &{dst}), (uint32_t) &{src}, {m}, {n} );")
@instr(_gemm_st_acc)
def do_st_acc(
    n     : size,
    m     : size,
    src   : [R][n, 16] @ GEMM_ACCUM,
    dst   : [R][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i,j]


st_acc_v2 = st_acc.rename("st_acc_v2")
st_acc_v2 = st_acc_v2.configwrite_root(ConfigStore, 'dst_stride', 'stride(dst, 0)')
st_acc_v2 = st_acc_v2.replace(do_st_acc, 'for i in _:_')
st_acc_v2 = st_acc_v2.replace(config_st_acc, 'ConfigStore.dst_stride = _')



def matmul_algorithm():
    @proc
    def matmul( N : size, M : size, K : size, A : R[N,K], B : R[K,M], C : R[N,M] ):
        for i in par(0,N):
            for j in par(0,M):
                for k in par(0,K):
                    C[i,j] += A[i,k]*B[k,j]

    return matmul


def inline_lift_config(gemmini):
    gemmini = gemmini.call_eqv(ld_acc_v2, "ld_acc(_)")
    gemmini = gemmini.inline("ld_acc_v2(_)")
    gemmini = gemmini.inline_window("src = C[_]")
    gemmini = gemmini.inline_window("dst = res[_]")

    gemmini = gemmini.call_eqv(ld_data_v2, "ld_data(_)")
    gemmini = gemmini.inline("ld_data_v2(_)")
    gemmini = gemmini.inline_window("src = A[_]")
    gemmini = gemmini.inline_window("dst = a[_]")

    gemmini = gemmini.call_eqv(ld_data_v2, "ld_data(_)")
    gemmini = gemmini.inline("ld_data_v2(_)")
    gemmini = gemmini.inline_window("src = B[_]")
    gemmini = gemmini.inline_window("dst = b[_]")

    gemmini = gemmini.call_eqv(matmul_v2, "matmul(_)")
    gemmini = gemmini.inline("matmul_v2(_)")
    gemmini = gemmini.inline_window("A = a[_]")
    gemmini = gemmini.inline_window("B = b[_]")
    gemmini = gemmini.inline_window("C = res[_]")

    gemmini = gemmini.call_eqv(st_acc_v2, "st_acc(_)")
    gemmini = gemmini.inline("st_acc_v2(_)")
    gemmini = gemmini.inline_window("src = res[_]")
    gemmini = gemmini.inline_window("dst = C[_]")

    gemmini = lift_config(gemmini, 'config_matmul(_)')
    gemmini = lift_config(gemmini, 'config_st_acc(_)')

    return gemmini

def test_matmul_paper():
    NN = 128
    MM = 128
    KK = 128

    cpu = matmul_algorithm().rename("matmul_on_cpu").partial_eval(NN, MM, KK)

    gemmini = matmul_algorithm().rename("matmul_on_gemmini").partial_eval(NN, MM, KK)

    # Stage memories, so that we can use gemmini scratchpad & accumulator
    gemmini = gemmini.stage_assn('res', 'C[_] += _')
    gemmini = gemmini.double_fission('res = _', 'res += _')
    gemmini = gemmini.bind_expr('a', 'A[_]')
    gemmini = gemmini.bind_expr('b', 'B[_]')

    # Tile dimensions
    gemmini = gemmini.split('i', 16, ['io', 'ii'], perfect=True)
    gemmini = gemmini.split('j', 16, ['jo', 'ji'], perfect=True)
    gemmini = gemmini.reorder('ii', 'jo')
    gemmini = gemmini.split('k', 16, ['ko', 'ki'], perfect=True)

    # Fission inner dimensions
    gemmini = gemmini.lift_alloc('res:_', n_lifts=2)
    gemmini = gemmini.fission_after('res = _', n_lifts=2)
    gemmini = gemmini.fission_after('for ko in _:_', n_lifts=2)
    gemmini = gemmini.reorder('ji', 'ko')
    gemmini = gemmini.reorder('ii', 'ko')
    gemmini = gemmini.lift_alloc('a:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('b:_')
    gemmini = gemmini.lift_alloc('b:_', mode='col', n_lifts=2)
    gemmini = gemmini.fission_after('a[_] = _', n_lifts=3)
    gemmini = gemmini.fission_after('b[_] = _', n_lifts=3)
    gemmini = gemmini.lift_alloc('res:_', n_lifts=2)
    gemmini = gemmini.lift_alloc('a:_', n_lifts=3)
    gemmini = gemmini.lift_alloc('b:_', n_lifts=3)

    # replace loops with accelerator instructions
    gemmini = gemmini.replace(ld_acc, 'for ii in _:_ #0')
    gemmini = gemmini.replace(ld_data, 'for ii in _:_ #0')
    gemmini = gemmini.reorder('ji','ki')
    gemmini = gemmini.replace(ld_data, 'for ki in _:_ #0')
    gemmini = gemmini.reorder('ki','ji')
    gemmini = gemmini.replace(matmul, 'for ii in _:_ #0')
    gemmini = gemmini.replace(st_acc, 'for ii in _:_ #0')
    gemmini = gemmini.simplify()

    # inline and lift config
    gemmini = inline_lift_config(gemmini)
    gemmini = gemmini.simplify()

    print(gemmini)

