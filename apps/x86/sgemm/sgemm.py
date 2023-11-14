from __future__ import annotations

from exo import *
from exo.libs.memories import get_DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *


def reorder_up(p, stmt_pattern, n=1):
    for _ in range(n):
        c = p.find(stmt_pattern).expand(1, 0)
        p = reorder_stmts(p, c)
    return p


def fuse_after(p, stmt):
    c = p.find_loop(stmt)
    c2 = c.next()
    return fuse(p, c, c2)


# noinspection PyPep8Naming
@proc
def SGEMM(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    assert M >= 1
    assert N >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for k in seq(0, K):
        for i in seq(0, M):
            for j in seq(0, N):
                C[i, j] += A[i, k] * B[k, j]


def make_win(p):
    p = rename(p, "SGEMM_WINDOW")
    p = set_window(p, "A", True)
    p = set_window(p, "B", True)
    p = set_window(p, "C", True)
    return p


SGEMM_WINDOW = make_win(SGEMM)

# Constants for scheduling
VEC_W = 16

M_REG_BLK = 6
N_REG_BLK = 4 * VEC_W

M_L1_FAC = 44
N_L1_FAC = 1

M_L1_BLK = M_REG_BLK * M_L1_FAC
N_L1_BLK = N_REG_BLK * N_L1_FAC
K_L1_BLK = 512

AVX512F_instructions = [
    mm512_loadu_ps,
    mm512_storeu_ps,
    mm512_fmadd_ps,
    mm512_set1_ps,
    mm512_maskz_loadu_ps,
    mm512_mask_storeu_ps,
    mm512_mask_fmadd_ps,
    mm512_mask_set1_ps,
]

COPY_STREAMS = 3

basic_kernel_Mx4 = {}
sgemm_kernel_avx512_Mx4 = {}
for M in range(1, M_REG_BLK + 1):

    def make_basic(p):
        p = rename(p, f"basic_kernel_{M}x4")
        p = p.partial_eval(M, N_REG_BLK)
        p = simplify(p)
        return p

    basic_kernel_Mx4[M] = make_basic(SGEMM_WINDOW)

    def make_avx512_kernel(p):
        p = rename(p, f"sgemm_kernel_avx512_{M}x4")
        p = simplify(auto_stage_mem(p, p.find("C[_] += _"), "C_reg", n_lifts=3))
        C_reg = p.body()[0]
        p = set_memory(divide_dim(p, C_reg, 1, VEC_W), C_reg, AVX512)
        for loop_iter in ["i1", "j", "i1"]:
            p = vectorize(
                p,
                p.find_loop(loop_iter),
                VEC_W,
                1,
                1,
                AVX512,
                "f32",
                AVX512F_instructions,
                vectorize_tail=False,
            )
        p = apply_to_block(p, p.find_loop("jo").body(), hoist_stmt)
        return p

    sgemm_kernel_avx512_Mx4[M] = make_avx512_kernel(basic_kernel_Mx4[M])


def make_bottom_panel_kernel(p):
    p = rename(p, "bottom_panel_kernel")
    p = p.partial_eval(N=N_REG_BLK)
    p = p.add_assertion(f"M < {M_REG_BLK}")
    p = simplify(p)
    return p


bottom_panel_kernel = make_bottom_panel_kernel(SGEMM_WINDOW)


def make_bottom_panel_kernel_scheduled(p=bottom_panel_kernel):
    p = rename(p, "bottom_panel_kernel_scheduled")
    p = specialize(p, "for k in _: _ #0", [f"M == {i}" for i in range(1, M_REG_BLK)])
    p = simplify(p)
    p = eliminate_dead_code_pass(p)
    for M in range(1, M_REG_BLK):
        p = replace_all(p, basic_kernel_Mx4[M])
        p = call_eqv(p, f"basic_kernel_{M}x4(_)", sgemm_kernel_avx512_Mx4[M])
    p = simplify(p)
    return p


bottom_panel_kernel_scheduled = make_bottom_panel_kernel_scheduled()


def make_right_panel_kernel(p=SGEMM_WINDOW):
    p = rename(p, "right_panel_kernel")
    p = p.partial_eval(M=M_REG_BLK)
    p = p.add_assertion(f"N < {N_REG_BLK}")
    p = simplify(p)
    return p


right_panel_kernel = make_right_panel_kernel()


def make_right_panel_kernel_opt(p=right_panel_kernel):
    p, _ = auto_divide_loop(p, p.find_loop("j"), VEC_W)
    p = specialize(
        p,
        p.body()[0],
        [
            f"((N + {VEC_W - 1}) / {VEC_W}) == {i}"
            for i in range(1, 1 + (N_REG_BLK // VEC_W))
        ],
    )

    for fma in p.find("C[_] += _", many=True):
        p = simplify(auto_stage_mem(p, fma, "C_reg", 4))
    p = eliminate_dead_code_pass(p)
    for C_reg in p.find("C_reg : _", many=True):
        p = simplify(divide_dim(p, C_reg, 1, VEC_W))
        p = set_memory(p, C_reg, AVX512)

    for loop in p.find_loop("i1", many=True):
        p = vectorize_to_loops(p, loop, VEC_W, AVX512, "f32")
    for loop in p.find_loop("ji", many=True):
        p = vectorize_to_loops(p, loop, VEC_W, AVX512, "f32")
    p = simplify(p)
    for loop in p.find_loop("i1o", many=True):
        p = unroll_loop(p, loop)
    for loop in p.find_loop("jo", many=True):
        p = unroll_loop(p, loop)
    p = simplify(p)
    p = eliminate_dead_code_pass(p)
    p = replace_all(p, AVX512F_instructions)
    p = rename(p, "right_panel_kernel_scheduled")
    return p


right_panel_kernel_scheduled = make_right_panel_kernel_opt()


def make_sgemm_above_kernel(p=SGEMM_WINDOW):
    p = rename(p, "sgemm_above_kernel")
    p = tile_loops_bottom_up(p, p.body()[0], (None, M_REG_BLK, N_REG_BLK))
    # Main block
    p = replace_all(p, basic_kernel_Mx4[M_REG_BLK])
    p = call_eqv(p, basic_kernel_Mx4[M_REG_BLK], sgemm_kernel_avx512_Mx4[M_REG_BLK])
    # Right panel
    p = replace_all(p, right_panel_kernel)
    p = call_eqv(p, right_panel_kernel, right_panel_kernel_scheduled)
    # Bottom panel
    p = replace_all(p, bottom_panel_kernel)
    p = call_eqv(p, bottom_panel_kernel, bottom_panel_kernel_scheduled)
    ## TODO: bottom-right tile
    p = simplify(p)
    return p


sgemm_above_kernel = make_sgemm_above_kernel()


@proc
def copy_submatrix(M: size, N: size, dst: [f32][M, N], src: [f32][M, N]):
    assert M >= 1
    assert N >= 1
    assert stride(dst, 1) == 1
    assert stride(src, 1) == 1
    for i in seq(0, M):
        for j in seq(0, N):
            dst[i, j] = src[i, j]


def make_sgemm_exo(p=SGEMM):
    p = rename(p, "sgemm_exo")
    p = tile_loops_bottom_up(p, p.body()[0], (K_L1_BLK, M_L1_BLK, N_L1_BLK))
    for i in range(0, 8):
        B_name = "B_cache"
        p = auto_stage_mem(p, p.find(f"B[_] #{i}"), B_name, n_lifts=3)
        B_alloc_ref = B_name + f":_ #{i}"
        p = bound_alloc(
            p, B_alloc_ref, [K_L1_BLK, N_L1_BLK], unsafe_disable_checks=True
        )
        p = set_memory(p, B_alloc_ref, get_DRAM_STATIC(4096))
    p = auto_stage_mem(p, p.find(f"A[_] #0"), "A_cache", n_lifts=3)
    p = bound_alloc(
        p, f"A_cache : _ #0", [M_L1_BLK, K_L1_BLK], unsafe_disable_checks=True
    )
    p = set_memory(p, f"A_cache : _ #0", get_DRAM_STATIC(4096))
    A_alloc_parent = p.find(f"A_cache : _ #0").parent()
    if isinstance(A_alloc_parent, ForSeqCursor):
        p = apply_to_block(p, A_alloc_parent.body(), hoist_stmt)
    p = replace_all(p, SGEMM_WINDOW)
    p = repeat(call_eqv)(p, SGEMM_WINDOW, sgemm_above_kernel)
    p = replace_all(p, copy_submatrix)
    p = simplify(p)
    return p


sgemm_exo = make_sgemm_exo()

if __name__ == "__main__":
    # print(sgemm_above_kernel)
    print(sgemm_exo)

__all__ = ["sgemm_exo"]
