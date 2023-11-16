from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *


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


SGEMM_WINDOW = rename(SGEMM, "SGEMM_WINDOW")
SGEMM_WINDOW = set_window(SGEMM_WINDOW, "A", True)
SGEMM_WINDOW = set_window(SGEMM_WINDOW, "B", True)
SGEMM_WINDOW = set_window(SGEMM_WINDOW, "C", True)

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


def schedule_kernel(p, cond):
    og_p = p.add_assertion(cond)
    p, _ = auto_divide_loop(og_p, og_p.find_loop("j"), VEC_W)
    p = simplify(specialize(p, p.find_loop("k"), cond))
    p = eliminate_dead_code_pass(p)
    p = simplify(auto_stage_mem(p, p.find("C[_] += _"), "C_reg", 4))
    p = simplify(divide_dim(p, "C_reg", 1, VEC_W))
    p = set_memory(p, "C_reg", AVX512)
    for it in ("i1", "ji", "i1"):
        p = scalar_loop_to_simd_loops(p, p.find_loop(it), VEC_W, AVX512)
    for it in ("i1o", "jio", "jo", "i1o"):
        p = unroll_loop(p, it)
    p = eliminate_dead_code_pass(p)
    p = replace_all(p, AVX512F_instructions)
    return og_p, simplify(p)


basic_kernel_Mx4 = {}
sgemm_kernel_avx512_Mx4 = {}
for M in range(1, M_REG_BLK + 1):

    def make_basic(p):
        p = rename(p, f"basic_kernel_{M}x4")
        p = p.partial_eval(M, N_REG_BLK)
        p = simplify(p)
        return p

    p = make_basic(SGEMM_WINDOW)
    p, p_sched = schedule_kernel(p, "0 == 0")
    basic_kernel_Mx4[M] = p
    sgemm_kernel_avx512_Mx4[M] = p_sched


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
    conds = [
        f"((N + {VEC_W - 1}) / {VEC_W}) == {i}"
        for i in range(1, 1 + (N_REG_BLK // VEC_W))
    ]
    p = specialize(p, p.body()[0], conds)
    p = eliminate_dead_code_pass(p)
    for i, cond in enumerate(conds):
        case, case_sched = schedule_kernel(right_panel_kernel, cond)
        case_sched = rename(case_sched, f"{case_sched.name()}{i}")
        p = replace(p, p.find_loop("k"), case)
        p = call_eqv(p, case, case_sched)
    p = rename(p, "right_panel_kernel_scheduled")
    return p


right_panel_kernel_scheduled = make_right_panel_kernel_opt()


def make_sgemm_above_kernel(p=SGEMM_WINDOW):
    p = rename(p, "sgemm_above_kernel")
    p = tile_loops_bottom_up(p, p.body()[0], (None, M_REG_BLK, N_REG_BLK))
    p = fission(p, p.find_loop("jo").after())
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


def make_sgemm_exo(p=SGEMM):
    p = rename(p, "sgemm_exo")
    p = tile_loops_bottom_up(p, p.body()[0], (K_L1_BLK, M_L1_BLK, N_L1_BLK))
    for i in range(0, 8):
        B_name = "B_cache"
        p = auto_stage_mem(p, p.find(f"B[_] #{i}"), B_name, n_lifts=3)
        B_alloc_ref = B_name + f":_ #{i}"
        p = bound_alloc(p, B_alloc_ref, [K_L1_BLK, N_L1_BLK], True)
        p = set_memory(p, B_alloc_ref, DRAM_STATIC)
    p = auto_stage_mem(p, p.find(f"A[_]"), "A_cache", n_lifts=3)
    p = bound_alloc(p, "A_cache : _ ", [M_L1_BLK, K_L1_BLK], True)
    p = set_memory(p, "A_cache:_ ", DRAM_STATIC)
    p = apply_to_block(p, p.find_loop("jo").body(), hoist_stmt)
    p = fission(p, p.find_loop("jo").after(), n_lifts=2)
    p = replace_all(p, SGEMM_WINDOW)
    p = repeat(call_eqv)(p, SGEMM_WINDOW, sgemm_above_kernel)
    return simplify(p)


sgemm_exo = make_sgemm_exo()

if __name__ == "__main__":
    # print(sgemm_above_kernel)
    print(sgemm_exo)

__all__ = ["sgemm_exo"]
