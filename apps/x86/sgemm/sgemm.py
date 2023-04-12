from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


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
    # (
    #    SGEMM_WINDOW
    #        .rename(f'basic_kernel_{M}x4')
    #        .partial_eval(M, N_REG_BLK)
    #        .simplify()
    # )
    def make_avx512_kernel(p):
        p = rename(p, f"sgemm_kernel_avx512_{M}x4")
        # Vectorize columns
        p = divide_loop(p, "j", VEC_W, ["jo", "ji"], perfect=True)
        # Stage C for reduction
        p = stage_mem(p, "C[_] += _", f"C[i, {VEC_W} * jo + ji]", "C_reg")
        p = set_memory(p, "C_reg", AVX512)
        p = autolift_alloc(p, "C_reg: _", n_lifts=3, keep_dims=True)
        p = autolift_alloc(p, "C_reg: _")
        p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=4)
        p = autofission(p, p.find("C[_] = _").before(), n_lifts=4)
        # Stage A & B
        def stage_input(p, expr, new_buf):
            p = bind_expr(p, expr, new_buf)
            p = expand_dim(p, new_buf, 16, "ji")
            p = lift_alloc(p, new_buf)
            p = set_memory(p, new_buf, AVX512)
            p = fission(p, p.find(f"{new_buf} = _").after())
            return p

        p = stage_input(p, "A[_]", "A_vec")
        p = stage_input(p, "B[_]", "B_vec")
        # Schedule ops
        p = replace(p, "for ji in _: _ #0", mm512_loadu_ps)
        p = replace(p, "for ji in _: _ #3", mm512_storeu_ps)
        p = replace_all(p, mm512_set1_ps)
        p = replace_all(p, mm512_loadu_ps)
        p = replace_all(p, mm512_fmadd_ps)
        # LICM
        p = autolift_alloc(p, "A_vec: _", keep_dims=True)
        p = autofission(p, p.find("mm512_set1_ps(_)").after())
        # Clean up
        p = simplify(p)
        print(p)
        return p

    sgemm_kernel_avx512_Mx4[M] = make_avx512_kernel(basic_kernel_Mx4[M])


def make_bottom_panel_kernel(p):
    p = rename(p, "bottom_panel_kernel")
    p = p.partial_eval(N=N_REG_BLK)
    p = p.add_assertion("M < 6")
    p = simplify(p)
    return p


bottom_panel_kernel = make_bottom_panel_kernel(SGEMM_WINDOW)


def make_bottom_panel_kernel_scheduled(p=bottom_panel_kernel):
    p = rename(p, "bottom_panel_kernel_scheduled")
    p = specialize(p, "for k in _: _ #0", [f"M == {i}" for i in range(1, M_REG_BLK)])
    p = simplify(p)
    for M in range(1, 6):
        p = replace_all(p, basic_kernel_Mx4[M])
        p = call_eqv(p, f"basic_kernel_{M}x4(_)", sgemm_kernel_avx512_Mx4[M])
    p = simplify(p)
    return p


bottom_panel_kernel_scheduled = make_bottom_panel_kernel_scheduled()


def make_right_panel_kernel(p=SGEMM_WINDOW):
    p = rename(p, "right_panel_kernel")
    p = p.partial_eval(M=M_REG_BLK)
    p = p.add_assertion("N / 16 < 4")
    p = simplify(p)
    return p


right_panel_kernel = make_right_panel_kernel()


def make_right_panel_kernel_opt(p=right_panel_kernel):
    p = rename(p, "right_panel_kernel_opt")
    #
    p = stage_mem(p, "C[_] += _", "C[i, j]", "C_reg")
    p = divide_loop(p, "j", VEC_W, ["jo", "ji"], tail="cut")
    p = bound_and_guard(p, "for ji in _: _ #1")
    p = fission(p, p.find("for jo in _: _").after(), n_lifts=2)
    #
    p = autolift_alloc(p, "C_reg: _", n_lifts=3, keep_dims=True)
    p = autolift_alloc(p, "C_reg: _ #1", n_lifts=3, keep_dims=True)
    p = autolift_alloc(p, "C_reg: _")
    p = autolift_alloc(p, "C_reg: _ #1")
    p = reorder_up(p, "C_reg : _ #1")
    # p = reorder_stmts(p, 'for k in _ : _\n'
    #                     'C_reg: _')
    #
    p = autofission(p, p.find("C_reg[_] = _ #0").after(), n_lifts=4)
    p = autofission(p, p.find("C_reg[_] = _ #1").after(), n_lifts=4)
    p = autofission(p, p.find("C_reg[_] += _ #0").after(), n_lifts=4)
    p = autofission(p, p.find("C_reg[_] += _ #1").after(), n_lifts=4)
    #
    p = reorder_up(p, "for i in _: _ #3")
    p = reorder_up(p, "for i in _: _ #2")
    p = reorder_up(p, "for k in _: _ #1")
    #
    p = set_memory(p, "C_reg", AVX512)
    p = set_memory(p, "C_reg #1", AVX512)
    #
    def stage_input(p, expr, new_buf, n_lifts=1):
        p = bind_expr(p, expr, new_buf)
        p = expand_dim(p, new_buf, 16, "ji", unsafe_disable_checks=True)
        p = lift_alloc(p, new_buf, n_lifts=n_lifts)
        p = set_memory(p, new_buf, AVX512)
        p = fission(p, p.find(f"{new_buf} = _").after(), n_lifts=n_lifts)
        return p

    p = stage_input(p, "A[_]", "A_reg")
    p = stage_input(p, "B[_]", "B_reg")
    #
    p = replace_all(p, mm512_set1_ps)
    p = replace_all(p, mm512_fmadd_ps)
    p = replace(p, "for ji in _:\n" "  C[_] = _", mm512_storeu_ps)
    p = replace_all(p, mm512_loadu_ps)
    #
    p = replace(p, "for ji in _: _ #0", mm512_maskz_loadu_ps)
    p = replace(p, "for ji in _: _ #1", mm512_mask_storeu_ps)
    #
    p = stage_input(p, "A[_] #1", "A_reg2", n_lifts=2)
    p = stage_input(p, "B[_] #1", "B_reg2", n_lifts=2)
    #
    p = replace_all(p, mm512_mask_set1_ps)
    p = replace_all(p, mm512_mask_fmadd_ps)
    p = replace_all(p, mm512_maskz_loadu_ps)
    #
    for tgt in ["i #0", "k #0", "i #1", "i #2"]:
        p = fuse_after(p, tgt)
    #
    p = simplify(p)
    return p


right_panel_kernel_opt = make_right_panel_kernel_opt()


def make_right_panel_kernel_scheduled(p=right_panel_kernel):
    p = rename(p, "right_panel_kernel_scheduled")
    p = replace_all(p, right_panel_kernel)
    #
    p = specialize(
        p,
        "right_panel_kernel(_)",
        [f"(N / 16) == {i}" for i in range(N_REG_BLK // VEC_W)],
    )
    #
    p = repeat(call_eqv)(p, "right_panel_kernel(_)", right_panel_kernel_opt)
    p = repeat(inline)(p, "right_panel_kernel_opt")
    #
    p = repeat(inline_window)(p, "A = _")
    p = repeat(inline_window)(p, "B = _")
    p = repeat(inline_window)(p, "C = _")
    #
    p = simplify(p)
    return p


right_panel_kernel_scheduled = make_right_panel_kernel_scheduled()


def make_sgemm_above_kernel(p=SGEMM_WINDOW):
    p = rename(p, "sgemm_above_kernel")
    # Split up into cases
    p = divide_loop(p, "j", N_REG_BLK, ["jo", "ji"], tail="cut_and_guard")
    p = divide_loop(p, "i", M_REG_BLK, ["io", "ii"], tail="cut_and_guard")
    p = fission(p, p.find("for jo in _: _ #0").after(), n_lifts=2)
    p = reorder_loops(p, "ii jo #0")
    p = fission(p, p.find("for io in _: _").after())
    p = fission(p, p.find("for io in _: _ #1").after())
    p = reorder_loops(p, "k io #0")
    p = reorder_loops(p, "k jo #0")
    p = lift_if(p, "if N % 64 > 0: _ #0", n_lifts=3)
    p = reorder_loops(p, "k io")
    p = lift_if(p, "if M % 6 > 0: _ #0")
    p = fission(p, p.find("for jo in _: _ #1").after(), n_lifts=2)
    p = reorder_loops(p, "ii jo")
    p = reorder_loops(p, "k jo")
    p = lift_if(p, "if N % 64 > 0: _ #1", n_lifts=2)
    # Main block
    p = replace_all(p, basic_kernel_Mx4[6])
    p = call_eqv(p, basic_kernel_Mx4[6], sgemm_kernel_avx512_Mx4[6])
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
    # Split all loops
    p = divide_loop(p, "k", K_L1_BLK, ["ko", "ki"], tail="cut_and_guard")
    p = repeat(divide_loop)(p, "i", M_L1_BLK, ["io", "ii"], tail="cut_and_guard")
    p = repeat(divide_loop)(p, "j", N_L1_BLK, ["jo", "ji"], tail="cut_and_guard")
    # Explode into 8 cases
    for i in range(0, 2):
        p = fission(p, p.find(f"for io in _:_ #{i}").after(), n_lifts=2)
    for i in range(0, 4):
        p = fission(p, p.find(f"for jo in _:_ #{i}").after(), n_lifts=4)
    # Case 1:
    p = repeat(reorder_loops)(p, "ki io")
    p = repeat(reorder_loops)(p, "ii jo")
    p = repeat(reorder_loops)(p, "ki jo")
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 2:
    p = lift_if(p, "if N%64 > 0: _ #0", n_lifts=4)
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 3:
    p = lift_if(p, "if M%264 > 0: _ #0", n_lifts=2)
    p = repeat(reorder_loops)(p, "ki jo")
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 4:
    p = lift_if(p, "if M%264 > 0: _ #1", n_lifts=2)
    p = lift_if(p, "if N%64 > 0: _ #1", n_lifts=3)
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 5:
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 6:
    p = lift_if(p, "if N%64 > 0: _ #2", n_lifts=3)
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 7:
    p = lift_if(p, "if M%264 > 0: _ #2")
    p = repeat(reorder_loops)(p, "ki jo")
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    # Case 8:
    p = lift_if(p, "if M%264 > 0: _ #3")
    p = lift_if(p, "if N%64 > 0: _ #3", n_lifts=2)
    p = replace(p, "for ki in _: _ #0", SGEMM_WINDOW)
    ##
    ## Case 1 memory staging
    p = stage_window(p, "A[_] #0", "A1_cache", DRAM_STATIC)
    p = stage_window(p, "B[_] #0", "B1_cache", DRAM_STATIC)
    p = autolift_alloc(p, "A1_cache : _", n_lifts=3)
    p = autolift_alloc(p, "B1_cache : _", n_lifts=3)
    p = autofission(p, p.find_loop("i0 #0").after())
    ### Case 2 memory staging
    p = stage_window(p, "B[_] #1", "B2_cache", DRAM_STATIC)
    p = bound_alloc(p, "B2_cache", [None, "64"], unsafe_disable_checks=True)
    p = lift_alloc(p, "B2_cache")
    p = autofission(p, p.find_loop("i0 #2").after())
    ## Case 3 memory staging
    p = stage_window(p, "B[_] #2", "B3_cache", DRAM_STATIC)
    ## Case 4 memory staging
    p = stage_window(p, "B[_] #3", "B4_cache", DRAM_STATIC)
    p = bound_alloc(p, "B4_cache", [None, "64"], unsafe_disable_checks=True)
    ## Case 5 memory staging
    p = stage_window(p, "B[_] #4", "B5_cache", DRAM_STATIC)
    p = bound_alloc(p, "B5_cache", ["512", None], unsafe_disable_checks=True)
    ## Case 6 memory staging
    p = stage_window(p, "B[_] #5", "B6_cache", DRAM_STATIC)
    p = bound_alloc(p, "B6_cache", ["512", "64"], unsafe_disable_checks=True)
    ## Case 7 memory staging
    p = stage_window(p, "B[_] #6", "B7_cache", DRAM_STATIC)
    p = bound_alloc(p, "B7_cache", ["512", None], unsafe_disable_checks=True)
    ## Case 8 memory staging
    p = stage_window(p, "B[_] #7", "B8_cache", DRAM_STATIC)
    p = bound_alloc(p, "B8_cache", ["512", "64"], unsafe_disable_checks=True)
    ## Replace SGEMM_WINDOW with optimized form
    # These must come AFTER bound_alloc since the internal check-effects
    # is a whole program analysis that is VERY expensive
    p = repeat(call_eqv)(p, SGEMM_WINDOW, sgemm_above_kernel)
    # Clean up
    p = simplify(p)
    return p


sgemm_exo = make_sgemm_exo()

if __name__ == "__main__":
    # print(sgemm_above_kernel)
    print(sgemm_exo)

__all__ = ["sgemm_exo"]
