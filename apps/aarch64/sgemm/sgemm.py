from __future__ import annotations

from exo import *
from exo.platforms.neon import *
from exo.stdlib.scheduling import *

# Compute Matrix-Matrix Multiplication C += A * B
@proc
def SGEMM(M: size, N: size, K: size, A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    assert M >= 1
    assert N >= 1
    assert K >= 1
    assert stride(A, 1) == 1
    assert stride(B, 1) == 1
    assert stride(C, 1) == 1

    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]


def make_sgemm_win(p=SGEMM):
    p = rename(SGEMM, "sgemm_win")
    p = set_window(p, "A", True)
    p = set_window(p, "B", True)
    p = set_window(p, "C", True)
    return p


sgemm_win = make_sgemm_win()


micro_N = 4
micro_M = 16
assert micro_M % 4 == 0

L1_N = 64
L1_M = 64
L1_K = 64

assert L1_N % micro_N == 0
assert L1_M % micro_M == 0
mid_N = L1_N // micro_N
mid_M = L1_M // micro_M
mid_K = L1_K

microkernel = rename(sgemm_win, "microkernel")
microkernel = microkernel.partial_eval(micro_N, micro_M)
microkernel = simplify(microkernel)


def make_sgemm_tiled(p=SGEMM):
    p = rename(p, "sgemm_tiled")
    # tile i & j for the kernel
    p = divide_loop(p, "i", micro_N, ["io", "ii"], tail="cut_and_guard")
    p = divide_loop(p, "j #0", micro_M, ["jo", "ji"], tail="cut_and_guard")
    # isolate the main chunk of work
    p = fission(p, p.find("for jo in _: _").after(), n_lifts=2)
    p = reorder_loops(p, "ii jo")
    # tile k now, before we do the microkernel replacement
    p = divide_loop(p, "k #0", mid_K, ["ko", "ki"], tail="cut_and_guard")
    p = fission(p, p.find("for ko in _: _").after(), n_lifts=2)
    p = reorder_loops(p, "ji ko")
    p = reorder_loops(p, "ii ko")
    p = replace_all(p, microkernel)
    p = simplify(p)
    return p


sgemm_tiled = make_sgemm_tiled()
print(sgemm_tiled)


def make_neon_microkernel(p=microkernel):
    p = rename(p, "neon_microkernel")
    # Move k to the outermost loop
    p = reorder_loops(p, "j k")
    p = reorder_loops(p, "i k")
    # expose inner-loop for 4-wide vectorization
    p = divide_loop(p, "j", 4, ["jo", "ji"], perfect=True)
    return p


neon_microkernel = make_neon_microkernel()


def stage_C_microkernel(p=neon_microkernel):
    p = stage_mem(p, "C[_] += _", "C[i, 4 * jo + ji]", "C_reg")
    for iname in reversed(["i", "jo", "ji"]):
        p = expand_dim(p, "C_reg", 4, iname, unsafe_disable_checks=True)
    p = lift_alloc(p, "C_reg", n_lifts=4)
    p = autofission(p, p.find("C_reg[_] = _").after(), n_lifts=4)
    p = autofission(p, p.find("C[_] = _").before(), n_lifts=4)
    #
    p = replace(p, "for ji in _: _ #0", neon_vld_4xf32)
    p = replace(p, "for ji in _: _ #1", neon_vst_4xf32)
    p = set_memory(p, "C_reg", Neon)
    return p


neon_microkernel = stage_C_microkernel()


def stage_A_B_microkernel(p=neon_microkernel):
    for buf in ("A", "B"):
        p = bind_expr(p, f"{buf}[_]", f"{buf}_vec")
        p = expand_dim(p, f"{buf}_vec", 4, "ji", unsafe_disable_checks=True)
        p = lift_alloc(p, f"{buf}_vec")
        p = fission(p, p.find(f"{buf}_vec[_] = _").after())
        p = set_memory(p, f"{buf}_vec", Neon)
    #
    p = replace_all(p, neon_vld_4xf32)
    p = replace_all(p, neon_broadcast_4xf32)
    p = replace_all(p, neon_vfmadd_4xf32_4xf32)
    p = autolift_alloc(p, "A_vec", n_lifts=2)
    p = autofission(p, p.find("B_vec : _").before(), n_lifts=2)
    p = autolift_alloc(p, "B_vec", n_lifts=2)
    p = autofission(p, p.find("neon_vld_4xf32(_) #1").after(), n_lifts=2)
    return p


neon_microkernel = stage_A_B_microkernel()

neon_microkernel = simplify(neon_microkernel)
print(neon_microkernel)


def finish_sgemm_tiled(p=sgemm_tiled):
    p = call_eqv(p, microkernel, neon_microkernel)
    # clean up tail case from earlier
    p = autofission(p, p.find("for ko in _: _ #0").after(), n_lifts=2)
    # actually tile for L1 cache
    p = reorder_loops(p, "jo ko #0")
    p = reorder_loops(p, "io ko #0")
    p = divide_loop(p, "io #0", mid_N, ["io", "im"], tail="cut")
    p = divide_loop(p, "jo #0", mid_M, ["jo", "jm"], tail="cut")
    p = fission(p, p.find("for jo in _: _ #0").after(), n_lifts=3)
    p = repeat(reorder_loops)(p, "im jm")
    p = repeat(reorder_loops)(p, "im jo")
    p = simplify(p)
    # stage per-tile memory at appropriate levels
    p = stage_mem(
        p,
        "for jo in _: _ #0",
        f"A[{L1_N}*io : {L1_N}*io + {L1_N}, {L1_K}*ko : {L1_K}*ko + {L1_K}]",
        "Atile",
    )
    p = lift_alloc(simplify(p), "Atile", n_lifts=2)
    p = stage_mem(
        p,
        "for im in _: _ #0",
        f"B[{L1_K}*ko : {L1_K}*ko + {L1_K}, {L1_M}*jo : {L1_M}*jo + {L1_M}]",
        "Btile",
    )
    p = lift_alloc(simplify(p), "Btile", n_lifts=3)
    # cleanup
    p = simplify(p)
    return p


sgemm_tiled = finish_sgemm_tiled()

sgemm_exo = rename(sgemm_tiled, "sgemm_exo")
print(sgemm_exo)

__all__ = ["sgemm_exo"]
