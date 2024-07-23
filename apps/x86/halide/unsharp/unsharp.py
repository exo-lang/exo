from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STACK
from exo.platforms.x86 import *

from exo.stdlib.scheduling import *
from exo.stdlib.halide_scheduling_ops import *
from exo.stdlib.inspection import get_enclosing_loop_by_name, get_enclosing_loop
from exo.stdlib.stdlib import vectorize


avx_f32_insts = [
    mm256_loadu_ps,
    mm256_storeu_ps,
    mm256_mul_ps,
    mm256_add_ps,
    mm256_sub_ps,
    mm256_div_ps,
    mm256_broadcast_ss,
    mm256_prefix_load_ps,
    mm256_prefix_store_ps,
    mm256_prefix_add_ps,
    mm256_prefix_mul_ps,
    mm256_prefix_sub_ps,
    mm256_prefix_div_ps,
    mm256_prefix_broadcast_ss,
]


def halide_vectorize(p, buffer: str, loop: str, width: int):
    loop = get_enclosing_loop_by_name(p, p.find(f"{buffer} = _"), loop)
    p = vectorize(p, loop, width, "f32", AVX2, avx_f32_insts, rules=[])

    return p


from math import exp, pi, sqrt

sigma = 1.5
k0, k1, k2, k3 = [
    exp(-x * x / (2 * sigma**2)) / (sqrt(2 * pi) * sigma) for x in range(4)
]
r_to_gray, g_to_gray, b_to_gray = 0.299, 0.587, 0.114


@proc
def exo_unsharp_base(
    W: size, H: size, output: f32[3, H, W], input: f32[3, H + 6, W + 6]
):
    # TODO: remove the H % 32 constraint and handle tail cases in the y direction
    assert H % 32 == 0

    gray: f32[H + 6, W + 6]
    for y in seq(0, H + 6):
        for x in seq(0, W + 6):
            gray[y, x] = (
                r_to_gray * input[0, y, x]
                + g_to_gray * input[1, y, x]
                + b_to_gray * input[2, y, x]
            )

    blur_y: f32[H, W + 6]
    for y in seq(0, H):
        for x in seq(0, W + 6):
            blur_y[y, x] = (
                k0 * gray[y + 3, x]
                + k1 * (gray[y + 2, x] + gray[y + 4, x])
                + k2 * (gray[y + 1, x] + gray[y + 5, x])
                + k3 * (gray[y + 0, x] + gray[y + 6, x])
            )

    blur_x: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            blur_x[y, x] = (
                k0 * blur_y[y, x + 3]
                + k1 * (blur_y[y, x + 2] + blur_y[y, x + 4])
                + k2 * (blur_y[y, x + 1] + blur_y[y, x + 5])
                + k3 * (blur_y[y, x + 0] + blur_y[y, x + 6])
            )

    sharpen: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            sharpen[y, x] = 2.0 * gray[y + 3, x + 3] - blur_x[y, x]

    ratio: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            ratio[y, x] = sharpen[y, x] / gray[y + 3, x + 3]

    for y in seq(0, H):
        for c in seq(0, 3):
            for x in seq(0, W):
                output[c, y, x] = ratio[y, x] * input[c, y + 3, x + 3]


def halide_schedule(p):
    consts = {
        "r_to_gray": r_to_gray,
        "g_to_gray": g_to_gray,
        "b_to_gray": b_to_gray,
        "k0": k0,
        "k1": k1,
        "k2": k2,
        "k3": k3,
    }
    for name, c in consts.items():
        consts[name] = p.find(str(c), many=True)
    consts["two"] = p.find("sharpen[_] = _").rhs().lhs().lhs()

    for name, cursors in consts.items():
        p = bind_expr(p, cursors, name)
        alloc = p.find(f"{name}: _")
        p = expand_dim(p, alloc, 1, 0)
        p = set_precision(p, alloc, "f32")
        p = set_memory(p, alloc, DRAM_STACK)
        p = repeat(lift_alloc)(p, alloc)

        assign = p.find(f"{name} = _")
        while not isinstance(assign.parent(), InvalidCursor):
            p = fission(p, assign.after())
            assign = p.forward(assign)
            p = remove_loop(p, assign.parent())
            assign = p.forward(assign)

    p = halide_split(p, "output", "y", "y", "yi", 32, tail="perfect")
    p = halide_parallel(p, "y")

    p = halide_compute_and_store_at(p, "ratio", "output", "yi", "y")
    p = halide_fully_inline(p, "sharpen", "ratio")
    p = halide_fully_inline(p, "blur_x", "ratio")
    p = halide_compute_and_store_at(p, "blur_y", "output", "yi", "y")
    p = halide_compute_and_store_at(p, "gray", "output", "yi", "y")

    # Circular buffer optimization
    p = resize_dim(p, p.find("ratio: _"), 0, 1, 0, fold=True)
    p = resize_dim(p, p.find("blur_y: _"), 0, 1, 0, fold=True)
    p = resize_dim(p, p.find("gray: _"), 0, 8, 0, fold=True)

    p = simplify(p)  # for deterministic codegen
    print("Before vectorization:\n")
    print(p)
    return p


def vectorize_schedule(p):
    p = halide_vectorize(p, "gray", "x", 8)
    p = halide_vectorize(p, "gray", "x", 8)
    p = halide_vectorize(p, "blur_y", "x", 8)
    p = halide_vectorize(p, "ratio", "x", 8)
    p = halide_vectorize(p, "output", "x", 8)

    p = simplify(p)  # for deterministic codegen
    print("After vectorization:\n")
    print(p)
    return p


exo_unsharp = rename(halide_schedule(exo_unsharp_base), "exo_unsharp")
exo_unsharp_vectorized = rename(
    vectorize_schedule(exo_unsharp), "exo_unsharp_vectorized"
)
