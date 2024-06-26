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
kernel = [exp(-x * x / (2 * sigma * sigma)) / (sqrt(2 * pi) * sigma) for x in range(4)]
k0, k1, k2, k3 = kernel


@proc
def exo_unsharp_base(
    W: size, H: size, output: f32[3, H, W], input: f32[3, H + 6, W + 6]
):
    # TODO: remove the H % 32 constraint and handle tail cases in the y direction
    assert H % 32 == 0
    # constants
    rgb_to_gray: f32[3]
    rgb_to_gray[0] = 0.299
    rgb_to_gray[1] = 0.587
    rgb_to_gray[2] = 0.114

    kernel: f32[4]
    kernel[0] = k0
    kernel[1] = k1
    kernel[2] = k2
    kernel[3] = k3

    # TODO: this is kind of silly
    two: f32[1]
    two[0] = 2

    gray: f32[H + 6, W + 6]
    for y in seq(0, H + 6):
        for x in seq(0, W + 6):
            gray[y, x] = (
                rgb_to_gray[0] * input[0, y, x]
                + rgb_to_gray[1] * input[1, y, x]
                + rgb_to_gray[2] * input[2, y, x]
            )

    blur_y: f32[H, W + 6]
    for y in seq(0, H):
        for x in seq(0, W + 6):
            blur_y[y, x] = (
                kernel[0] * gray[y + 3, x]
                + kernel[1] * (gray[y + 2, x] + gray[y + 4, x])
                + kernel[2] * (gray[y + 1, x] + gray[y + 5, x])
                + kernel[3] * (gray[y + 0, x] + gray[y + 6, x])
            )

    blur_x: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            blur_x[y, x] = (
                kernel[0] * blur_y[y, x + 3]
                + kernel[1] * (blur_y[y, x + 2] + blur_y[y, x + 4])
                + kernel[2] * (blur_y[y, x + 1] + blur_y[y, x + 5])
                + kernel[3] * (blur_y[y, x + 0] + blur_y[y, x + 6])
            )

    sharpen: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            sharpen[y, x] = two[0] * gray[y + 3, x + 3] - blur_x[y, x]

    ratio: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            ratio[y, x] = sharpen[y, x] / gray[y + 3, x + 3]

    for y in seq(0, H):
        for c in seq(0, 3):
            for x in seq(0, W):
                output[c, y, x] = ratio[y, x] * input[c, y + 3, x + 3]


def halide_schedule(p):
    p = halide_split(p, "output", "y", "y", "yi", 32, tail="perfect")
    p = halide_parallel(p, "y")

    p = halide_compute_and_store_at(p, "ratio", "output", "yi")
    p = lift_alloc(p, "ratio")

    p = halide_fully_inline(p, "sharpen", "ratio")
    p = halide_fully_inline(p, "blur_x", "ratio")

    p = halide_compute_and_store_at(p, "blur_y", "output", "yi")
    p = lift_alloc(p, "blur_y")

    # NOTE: when compute_at is at a higher loop level than store_at, we actually want to
    # divide and front load some work with a guard instead of divide_with_recompute.
    p = halide_compute_and_store_at(p, "gray", "output", "y")
    p = halide_compute_at(p, "gray", "output", "yi", divide_with_recompute=False)

    # Circular buffer optimization
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
