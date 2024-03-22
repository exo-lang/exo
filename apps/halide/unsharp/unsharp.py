from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STACK
from exo.platforms.x86 import *

from exo.stdlib.scheduling import *
from exo.stdlib.halide_scheduling_ops import *
from exo.stdlib.inspection import get_enclosing_loop_by_name
from exo.stdlib.stdlib import vectorize


def halide_tile(p, buffer, y, x, yi, xi, yTile, xTile):
    assign = p.find(f"{buffer} = _")
    y_loop = get_enclosing_loop_by_name(p, assign, y)
    x_loop = get_enclosing_loop_by_name(p, assign, x)

    return tile(p, y_loop, x_loop, [y, yi], [x, xi], yTile, xTile, perfect=True)


def halide_split(p, stage, x, xo, xi, split_factor):
    loop = get_enclosing_loop_by_name(p, p.find(f"{stage} = _"), x)
    return split(p, loop, xo, xi, split_factor)


def halide_compute_at(p, producer: str, consumer: str, loop: str):
    x_loop = get_enclosing_loop_by_name(p, p.find(f"{consumer} = _"), loop)
    return compute_at(p, producer, consumer, x_loop)


def halide_parallel(p, loop: str):
    return parallelize_loop(p, p.find_loop(loop))


avx_f32_insts = [
    mm256_loadu_si256,
    mm256_storeu_si256,
    mm256_add_epi16,
]


def halide_vectorize(p, buffer: str, loop: str, width: int):
    loop = get_enclosing_loop_by_name(p, p.find(f"{buffer} = _"), loop)
    p = vectorize(p, loop, width, "f32", AVX2, avx_f32_insts, rules=[], tail="perfect")

    return p


# TODO: define a special case of compute_at which fully inlines computations
# TODO: implement halide's reorder over arbitrary loop nests


# from math import exp, pi, sqrt
# sigma = 1.5
# kernel = [exp(-x * x / (2 * sigma * sigma)) / (sqrt(2 * pi) * sigma) for x in range(4)]
# print(kernel)
# >> [0.2659615202676218, 0.2129653370149015, 0.10934004978399575, 0.035993977675458706]
# TODO: it would be nice if meta-programming allowed pre-computing values outside of the proc


@proc
def unsharp(W: size, H: size, output: f32[H, W], input: f32[H + 6, W + 6, 3]):
    assert H % 32 == 0
    assert W % 256 == 0

    gray: f32[H + 6, W + 6]
    for y in seq(0, H + 6):
        for x in seq(0, W + 6):
            gray[y, x] = (
                0.299 * input[y, x, 0] + 0.587 * input[y, x, 1] + 0.114 * input[y, x, 2]
            )

    blur_y: f32[H, W + 6]
    for y in seq(0, H):
        for x in seq(0, W + 6):
            blur_y[y, x] = (
                0.2659615202676218 * gray[y + 3, x]
                + 0.2129653370149015 * (gray[y + 2, x] + gray[y + 4, x])
                + 0.10934004978399575 * (gray[y + 1, x] + gray[y + 5, x])
                + 0.035993977675458706 * (gray[y + 0, x] + gray[y + 6, x])
            )

    blur_x: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            blur_x[y, x] = (
                0.2659615202676218 * blur_y[y, x + 3]
                + 0.2129653370149015 * (blur_y[y, x + 2] + blur_y[y, x + 4])
                + 0.10934004978399575 * (blur_y[y, x + 1] + blur_y[y, x + 5])
                + 0.035993977675458706 * (blur_y[y, x + 0] + blur_y[y, x + 6])
            )

    sharpen: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            sharpen[y, x] = 2 * gray[y + 3, x + 3] - blur_x[y, x]

    ratio: f32[H, W]
    for y in seq(0, H):
        for x in seq(0, W):
            ratio[y, x] = sharpen[y, x] / gray[y + 3, x + 3]

    for y in seq(0, H):
        for x in seq(0, W):
            for c in seq(0, 3):
                output[y, x] = ratio[y, x] * input[y + 3, x + 3, c]


unsharp = halide_split(unsharp, "output", "y", "y", "yi", 32)
unsharp = halide_compute_at(unsharp, "ratio", "output", "yi")
unsharp = halide_compute_at(unsharp, "sharpen", "ratio", "x")

print(unsharp)
