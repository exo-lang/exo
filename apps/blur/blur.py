from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STACK
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import get_enclosing_loop
from exo.stdlib.stdlib import vectorize, is_div, is_literal

# TODO: fix duplicated functions, e.g. get_enclosing_loop
# TODO: change check_replace to work for blocks


def halide_tile(p, buffer, y, x, yi, xi, yTile, xTile):
    assign = p.find(f"{buffer} = _")
    y_loop = get_enclosing_loop(assign, y)
    x_loop = get_enclosing_loop(assign, x)

    return tile(p, y_loop, x_loop, [y, yi], [x, xi], yTile, xTile, perfect=True)


def halide_compute_at(p, producer: str, consumer: str, loop: str):
    x_loop = get_enclosing_loop(p.find(f"{consumer} = _"), loop)
    return compute_at(p, "blur_x", "blur_y", x_loop)


def halide_parallel(p, loop: str):
    return parallelize_loop(p, p.find_loop(loop))


avx_ui16_insts = [
    mm256_loadu_si256,
    mm256_storeu_si256,
    mm256_add_epi16,
    avx2_ui16_divide_by_3,
]


def divide_by_3_rule(proc, expr):
    expr = proc.forward(expr)

    if is_div(proc, expr) and is_literal(proc, expr.rhs(), value=3):
        return [expr.lhs()]


def halide_vectorize(p, buffer: str, loop: str, width: int):
    loop = get_enclosing_loop(p.find(f"{buffer} = _"), loop)
    rules = [divide_by_3_rule]
    p = vectorize(
        p, loop, width, "ui16", AVX2, avx_ui16_insts, rules=rules, tail="perfect"
    )

    return p


@proc
def blur(W: size, H: size, blur_y: ui16[H, W], inp: ui16[H + 2, W + 2]):
    assert H % 32 == 0
    assert W % 256 == 0

    blur_x: ui16[H + 2, W]
    for y in seq(0, H + 2):
        for x in seq(0, W):
            blur_x[y, x] = (inp[y, x] + inp[y, x + 1] + inp[y, x + 2]) / 3.0
    for y in seq(0, H):
        for x in seq(0, W):
            blur_y[y, x] = (blur_x[y, x] + blur_x[y + 1, x] + blur_x[y + 2, x]) / 3.0


def prod_halide(p):
    p = halide_tile(p, "blur_y", "y", "x", "yi", "xi", 32, 256)
    p = halide_compute_at(p, "blur_x", "blur_y", "x")
    p = halide_parallel(p, "y")
    p = halide_vectorize(p, "blur_x", "xi", 16)
    p = halide_vectorize(p, "blur_y", "xi", 16)
    p = set_memory(p, p.find(f"blur_x : _"), DRAM_STACK)
    p = simplify(
        p
    )  # necessary because unification is not deterministic, which breaks test cases
    return p


blur_halide = rename(prod_halide(blur), "exo_blur_halide")
print("blur_halide")
print(blur_halide)
