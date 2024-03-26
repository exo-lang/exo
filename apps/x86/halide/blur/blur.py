from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STACK
from exo.platforms.x86 import *

from exo.stdlib.scheduling import *
from exo.stdlib.halide_scheduling_ops import *
from exo.stdlib.inspection import get_enclosing_loop_by_name
from exo.stdlib.stdlib import vectorize, is_div, is_literal


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
    loop = get_enclosing_loop_by_name(p, p.find(f"{buffer} = _"), loop)
    rules = [divide_by_3_rule]

    # Ensure that it is the innermost loop
    while len(loop.body()) == 1 and isinstance(loop.body()[0], ForCursor):
        p = reorder_loops(p, loop)
        loop = p.forward(loop)

    p = vectorize(
        p,
        loop,
        width,
        "ui16",
        AVX2,
        avx_ui16_insts,
        rules=[divide_by_3_rule],
        tail="perfect",
    )

    return p


@proc
def exo_base_blur(W: size, H: size, blur_y: ui16[H, W], inp: ui16[H + 2, W + 2]):
    assert H % 32 == 0
    assert W % 256 == 0

    blur_x: ui16[H + 2, W]
    for y in seq(0, H + 2):
        for x in seq(0, W):
            blur_x[y, x] = (inp[y, x] + inp[y, x + 1] + inp[y, x + 2]) / 3.0
    for y in seq(0, H):
        for x in seq(0, W):
            blur_y[y, x] = (blur_x[y, x] + blur_x[y + 1, x] + blur_x[y + 2, x]) / 3.0


def halide_schedule(p):
    p = halide_tile(p, "blur_y", "y", "x", "yi", "xi", 32, 256)
    p = halide_compute_and_store_at(p, "blur_x", "blur_y", "x")
    p = halide_parallel(p, "y")
    p = halide_vectorize(p, "blur_x", "xi", 16)
    p = halide_vectorize(p, "blur_y", "xi", 16)
    p = set_memory(p, p.find("blur_x : _"), DRAM_STACK)
    p = simplify(p)  # necessary because unification is not deterministic
    return p


exo_blur = rename(halide_schedule(exo_base_blur), "exo_blur_halide")
print(exo_blur)
