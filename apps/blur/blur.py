from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


@proc
def do_blur_x(W: size, H: size, inp: ui16[H + 2, W + 2], out: ui16[H + 2, W]):
    for y in seq(0, H + 2):
        for x in seq(0, W):
            out[y, x] = (inp[y, x] + inp[y, x + 1] + inp[y, x + 2]) / 3.0


@proc
def do_blur_y(W: size, H: size, inp: ui16[H + 2, W], out: ui16[H, W]):
    for y in seq(0, H):
        for x in seq(0, W):
            out[y, x] = (inp[y, x] + inp[y + 1, x] + inp[y + 2, x]) / 3.0


@proc
def blur(W: size, H: size, blur_y: ui16[H, W], inp: ui16[H + 2, W + 2]):
    assert H % 32 == 0
    assert W % 16 == 0

    blur_x: ui16[H + 2, W]
    do_blur_x(W, H, inp, blur_x)
    do_blur_y(W, H, blur_x, blur_y)


def inline_stages(p):
    p = inline(p, "do_blur_y(_)")
    p = inline(p, "do_blur_x(_)")
    return p


def prod_halide(p):
    p = inline(p, "do_blur_y(_)")
    p = inline(p, "do_blur_x(_)")

    p = divide_loop(p, p.find_loop("y #1"), 32, ["y", "yi"], perfect=True)

    # blur_x.compute_at(blur_y, x)
    # TODO: would rather not have to find the loop every time,
    # but fuse invalidates the second loop...
    p = fuse_at(p, "blur_x", "blur_y", p.find_loop("y #1"), reorder=False)
    # TODO: This simplify is ugly
    p = rewrite_expr(p, "H % 32", 0)
    p = simplify(p)

    p = fuse_at(p, "blur_x", "blur_y", p.find_loop("yi #1"))

    p = fuse_at(p, "blur_x", "blur_y", p.find_loop("x #1"))
    p = unroll_loop(p, "xi")

    # blur_x.store_at(blur_y, y)
    p = store_at(p, "blur_x", "blur_y", p.find_loop("y"))

    p = lift_alloc(p, p.find("blur_x: _"))

    return p


def prod_tile(p, i_tile=32, j_tile=32):
    p = inline(p, "producer(_)")
    p = inline(p, "consumer(_)")
    p = tile(p, "g", "i", "j", ["io", "ii"], ["jo", "ji"], i_tile, j_tile, perfect=True)
    p = simplify(p)

    loop = p.find_loop("io")
    p = fuse_at(p, "f", "g", loop)

    loop = p.find_loop("jo")
    p = fuse_at(p, "f", "g", loop)

    # TODO: eliminate this
    p = rewrite_expr(p, "n % 128", 0)
    p = rewrite_expr(p, "m % 256", 0)
    p = simplify(p)

    p = store_at(p, "f", "g", p.find_loop("io"))
    p = store_at(p, "f", "g", p.find_loop("jo"))
    p = lift_alloc(p, "f: _", n_lifts=2)

    p = simplify(p)
    return p


blur_staged = rename(inline_stages(blur), "exo_blur_staged")
print("blur_staged")
print(blur_staged)
blur_halide = rename(prod_halide(blur), "exo_blur_halide")
print("blur_halide")
print(blur_halide)
# blur_inline = rename(prod_inline(blur), "blur_inline")
# print("blur_inline")
# print(blur_inline)
# blur_tiled = rename(prod_tile(blur, i_tile=128, j_tile=256), "blur_tiled")
# print("blur_tiled")
# print(blur_tiled)

if __name__ == "__main__":
    print(blur)

# __all__ = ["blur_staged", "blur_inline", "blur_tiled"]
__all__ = ["blur_staged", "blur_halide"]
