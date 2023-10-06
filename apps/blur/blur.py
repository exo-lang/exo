from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


@proc
def producer(n: size, m: size, f: ui8[n + 4, m + 4], inp: ui8[n + 4, m + 4]):
    for i in seq(0, n + 4):
        for j in seq(0, m):
            f[i, j] = (
                inp[i, j]
                + inp[i, j + 1]
                + inp[i, j + 2]
                + inp[i, j + 3]
                + inp[i, j + 4]
            ) / 5.0


@proc
def consumer(n: size, m: size, f: ui8[n + 4, m + 4], g: ui8[n + 4, m + 4]):
    for i in seq(0, n):
        for j in seq(0, m):
            g[i, j] = (
                f[i, j] + f[i + 1, j] + f[i + 2, j] + f[i + 3, j] + f[i + 4, j]
            ) / 5.0


@proc
def blur(n: size, m: size, g: ui8[n + 4, m + 4], inp: ui8[n + 4, m + 4]):
    assert n % 256 == 0
    assert m % 256 == 0

    f: ui8[n + 4, m + 4]
    producer(n, m, f, inp)
    consumer(n, m, f, g)


def prod_inline(p):
    p = inline(p, "producer(_)")
    p = inline(p, "consumer(_)")

    c_bounds = (0, "i", 0, 1)
    p_bounds = (0, "i", 0, 5)
    p_compute_at_store_root = compute_at(
        p, "f", "g", p.find_loop("i #1"), c_bounds, p_bounds
    )

    loop = p_compute_at_store_root.find_loop("i")
    p_compute_at_store_at = store_at(p_compute_at_store_root, "f", "g", loop, p_bounds)

    c_bounds_2 = (1, "j", 0, 1)
    p_bounds_2 = (1, "j", 0, 1)
    p_compute_at_store_root_j = compute_at(
        p_compute_at_store_at,
        "f",
        "g",
        p_compute_at_store_at.find_loop("j #1"),
        c_bounds_2,
        p_bounds_2,
    )

    p_inline = p_compute_at_store_root_j
    for i in range(5):
        p_inline = inline_assign(
            p_inline,
            p_inline.find("g[_] = _").as_block().expand(delta_lo=1, delta_hi=0),
        )
    p_inline = delete_buffer(p_inline, "f : _")
    p_inline = rename(p_inline, "p_inline")

    return p_inline


def prod_tile(p, tile_size=32):
    p = inline(p, "producer(_)")
    p = inline(p, "consumer(_)")
    p = tile(
        p, "g", "i", "j", ["io", "ii"], ["jo", "ji"], tile_size, tile_size, perfect=True
    )

    tiled_c_io_bounds = (0, f"{tile_size} * io", 0, tile_size)
    tiled_p_io_bounds = (0, f"{tile_size} * io", 0, tile_size + 5)
    tiled_c_jo_bounds = (1, f"{tile_size} * jo", 0, tile_size)
    tiled_p_jo_bounds = (1, f"{tile_size} * jo", 0, tile_size)

    loop = p.find_loop("io")
    p = compute_at(p, "f", "g", loop, tiled_c_io_bounds, tiled_p_io_bounds, hardcode=4)
    p = reorder_loops(p, "ii j")

    loop = p.find_loop("jo")
    p = compute_at(p, "f", "g", loop, tiled_c_jo_bounds, tiled_p_jo_bounds, hardcode=0)

    p = store_at(p, "f", "g", p.find_loop("io"), tiled_p_io_bounds)
    p = store_at(p, "f", "g", p.find_loop("jo"), tiled_p_jo_bounds)
    p = lift_alloc(p, "f: _", n_lifts=2)

    return p


blur_staged = rename(blur, "blur_staged")
print("blur_staged")
print(blur_staged)
blur_inline = rename(prod_inline(blur), "blur_inline")
print("blur_inline")
print(blur_inline)
blur_tiled = rename(prod_tile(blur, tile_size=128), "blur_tiled")
print("blur_tiled")
print(blur_tiled)

if __name__ == "__main__":
    print(blur)

__all__ = ["blur_staged", "blur_inline", "blur_tiled"]
