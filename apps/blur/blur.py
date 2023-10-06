from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


@proc
def producer(n: size, m: size, f: ui8[n, m], inp: ui8[n, m]):
    assert m > 5
    for i in seq(0, n):
        for j in seq(0, m - 4):
            f[i, j] = (
                inp[i, j]
                + inp[i, j + 1]
                + inp[i, j + 2]
                + inp[i, j + 3]
                + inp[i, j + 4]
            ) / 5.0


@proc
def consumer(n: size, m: size, f: ui8[n, m], g: ui8[n, m]):
    assert n > 5
    assert m > 5
    for i in seq(0, n - 4):
        for j in seq(0, m - 4):
            g[i, j] = (
                f[i, j] + f[i + 1, j] + f[i + 2, j] + f[i + 3, j] + f[i + 4, j]
            ) / 5.0


@proc
def blur(n: size, m: size, g: ui8[n, m], inp: ui8[n, m]):
    assert n > 5
    assert m > 5
    f: ui8[n, m]
    producer(n, m, f, inp)
    consumer(n, m, f, g)


def prod_inline(p):
    p = inline(p, "producer(_)")
    p = inline(p, "consumer(_)")

    print(p)
    c_bounds = (0, "i", 0, 1)
    p_bounds = (0, "i", 0, 5)
    p_compute_at_store_root = compute_at(
        p, "f", "g", p.find_loop("i #1"), c_bounds, p_bounds
    )
    print("p_compute_at_store_root")
    print(p_compute_at_store_root)
    print()

    loop = p_compute_at_store_root.find_loop("i")
    p_compute_at_store_at = store_at(p_compute_at_store_root, "f", "g", loop, p_bounds)
    print("p_compute_at_store_at")
    print(p_compute_at_store_at)
    print()

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

    print(p_compute_at_store_root_j)

    p_inline = p_compute_at_store_root_j
    for i in range(5):
        p_inline = inline_assign(
            p_inline,
            p_inline.find("g[_] = _").as_block().expand(delta_lo=1, delta_hi=0),
        )
    p_inline = delete_buffer(p_inline, "f : _")
    p_inline = rename(p_inline, "p_inline")
    print(p_inline)

    return p_inline


blur_staged = rename(blur, "blur_staged")
blur_inline = rename(prod_inline(blur), "blur_inlined")

if __name__ == "__main__":
    print(blur)

__all__ = ["blur_staged", "blur_inline"]
