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
    assert n % 128 == 0
    assert m % 256 == 0

    f: ui8[n + 4, m + 4]
    producer(n, m, f, inp)
    consumer(n, m, f, g)


def prod_inline(p):
    p = inline(p, "producer(_)")
    p = inline(p, "consumer(_)")

    p = fuse_at(p, "f", "g", p.find_loop("i #1"))

    loop = p.find_loop("i")
    p = store_at(p, "f", "g", loop)

    p = fuse_at(p, "f", "g", p.find_loop("j #1"))

    loop = p.find_loop("j")
    p = store_at(p, "f", "g", loop)

    p = p
    p = unroll_loop(p, "ji")
    p = unroll_loop(p, "ii")
    for i in range(5):
        p = inline_assign(p, p.find("g[_] = _").prev())
    p = delete_buffer(p, "f : _")
    p = rename(p, "p_inline")

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


blur_staged = rename(blur, "blur_staged")
print("blur_staged")
print(blur_staged)
blur_inline = rename(prod_inline(blur), "blur_inline")
print("blur_inline")
print(blur_inline)
blur_tiled = rename(prod_tile(blur, i_tile=128, j_tile=256), "blur_tiled")
print("blur_tiled")
print(blur_tiled)

if __name__ == "__main__":
    print(blur)

__all__ = ["blur_staged", "blur_inline", "blur_tiled"]
