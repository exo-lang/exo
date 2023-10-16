from __future__ import annotations

import os
import sys

from exo import proc
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *

# Hide output when running through exocc.
if __name__ != "__main__" and hasattr(os, "devnull"):
    sys.stdout = open(os.devnull, "w")


@proc
def blur1d_compute_root(n: size, consumer: i8[n], inp: i8[n + 6]):
    producer: i8[n + 1]
    for i in seq(0, n + 1):
        producer[i] = (
            inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] + inp[i + 5]
        ) / 6.0

    for i in seq(0, n):
        consumer[i] = (producer[i] + producer[i + 1]) / 2.0


def schedule_blur1d():
    p = blur1d_compute_root
    print(blur1d_compute_root)

    p_bounds = (0, "i", 0, 2)
    c_bounds = (0, "i", 0, 1)

    loop = p.find_loop("i #1")
    p = fuse_at(p, "producer", "consumer", loop, c_bounds, p_bounds)
    p = rename(p, "blur1d_compute_at_store_root")
    print(p)

    loop = p.find_loop("i")
    p = store_at(p, "producer", "consumer", loop, p_bounds)
    p = rename(p, "blur1d_compute_at")
    print(p)

    p = unroll_loop(p, "ii")
    for i in range(2):
        p = inline_assign(
            p,
            p.find("consumer[_] = _").as_block().expand(delta_lo=1, delta_hi=0),
        )
    p = delete_buffer(p, "producer: _")
    p = rename(p, "blur1d_inline")
    print(p)


@proc
def blur2d_compute_root(n: size, consumer: i8[n, n], sin: i8[n + 1, n + 1]):
    assert n % 4 == 0
    producer: i8[n + 1, n + 1]
    for i in seq(0, n + 1):
        for j in seq(0, n + 1):
            producer[i, j] = sin[
                i, j
            ]  # just a placeholder since sine can't evalute on index exprs

    for i in seq(0, n):
        for j in seq(0, n):
            consumer[i, j] = (
                producer[i, j]
                + producer[i, j + 1]
                + producer[i + 1, j]
                + producer[i + 1, j + 1]
            ) / 4.0


def schedule_blur2d():
    p = blur2d_compute_root
    print(p)

    c_i_bounds = (0, "i", 0, 1)
    p_i_bounds = (0, "i", 0, 2)
    c_j_bounds = (1, "j", 0, 1)
    p_j_bounds = (1, "j", 0, 2)

    loop = p.find_loop("i #1")
    p = fuse_at(p, "producer", "consumer", loop, c_i_bounds, p_i_bounds)
    p = rename(
        p,
        "blur2d_compute_at_i_store_root",
    )
    print(p)
    p_tmp = p  # For testing different branches of scheduling

    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"), c_j_bounds, p_j_bounds)
    p = rename(p, "blur2d_compute_at_j_store_root")
    print(p)

    p = store_at(p_tmp, "producer", "consumer", p_tmp.find_loop("i"), p_i_bounds)
    p = rename(p, "blur2d_compute_at_i")
    print(p)

    p = fuse_at(
        p,
        "producer",
        "consumer",
        p.find_loop("j #1"),
        c_j_bounds,
        p_j_bounds,
    )
    p = simplify(p)
    p = rename(p, "blur2d_compute_at_j_store_at_i")
    print(p)

    p = store_at(p, "producer", "consumer", p.find_loop("j"), p_j_bounds)
    p = unroll_loop(p, "ji")
    p = unroll_loop(p, "ii")
    for i in range(4):
        p = inline_assign(
            p,
            p.find("consumer[_] = _").as_block().expand(delta_lo=1, delta_hi=0),
        )
    p = delete_buffer(p, "producer: _")
    p = rename(p, "blur2d_inline")
    print(p)


def schedule_blur2d_tiled():
    # TODO: make a tile composed schedule from the below
    compute_root = blur2d_compute_root

    tiled = tile(
        compute_root,
        "consumer",
        "i",
        "j",
        ["io", "ii"],
        ["jo", "ji"],
        4,
        4,
        perfect=True,
    )
    tiled = rename(tiled, "blur2d_tiled")
    print(tiled)

    # Bounds inference for the consumer and producer at various
    # loop levels in the tiled implementation. Comes in the form
    # (dim_idx, base, lo, hi), which means it affects the [dim_idx]
    # dimension of the buffer, and it ranges from [base+lo, base+hi)
    # TODO: we should be able to do this automatically
    tiled_c_io_bounds = (0, "4 * io", 0, 4)
    tiled_p_io_bounds = (0, "4 * io", 0, 5)
    tiled_c_jo_bounds = (1, "4 * jo", 0, 4)
    tiled_p_jo_bounds = (1, "4 * jo", 0, 5)
    tiled_c_ii_bounds = (0, "4 * io + ii", 0, 1)
    tiled_p_ii_bounds = (0, "4 * io + ii", 0, 2)
    tiled_c_ji_bounds = (1, "4 * jo + ji", 0, 1)
    tiled_p_ji_bounds = (1, "4 * jo + ji", 0, 2)

    loop = tiled.find_loop("io")
    tiled_compute_at_io = fuse_at(
        tiled, "producer", "consumer", loop, tiled_c_io_bounds, tiled_p_io_bounds
    )
    # TODO: maybe rewrite_expr of predicates should be in simplify
    tiled_compute_at_io = simplify(rewrite_expr(tiled_compute_at_io, "n%4", 0))
    tiled_compute_at_io = rename(
        tiled_compute_at_io,
        "blur2d_tiled_compute_at_io",
    )
    print(tiled_compute_at_io)

    loop = tiled_compute_at_io.find_loop("jo")
    tiled_compute_at_jo = fuse_at(
        tiled_compute_at_io,
        "producer",
        "consumer",
        loop,
        tiled_c_jo_bounds,
        tiled_p_jo_bounds,
    )
    tiled_compute_at_jo = simplify(rewrite_expr(tiled_compute_at_jo, "n%4", 0))
    tiled_compute_at_jo = rename(
        tiled_compute_at_jo,
        "blur2d_tiled_compute_at_jo",
    )
    print(tiled_compute_at_jo)

    loop = tiled_compute_at_jo.find_loop("ii #1")
    tiled_compute_at_ii = fuse_at(
        tiled_compute_at_jo,
        "producer",
        "consumer",
        loop,
        tiled_c_ii_bounds,
        tiled_p_ii_bounds,
    )
    tiled_compute_at_ii = rename(
        tiled_compute_at_ii,
        "blur2d_tiled_compute_at_ii",
    )
    print(tiled_compute_at_ii)

    loop = tiled_compute_at_ii.find_loop("ji #1")
    tiled_compute_at_ji = fuse_at(
        tiled_compute_at_ii,
        "producer",
        "consumer",
        loop,
        tiled_c_ji_bounds,
        tiled_p_ji_bounds,
    )
    tiled_compute_at_ji = rename(
        tiled_compute_at_ji,
        "blur2d_tiled_compute_at_ji",
    )
    print(tiled_compute_at_ji)


schedule_blur1d()
schedule_blur2d()
schedule_blur2d_tiled()
