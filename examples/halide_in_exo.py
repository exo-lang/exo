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
    print(blur1d_compute_root)

    bounds = [("i", 0, 2)]

    blur1d_compute_at_store_root = rename(
        blur1d_compute_root, "blur1d_compute_at_store_root"
    )
    loop = blur1d_compute_at_store_root.find_loop("i #1")
    blur1d_compute_at_store_root = compute_at(
        blur1d_compute_at_store_root, "producer", "consumer", loop, bounds
    )
    print(blur1d_compute_at_store_root)

    blur1d_compute_at = rename(blur1d_compute_at_store_root, "blur1d_compute_at")
    loop = blur1d_compute_at_store_root.find_loop("i")
    blur1d_compute_at = store_at(
        blur1d_compute_at_store_root, "producer", "consumer", loop, bounds
    )
    print(blur1d_compute_at)

    blur1d_inline = rename(blur1d_compute_at, "blur1d_inline")
    for i in range(2):
        blur1d_inline = inline_assign(
            blur1d_inline,
            blur1d_inline.find("consumer[_] = _")
            .as_block()
            .expand(delta_lo=1, delta_hi=0),
        )
    blur1d_inline = delete_buffer(blur1d_inline, "producer: _")
    print(blur1d_inline)


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
    compute_root = blur2d_compute_root
    print(compute_root)

    c_i_bounds = (0, "i", 0, 1)
    p_i_bounds = (0, "i", 0, 2)
    c_j_bounds = (1, "j", 0, 1)
    p_j_bounds = (1, "j", 0, 2)

    loop = compute_root.find_loop("i #1")
    compute_at_i_store_root = compute_at(
        compute_root, "producer", "consumer", loop, c_i_bounds, p_i_bounds
    )
    compute_at_i_store_root = rename(
        compute_at_i_store_root,
        "blur2d_compute_at_i_store_root",
    )
    print(compute_at_i_store_root)

    loop = compute_at_i_store_root.find_loop("i")
    compute_at_i = store_at(
        compute_at_i_store_root, "producer", "consumer", loop, p_i_bounds
    )
    compute_at_i = rename(compute_at_i, "blur2d_compute_at_i")
    print(compute_at_i)

    loop = compute_at_i_store_root.find_loop("j #1")
    compute_at_j_store_root = compute_at(
        compute_at_i_store_root, "producer", "consumer", loop, c_j_bounds, p_j_bounds
    )
    compute_at_j_store_root = rename(
        compute_at_j_store_root, "blur2d_compute_at_j_store_root"
    )
    print(compute_at_j_store_root)

    loop = compute_at_i.find_loop("j #1")
    compute_at_j_store_at_i = compute_at(
        compute_at_i, "producer", "consumer", loop, c_j_bounds, p_j_bounds
    )
    compute_at_j_store_at_i = rename(
        compute_at_j_store_at_i, "blur2d_compute_at_j_store_at_i"
    )
    compute_at_j_store_at_i = simplify(compute_at_j_store_at_i)
    print(compute_at_j_store_at_i)

    loop = compute_at_j_store_at_i.find_loop("j")
    inline = store_at(compute_at_j_store_at_i, "producer", "consumer", loop, p_j_bounds)
    for i in range(4):
        inline = inline_assign(
            inline,
            inline.find("consumer[_] = _").as_block().expand(delta_lo=1, delta_hi=0),
        )
    inline = delete_buffer(inline, "producer: _")
    inline = rename(inline, "blur2d_inline")
    print(inline)


def schedule_blur2d_tiled():
    # TODO: make a tile composed schedule from the below
    compute_root = blur2d_compute_root

    i_loop = compute_root.find_loop("i #1")
    j_loop = compute_root.find_loop("j #1")
    tiled = divide_loop(compute_root, i_loop, 4, ["io", "ii"], perfect=True)
    tiled = divide_loop(tiled, j_loop, 4, ["jo", "ji"], perfect=True)
    tiled = reorder_loops(tiled, "ii jo")
    tiled = rename(tiled, "blur2d_tiled")
    print(tiled)

    tiled_c_io_bounds = (0, "4 * io", 0, 4)
    tiled_p_io_bounds = (0, "4 * io", 0, 5)
    tiled_c_jo_bounds = (1, "4 * jo", 0, 4)
    tiled_p_jo_bounds = (1, "4 * jo", 0, 5)
    tiled_c_ii_bounds = (0, "4 * io + ii", 0, 1)
    tiled_p_ii_bounds = (0, "4 * io + ii", 0, 2)
    tiled_c_ji_bounds = (1, "4 * jo + ji", 0, 1)
    tiled_p_ji_bounds = (1, "4 * jo + ji", 0, 2)

    loop = tiled.find_loop("io")
    tiled_compute_at_io = compute_at(
        tiled, "producer", "consumer", loop, tiled_c_io_bounds, tiled_p_io_bounds
    )
    tiled_compute_at_io = rename(
        tiled_compute_at_io,
        "blur2d_tiled_compute_at_io",
    )
    print(tiled_compute_at_io)

    tiled_compute_at_jo = reorder_loops(tiled_compute_at_io, "ii j")
    loop = tiled_compute_at_jo.find_loop("jo")
    tiled_compute_at_jo = compute_at(
        tiled_compute_at_jo,
        "producer",
        "consumer",
        loop,
        tiled_c_jo_bounds,
        tiled_p_jo_bounds,
    )
    tiled_compute_at_jo = rename(
        tiled_compute_at_jo,
        "blur2d_tiled_compute_at_jo",
    )
    print(tiled_compute_at_jo)

    tiled_compute_at_ii = reorder_loops(tiled_compute_at_jo, "ji ii")
    loop = tiled_compute_at_ii.find_loop("ii #1")
    tiled_compute_at_ii = compute_at(
        tiled_compute_at_ii,
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
    tiled_compute_at_ji = compute_at(
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


# schedule_blur1d()
# schedule_blur2d()
schedule_blur2d_tiled()
