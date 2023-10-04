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
    print(blur2d_compute_root)

    loop = blur2d_compute_root.find_loop("i #1")
    blur2d_compute_at_i_store_root = compute_at(
        blur2d_compute_root, "producer", "consumer", loop, (0, "i", 0, 2)
    )
    blur2d_compute_at_i_store_root = rename(
        blur2d_compute_at_i_store_root,
        "blur2d_compute_at_i_store_root",
    )
    print(blur2d_compute_at_i_store_root)

    loop = blur2d_compute_at_i_store_root.find_loop("i")
    blur2d_compute_at_i = store_at(
        blur2d_compute_at_i_store_root, "producer", "consumer", loop, (0, "i", 0, 2)
    )
    blur2d_compute_at_i = rename(blur2d_compute_at_i, "blur2d_compute_at_i")
    print(blur2d_compute_at_i)

    loop = blur2d_compute_at_i_store_root.find_loop("j #1")
    blur2d_compute_at_j_store_root = compute_at(
        blur2d_compute_at_i_store_root, "producer", "consumer", loop, (1, "j", 0, 2)
    )
    blur2d_compute_at_j_store_root = rename(
        blur2d_compute_at_j_store_root, "blur2d_compute_at_j_store_root"
    )
    print(blur2d_compute_at_j_store_root)

    loop = blur2d_compute_at_i.find_loop("j #1")
    blur2d_compute_at_j_store_at_i = compute_at(
        blur2d_compute_at_i, "producer", "consumer", loop, (1, "j", 0, 2)
    )
    blur2d_compute_at_j_store_at_i = rename(
        blur2d_compute_at_j_store_at_i, "blur2d_compute_at_j_store_at_i"
    )
    blur2d_compute_at_j_store_at_i = simplify(blur2d_compute_at_j_store_at_i)
    print(blur2d_compute_at_j_store_at_i)

    loop = blur2d_compute_at_j_store_at_i.find_loop("j")
    blur2d_inline = store_at(
        blur2d_compute_at_j_store_at_i, "producer", "consumer", loop, (1, "j", 0, 2)
    )
    for i in range(4):
        blur2d_inline = inline_assign(
            blur2d_inline,
            blur2d_inline.find("consumer[_] = _")
            .as_block()
            .expand(delta_lo=1, delta_hi=0),
        )
    blur2d_inline = delete_buffer(blur2d_inline, "producer: _")
    blur2d_inline = rename(blur2d_inline, "blur2d_inline")
    print(blur2d_inline)


# schedule_blur1d()
schedule_blur2d()
