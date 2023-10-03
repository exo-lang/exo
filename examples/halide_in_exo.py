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

    loop = blur1d_compute_root.find_loop("i #1")
    blur1d_compute_at_store_root = rename(
        compute_at(blur1d_compute_root, "producer", "consumer", loop, bounds),
        "blur1d_compte_at_store_root",
    )
    print(blur1d_compute_at_store_root)

    loop = blur1d_compute_at_store_root.find_loop("i")
    blur1d_compute_at = rename(
        store_at(blur1d_compute_at_store_root, "producer", "consumer", loop, bounds),
        "blur1d_compute_at",
    )
    print(blur1d_compute_at)

    blur1d_compute_at = inline_assign(
        blur1d_compute_at,
        blur1d_compute_at.find("producer_tmp[_] = _ #1")
        .as_block()
        .expand(delta_lo=0, delta_hi=1),
    )
    blur1d_compute_at = inline_assign(
        blur1d_compute_at,
        blur1d_compute_at.find("producer_tmp[_] = _")
        .as_block()
        .expand(delta_lo=0, delta_hi=1),
    )
    blur1d_compute_at = delete_buffer(blur1d_compute_at, "producer_tmp : _")
    print(blur1d_compute_at)


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

    bounds = [("i", 0, 2), ("0", 0, "n+1")]

    loop = blur2d_compute_root.find_loop("i #1")
    blur2d_compute_at_i_store_root = rename(
        compute_at(blur2d_compute_root, "producer", "consumer", loop, bounds),
        "blur2d_compute_at_i_store_root",
    )
    print(blur2d_compute_at_i_store_root)

    loop = blur2d_compute_at_i_store_root.find_loop("j #1")
    blur2d_compute_at_j_store_root = rename(
        compute_at(
            blur2d_compute_at_i_store_root, "producer", "consumer", loop, bounds
        ),
        "blur2d_compute_at_j_store_root",
    )
    print(blur2d_compute_at_j_store_root)

    # TODO: current strategy doesn't work with non-constant sizes...
    # loop = blur2d_compute_at_i_store_root.find_loop("i")
    # blur2d_compute_at = rename(
    #     store_at(
    #         blur2d_compute_at_i_store_root, "producer", "consumer", loop, bounds
    #     ),
    #     "blur2d_compute_at"
    # )
    # print(blur2d_compute_at)


schedule_blur1d()
schedule_blur2d()
