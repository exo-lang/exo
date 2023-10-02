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


@proc
def goal(n: size, sin: i8[n + 1]):
    producer: i8[n + 1]
    consumer: i8[n]
    for i in seq(0, n):
        producer[i] = sin[i]
        producer[i + 1] = sin[i + 1]
        consumer[i] = producer[i] + producer[i + 1]


@proc
def blur2d_compute_root(n: size, consumer: i8[n, n], sin: i8[n + 1, n + 1]):
    producer: i8[n + 1, n + 1]
    for i in seq(0, n + 1):
        for j in seq(0, n + 1):
            producer[i, j] = sin[i, j]

    for i in seq(0, n):
        for j in seq(0, n):
            consumer[i, j] = (
                producer[i, j]
                + producer[i, j + 1]
                + producer[i + 1, j]
                + producer[i + 1, j + 1]
            )


"""
Compute_at(g, f, i) means compute the necessary g values within loop i over f
Store_at(g, f, i) means to allocate g within the loop i over f

caveats to consider:
 - no bounds inference
"""


def schedule_blur1d_compute_at():
    # TODO: this approach only works for constant sized kernels
    blur = rename(blur1d_compute_root, "blur_scheduled_compute_root")
    print("Initial blur:\n", blur)

    prod_loop = blur.find_loop("i")
    consumer_loop = blur.find_loop("i #1")
    accesses = range(2)  # TODO: need introspection to get this

    # Assumes constant, consecutive windows
    first_prod_loop = prod_loop
    for i in accesses[1:]:
        # surgery
        blur = cut_loop(blur, prod_loop, f"n + {i} - 1")
        blur = cut_loop(blur, prod_loop, i)

        # duplicate work
        middle_prod_loop = blur.forward(prod_loop).next()
        blur = add_loop(blur, middle_prod_loop, "ii", 2)
        blur = unroll_loop(blur, blur.forward(middle_prod_loop).parent())

        # stitch together
        blur = join_loops(blur, prod_loop, blur.forward(prod_loop).next())
        next_loop = blur.forward(prod_loop).next()
        blur = join_loops(blur, next_loop, next_loop.next())
        blur = simplify(blur)
        prod_loop = next_loop

    # merge producer loops
    prod_loop = blur.forward(first_prod_loop)
    for i in accesses[1:]:
        next_loop = prod_loop.next()
        blur = shift_loop(blur, next_loop, 0)
        blur = fuse(blur, prod_loop, next_loop, unsafe_disable_check=True)
        prod_loop = blur.forward(prod_loop)

    # fuse with consumer
    blur = fuse(blur, prod_loop, consumer_loop, unsafe_disable_check=True)

    return simplify(blur)


blur1d_compute_at = schedule_blur1d_compute_at()
print("Compute at blur:\n", blur1d_compute_at)
