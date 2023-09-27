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
def blur1d_compute_root(n: size, consumer: i8[n], sin: i8[n + 1]):
    producer: i8[n + 1]
    for i in seq(0, n + 1):
        producer[i] = sin[i]

    consumer: i8[n]
    for i in seq(0, n):
        consumer[i] = producer[i] + producer[i + 1]


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

    prod_loop = blur.find_loop("i")
    consumer_loop = blur.find_loop("i #1")

    # loop surgery
    blur = cut_loop(blur, prod_loop, "n")
    blur = cut_loop(blur, prod_loop, 1)
    middle_prod_loop = blur.forward(prod_loop).next()
    blur = add_loop(
        blur, middle_prod_loop, "ii", 2
    )  # I don't like the manual forwarding that's needed because of the next() call...
    blur = unroll_loop(blur, blur.forward(middle_prod_loop).parent())
    blur = join_loops(blur, prod_loop, blur.forward(prod_loop).next())
    next_loop = blur.forward(prod_loop).next()
    blur = join_loops(blur, next_loop, next_loop.next())
    blur = shift_loop(blur, next_loop, 0)

    # loop fusion
    blur = simplify(blur)
    blur = fuse(blur, prod_loop, next_loop, unsafe_disable_check=True)
    blur = reorder_stmts(blur, blur.forward(prod_loop).expand(0, 1))
    blur = fuse(blur, prod_loop, consumer_loop, unsafe_disable_check=True)

    return simplify(blur)


blur1d_compute_at = schedule_blur1d_compute_at()
print(blur1d_compute_at)

"""
Compute_at (without store_at)
=================================
Add bare minimum extra compute, and then perform loop manipulation

NOTE: For the introspection, this could be good because we make scheduling
decisions based on the consumer's access pattern of the producer.
----------
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n + 1):
        for j in seq(0, n + 1):
            producer[i, j] = sin[i, j]
-- TODO: generalized cut_loops (NOTE: this depends on access pattern of consumer) ->
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n + 1):
        for j in seq(0, 1):
            producer[i, j] = sin[i, j]
        for j in seq(1, n):
            producer[i, j] = sin[i, j]
        for j in seq(n, n + 1):
            producer[i, j] = sin[i, j]
-- add_loop/unroll (NOTE: this depends on access patern of consumer) ->
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n + 1):
        for j in seq(0, 1):
            producer[i, j] = sin[i, j]
        for j in seq(1, n):
            producer[i, j] = sin[i, j]
        for j in seq(1, n):
            producer[i, j] = sin[i, j]
        for j in seq(n, n + 1):
            producer[i, j] = sin[i, j]
-- TODO: merge loop bounds ->
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n + 1):
        for j in seq(0, n):
            producer[i, j] = sin[i, j]
        for j in seq(1, n + 1):
            producer[i, j] = sin[i, j]
-- TODO: shift loop bounds ->
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n + 1):
        for j in seq(0, n):
            producer[i, j] = sin[i, j]
        for j in seq(0, n):
            producer[i, j + 1] = sin[i, j + 1]
-- fuse_loops ->
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n + 1):
        for j in seq(0, n):
            producer[i, j] = sin[i, j]
            producer[i, j + 1] = sin[i, j + 1]

Now if we repeat for i and add back the rest of the code:
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n):
        for j in seq(0, n):
            producer[i, j] = sin[i, j]
            producer[i, j + 1] = sin[i, j + 1]
            producer[i + 1, j] = sin[i + 1, j]
            producer[i + 1, j + 1] = sin[i + 1, j + 1]
    for i in seq(0, n):
        for j in seq(0, n):
            consumer[i, j] = producer[i, j] + producer[i, j + 1] + producer[i + 1, j] + producer[i + 1, j + 1]
-- fuse_loops ->
    producer: i8[n + 1, n + 1] @ DRAM
    for i in seq(0, n):
        for j in seq(0, n):
            producer[i, j] = sin[i, j]
            producer[i, j + 1] = sin[i, j + 1]
            producer[i + 1, j] = sin[i + 1, j]
            producer[i + 1, j + 1] = sin[i + 1, j + 1]
            consumer[i, j] = producer[i, j] + producer[i, j + 1] + producer[i + 1, j] + producer[i + 1, j + 1]
"""

"""

producer: i8[n + 1, n + 1] @ DRAM
for i in seq(0, n):
    for j in seq(0, n):
        for ii in seq(0, 2):
            for ji in seq(0, 2):
                producer[i+ii, j+ji] = sin[i+ii, j+ji]
        consumer[i, j] = 4 producers added

i8[n, 2] -> i8[n+1]
"""
