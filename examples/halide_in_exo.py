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


def schedule_compute_at(algorithm, name):
    blur = rename(algorithm, name)

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


def schedule_store_at(algorithm, producer, consumer, loop, algorithm_name):
    blur = rename(algorithm, algorithm_name)

    producer_alloc = blur.find(f"{producer}:_")
    consumer_assign = blur.find(f"{consumer} = _")
    loop = blur.forward(loop)  # need to forward because rename has a fwding function

    name = producer_alloc.name()
    bound = 2  # TODO: need bounds inference to figure this out

    before_consumer = consumer_assign.prev().as_block().expand(delta_hi=0)
    blur = stage_mem(blur, before_consumer, f"{name}[i:i+{bound}]", f"{name}_tmp")
    blur = simplify(blur)

    blur = sink_alloc(blur, producer_alloc)

    blur = unroll_loop(blur, blur.forward(consumer_assign).prev())
    blur = simplify(blur)

    for i in range(bound):
        block = blur.forward(consumer_assign).expand(delta_lo=1, delta_hi=0)
        blur = inline_assign(blur, block)

    blur = delete_buffer(blur, producer_alloc)

    return simplify(blur)


print("Original blur:\n", blur1d_compute_root)
blur1d_compute_at_store_root = schedule_compute_at(
    blur1d_compute_root, "blur_compute_at_store_root"
)
print("Compute at, store root blur:\n", blur1d_compute_at_store_root)

loop = blur1d_compute_at_store_root.find_loop("i")
blur1d_compute_at = schedule_store_at(
    blur1d_compute_at_store_root, "producer", "consumer", loop, "blur_compute_at"
)
print("Compute at blur:\n", blur1d_compute_at)
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
print("Inline:\n", blur1d_compute_at)
