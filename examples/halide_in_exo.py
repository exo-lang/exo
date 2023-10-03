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


print("Original blur:\n", blur1d_compute_root)
blur1d_compute_at_store_root = compute_at(
    blur1d_compute_root, "blur_compute_at_store_root"
)
print("Compute at, store root blur:\n", blur1d_compute_at_store_root)

loop = blur1d_compute_at_store_root.find_loop("i")
blur1d_compute_at = store_at(
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
