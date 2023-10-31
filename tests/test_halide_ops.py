from __future__ import annotations

import os
import sys

from exo import proc
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *


@proc
def blur1d_compute_root(n: size, consumer: i8[n], inp: i8[n + 6]):
    producer: i8[n + 1]
    for i in seq(0, n + 1):
        producer[i] = (
            inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] + inp[i + 5]
        ) / 6.0

    for i in seq(0, n):
        consumer[i] = (producer[i] + producer[i + 1]) / 2.0


def test_schedule_blur1d(golden):
    p = blur1d_compute_root
    procs = []

    loop = p.find_loop("i #1")
    p = fuse_at(p, "producer", "consumer", loop)
    p = rename(p, "blur1d_compute_at_store_root")
    procs.append(p)

    loop = p.find_loop("i")
    p_bounds = (0, "i", 0, 2)
    p = store_at(p, "producer", "consumer", loop)
    p = rename(p, "blur1d_compute_at")
    procs.append(p)

    p = unroll_loop(p, "ii")
    for i in range(2):
        p = inline_assign(p, p.find("consumer[_] = _").prev())
    p = delete_buffer(p, "producer: _")
    p = simplify(p)
    p = rename(p, "blur1d_inline")
    procs.append(p)

    print("\n\n".join([str(p) for p in procs]))
    assert "\n\n".join([str(p) for p in procs]) == golden


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


def test_schedule_blur2d(golden):
    p = blur2d_compute_root
    procs = []

    c_i_bounds = (0, "i", 0, 1)
    p_i_bounds = (0, "i", 0, 2)
    c_j_bounds = (1, "j", 0, 1)
    p_j_bounds = (1, "j", 0, 2)

    p = fuse_at(p, "producer", "consumer", p.find_loop("i #1"))
    p = rename(p, "blur2d_compute_at_i_store_root")
    procs.append(p)
    p_tmp = p  # For testing different branches of scheduling

    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"))
    p = rename(p, "blur2d_compute_at_j_store_root")
    procs.append(p)

    p = store_at(p_tmp, "producer", "consumer", p_tmp.find_loop("i"))
    p = rename(p, "blur2d_compute_at_i")
    procs.append(p)

    p = fuse_at(p, "producer", "consumer", p.find_loop("j #1"))
    p = simplify(p)
    p = rename(p, "blur2d_compute_at_j_store_at_i")
    procs.append(p)

    p = store_at(p, "producer", "consumer", p.find_loop("j"))
    p = unroll_loop(p, "ji")
    p = unroll_loop(p, "ii")
    for i in range(4):
        p = inline_assign(p, p.find("consumer[_] = _").prev())
    p = delete_buffer(p, "producer: _")
    p = rename(p, "blur2d_inline")
    procs.append(p)

    print("\n\n".join([str(p) for p in procs]))
    assert "\n\n".join([str(p) for p in procs]) == golden


def test_schedule_tiled_blur2d(golden):
    compute_root = blur2d_compute_root
    procs = []

    p = tile(
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
    p = rename(p, "blur2d_tiled")
    procs.append(p)

    p = fuse_at(p, "producer", "consumer", p.find_loop("io"))
    # TODO: maybe rewrite_expr of predicates should be in simplify
    p = simplify(rewrite_expr(p, "n%4", 0))
    p = rename(p, "blur2d_tiled_compute_at_io")
    procs.append(p)

    p = fuse_at(p, "producer", "consumer", p.find_loop("jo"))
    p = simplify(rewrite_expr(p, "n%4", 0))
    p = rename(p, "blur2d_tiled_compute_at_jo")
    procs.append(p)

    p = fuse_at(p, "producer", "consumer", p.find_loop("ii #1"))
    p = rename(p, "blur2d_tiled_compute_at_ii")
    procs.append(p)

    p = fuse_at(p, "producer", "consumer", p.find_loop("ji #1"))
    p = rename(p, "blur2d_tiled_compute_at_ji")
    procs.append(p)

    print("\n\n".join([str(p) for p in procs]))
    assert "\n\n".join([str(p) for p in procs]) == golden
