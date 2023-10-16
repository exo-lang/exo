from __future__ import annotations

from exo.platforms.gemmini import *
from exo.stdlib.scheduling import *


@proc
def matmul_on_cpu(
    N: size,
    M: size,
    K: size,
    scale: f32,
    act: bool,
    A: i8[N, K] @ DRAM,
    B: i8[K, M] @ DRAM,
    C: i8[N, M] @ DRAM,
):
    for i in seq(0, N):
        for j in seq(0, M):
            res: i32 @ DRAM
            res = 0.0
            for k in seq(0, K):
                a: i8 @ DRAM
                a = A[i, k]

                b: i8 @ DRAM
                b = B[k, j]

                a2: i32
                b2: i32
                a2 = a
                b2 = b
                res += a2 * b2

            src_tmp: i32
            src_tmp = res
            tmp_res1: f32
            acc_scale(src_tmp, tmp_res1, scale)
            tmp_res2: i8
            clamp(tmp_res1, tmp_res2)
            if act == True:
                tmp_res2 = relu(tmp_res2)
            C[i, j] = tmp_res2


def sched_matmul(
    name,
    NN,
    MM,
    KK,
):
    cpu = rename(matmul_on_cpu, f"cpu_{name}")
    cpu = cpu.partial_eval(NN, MM, KK)

    gemmini = rename(cpu, name)
    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)

    # Tile outer loops
    gemmini = tile_outer_loops(gemmini)

    # Lift res, so that we can fission the inner loop to use gemmini instructions
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=1, mode="col", size=16)

    # fission loops to zero accum code block, main block, and store block and reorder k up
    gemmini = fission_outer_blocks(gemmini)

    # fission the main block to 4x16x16 blocks, so that we can use gemmini instr
    gemmini = fission_inner_blocks(gemmini)

    # replace to gemmini calls
    gemmini = replace_gemmini_calls(gemmini)

    # inline and lift config
    gemmini = inline_lift_config(gemmini)

    return cpu, gemmini


# Best for 512x512x512
def schedule_matmul_512x512x512():
    NN = 512
    MM = 512
    KK = 512

    cpu, gemmini = sched_matmul("matmul_512x512x512", NN, MM, KK)

    # Real optimization
    # tile
    gemmini = matmul_tile(gemmini)

    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=4)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=3)

    for (s, n) in [("a : i8", 1), ("b : i8", 2), ("res : _", 4)]:
        gemmini = old_lift_alloc(gemmini, s, n_lifts=n, keep_dims=False)

    gemmini = simplify(gemmini)

    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 5)
    do_fission("do_ld_i8_block_id1(_)", 6)
    do_fission("for k in _:_", 6)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "ji", 4, guard=True)
    gemmini = add_loop(gemmini, "if ji == 0: _", "jo", 2, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 8, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 2, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for jo in _:_ #1", "ioo", 2)
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "ioo", 2)
    gemmini = fuse(gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1")
    gemmini = fuse(gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1")
    gemmini = fuse(
        gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "jo", 2)
    gemmini = old_reorder(gemmini, "ji jo")
    gemmini = old_reorder(gemmini, "ko jo")
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = old_reorder(gemmini, "io jo")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(
        gemmini, "for jo in _:_ #0", "for jo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = old_reorder(gemmini, "ji io")
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "io", 2)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "ji i")
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "i", 8)
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko ji")
    gemmini = fuse(gemmini, "for ji in _:_ #0", "for ji in _:_ #1")
    gemmini = fuse(gemmini, "for ji in _:_ #0", "for ji in _:_ #1")
    gemmini = fuse(gemmini, "for ji in _:_ #0", "for ji in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = fuse(gemmini, "for k in _:_ #0", "for k in _:_ #1")
    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")
    gemmini = simplify(gemmini)

    return cpu, gemmini


def schedule_matmul_4():
    NN = 12544
    MM = 256
    KK = 64

    cpu, gemmini = sched_matmul("matmul_4", NN, MM, KK)

    # Real optimization
    gemmini = old_unroll(gemmini, "ko")
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = simplify(gemmini)

    gemmini = divide_loop(gemmini, "i", 196, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)

    # tile
    gemmini = old_lift_alloc(gemmini, "a : i8", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=1, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=2, keep_dims=False)

    # Previously add_guard
    gemmini = simplify(gemmini)

    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 5)
    do_fission("do_ld_i8_block_id1(_)", 6)
    do_fission("for k in _:_", 6)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 4, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 196, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 4, guard=True)
    # Fuse_loop cleanup
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "j io")
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 4)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j_in_o in _:_ #0", "i", 196)
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_6():
    NN = 12544
    MM = 64
    KK = 256

    cpu, gemmini = sched_matmul("matmul_6", NN, MM, KK)

    # Real optimization
    gemmini = old_unroll(gemmini, "j")
    gemmini = divide_loop(gemmini, "i", 8, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _")
    gemmini = old_lift_alloc(gemmini, "b : _")
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 2)
    do_fission("do_ld_i8_block_id1(_)", 3)
    do_fission("for k in _:_", 3)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 8, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 98, guard=True)
    # Fuse_loop cleanup
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = add_loop(gemmini, "for i in _:_ #0", "io", 98)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_14():
    NN = 3136
    MM = 512
    KK = 128

    cpu, gemmini = sched_matmul("matmul_14", NN, MM, KK)

    # Real optimization
    gemmini = divide_loop(gemmini, "i", 49, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", n_lifts=2, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 3)
    do_fission("do_ld_i8_block_id1(_)", 4)
    do_fission("for k in _:_", 4)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 8, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 49, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 4, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 4)
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = old_reorder(gemmini, "j io")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j in _:_ #0", "i", 49)
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko j")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_16():
    NN = 3136
    MM = 128
    KK = 512

    cpu, gemmini = sched_matmul("matmul_16", NN, MM, KK)

    # Real optimization
    gemmini = divide_loop(gemmini, "i", 14, ["io", "i"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", n_lifts=2, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 3)
    do_fission("do_ld_i8_block_id1(_)", 4)
    do_fission("for k in _:_", 4)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 2, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 14, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 14, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 14)
    gemmini = old_reorder(gemmini, "i io")
    gemmini = old_reorder(gemmini, "k io")
    gemmini = old_reorder(gemmini, "ko io")
    gemmini = old_reorder(gemmini, "j io")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j in _:_ #0", "i", 14)
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko j")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


def schedule_matmul_27():
    NN = 784
    MM = 1024
    KK = 256

    cpu, gemmini = sched_matmul("matmul_27", NN, MM, KK)

    # Real optimization
    gemmini = divide_loop(gemmini, "i", 7, ["io", "i"], perfect=True)
    gemmini = divide_loop(gemmini, "j", 8, ["jo", "j"], perfect=True)
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = old_lift_alloc(gemmini, "res:_")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2)
    gemmini = simplify(gemmini)

    # tile
    gemmini = old_lift_alloc(gemmini, "res:_", n_lifts=3, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=2, keep_dims=False)

    gemmini = simplify(gemmini)

    # Previously add_guard
    def do_fission(pattern, n):
        nonlocal gemmini
        gemmini = autofission(gemmini, gemmini.find(pattern).after(), n_lifts=n)

    do_fission("for j_in_o in _:_", 4)
    do_fission("do_ld_i8_block_id1(_)", 5)
    do_fission("for k in _:_", 5)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id1(_)", "j", 8, guard=True)
    gemmini = add_loop(gemmini, "if j == 0: _", "jo", 2, guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 7, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for jo in _:_ #1", "io", 7)
    gemmini = add_loop(gemmini, "for j in _:_ #0", "io", 7)
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(gemmini, "for io in _:_ #0", "for io in _:_ #1")
    gemmini = fuse(
        gemmini, "for io in _:_ #0", "for io in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for j in _:_ #0", "jo", 2)
    gemmini = old_reorder(gemmini, "j jo")
    gemmini = old_reorder(gemmini, "ko jo")
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(gemmini, "for jo in _:_ #0", "for jo in _:_ #1")
    gemmini = fuse(
        gemmini, "for jo in _:_ #0", "for jo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "k i")
    gemmini = old_reorder(gemmini, "ko i")
    gemmini = old_reorder(gemmini, "j i")
    gemmini = add_loop(gemmini, "for j in _:_ #0", "i", 7)
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(gemmini, "for i in _:_ #0", "for i in _:_ #1")
    gemmini = fuse(
        gemmini, "for i in _:_ #0", "for i in _:_ #1", unsafe_disable_check=True
    )
    gemmini = old_reorder(gemmini, "ko j")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for j in _:_ #0", "for j in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")
    gemmini = fuse(gemmini, "for ko in _:_ #0", "for ko in _:_ #1")

    gemmini = old_unroll(gemmini, "j_in_o")
    gemmini = old_unroll(gemmini, "k")

    return cpu, gemmini


cpu_matmul_512x512x512, matmul_512x512x512 = schedule_matmul_512x512x512()
cpu_matmul_4, matmul_4 = schedule_matmul_4()
cpu_matmul_6, matmul_6 = schedule_matmul_6()
cpu_matmul_14, matmul_14 = schedule_matmul_14()
cpu_matmul_16, matmul_16 = schedule_matmul_16()
cpu_matmul_27, matmul_27 = schedule_matmul_27()

__all__ = [
    "cpu_matmul_512x512x512",
    "matmul_512x512x512",
    "cpu_matmul_4",
    "matmul_4",
    "cpu_matmul_6",
    "matmul_6",
    "cpu_matmul_14",
    "matmul_14",
    "cpu_matmul_16",
    "matmul_16",
    "cpu_matmul_27",
    "matmul_27",
]
