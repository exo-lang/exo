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


# Best for 512x512x512
def test_matmul(golden):
    KK = 512

    gemmini = matmul_on_cpu.partial_eval(K=KK)
    gemmini = gemmini.add_assertion("N % 256 == 0")
    gemmini = gemmini.add_assertion("M % 256 == 0")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)

    # Schedule starts here!!

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
    gemmini = add_loop(gemmini, "if ji == 0: _", "jo", "M / 256", guard=True)
    gemmini = add_loop(gemmini, "do_ld_i8_block_id2(_)", "i", 8, guard=True)
    gemmini = add_loop(gemmini, "if i == 0: _", "io", 2, guard=True)
    # Fuse_loop cleanup
    gemmini = add_loop(gemmini, "for jo in _:_ #1", "ioo", "N / 256")
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "ioo", "N / 256")
    gemmini = fuse(gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1")
    gemmini = fuse(gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1")
    gemmini = fuse(
        gemmini, "for ioo in _:_ #0", "for ioo in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for ji in _:_ #0", "jo", "M / 256")
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

    # Schedule ends here!!
    # 61 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")

    assert gemmini.c_code_str() == golden
