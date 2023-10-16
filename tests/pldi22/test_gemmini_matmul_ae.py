from __future__ import annotations

from exo.platforms.gemmini import *


def matmul_algorithm():
    @proc
    def matmul(
        N: size,
        M: size,
        K: size,
        scale: f32,
        act: bool,
        A: i8[N, K] @ DRAM,
        B: i8[K, M] @ DRAM,
        C: i8[N, M] @ DRAM,
    ):

        # Algorithm starts here
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
        # Algorithm ends here. 23 lines excluding newlines

    return matmul


# Matmul test for artifact evaluation. The same algorithm and schedule
# was used for Table 2 (512x521x512) and Table 3 (code size)
def test_matmul_ae(golden):
    NN = 512
    MM = 512
    KK = 512

    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(NN, MM, KK)

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Schedule starts here. Below sets buffer to use GEMMINI memories.
    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "a", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "b", GEMM_SCRATCH)

    # Tile outer loops
    gemmini = tile_outer_loops(gemmini)

    # Lift res, so that we can fission the inner loop to use gemmini instructions
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=1, mode="col", size=16)

    # Fission loops to zero accum code block, main block, and store block and reorder k up
    gemmini = fission_outer_blocks(gemmini)

    # Fission the main block to 4x16x16 blocks, so that we can use gemmini instr
    gemmini = fission_inner_blocks(gemmini)

    # Replace to gemmini calls
    gemmini = replace_gemmini_calls(gemmini)

    # Inline and lift the configuration as high as possible
    # Lift config_zero
    gemmini = call_eqv(gemmini, "zero_acc_i32(_, _, _)", zero_acc_i32_v2)
    gemmini = inline(gemmini, "zero_acc_i32_v2(_, _, _)")
    gemmini = inline_window(gemmini, "dst = res[_]")
    gemmini = lift_config(gemmini, "config_zero()")
    # Lift config_ld_i8_id1
    gemmini = call_eqv(gemmini, "ld_i8_block_id1(_)", ld_i8_block_id1_v2)
    gemmini = inline(gemmini, "ld_i8_block_id1_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = A[_]")
    gemmini = inline_window(gemmini, "dst = a[_]")
    gemmini = lift_config(gemmini, "config_ld_i8_id1()")
    # Lift config_ld_i8_id2
    gemmini = call_eqv(gemmini, "ld_i8_block_id2(_)", ld_i8_block_id2_v2)
    gemmini = inline(gemmini, "ld_i8_block_id2_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = B[_]")
    gemmini = inline_window(gemmini, "dst = b[_]")
    gemmini = lift_config(gemmini, "config_ld_i8_id2()")
    # Lift config_matmul
    gemmini = call_eqv(gemmini, "matmul_acc_i8(_, _, _, _, _)", matmul_acc_i8_v2)
    gemmini = inline(gemmini, "matmul_acc_i8_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "A = a[_]")
    gemmini = inline_window(gemmini, "B = b[_]")
    gemmini = inline_window(gemmini, "C = res[_]")
    gemmini = lift_config(gemmini, "config_matmul()")
    # Lift config_st_acc_i8
    gemmini = call_eqv(gemmini, "st_acc_i8(_, _, _, _, _, _)", st_acc_i8_v2)
    gemmini = inline(gemmini, "st_acc_i8_v2(_, _, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = res[_]")
    gemmini = inline_window(gemmini, "dst = C[_]")
    gemmini = lift_config(gemmini, "config_st_acc_i8(_)")

    # Futher tile the inner loops
    gemmini = matmul_tile(gemmini)

    # Lift the allocations
    gemmini = old_lift_alloc(gemmini, "res : _", n_lifts=1)
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=4)
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=3)

    for (s, n) in [("a : i8", 1), ("b : i8", 2), ("res : _", 4)]:
        gemmini = old_lift_alloc(gemmini, s, n_lifts=n, keep_dims=False)

    # These schedules correspond to the previous add_guard
    gemmini = simplify(gemmini)
    gemmini = autofission(gemmini, gemmini.find("for j_in_o in _:_").after(), n_lifts=5)
    gemmini = autofission(gemmini, gemmini.find("for k in _:_").after(), n_lifts=6)
    gemmini = autofission(
        gemmini, gemmini.find("do_ld_i8_block_id1(_)").after(), n_lifts=6
    )
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
    gemmini = simplify(gemmini)

    # Schedule ends here. 43 lines excluding comments and newlines

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")

    assert gemmini.c_code_str() == golden
