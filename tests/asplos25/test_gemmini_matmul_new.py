from __future__ import annotations

from exo.platforms.gemmini import (
    zero_acc_i32,
    zero_acc_i32_v2,
    ld_i8_block_id1,
    ld_i8_block_id1_v2,
    ld_i8_block_id2,
    ld_i8_block_id2_v2,
    matmul_acc_i8,
    matmul_acc_i8_v2,
    st_acc_i8,
    st_acc_i8_v2,
    acc_scale,
    clamp,
)
import exo.API_cursors as pc
from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
from exo import proc, instr, DRAM, config, ExoType
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *
from exo.stdlib.inspection import *
from gemmini_schedules import *

ld_i8_block_id1 = reorder_loops(ld_i8_block_id1, "i j")
ld_i8_block_id2 = reorder_loops(ld_i8_block_id2, "i j")


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
                    a2: i32
                    b2: i32
                    a2 = A[i, k]
                    b2 = B[k, j]
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


def test_matmul(golden):
    KK = 512

    cpu = rename(
        matmul_algorithm(), "matmul_on_cpu"
    )  # Rename "matmul" to "matmul_on_cpu"
    cpu = cpu.partial_eval(K=KK)
    cpu = cpu.add_assertion("N % 256 == 0")
    cpu = cpu.add_assertion("M % 256 == 0")

    # Rename the procedure to "matmul_on_gemmini"
    gemmini = rename(cpu, "matmul_on_gemmini")

    print("")
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE ORIGINAL MATMUL ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Parameters
    accum_size = 16 * 1024
    sc_size = 256 * 1024

    # Grab cursors
    i_loop = gemmini.find_loop("i")
    j_loop = gemmini.find_loop("j")
    k_loop = gemmini.find_loop("k")
    res_load = gemmini.find("res = 0.0")
    a_assign = gemmini.find("a2 = A[_]")
    b_assign = gemmini.find("b2 = B[_]")
    res_alloc = res_load.prev()

    # Schedule starts here!!

    # Tile loops for a scratchpad and an accumulator
    gemmini, _ = tile_loops(gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True)
    gemmini, [_, j_outer] = tile_loops(
        gemmini, [(i_loop, 16), (j_loop, 16)], perfect=True
    )
    gemmini, _ = tile_loops(gemmini, [(k_loop, 16)])

    # Bind and lift scratchpad & accumulator memories
    gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
    gemmini, a_load, a_alloc = bind_and_lift(
        gemmini, gemmini.forward(a_assign).rhs(), max_size=sc_size
    )
    gemmini, b_load, b_alloc = bind_and_lift(
        gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
    )

    # Divide by 4 to use load_blocks
    gemmini, _ = tile_loops(gemmini, [(k_loop, 4)])
    gemmini, [j_imost] = tile_loops(gemmini, [(j_outer, 4)], perfect=True)

    # Fission all the loops
    gemmini = fission_as_much_as_possible(gemmini, res_load)
    gemmini = fission_as_much_as_possible(gemmini, k_loop)
    gemmini = fission_as_much_as_possible(gemmini, a_load)
    gemmini = fission_as_much_as_possible(gemmini, b_load)

    # Fix indexing
    gemmini = rearrange_dim(gemmini, a_alloc, [0, 2, 1, 3])
    gemmini = rearrange_dim(gemmini, b_alloc, [2, 0, 3, 1])
    gemmini = reorder_loops_from_idx(gemmini, a_load)
    gemmini = reorder_loops_from_idx(gemmini, b_load)
    gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
    gemmini = remove_redundant_loops(gemmini, a_load, num=2)
    gemmini = remove_redundant_loops(gemmini, b_load, num=1)

    # Replace to gemmini calls, inline to v2, and hoist all the configurations
    gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
    gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
    gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
    tuples = [
        (zero_acc_i32, zero_acc_i32_v2),
        (ld_i8_block_id1, ld_i8_block_id1_v2),
        (ld_i8_block_id2, ld_i8_block_id2_v2),
        (matmul_acc_i8, matmul_acc_i8_v2),
        (st_acc_i8, st_acc_i8_v2),
    ]
    for t in tuples:
        gemmini = simplify(replace_and_inline(gemmini, t))

    # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id2(_)"))

    # fuse loops, and unroll
    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])
    gemmini = unroll_all(gemmini, gemmini.find_loop(j_imost.name(), many=True))

    # Schedule ends here!!
    # 29 lines excluding comments and newlines

    gemmini = simplify(gemmini)

    print("")
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED MATMUL ===============")
    print("")

    assert gemmini.c_code_str() == golden
