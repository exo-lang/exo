from __future__ import annotations

from exo.platforms.gemmini import (
    ld_acc_i32_vector,
    ld_acc_i32_vector_v2,
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


def conv_algorithm():
    @proc
    def conv(
        batch_size: size,
        out_dim: size,
        out_channel: size,
        kernel_dim: size,
        in_channel: size,
        in_dim: size,
        output: i8[batch_size, out_dim, out_dim, out_channel],
        bias: i32[1, out_channel],
        inp: i8[batch_size, in_dim, in_dim, in_channel],
        weights: i8[kernel_dim, kernel_dim, in_channel, out_channel],
        act: bool,
        scale: f32,
    ):

        assert out_dim == in_dim - kernel_dim + 1

        # Algorithm starts here
        for b in seq(0, batch_size):
            for ocol in seq(0, out_dim):
                for orow in seq(0, out_dim):
                    for och in seq(0, out_channel):
                        res: i32
                        res = bias[0, och]
                        for krow in seq(0, kernel_dim):
                            for kcol in seq(0, kernel_dim):
                                for kch in seq(0, in_channel):
                                    a2: i32
                                    b2: i32
                                    a2 = inp[b, orow + krow, ocol + kcol, kch]
                                    b2 = weights[krow, kcol, kch, och]
                                    res += a2 * b2

                        src_tmp: i32
                        src_tmp = res
                        tmp_res1: f32
                        acc_scale(src_tmp, tmp_res1, scale)
                        tmp_res2: i8
                        clamp(tmp_res1, tmp_res2)
                        if act == True:
                            tmp_res2 = relu(tmp_res2)

                        output[b, orow, ocol, och] = tmp_res2
        # Algorithm ends here. 26 lines excluding newlines

    return conv


# Conv test for the artifact evaluation. The same algorithm and schedule
# was used for Table 2 (first row) and Table 3 (code size)
def test_conv_ae(golden):
    batch_size = 4
    out_channel = 64
    kernel_dim = 3
    in_channel = 64
    in_dim = 58
    out_dim = int((in_dim - kernel_dim) / 1 + 1)
    assert out_dim == 56

    # Rename the conv algorithm to "conv_on_gemmini"
    gemmini = rename(conv_algorithm(), "conv_on_gemmini")

    print("")
    print("===== THIS IS THE CONV ALGORITHM BEFORE SCHEDULING ====")
    print(gemmini)
    print("===== THIS IS THE CONV ALGORITHM BEFORE SCHEDULING ====")
    print("")

    # Schedule starts here. Below schedule partially evaluates the proc with conv parameters
    gemmini = gemmini.partial_eval(
        batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim
    )

    # Parameters
    accum_size = 4 * 16 * 16
    sc_size = 16 * 16 * 4 * 4 * 3 * 3

    def sched_one_block(
        gemmini,
        res_alloc,
        res_load,
        krow_loop,
        kcol_loop,
        b_assign,
        a_assign,
        asize=accum_size,
        isize=0,
    ):
        # Bind and lift scrathpad & accumulator memories
        gemmini = autolift_alloc(gemmini, res_alloc, max_size=accum_size)
        gemmini = fission_as_much_as_possible(gemmini, res_load)
        gemmini = fission_as_much_as_possible(gemmini, krow_loop)

        gemmini = lift_scope(gemmini, krow_loop)
        gemmini = lift_scope(gemmini, krow_loop)
        gemmini = lift_scope(gemmini, krow_loop)

        gemmini = lift_scope(gemmini, kcol_loop)
        gemmini = lift_scope(gemmini, kcol_loop)
        gemmini = lift_scope(gemmini, kcol_loop)

        gemmini, b_load, b_alloc = bind_and_lift(
            gemmini, gemmini.forward(b_assign).rhs(), max_size=sc_size
        )
        gemmini, a_load, a_alloc = bind_and_lift(
            gemmini, gemmini.forward(a_assign).rhs(), max_size=isize, lift=False
        )

        gemmini = expand_dim(gemmini, a_alloc, "30", "krow + orowi")
        gemmini = repeat(gemmini, lift_alloc, a_alloc)
        gemmini = reorder_top(gemmini, gemmini.forward(res_alloc))
        gemmini = reorder_top(gemmini, gemmini.forward(b_alloc))
        gemmini = reorder_top(gemmini, gemmini.forward(a_alloc))
        gemmini = fission_as_much_as_possible(gemmini, b_load)

        # Fission all the loops
        gemmini = fission_as_much_as_possible(gemmini, a_load)

        # Fix indexing
        gemmini = rearrange_dim(gemmini, a_alloc, [0, 1, 3, 2, 4])
        gemmini = rearrange_dim(gemmini, b_alloc, [0, 1, 4, 2, 5, 3])

        gemmini = reorder_loops_from_idx(gemmini, a_load)
        gemmini = reorder_loops_from_idx(gemmini, b_load)
        gemmini = reorder_loops_from_idx(gemmini, gemmini.forward(a_assign).rhs())
        gemmini = remove_redundant_loops(gemmini, a_load, num=2)
        gemmini = remove_redundant_loops(gemmini, b_load, num=1)

        # Replace to gemmini calls, inline to v2, and hoist all the configurations
        gemmini = set_memory(gemmini, res_alloc, GEMM_ACCUM)
        gemmini = set_memory(gemmini, a_alloc, GEMM_SCRATCH)
        gemmini = set_memory(gemmini, b_alloc, GEMM_SCRATCH)
        gemmini = reorder_loops(gemmini, "ocho kchi")
        gemmini = reorder_loops(gemmini, "kcho ocoli")

        tuples = [
            (ld_acc_i32_vector, ld_acc_i32_vector_v2),
            (ld_i8_block_id1, ld_i8_block_id1_v2),
            (ld_i8_block_id2, ld_i8_block_id2_v2),
            (matmul_acc_i8, matmul_acc_i8_v2),
            (st_acc_i8, st_acc_i8_v2),
        ]
        for t in tuples:
            gemmini = simplify(replace_and_inline(gemmini, t))

        return gemmini

    # Grab cursors
    ocol_loop = gemmini.find_loop("ocol")
    och_loop = gemmini.find_loop("och")
    orow_loop = gemmini.find_loop("orow")
    kch_loop = gemmini.find_loop("kch")
    kcol_loop = gemmini.find_loop("kcol")
    krow_loop = gemmini.find_loop("krow")
    res_load = gemmini.find("res = bias[_]")
    a_assign = gemmini.find("a2 = inp[_]")
    b_assign = gemmini.find("b2 = weights[_]")
    res_alloc = res_load.prev()

    print(gemmini)

    # Tile loops for a scratchpad and an accumulator
    gemmini, cursors = tile_loops_top_down(gemmini, [(orow_loop, 28), (och_loop, 16)])
    gemmini, _ = tile_loops_top_down(gemmini, [(kch_loop, 16)])
    gemmini, kch_inner = tile_loops_top_down(gemmini, [(ocol_loop, 16)])

    # Loop reorganization, would be nice to clean this up...
    gemmini = lift_scope(gemmini, orow_loop)
    gemmini = lift_scope(gemmini, cursors[0])
    gemmini = lift_scope(gemmini, cursors[0])
    tail = gemmini.forward(ocol_loop).next()
    tail_orowo = tail.body()[0]
    tail_orowi = tail_orowo.body()[0].body()[0]
    gemmini = lift_scope(gemmini, tail_orowo)
    gemmini = lift_scope(gemmini, tail_orowi)
    gemmini = lift_scope(gemmini, tail_orowi)
    gemmini = lift_scope(gemmini, gemmini.find_loop("ocho"))
    gemmini = lift_scope(gemmini, gemmini.find_loop("ocho #1"))

    # Get cursors for the tail loop
    tail_res_load = gemmini.find("res = bias[_] #1")
    tail_res_alloc = tail_res_load.prev()
    tail_krow_loop = tail_res_load.next()
    tail_kcol_loop = tail_krow_loop.body()[0]
    tail_a_assign = gemmini.find("a2 = inp[_] #1")
    tail_b_assign = gemmini.find("b2 = weights[_] #1")

    # Schedule the main loop
    gemmini = sched_one_block(
        gemmini,
        res_alloc,
        res_load,
        krow_loop,
        kcol_loop,
        b_assign,
        a_assign,
        isize=3 * 4 * 16 * 16,
    )
    # Schedule the tail loop
    gemmini = sched_one_block(
        gemmini,
        tail_res_alloc,
        tail_res_load,
        tail_krow_loop,
        tail_kcol_loop,
        tail_b_assign,
        tail_a_assign,
        asize=4 * 9 * 16,
        isize=3 * 4 * 8 * 16,
    )

    # Add a guard to redundant loads
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_)"))
    gemmini = add_unsafe_guard(
        gemmini, "do_ld_i8_block_id2(_) #0", "orowi == 0 or krow == 2"
    )
    gemmini = add_guard(gemmini, gemmini.find("do_ld_i8_block_id1(_) #1"))
    gemmini = add_unsafe_guard(
        gemmini, "do_ld_i8_block_id2(_) #1", "orowi == 0 or krow == 2"
    )

    gemmini = fuse_all_loops(gemmini, gemmini.body()[0])

    # Delete  redundant config from the tail loop
    gemmini = delete_config(gemmini, "config_ld_acc_i32_vector(_) #1")
    gemmini = delete_config(gemmini, "config_ld_i8_id1(_) #1")
    gemmini = delete_config(gemmini, "config_matmul(_) #1")
    gemmini = delete_config(gemmini, "config_st_acc_i8(_) #1")

    print(gemmini)
    # quit()

    print("")
    print("============= THIS IS THE SCHEDULED CONV ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED CONV ===============")
    print("")

    assert gemmini.c_code_str() == golden
