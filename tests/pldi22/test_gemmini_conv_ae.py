from __future__ import annotations

import pytest

from exo.platforms.gemmini import *


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
            for orow in seq(0, out_dim):
                for ocol in seq(0, out_dim):
                    for och in seq(0, out_channel):

                        res: i32
                        res = bias[0, och]
                        for krow in seq(0, kernel_dim):
                            for kcol in seq(0, kernel_dim):
                                for kch in seq(0, in_channel):
                                    w_s: i8 @ DRAM
                                    w_s = weights[krow, kcol, kch, och]

                                    i_s: i8 @ DRAM
                                    i_s = inp[b, orow + krow, ocol + kcol, kch]

                                    a2: i32
                                    b2: i32
                                    a2 = i_s
                                    b2 = w_s

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
@pytest.mark.slow
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

    # Split the outer dimension and replace the code with gemmini instructions
    gemmini = split_fission_dim(gemmini)
    gemmini = replace_div_part(gemmini)
    gemmini = replace_mod_part(gemmini)

    # Set buffers to use gemmini memories
    gemmini = set_memory(gemmini, "res", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "i_s", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "w_s", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "res #1", GEMM_ACCUM)
    gemmini = set_memory(gemmini, "i_s #1", GEMM_SCRATCH)
    gemmini = set_memory(gemmini, "w_s #1", GEMM_SCRATCH)

    # Inline and lift the configuration as high as possible for the "div" part
    gemmini = inline_vector(gemmini)
    gemmini = lift_config(gemmini, "config_ld_acc_i32_vector(_)")
    gemmini = inline_ld_id1(gemmini)
    gemmini = lift_config(gemmini, "config_ld_i8_id1(_)")
    gemmini = inline_matmul(gemmini)
    gemmini = lift_config(gemmini, "config_matmul(_)")
    gemmini = inline_st(gemmini)
    gemmini = lift_config(gemmini, "config_st_acc_i8(_)")

    # Inline and lift the configuration as high as possible for the "mod" part (tail case)
    gemmini = inline_vector(gemmini)
    gemmini = inline_ld_id1(gemmini)
    gemmini = inline_matmul(gemmini)
    gemmini = inline_st(gemmini)
    gemmini = delete_config(gemmini, "config_ld_acc_i32_vector(_) #1")
    gemmini = delete_config(gemmini, "config_ld_i8_id1(_) #1")
    gemmini = delete_config(gemmini, "config_matmul(_) #1")
    gemmini = delete_config(gemmini, "config_st_acc_i8(_) #1")
    gemmini = simplify(gemmini)

    # Real optimization
    gemmini = old_lift_alloc(gemmini, "w_s : _", n_lifts=2)
    gemmini = old_fission_after(gemmini, "for ocol_o in _:_ #0")
    gemmini = old_reorder(gemmini, "orow ocol_o")
    gemmini = old_split(gemmini, "orow", 28, ["orow_o", "orow_i"], perfect=True)
    gemmini = expand_dim(
        gemmini, "i_s: i8[_]", "30", "krow + orow_i", unsafe_disable_checks=True
    )
    # TODO: Use cursor + repeat to improve this!
    gemmini = old_lift_alloc(gemmini, "i_s : _ #0", n_lifts=5, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "i_s : _ #1", n_lifts=4, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "w_s : _ #0", n_lifts=4, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "w_s : _ #1", n_lifts=3, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "res : _ #0", n_lifts=4, keep_dims=False)
    gemmini = old_lift_alloc(gemmini, "res : _ #1", n_lifts=3, keep_dims=False)

    gemmini = old_fission_after(gemmini, "for kch_o in _:_ #0", n_lifts=6)
    gemmini = old_fission_after(gemmini, "for kch_o in _:_ #2", n_lifts=5)
    gemmini = old_fission_after(gemmini, "for och_o in _:_ #3")
    gemmini = add_loop(gemmini, "for kch_o in _:_ #0", "orow_i", 28, guard=True)
    gemmini = add_loop(gemmini, "if orow_i == 0:_", "orow_o", 2, guard=True)
    gemmini = add_loop(gemmini, "if orow_o == 0:_", "b", 4, guard=True)
    gemmini = add_loop(gemmini, "if b == 0:_", "ocol_o", 3, guard=True)
    gemmini = add_loop(gemmini, "for kch_o in _:_ #2", "orow_i", 28, guard=True)
    gemmini = add_loop(gemmini, "if orow_i == 0:_ #1", "orow_o", 2, guard=True)
    gemmini = add_loop(gemmini, "if orow_o == 0:_ #1", "b", 4, guard=True)
    # Start fissioning loops
    gemmini = add_loop(gemmini, "for och_o in _:_ #0", "b", 4)
    gemmini = old_reorder(gemmini, "orow_o b")
    gemmini = old_reorder(gemmini, "orow_i b")
    gemmini = old_reorder(gemmini, "kcol b")
    gemmini = old_reorder(gemmini, "krow b")
    gemmini = fuse(gemmini, "for b in _:_ #0", "for b in _:_ #1")
    gemmini = fuse(
        gemmini, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True
    )
    gemmini = fuse(gemmini, "for b in _:_ #0", "for b in _:_ #1")
    gemmini = fuse(
        gemmini, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True
    )
    gemmini = add_loop(gemmini, "for och_o in _:_ #0", "ocol_o", 3)
    gemmini = old_reorder(gemmini, "orow_o ocol_o")
    gemmini = old_reorder(gemmini, "orow_i ocol_o")
    gemmini = old_reorder(gemmini, "kcol ocol_o")
    gemmini = old_reorder(gemmini, "krow ocol_o")
    gemmini = fuse(gemmini, "for ocol_o in _:_ #0", "for ocol_o in _:_ #1")
    gemmini = fuse(
        gemmini,
        "for ocol_o in _:_ #0",
        "for ocol_o in _:_ #1",
        unsafe_disable_check=True,
    )
    gemmini = add_loop(gemmini, "for och_o in _:_ #0", "orow_o", 2)
    gemmini = old_reorder(gemmini, "orow_i orow_o")
    gemmini = old_reorder(gemmini, "kcol orow_o")
    gemmini = old_reorder(gemmini, "krow orow_o")
    gemmini = fuse(gemmini, "for orow_o in _:_ #0", "for orow_o in _:_ #1")
    gemmini = fuse(
        gemmini,
        "for orow_o in _:_ #0",
        "for orow_o in _:_ #1",
        unsafe_disable_check=True,
    )
    gemmini = add_loop(gemmini, "for och_o in _:_ #0", "orow_i", 28)
    gemmini = old_reorder(gemmini, "kcol orow_i")
    gemmini = old_reorder(gemmini, "krow orow_i")
    gemmini = fuse(gemmini, "for orow_i in _:_ #0", "for orow_i in _:_ #1")
    gemmini = fuse(
        gemmini,
        "for orow_i in _:_ #0",
        "for orow_i in _:_ #1",
        unsafe_disable_check=True,
    )

    gemmini = add_loop(gemmini, "for och_o in _:_ #3", "orow_o", 2)
    gemmini = fuse(gemmini, "for orow_o in _:_ #1", "for orow_o in _:_ #2")
    gemmini = fuse(
        gemmini,
        "for orow_o in _:_ #1",
        "for orow_o in _:_ #2",
        unsafe_disable_check=True,
    )
    gemmini = add_loop(gemmini, "for och_o in _:_ #3", "orow_i", 28)
    gemmini = fuse(gemmini, "for orow_i in _:_ #1", "for orow_i in _:_ #2")
    gemmini = fuse(
        gemmini,
        "for orow_i in _:_ #1",
        "for orow_i in _:_ #2",
        unsafe_disable_check=True,
    )

    gemmini = fuse(gemmini, "for krow in _:_ #0", "for krow in _:_ #1")
    gemmini = fuse(gemmini, "for kcol in _:_ #0", "for kcol in _:_ #1")
    gemmini = fuse(gemmini, "for krow in _:_ #1", "for krow in _:_ #2")
    gemmini = fuse(gemmini, "for kcol in _:_ #1", "for kcol in _:_ #2")

    gemmini = add_unsafe_guard(
        gemmini, "ld_i8_block_id2(_) #0", "orow_i == 0 or krow == 2"
    )
    gemmini = add_unsafe_guard(
        gemmini, "ld_i8_block_id2(_) #1", "orow_i == 0 or krow == 2"
    )

    gemmini = old_split(gemmini, "orow_i", 7, ["orow_io", "orow_ii"], perfect=True)
    gemmini = old_unroll(gemmini, "och_o")
    gemmini = old_unroll(gemmini, "kch_o")
    gemmini = old_unroll(gemmini, "kcol")
    gemmini = old_unroll(gemmini, "krow")
    gemmini = simplify(gemmini)

    # Schedule ends here, 44 lines excluding comments and newlines

    print("")
    print("============= THIS IS THE SCHEDULED CONV ===============")
    print(gemmini)
    print("============= THIS IS THE SCHEDULED CONV ===============")
    print("")

    assert gemmini.c_code_str() == golden
