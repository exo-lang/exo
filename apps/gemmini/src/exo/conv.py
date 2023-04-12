from __future__ import annotations

from exo.platforms.gemmini import *


@proc
def conv_on_cpu_stride_1(
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


def split_fission_dim(conv):
    conv = old_split(conv, "ocol", 16, ["ocol_o", "ocol_i"], tail="cut_and_guard")
    conv = old_split(conv, "och", 16, ["och_o", "och_i"], perfect=True)
    conv = old_split(conv, "kch", 16, ["kch_o", "kch_i"], perfect=True)
    conv = old_reorder(conv, "ocol_i och_o")
    conv = old_lift_alloc(conv, "res : _", n_lifts=3)
    conv = old_fission_after(conv, "res[_] = _", n_lifts=3)
    conv = old_fission_after(conv, "for krow in _:_", n_lifts=3)
    conv = old_reorder(conv, "och_i krow")
    conv = old_reorder(conv, "och_i kcol")
    conv = old_reorder(conv, "och_i kch_o")
    conv = old_reorder(conv, "ocol_i krow")
    conv = old_reorder(conv, "ocol_i kcol")
    conv = old_reorder(conv, "ocol_i kch_o")
    conv = old_reorder(conv, "och_o krow")
    conv = simplify(conv)
    conv = old_lift_alloc(conv, "i_s : _", n_lifts=6)
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=1)
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=1, mode="col")
    conv = old_reorder(conv, "och_o kcol")
    conv = old_reorder(conv, "och_o kch_o")
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=3)
    conv = old_fission_after(conv, "w_s = _", n_lifts=5)
    conv = old_fission_after(conv, "i_s = _", n_lifts=5)

    return conv


def replace_div_part(conv):
    conv = replace(conv, "for ocol_i in _:_ #0", ld_acc_i32_vector)
    conv = old_reorder(conv, "och_i kch_i")
    conv = old_reorder(conv, "och_o kch_i")
    conv = replace(conv, "for kch_i in _:_ #0", ld_i8_block_id1)
    conv = old_reorder(conv, "kch_o ocol_i")
    conv = replace(conv, "for ocol_i in _:_ #0", ld_i8_block_id2)
    conv = old_reorder(conv, "kch_i och_i")
    conv = replace(conv, "for ocol_i in _:_ #0", matmul_acc_i8)
    conv = replace(conv, "for ocol_i in _:_ #0", st_acc_i8)

    return conv


def replace_mod_part(conv):
    conv = replace(conv, "for ocol_i in _:_ #0", ld_acc_i32_vector)
    conv = old_reorder(conv, "och_i kch_i")
    conv = replace(conv, "for kch_i in _:_ #0", ld_i8_block_id1)
    conv = replace(conv, "for ocol_i in _:_ #0", ld_i8_block_id2)
    conv = old_reorder(conv, "kch_i och_i")
    conv = replace(conv, "for ocol_i in _:_ #0", matmul_acc_i8)
    conv = replace(conv, "for ocol_i in _:_ #0", st_acc_i8)

    return conv


def inline_div_part(conv):
    conv = inline_vector(conv)
    conv = lift_config(conv, "config_ld_acc_i32_vector(_)")
    conv = inline_ld_id1(conv)
    conv = lift_config(conv, "config_ld_i8_id1(_)")
    conv = inline_matmul(conv)
    conv = lift_config(conv, "config_matmul(_)")
    conv = inline_st(conv)
    conv = lift_config(conv, "config_st_acc_i8(_)")

    return conv


def inline_mod_part(conv):
    conv = inline_vector(conv)
    conv = inline_ld_id1(conv)
    conv = inline_matmul(conv)
    conv = inline_st(conv)
    conv = delete_config(conv, "config_ld_acc_i32_vector(_) #1")
    conv = delete_config(conv, "config_ld_i8_id1(_) #1")
    conv = delete_config(conv, "config_matmul(_) #1")
    conv = delete_config(conv, "config_st_acc_i8(_) #1")
    conv = simplify(conv)

    return conv


def set_gemm_memories(conv):
    conv = set_memory(conv, "res", GEMM_ACCUM)
    conv = set_memory(conv, "i_s", GEMM_SCRATCH)
    conv = set_memory(conv, "w_s", GEMM_SCRATCH)
    try:
        conv = set_memory(conv, "res #1", GEMM_ACCUM)
        conv = set_memory(conv, "i_s #1", GEMM_SCRATCH)
        conv = set_memory(conv, "w_s #1", GEMM_SCRATCH)
    except:
        pass

    return conv


def schedule_conv_3():
    batch_size = 4
    out_channel = 64
    kernel_dim = 3
    in_channel = 64
    in_dim = 58
    out_dim = int((in_dim - kernel_dim) / 1 + 1)
    assert out_dim == 56

    cpu = rename(conv_on_cpu_stride_1, "conv_3_cpu")
    cpu = cpu.partial_eval(
        batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim
    )

    conv = rename(cpu, "conv_3")

    conv = split_fission_dim(conv)

    conv = replace_div_part(conv)
    conv = replace_mod_part(conv)

    conv = set_gemm_memories(conv)

    conv = inline_div_part(conv)
    conv = inline_mod_part(conv)

    # Real optimization
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=2)
    conv = old_fission_after(conv, "for ocol_o in _:_ #0")
    conv = old_reorder(conv, "orow ocol_o")
    conv = old_split(conv, "orow", 28, ["orow_o", "orow_i"], perfect=True)
    # FIXME(#133): Remove unsafe_disable_checks once we have new effectcheck working
    conv = expand_dim(
        conv, "i_s: i8[_]", "30", "krow + orow_i", unsafe_disable_checks=True
    )
    conv = expand_dim(
        conv, "i_s: i8[_] #1", "30", "krow + orow_i", unsafe_disable_checks=True
    )
    # TODO: We should definitely use repeat for this!
    conv = old_lift_alloc(conv, "i_s : _ #0", n_lifts=5, keep_dims=False)
    conv = old_lift_alloc(conv, "i_s : _ #1", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "w_s : _ #0", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "w_s : _ #1", n_lifts=3, keep_dims=False)
    conv = old_lift_alloc(conv, "res : _ #0", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "res : _ #1", n_lifts=3, keep_dims=False)

    conv = old_fission_after(conv, "for kch_o in _:_ #0", n_lifts=6)
    conv = old_fission_after(conv, "for kch_o in _:_ #2", n_lifts=5)
    conv = old_fission_after(conv, "for och_o in _:_ #3")
    conv = add_loop(conv, "for kch_o in _:_ #0", "orow_i", 28, guard=True)
    conv = add_loop(conv, "if orow_i == 0:_", "orow_o", 2, guard=True)
    conv = add_loop(conv, "if orow_o == 0:_", "b", 4, guard=True)
    conv = add_loop(conv, "if b == 0:_", "ocol_o", 3, guard=True)
    conv = add_loop(conv, "for kch_o in _:_ #2", "orow_i", 28, guard=True)
    conv = add_loop(conv, "if orow_i == 0:_ #1", "orow_o", 2, guard=True)
    conv = add_loop(conv, "if orow_o == 0:_ #1", "b", 4, guard=True)
    # Start fissioning loops
    conv = add_loop(conv, "for och_o in _:_ #0", "b", 4)
    conv = old_reorder(conv, "orow_o b")
    conv = old_reorder(conv, "orow_i b")
    conv = old_reorder(conv, "kcol b")
    conv = old_reorder(conv, "krow b")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True)
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True)
    conv = add_loop(conv, "for och_o in _:_ #0", "ocol_o", 3)
    conv = old_reorder(conv, "orow_o ocol_o")
    conv = old_reorder(conv, "orow_i ocol_o")
    conv = old_reorder(conv, "kcol ocol_o")
    conv = old_reorder(conv, "krow ocol_o")
    conv = fuse(conv, "for ocol_o in _:_ #0", "for ocol_o in _:_ #1")
    conv = fuse(
        conv, "for ocol_o in _:_ #0", "for ocol_o in _:_ #1", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #0", "orow_o", 2)
    conv = old_reorder(conv, "orow_i orow_o")
    conv = old_reorder(conv, "kcol orow_o")
    conv = old_reorder(conv, "krow orow_o")
    conv = fuse(conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1")
    conv = fuse(
        conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #0", "orow_i", 28)
    conv = old_reorder(conv, "kcol orow_i")
    conv = old_reorder(conv, "krow orow_i")
    conv = fuse(conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1")
    conv = fuse(
        conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1", unsafe_disable_check=True
    )

    conv = add_loop(conv, "for och_o in _:_ #3", "orow_o", 2)
    conv = fuse(conv, "for orow_o in _:_ #1", "for orow_o in _:_ #2")
    conv = fuse(
        conv, "for orow_o in _:_ #1", "for orow_o in _:_ #2", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #3", "orow_i", 28)
    conv = fuse(conv, "for orow_i in _:_ #1", "for orow_i in _:_ #2")
    conv = fuse(
        conv, "for orow_i in _:_ #1", "for orow_i in _:_ #2", unsafe_disable_check=True
    )

    conv = fuse(conv, "for krow in _:_ #0", "for krow in _:_ #1")
    conv = fuse(conv, "for kcol in _:_ #0", "for kcol in _:_ #1")
    conv = fuse(conv, "for krow in _:_ #1", "for krow in _:_ #2")
    conv = fuse(conv, "for kcol in _:_ #1", "for kcol in _:_ #2")

    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #0", "orow_i == 0 or krow == 2")
    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #1", "orow_i == 0 or krow == 2")

    conv = old_split(conv, "orow_i", 7, ["orow_io", "orow_ii"], perfect=True)
    conv = old_unroll(conv, "och_o")
    # conv = old_unroll(conv, 'kch_o')
    # conv = old_unroll(conv, 'kcol')
    conv = simplify(conv)

    return conv, cpu


def schedule_conv_17():
    batch_size = 4
    out_channel = 128
    kernel_dim = 3
    in_channel = 128
    in_dim = 30
    out_dim = int((in_dim - kernel_dim) / 1 + 1)
    assert out_dim == 28

    cpu = rename(conv_on_cpu_stride_1, "conv_17_cpu")
    cpu = cpu.partial_eval(
        batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim
    )

    conv = rename(cpu, "conv_17")

    conv = split_fission_dim(conv)
    conv = replace(conv, "for ocol_i in _:_ #0", ld_acc_i32_vector)
    conv = divide_loop(conv, "och_o #1", 4, ["och_o_o", "och_o_i"], perfect=True)
    conv = old_reorder(conv, "och_i kch_i")
    conv = old_reorder(conv, "och_o_i kch_i")
    conv = replace(conv, "for kch_i in _:_ #0", ld_i8_block_id1)
    conv = divide_loop(conv, "kch_o #1", 4, ["kch_o_o", "kch_o_i"], perfect=True)
    conv = old_reorder(conv, "kch_o_i ocol_i")
    conv = replace(conv, "for ocol_i in _:_ #0", ld_i8_block_id2)
    conv = old_reorder(conv, "kch_i och_i")
    conv = replace(conv, "for ocol_i in _:_ #0", matmul_acc_i8)
    conv = replace(conv, "for ocol_i in _:_ #0", st_acc_i8)

    conv = replace(conv, "for ocol_i in _:_ #0", ld_acc_i32_vector)
    conv = divide_loop(conv, "och_o #4", 4, ["och_o_o", "och_o_i"], perfect=True)
    conv = old_reorder(conv, "och_i kch_i")
    conv = old_reorder(conv, "och_o_i kch_i")
    conv = replace(conv, "for kch_i in _:_ #0", ld_i8_block_id1)
    conv = divide_loop(conv, "kch_o #3", 4, ["kch_o_o", "kch_o_i"], perfect=True)
    conv = old_reorder(conv, "kch_o_i ocol_i")
    conv = replace(conv, "for ocol_i in _:_ #0", ld_i8_block_id2)
    conv = old_reorder(conv, "kch_i och_i")
    conv = replace(conv, "for ocol_i in _:_ #0", matmul_acc_i8)
    conv = replace(conv, "for ocol_i in _:_ #0", st_acc_i8)

    conv = set_gemm_memories(conv)
    conv = inline_div_part(conv)
    conv = inline_mod_part(conv)

    # Real optimization
    conv = old_unroll(conv, "ocol_o")
    conv = old_fission_after(conv, "for och_o in _:_ #2", n_lifts=2)
    conv = old_split(conv, "orow", 14, ["orow_o", "orow_i"], perfect=True)
    # FIXME(#133): Remove unsafe_disable_checks once we have new effectcheck working
    conv = expand_dim(
        conv, "i_s: i8[_]", "16", "krow + orow_i", unsafe_disable_checks=True
    )
    conv = expand_dim(
        conv, "i_s: i8[_] #1", "16", "krow + orow_i", unsafe_disable_checks=True
    )
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=2)
    conv = old_split(conv, "b", 4, ["bo", "bi"], perfect=True)
    conv = old_lift_alloc(conv, "i_s : _", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=3, keep_dims=False)
    conv = old_lift_alloc(conv, "res : _", n_lifts=3, keep_dims=False)

    # conv = conv.add_guard('for kch_o in _:_', 'bi', 0)
    # conv = conv.add_guard('for kch_o in _:_', 'orow_o', 0)
    # conv = conv.add_guard('for kch_o in _:_', 'orow_i', 0)
    # conv = conv.add_guard('for kch_o in _:_ #2', 'bi #1', 0)
    # conv = conv.add_guard('for kch_o in _:_ #2', 'orow_o #1', 0)
    # conv = conv.add_guard('for kch_o in _:_ #2', 'orow_i #1', 0)

    conv = old_fission_after(conv, "for kch_o in _:_ #0", n_lifts=5)
    conv = old_fission_after(conv, "for kch_o in _:_ #2", n_lifts=5)
    conv = add_loop(conv, "for kch_o in _:_ #0", "orow_i", 14, guard=True)
    conv = add_loop(
        conv, "if orow_i == 0:_", "orow_o", 2, guard=True, unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "if orow_o == 0:_", "bi", 4, guard=True, unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "for kch_o in _:_ #2", "orow_i", 14, guard=True, unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "if orow_i == 0:_ #1", "orow_o", 2, guard=True, unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "if orow_o == 0:_ #1", "bi", 4, guard=True, unsafe_disable_check=True
    )
    # Start fissioning loops
    conv = add_loop(conv, "for och_o in _:_ #0", "bi", 4, unsafe_disable_check=True)
    conv = old_reorder(conv, "orow_o bi")
    conv = old_reorder(conv, "orow_i bi")
    conv = old_reorder(conv, "kcol bi")
    conv = old_reorder(conv, "krow bi")
    conv = fuse(conv, "for bi in _:_ #0", "for bi in _:_ #1")
    conv = fuse(conv, "for bi in _:_ #0", "for bi in _:_ #1", unsafe_disable_check=True)
    conv = add_loop(conv, "for och_o in _:_ #0", "orow_o", 2)
    conv = old_reorder(conv, "orow_i orow_o")
    conv = old_reorder(conv, "kcol orow_o")
    conv = old_reorder(conv, "krow orow_o")
    conv = fuse(conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1")
    conv = fuse(
        conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1", unsafe_disable_check=True
    )
    conv = add_loop(conv, "for och_o in _:_ #0", "orow_i", 14)
    conv = old_reorder(conv, "kcol orow_i")
    conv = old_reorder(conv, "krow orow_i")
    conv = fuse(
        conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for krow in _:_ #0", "for krow in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for kcol in _:_ #0", "for kcol in _:_ #1", unsafe_disable_check=True
    )

    conv = add_loop(conv, "for och_o in _:_ #3", "bi", 4, unsafe_disable_check=True)
    conv = fuse(conv, "for bi in _:_ #1", "for bi in _:_ #2", unsafe_disable_check=True)
    conv = fuse(conv, "for bi in _:_ #1", "for bi in _:_ #2", unsafe_disable_check=True)
    conv = add_loop(conv, "for och_o in _:_ #3", "orow_o", 2, unsafe_disable_check=True)
    conv = fuse(
        conv, "for orow_o in _:_ #1", "for orow_o in _:_ #2", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for orow_o in _:_ #1", "for orow_o in _:_ #2", unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "for och_o in _:_ #3", "orow_i", 14, unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for orow_i in _:_ #1", "for orow_i in _:_ #2", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for orow_i in _:_ #1", "for orow_i in _:_ #2", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for krow in _:_ #1", "for krow in _:_ #2", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for kcol in _:_ #1", "for kcol in _:_ #2", unsafe_disable_check=True
    )

    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #0", "orow_i == 0 or krow == 2")
    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #1", "orow_i == 0 or krow == 2")

    conv = old_unroll(conv, "och_o")
    conv = old_unroll(conv, "kch_o_o")
    conv = old_unroll(conv, "kch_o")
    conv = old_unroll(conv, "kcol")
    conv = simplify(conv)

    return conv, cpu


def schedule_conv_30():
    batch_size = 4
    out_channel = 256
    kernel_dim = 3
    in_channel = 256
    in_dim = 16
    out_dim = int((in_dim - kernel_dim) / 1 + 1)
    assert out_dim == 14

    cpu = rename(conv_on_cpu_stride_1, "conv_30_cpu")
    cpu = cpu.partial_eval(
        batch_size, out_dim, out_channel, kernel_dim, in_channel, in_dim
    )

    conv = rename(cpu, "conv_30")

    conv = divide_loop(conv, "och", 64, ["och_out", "och"], perfect=True)
    conv = old_reorder(conv, "ocol och_out")
    conv = old_reorder(conv, "orow och_out")
    conv = old_reorder(conv, "b och_out")
    conv = split_fission_dim(conv)
    conv = replace_div_part(conv)
    conv = set_gemm_memories(conv)
    conv = inline_div_part(conv)

    conv = old_lift_alloc(conv, "w_s : _", n_lifts=2)
    conv = old_split(conv, "orow", 7, ["orow_o", "orow_i"], perfect=True)
    # FIXME(#133): Remove unsafe_disable_checks once we have new effectcheck working
    conv = expand_dim(
        conv, "i_s: i8[_]", "9", "krow + orow_i", unsafe_disable_checks=True
    )
    conv = old_lift_alloc(conv, "res : _", n_lifts=1)
    conv = old_lift_alloc(conv, "i_s : _", n_lifts=5, keep_dims=False)
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=4, keep_dims=False)
    conv = old_lift_alloc(conv, "res : _", n_lifts=3, keep_dims=False)

    # conv = conv.add_guard('for kch_o in _:_', 'b', 0)
    # conv = conv.add_guard('for kch_o in _:_', 'orow_o', 0)
    # conv = conv.add_guard('for kch_o in _:_', 'orow_i', 0)
    conv = old_fission_after(conv, "for kch_o in _:_ #0", n_lifts=5)
    conv = old_fission_after(conv, "for och_o in _:_ #0", n_lifts=1)
    conv = add_loop(
        conv, "for kch_o in _:_ #0", "orow_i", 7, guard=True, unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "if orow_i == 0:_", "orow_o", 2, guard=True, unsafe_disable_check=True
    )
    conv = add_loop(
        conv, "if orow_o == 0:_", "b", 4, guard=True, unsafe_disable_check=True
    )
    # Start fissioning loops
    conv = add_loop(conv, "for orow_i in _:_ #0", "b", 4, unsafe_disable_check=True)
    conv = old_reorder(conv, "orow_o b")
    conv = old_reorder(conv, "orow_i b")
    conv = old_reorder(conv, "kcol b")
    conv = old_reorder(conv, "krow b")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1")
    conv = fuse(conv, "for b in _:_ #0", "for b in _:_ #1", unsafe_disable_check=True)
    conv = add_loop(
        conv, "for orow_i in _:_ #0", "orow_o", 2, unsafe_disable_check=True
    )
    conv = old_reorder(conv, "orow_i orow_o")
    conv = old_reorder(conv, "kcol orow_o")
    conv = old_reorder(conv, "krow orow_o")
    conv = fuse(
        conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for orow_o in _:_ #0", "for orow_o in _:_ #1", unsafe_disable_check=True
    )
    conv = old_reorder(conv, "kcol orow_i")
    conv = old_reorder(conv, "krow orow_i")
    conv = fuse(
        conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for orow_i in _:_ #0", "for orow_i in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for krow in _:_ #0", "for krow in _:_ #1", unsafe_disable_check=True
    )
    conv = fuse(
        conv, "for kcol in _:_ #0", "for kcol in _:_ #1", unsafe_disable_check=True
    )

    conv = add_unsafe_guard(conv, "ld_i8_block_id2(_) #0", "orow_i == 0 or krow == 2")

    conv = old_unroll(conv, "och_o")
    # conv = old_unroll(conv, 'kch_o')
    # conv = old_unroll(conv, 'kcol')
    conv = simplify(conv)

    return conv, cpu


conv_3, conv_3_cpu = schedule_conv_3()
conv_17, conv_17_cpu = schedule_conv_17()
conv_30, conv_30_cpu = schedule_conv_30()

__all__ = ["conv_3", "conv_3_cpu", "conv_17", "conv_17_cpu", "conv_30", "conv_30_cpu"]
