from __future__ import annotations

import os
import sys

import exo.API_cursors as pc
from exo import proc
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *


class RVM_TILE(StaticMemory):
    NUM_RVM_TILES = 8
    StaticMemory.init_state(NUM_RVM_TILES)
    tile_dict = {}

    # TODO: have a better way of doing this rather than manually
    # calling this after each test that fails to compile.
    @classmethod
    def reset_allocations(cls):
        cls.init_state(cls.NUM_RVM_TILES)
        cls.tile_dict = {}

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not (shape[0].isdecimal() and int(shape[0]) == 4):
            raise MemGenError("Number of tile rows must be 4.")
        if not (shape[1].isdecimal() and int(shape[1]) == 4):
            raise MemGenError("Number of tile columns must be 4.")

        tile_num = cls.find_free_chunk()
        cls.mark(tile_num)
        cls.tile_dict[new_name] = tile_num
        return f'#define {new_name} "m{7-tile_num}"'

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        tile_num = cls.tile_dict[new_name]
        del cls.tile_dict[new_name]
        cls.unmark(tile_num)
        return f"#undef {new_name}"


@instr(
    'asm volatile("mld.w "{dst_int}", (%1), %0" :: "r"(4*({src}.strides[0])), "r"(&{src_data}));'
)
def rvm_mld(dst: [i32][4, 4] @ RVM_TILE, src: [i32][4, 4] @ DRAM):
    assert stride(src, 1) == 1
    assert stride(dst, 1) == 1

    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j]


@instr('asm volatile("mzero "{dst_int});')
def rvm_mzero(dst: [i32][4, 4] @ RVM_TILE):
    assert stride(dst, 1) == 1

    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = 0.0


@instr(
    'asm volatile("mst.w "{src_int}", (%1), %0" :: "r"(4*({dst}.strides[0])), "r"(&{dst_data}));'
)
def rvm_mst(src: [i32][4, 4] @ RVM_TILE, dst: [i32][4, 4] @ DRAM):
    assert stride(src, 1) == 1
    assert stride(dst, 1) == 1

    for i in seq(0, 4):
        for j in seq(0, 4):
            dst[i, j] = src[i, j]


@instr('asm volatile("mmasa.w "{md_int}", "{ms1_int}", "{ms2_int});')
def rvm_mmasa(
    md: [i32][4, 4] @ RVM_TILE, ms1: [i32][4, 4] @ RVM_TILE, ms2: [i32][4, 4] @ RVM_TILE
):
    assert stride(md, 1) == 1
    assert stride(ms1, 1) == 1
    assert stride(ms2, 1) == 1
    for i in seq(0, 4):
        for j in seq(0, 4):
            for k in seq(0, 4):
                md[i, j] += ms2[i, k] * ms1[j, k]


# convert if else to bitwise using these instructions

# Look at _Select class in builtins and try to extend
# there are some examples

IW = 16
IC = 4
KW = 4
ICKW = IC * KW
OC = 16
TILE = 4


def gen_conv1d():
    K = OC
    W = IW
    C = IC
    R = KW

    @proc
    def generic_conv1d(
        data: i32[C, W],
        kernels: i32[K, C, R],
        out: i32[K, W],
    ):
        # do the convolution
        for k in seq(0, K):
            for i in seq(0, W):
                # zero out the result memory
                out[k, i] = 0.0
                for c in seq(0, C):
                    for r in seq(0, R):
                        y: i32
                        if i + r < W:
                            y = data[c, i + r]
                        else:
                            y = 0
                        out[k, i] += kernels[k, c, r] * y

    return generic_conv1d


def fuse_two_loops(p, c):
    """
    for i in ...:         <- c
        for j in ...:
            s1
    for k in ...:         <- c.next()
        for i in ...:
            s2
    ---->
    for i in ...:         <- c
        for j in ...:
            s1
        for k in ...:
            s2
    """
    try:
        next_c = c.next()
    except:
        return p, False

    if isinstance(c, pc.ForCursor) and isinstance(next_c, pc.ForCursor):
        if c.name() == next_c.name() and expr_to_string(c.hi()) == expr_to_string(
            next_c.hi()
        ):
            p = fuse(p, c, next_c, unsafe_disable_check=False)
            return p, True
        else:
            tgt_c, count = find_child_loop(next_c, c.name())
            if tgt_c:
                p = lift_scope_n(p, tgt_c, n_lifts=count)
                p = fuse(p, c, tgt_c, unsafe_disable_check=False)
                return p, True

    return p, False


def fuse_all_loops(p, cursor):
    """
    recursively calls fuse_two_loops to all the loops
    """
    while True:
        if isinstance(cursor, pc.ForCursor):
            p = fuse_all_loops(p, cursor.body()[0])

        # Fuse in current scope
        p, b = fuse_two_loops(p, cursor)

        if b:
            cursor = p.forward(cursor)
        else:
            try:
                cursor = p.forward(cursor).next()
            except:
                break

    return p


def autolift_alloc(p, alloc_c, dep_set=None, max_size=0, lift=True):
    """
    for i in seq(0, 10):
        for j in seq(0, 20):
            a : R          <- alloc_c, dep_set = {'i'}
            a[i] = ...
    ---->
    a : R[10]              <- if size is less than max_size
    for i in seq(0, n):
        for j in seq(0, m):
            a[i] = ...
    """
    alloc_c = p.forward(alloc_c)
    loop_c = get_enclosing_loop(p, alloc_c)
    accum_size = 1
    while True:
        try:
            if not isinstance(loop_c, pc.ForCursor):
                break
            if dep_set == None or loop_c.name() in dep_set:
                if (
                    isinstance(loop_c.hi(), LiteralCursor)
                    and accum_size * loop_c.hi().value() <= max_size
                ):
                    p = expand_dim(p, alloc_c, loop_c.hi().value(), loop_c.name())
                    accum_size = accum_size * loop_c.hi().value()
                    if lift:
                        p = lift_alloc(p, alloc_c)
            loop_c = loop_c.parent()
        except:
            break
    return p


def reorder_top(p, c):
    """
    for i in seq(0, 10):
        s1
        s2
        s3  <- c
    ---->
    for i in seq(0, 10):
        s3  <- c
        s1
        s2
    """
    c = p.forward(c)
    while True:
        try:
            p = reorder_stmts(p, c.expand(1, 0))
            c = p.forward(c)
        except:
            break
    return p


def fission_as_much_as_possible(p, cursor):
    """
    for i in ...:
        for j in ...:
            s1
            s2        <- cursor
            s3
    --->
    for i in ...:
        for j in ...:
            s2

    for i in ...:
        for j in ...:
            s1
            s3
    """
    cursor = p.forward(cursor)
    p = reorder_top(p, cursor)
    gap_c = cursor.after()
    while True:
        try:
            p = fission(p, gap_c)
            gap_c = p.forward(gap_c).parent().after()
        except:
            break

    return p


def lift_scope_n(p, c, n_lifts=1):
    """
    for i in seq(0, 10):
        for j in seq(0, 10):
            for k in seq(0, 10):
                if ...:  <- c
                    s1
    ----> if n_lifts == 2:
    for i in seq(0, 10):
        if ...:  <- c
            for j in seq(0, 10):
                for k in seq(0, 10):
                    s1
    """
    for i in range(0, n_lifts):
        p = lift_scope(p, c)
    return p


def remove_redundant_loops(p, c, num=0):
    """
    for i in ...:
        for j in ...:
            s1[j]      <- c
    --->
    for j in ...:
        s1[j]          <- c
    """
    c = p.forward(c)
    cur_depth = 0
    while True:
        c = c.parent()
        if not isinstance(c, pc.ForCursor):
            break
        try:
            if cur_depth >= num:
                break
            hi = c.hi().value()
            name = c.name()
            child = p.forward(c).body()[0]
            p = remove_loop(p, c)
            cur_depth += 1
        except:
            continue
    return p


def optimize_conv(p):
    p = rename(p, "exo_conv1d_tile_lt_kw")

    # Before scheduling, grab cursors to the object code.
    k_loop = p.find("for k in _:_")
    i_loop = p.find("for i in _:_")
    c_loop = p.find("for c in _:_")
    y_alloc = p.find("y : _")
    y_assign = p.find("y = data[_]")

    # Tile outer loops to TILE size for RVM
    p, _ = tile_loops(p, [(k_loop, TILE), (i_loop, TILE)], perfect=True)
    p = reorder_loops(p, k_loop)

    # Stage output to out_tile
    p, (_, out_tile, body, _) = auto_stage_mem(
        p, p.find_loop("c").expand(1, 0), "out", "out_tile", rc=True
    )
    p = autolift_alloc(p, out_tile, max_size=4 * 4 * 4)

    # Block the zero initialization and store blocks
    p = fission_as_much_as_possible(p, body)
    p = fission_as_much_as_possible(p, body[0])

    # Reorder c loop to the top
    p = lift_scope_n(p, c_loop, 3)

    # Stage y
    p = autolift_alloc(p, y_alloc, max_size=4 * 4)
    p = lift_alloc(p, y_alloc, n_lifts=2)

    # Fission the initialization loop and remove redundant loops
    p = fission_as_much_as_possible(p, y_assign.parent())
    p = remove_redundant_loops(p, y_assign.parent(), num=2)

    # Stage kernels to kernel_tile and y to data_tile
    ki_loop = p.forward(c_loop).body()[2].body()[0]
    p, (kernel_alloc, _, _, _) = auto_stage_mem(
        p, ki_loop, "kernels", "kernel_tile", rc=True
    )
    p = simplify(expand_dim(p, kernel_alloc, 4, ki_loop.parent().name()))
    p = lift_alloc(p, kernel_alloc)
    p, (data_alloc, _, _, _) = auto_stage_mem(
        p, ki_loop.parent(), "y", "data_tile", rc=True
    )

    # Set adequate memories
    p = set_memory(p, y_alloc, DRAM_STATIC)
    p = set_memory(p, out_tile, RVM_TILE)
    p = set_memory(p, kernel_alloc, RVM_TILE)
    p = set_memory(p, data_alloc, RVM_TILE)

    # Replace inner loops to calls to RVM instructions
    p = replace_all(p, [rvm_mzero, rvm_mst, rvm_mld, rvm_mmasa])

    # Clean up
    p = unroll_loop(p, "ko")
    p = unroll_loop(p, "ko")
    p = unroll_loop(p, "ko")
    p = simplify(p)
    p = unroll_buffer(p, kernel_alloc, 0)
    p = reuse_buffer(p, "kernel_tile_0: _", "kernel_tile_3: _")

    print(p)

    return p


def make_routine():
    generic_conv1d = gen_conv1d()
    rvm_optimized = optimize_conv(generic_conv1d)
    return rvm_optimized


exo_conv1d_tile_lt_kw = make_routine()
