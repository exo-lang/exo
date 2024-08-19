from __future__ import annotations

import os
import sys


from exo import proc
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *


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
            dst[i, j] = 0


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
        # zero out the result memory
        for k_init in seq(0, K):
            for i_init in seq(0, W):
                out[k_init, i_init] = 0.0

        # do the convolution
        for k in seq(0, K):
            for c in seq(0, C):
                for i in seq(0, W):
                    for r in seq(0, R):
                        y: i32
                        y = 0
                        if i + r < W:
                            y = data[c, i + r]
                        out[k, i] += kernels[k, c, r] * y

    return generic_conv1d


def make_im2col_from_generic(p):
    # Let's start applying scheduling
    p = rename(p, "im2col_conv")
    p = reorder_loops(p, "c i")
    p = fuse(p, "for k_init in _:_", "for k in _:_")
    p = fuse(p, "for i_init in _:_", "for i in _:_")
    # add the tiles corresponding to the size of our systolic array
    p = divide_loop(
        p, "for k_init in _:_", TILE, new_iters=["tile_i", "i"], perfect=True
    )
    p = divide_loop(
        p, "for i_init in _:_", TILE, new_iters=["tile_j", "j"], perfect=True
    )
    p = reorder_loops(p, "i tile_j")
    # tile once again since we have 4 registers, we want to do 4 computes on different output channels at once
    p = divide_loop(p, "for tile_i in _:_", 4, new_iters=["hi", "lo"], perfect=True)
    p = reorder_loops(p, "lo tile_j")

    # channels should be on the outside of each of these inner loops; these will all be part of the RVM instructions themselves
    p = autofission(p, p.find("for c in _:_").before(), 3)
    p = reorder_loops(p, "j c")
    p = reorder_loops(p, "i c")
    p = reorder_loops(p, "lo c")

    # next, we can start to lift that allocation
    # up and out of the loop
    p = autolift_alloc(p, "y:i32", 4, keep_dims=True)
    p = set_memory(p, "y: _", DRAM_STATIC)

    # Then, we can fission the loop correspondingly,
    # separating what is now a data-marshalling statement from
    # the actual compute statement in two subsequent
    # loop nests via fissioning
    p = autofission(p, p.find("out[_] += _").before(), 4)
    p = simplify(p)
    return p


def rvm_optimize(p):
    # Setting up data tile load
    p = stage_mem(p, "for lo in _:_ #1", f"y[0:{TILE}, 0:{TILE}]", "data_tile")
    p = set_memory(p, "data_tile", RVM_TILE)
    p = replace(p, "for i0 in _:_", rvm_mld)

    # Setting up kernel tile load
    p = stage_mem(
        p,
        "for i in _:_ #1",
        f"kernels[hi*{TILE}*4 + 4*lo:hi*{TILE}*4 + 4*lo+{TILE}, c, 0:{TILE}]",
        "kernel_tile",
    )
    p = set_memory(p, "kernel_tile", RVM_TILE)
    p = replace(p, "for i0 in _:_", rvm_mld)

    # Setting up output tile
    # Here we are specifiying all 4 output registers at once.
    p = stage_mem(
        p,
        "for c in _:_",
        f"out[hi*{TILE}*4: hi*{TILE}*4+{TILE}*4, tile_j*{TILE}:tile_j*{TILE}+{TILE}]",
        "output_tile",
    )
    # That means we need to divide the buffers further.
    p = set_memory(p, "output_tile", RVM_TILE)
    p = divide_dim(p, "output_tile:_", 0, 4)

    # In addition, the loops which do the clearing and storing of this buffer should be tiled,
    # that way the inner nest can be replaced with the appropriate matrix instructions.
    p = divide_loop(p, "for i0 in _:_", 4, ["d_lo", "d_i"], perfect=True)
    p = divide_loop(p, "for i0 in _:_", 4, ["s_lo", "s_i"], perfect=True)

    # Fuse output zeroing loop with generated stage_mem loop,
    # that way we can replace to mzero
    p = simplify(p)
    # loops need to be adjacent, get this alloc out of the way
    # need to simplify first otherwise it's dependent on the index
    p = lift_alloc(p, "output_tile: _")
    p = fuse(p, "for lo in _:_ #0", "for d_lo in _:_ #0")
    p = fuse(p, "for i in _:_ #0", "for d_i in _:_ #0")
    p = fuse(p, "for j in _:_ #0", "for i1 in _:_")
    p = sink_alloc(p, "output_tile: _")

    # remove the assignment to out[]
    # TODO: Not a bug, but it's interesting that this is correct?
    # Exo must be able to prove that I don't read from out[] later on in the program..
    p = inline_assign(p, "out[i + 4 * lo + 16 * hi, j + 4 * tile_j] = 0")
    p = replace(p, "for i in _:_ #0", rvm_mzero)

    # Setting up output tile store
    p = replace(p, "for s_i in _:_ ", rvm_mst)

    # Replace with matmul instruction
    p = replace(p, "for i in _:_", rvm_mmasa)

    # unroll everything!
    p = unroll_loop(p, "for lo in _:_")
    p = unroll_loop(p, "for s_lo in _:_")
    p = simplify(p)

    # some gymnastics for reusing the output of the load so we don't exceed the 8 available registers
    p = autolift_alloc(p, "kernel_tile: _", keep_dims=True)
    p = unroll_loop(p, "for lo in _:_")
    p = unroll_buffer(p, "kernel_tile: _", 0)
    p = reorder_stmts(p, "kernel_tile_3: _; rvm_mld(_)")
    p = reuse_buffer(p, "kernel_tile_0: _", "kernel_tile_3: _")

    p = unroll_buffer(p, "output_tile: _", 0)

    p = simplify(p)

    # Done!
    p = simplify(p)
    p = rename(p, "exo_conv1d_tile_lt_kw")
    return p


def make_routine():
    generic_conv1d = gen_conv1d()
    im2col_cpu = make_im2col_from_generic(generic_conv1d)
    rvm_optimized = rvm_optimize(im2col_cpu)
    return rvm_optimized


exo_conv1d_tile_lt_kw = make_routine()
