from __future__ import annotations

from exo import *
from exo.libs.memories import AVX2
from exo.stdlib.scheduling import *


@instr("{dst_data} = _mm256_loadu_ps(&{src_data});")
def vector_load(dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("_mm256_storeu_ps(&{dst_data}, {src_data});")
def vector_store(dst: [f32][8] @ DRAM, src: [f32][8] @ AVX2):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]


@instr("{out_data} = _mm256_mul_ps({x_data}, {y_data});")
def vector_multiply(out: [f32][8] @ AVX2, x: [f32][8] @ AVX2, y: [f32][8] @ AVX2):
    assert stride(out, 0) == 1
    assert stride(x, 0) == 1
    assert stride(y, 0) == 1

    for i in seq(0, 8):
        out[i] = x[i] * y[i]


@instr("{out_data} = _mm256_broadcast_ss(2.0);")
def vector_assign_two(out: [f32][8] @ AVX2):
    assert stride(out, 0) == 1

    for i in seq(0, 8):
        out[i] = 2.0


@proc
def vec_double(N: size, inp: f32[N], out: f32[N]):
    assert N % 8 == 0
    for i in seq(0, N):
        out[i] = 2.0 * inp[i]


def wrong_schedule(p):
    """
    Forgot to set the memory types to be AVX2 vectors, so replace instruction
    does not work as intended.
    """
    p = rename(p, "vec_double_optimized")
    p = divide_loop(p, "i", 8, ["io", "ii"], perfect=True)

    # Create a vector of twos
    p = bind_expr(p, "2.0", "two_vec")
    two_alloc = p.find("two_vec: _")
    two_assign = p.find("two_vec = _")
    p = expand_dim(p, two_alloc, 8, "ii")

    # Hoist the allocation and assignment of two vector
    p = lift_alloc(p, two_alloc, 2)
    p = fission(p, two_assign.after(), 2)
    p = remove_loop(p, two_assign.parent().parent())

    # Create vectors for the input and output values
    innermost_loop = p.find_loop("ii #1")
    p = stage_mem(p, innermost_loop, "out[8*io:8*io+8]", "out_vec")
    p = stage_mem(p, innermost_loop, "inp[8*io:8*io+8]", "inp_vec")
    p = simplify(p)

    # Replace with AVX instructinos
    avx_instrs = [vector_assign_two, vector_multiply, vector_load, vector_store]
    p = replace_all(p, avx_instrs)

    return p


w = wrong_schedule(vec_double)
print(w)
