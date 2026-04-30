from __future__ import annotations

import pytest

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *


def test_set_dim_size_annotation(compiler, golden):
    @proc
    def p(K: size):
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                # We will delete @ ring_buffer_by(3)
                # and add ring_buffer_by(4) to the K dimension.
                my_barrier: barrier[4 @ ring_buffer_by(3), 2, K] @ CudaMbarrier
                for k in seq(0, K):
                    for x in cuda_threads(0, 4, unit=2 * cuda_warp):
                        for y in cuda_threads(0, 2, unit=cuda_warp):
                            some_scalar: f32 @ CudaRmemUniform(32)
                            some_scalar = 1337
                            Arrive(cuda_in_order) >> my_barrier[x, y, k]
                            Await(my_barrier[x, y, k], cuda_in_order, 0)

    some_scalar_cursor = p.find_alloc_or_arg("some_scalar")
    my_barrier_cursor = p.find_alloc_or_arg("my_barrier")
    assert my_barrier_cursor.mem() == CudaMbarrier
    shape = my_barrier_cursor.shape()
    assert shape[0].size_annotation() == ring_buffer_by(3)
    assert shape[1].size_annotation() == None
    assert shape[2].size_annotation() == None

    p = set_dim_size_annotation(p, my_barrier_cursor, 2, ring_buffer_by(4))
    my_barrier_cursor = p.forward(my_barrier_cursor)
    shape = my_barrier_cursor.shape()
    assert shape[0].size_annotation() == ring_buffer_by(3)
    assert shape[1].size_annotation() == None
    assert shape[2].size_annotation() == ring_buffer_by(4)

    p = set_dim_size_annotation(p, my_barrier_cursor, 0, None)
    my_barrier_cursor = p.forward(my_barrier_cursor)
    shape = my_barrier_cursor.shape()
    assert shape[0].size_annotation() == None
    assert shape[1].size_annotation() == None
    assert shape[2].size_annotation() == ring_buffer_by(4)

    with pytest.raises(TypeError) as exc:
        set_dim_size_annotation(p, my_barrier_cursor, 0, 3.4)

    with pytest.raises(ValueError) as exc:
        set_dim_size_annotation(p, my_barrier_cursor, -1, None)

    with pytest.raises(ValueError) as exc:
        set_dim_size_annotation(p, my_barrier_cursor, 4, None)

    some_scalar_cursor = p.forward(some_scalar_cursor)
    assert some_scalar_cursor.name() == "some_scalar"

    with pytest.raises(ValueError) as exc:
        set_dim_size_annotation(p, some_scalar_cursor, 0, None)

    compiler.cuda_cpu_test(lambda: p, golden=golden)


def test_set_ring_guarded_by(compiler, golden):
    @proc
    def p(K: size):
        with CudaDeviceFunction(clusterDim=2, blockDim=256):
            for task in cuda_tasks(0, 1):
                dummy_smem: f32[16] @ CudaSmemLinear
                dummy_barrier: barrier[2, K @ ring_buffer_by(4)] @ CudaMbarrier
                my_barrier: barrier[2, K] @ CudaMbarrier
                smem: (
                    f32[2, K @ ring_buffer_by(4), 64, 128].ring_guarded_by(
                        dummy_barrier
                    )
                    @ CudaSmemLinear
                )
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for k in seq(0, K):
                        Await(my_barrier[cta, k], cuda_in_order, 0)
                        pass
                        Arrive(cuda_in_order) >> my_barrier[cta, k]

    my_barrier_cursor = p.find_alloc_or_arg("my_barrier")
    smem_cursor = p.find_alloc_or_arg("smem")
    assert smem_cursor.ring_guarded_by() == "dummy_barrier"

    p = set_ring_guarded_by(p, smem_cursor, None)
    smem_cursor = p.forward(smem_cursor)
    assert smem_cursor.ring_guarded_by() == None

    # XXX this should not be allowed before the above removal of
    # .ring_guarded_by(dummy_barrier), but it does.
    # The checks are based off mystery SMT solver stuff and not the normal
    # LoopIR get_reads etc. so I don't know how to fix this.
    p = delete_buffer(p, "dummy_barrier")

    p = set_ring_guarded_by(p, smem_cursor, my_barrier_cursor)
    smem_cursor = p.forward(smem_cursor)
    assert smem_cursor.ring_guarded_by() == "my_barrier"

    p = set_dim_size_annotation(p, my_barrier_cursor, 1, ring_buffer_by(4))

    with pytest.raises(TypeError) as exc:
        p = set_ring_guarded_by(p, smem_cursor, "dummy_smem")
    msg = str(exc.value)
    assert "dummy_smem: f32[16]" in msg
    assert "not a barrier" in msg

    compiler.cuda_cpu_test(lambda: p, golden=golden)


def test_multiple_ring_guarded_by_error(compiler):
    @proc
    def p(K: size):
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                my_barrier: barrier[4 @ ring_buffer_by(3), 2, K] @ CudaMbarrier
                for k in seq(0, K):
                    for x in cuda_threads(0, 4, unit=2 * cuda_warp):
                        for y in cuda_threads(0, 2, unit=cuda_warp):
                            some_scalar: f32 @ CudaRmemUniform(32)
                            some_scalar = 1337
                            Arrive(cuda_in_order) >> my_barrier[x, y, k]
                            Await(my_barrier[x, y, k], cuda_in_order, 0)

    p = set_dim_size_annotation(p, "my_barrier", 2, ring_buffer_by(3))
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(lambda: p)
    msg = str(exc.value)
    assert "multiple managed ring buffer dimensions in my_barrier: barrier[" in msg
