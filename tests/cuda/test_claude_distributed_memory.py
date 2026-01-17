"""Tests for error conditions in spork/distributed_memory.py

These tests cover error paths related to distributed memory deduction,
including permitted index expression requirements, range requirements,
and consistency checks across multiple usages.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *


# =============================================================================
# tile_count must match array extent (Range Requirement)
# =============================================================================


def mkproc_tile_count_mismatch(num_warps):
    """Test that iterator's tile_count must match the array dimension extent.

    The range requirement says the iterator used to index a distributed dimension
    must have range [0, extent - 1] matching the array dimension.

    num_warps=4: iterator range matches array extent (valid)
    num_warps=2: iterator range differs from array extent (invalid)
    """
    device_fn = CudaDeviceFunction(blockDim=128)

    # Array always has 4 warps, but iterator range is parameterized
    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                data: f32[4, 32, 2] @ CudaRmem
                for w in cuda_threads(0, num_warps, unit=cuda_warp):
                    for t in cuda_threads(0, 32, unit=cuda_thread):
                        for i in seq(0, 2):
                            data[w, t, i] = 1.0

    return simplify(test_proc)


def test_tile_count_mismatch_positive(compiler, golden):
    # Valid: iterator range (4) matches array extent (4)
    compiler.cuda_cpu_test(mkproc_tile_count_mismatch, golden=golden, num_warps=4)


def test_tile_count_mismatch_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: iterator range (2) doesn't match array extent (4)
        compiler.cuda_cpu_test(mkproc_tile_count_mismatch, num_warps=2)
    msg = str(exc.value)
    # Error: w.tile_count = 2; must be 4 to match underlying tensor
    assert "tile_count" in msg and "must be" in msg


# =============================================================================
# Expected cuda_threads-loop iterator (Required Iterator)
# =============================================================================


def mkproc_wrong_iterator_type(use_cuda_threads):
    """Test that distributed dimension index must be from cuda_threads loop.

    Distributed memory deduction requires the index to be a plain read of
    an iterator from a cuda_threads loop, not a seq loop or other variable.

    use_cuda_threads=True: use cuda_threads iterator (valid)
    use_cuda_threads=False: use seq loop iterator (invalid)
    """
    device_fn = CudaDeviceFunction(blockDim=64)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                # Register array distributed by warp then thread
                data: f32[2, 32, 4] @ CudaRmem
                if use_cuda_threads:
                    # Valid: cuda_threads iterators
                    for w in cuda_threads(0, 2, unit=cuda_warp):
                        for t in cuda_threads(0, 32, unit=cuda_thread):
                            for i in seq(0, 4):
                                data[w, t, i] = 1.0
                else:
                    # Invalid: seq loop iterator for first distributed dim
                    for t in cuda_threads(0, 32, unit=cuda_thread):
                        for j in seq(0, 2):
                            for i in seq(0, 4):
                                # j is from seq, not cuda_threads
                                data[j, t, i] = 1.0

    return simplify(test_proc)


def test_wrong_iterator_type_positive(compiler, golden):
    # Valid: cuda_threads iterator
    compiler.cuda_cpu_test(
        mkproc_wrong_iterator_type, golden=golden, use_cuda_threads=True
    )


def test_wrong_iterator_type_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: seq loop iterator
        compiler.cuda_cpu_test(mkproc_wrong_iterator_type, use_cuda_threads=False)
    msg = str(exc.value)
    # Error: Expected cuda_threads-loop iterator, not j
    assert "Expected cuda_threads-loop iterator" in msg


# =============================================================================
# Permitted index expression must be plain variable (Data)
# =============================================================================


def mkproc_non_plain_index_expr(use_plain_var):
    """Test that distributed dimension index must be a plain variable read.

    A permitted index expression is a plain read of a cuda_threads iterator.
    Expressions like `1 - cta` are not plain variable reads.

    use_plain_var=True: use plain variable (valid)
    use_plain_var=False: use expression (invalid)
    """
    device_fn = CudaDeviceFunction(clusterDim=2, blockDim=32)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                # SMEM distributed across 2 CTAs
                smem: f32[2, 32] @ CudaSmemLinear
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for tid in cuda_threads(0, 32):
                        if use_plain_var:
                            # Valid: plain variable read
                            smem[cta, tid] = 1.0
                        else:
                            # Invalid: expression, not plain variable
                            smem[1 - cta, tid] = 1.0

    return simplify(test_proc)


def test_non_plain_index_expr_positive(compiler, golden):
    # Valid: plain variable read
    compiler.cuda_cpu_test(
        mkproc_non_plain_index_expr, golden=golden, use_plain_var=True
    )


def test_non_plain_index_expr_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: expression instead of plain variable
        compiler.cuda_cpu_test(mkproc_non_plain_index_expr, use_plain_var=False)
    msg = str(exc.value)
    # Error: Expected single variable name, not 1 - cta
    assert "Expected single variable name" in msg


# =============================================================================
# Permitted index expression must be plain variable (Barrier)
# =============================================================================


def mkproc_barrier_non_plain_index(use_plain_var):
    """Test that barrier index must be a plain variable read.

    For barriers, all dimensions are distributed and must be indexed
    by plain reads of cuda_threads iterators, not constants or expressions.

    use_plain_var=True: use plain variable (valid)
    use_plain_var=False: use constant (invalid)
    """
    device_fn = CudaDeviceFunction(blockDim=64)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                bar: barrier[2] @ CudaMbarrier
                for w in cuda_threads(0, 2, unit=cuda_warp):
                    if use_plain_var:
                        # Valid: plain variable read
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Await(bar[w], cuda_in_order, ~1)
                    else:
                        # Invalid: constant instead of variable
                        Arrive(cuda_in_order, 1) >> bar[0]
                        Await(bar[0], cuda_in_order, ~1)

    return simplify(test_proc)


def test_barrier_non_plain_index_positive(compiler, golden):
    # Valid: plain variable read
    compiler.cuda_cpu_test(
        mkproc_barrier_non_plain_index, golden=golden, use_plain_var=True
    )


def test_barrier_non_plain_index_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: constant index
        compiler.cuda_cpu_test(mkproc_barrier_non_plain_index, use_plain_var=False)
    msg = str(exc.value)
    # Error: expected a plain variable, not 0
    assert "expected a plain variable" in msg


# =============================================================================
# Constant instead of permitted iterator for data (not a plain variable)
# =============================================================================


def mkproc_constant_distributed_index(use_iterator):
    """Test that distributed dimension cannot use a constant index.

    A permitted index expression must be a plain read of a cuda_threads iterator.
    A constant like 0 is not permitted.

    use_iterator=True: use cuda_threads iterator (valid)
    use_iterator=False: use constant 0 (invalid)
    """
    device_fn = CudaDeviceFunction(blockDim=64)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                # 2D distributed register array
                data: f32[2, 32, 4] @ CudaRmem
                for w in cuda_threads(0, 2, unit=cuda_warp):
                    for t in cuda_threads(0, 32, unit=cuda_thread):
                        for i in seq(0, 4):
                            if use_iterator:
                                # Valid: both w and t are permitted iterators
                                data[w, t, i] = 1.0
                            else:
                                # Invalid: 0 is not a permitted index expression
                                data[w, 0, i] = 1.0

    return simplify(test_proc)


def test_constant_distributed_index_positive(compiler, golden):
    # Valid: all indices are permitted iterators
    compiler.cuda_cpu_test(
        mkproc_constant_distributed_index, golden=golden, use_iterator=True
    )


def test_constant_distributed_index_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: constant instead of iterator
        compiler.cuda_cpu_test(mkproc_constant_distributed_index, use_iterator=False)
    msg = str(exc.value)
    # Error: Expected single variable name, not 0
    assert "Expected single variable name" in msg


# =============================================================================
# Usage Box Requirement: Missing subdivision at usage site
# =============================================================================


def mkproc_missing_subdivision(thread_unit):
    """Test that subdivided dimensions must have box=1 at usage site.

    For CudaRmem (native unit = cuda_thread), we must iterate down to
    individual threads at the usage site. If we only iterate by warps,
    the box will be 32 (not 1), causing "Missing subdivision" error.

    thread_unit=cuda_thread: iterate to thread level (valid)
    thread_unit=cuda_warp: only iterate by warps (invalid)

    Note: cuda_threads(0, 2) is valid - there is no requirement to use all threads.
    """
    device_fn = CudaDeviceFunction(blockDim=64)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                # CudaRmem requires thread-level subdivision
                data: f32[2, 4] @ CudaRmem
                for tid in cuda_threads(0, 2, unit=thread_unit):
                    for i in seq(0, 4):
                        data[tid, i] = 1.0

    return simplify(test_proc)


def test_missing_subdivision_positive(compiler, golden):
    # Valid: subdivided to thread level
    compiler.cuda_cpu_test(
        mkproc_missing_subdivision, golden=golden, thread_unit=cuda_thread
    )


def test_missing_subdivision_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: warp unit doesn't subdivide to thread level
        compiler.cuda_cpu_test(mkproc_missing_subdivision, thread_unit=cuda_warp)
    msg = str(exc.value)
    # Error: Missing subdivision on dims[0] to match cuda_thread
    assert "Missing subdivision" in msg and "cuda_thread" in msg


# =============================================================================
# Alloc Box Requirement: Missing threads at allocation site for SMEM
# =============================================================================


def mkproc_smem_inside_warp_loop(blockDim):
    """Test that SMEM allocation needs all CTA threads available.

    For CudaSmemLinear (native_unit = cuda_cta_in_cluster), allocation must
    happen at a point where all threads of the CTA are active. Allocating
    inside a warp loop (where only 1 warp is active) triggers "Missing threads".

    blockDim=32: single warp, warp loop has 1 iteration, all threads available (valid)
    blockDim=128: 4 warps, inside warp loop only 32 of 128 threads active (invalid)

    Note: This is the same error tested by test_smem_in_warp in test_coll_units.py,
    but included here for completeness of distributed memory error coverage.
    """
    device_fn = CudaDeviceFunction(blockDim=blockDim)
    num_warps = blockDim // 32

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                for w in cuda_threads(0, num_warps, unit=cuda_warp):
                    # With blockDim=32: 1 warp, all CTA threads available
                    # With blockDim=128: 4 warps, only 32 of 128 threads available
                    smem: f32[32] @ CudaSmemLinear
                    for t in cuda_threads(0, 32):
                        smem[t] = 1.0

    return simplify(test_proc)


def test_smem_inside_warp_loop_positive(compiler, golden):
    # Valid: blockDim=32 means 1 warp, all CTA threads available at alloc
    compiler.cuda_cpu_test(mkproc_smem_inside_warp_loop, golden=golden, blockDim=32)


def test_smem_inside_warp_loop_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: blockDim=128 means 4 warps, only 32 of 128 threads at alloc
        compiler.cuda_cpu_test(mkproc_smem_inside_warp_loop, blockDim=128)
    msg = str(exc.value)
    # Error: Missing threads to match cuda_cta_in_cluster
    assert "Missing threads" in msg and "cuda_cta_in_cluster" in msg
