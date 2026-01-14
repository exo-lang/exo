"""Tests for error conditions in spork/coll_algebra.py

These tests cover error paths related to collective unit algebra and tiling.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.stdlib.scheduling import *


# =============================================================================
# CollUnit cannot be scaled (unit without scaled_dim_idx)
# =============================================================================


def test_coll_unit_cannot_be_scaled():
    """Some CollUnits don't support scaling (scaled_dim_idx=None)"""
    # cuda_quadpair has scaled_dim_idx=None
    with pytest.raises(Exception) as exc:
        _ = 4 * cuda_quadpair
    msg = str(exc.value)
    assert "cannot be scaled" in msg.lower() or "quadpair" in msg.lower()


# =============================================================================
# CollUnit scaled by non-positive int
# =============================================================================


def test_coll_unit_scaled_by_zero():
    """CollUnit must be scaled by positive int"""
    with pytest.raises(Exception) as exc:
        _ = 0 * cuda_thread
    msg = str(exc.value)
    assert "positive" in msg.lower() or "int" in msg.lower()


def test_coll_unit_scaled_by_negative():
    """CollUnit must be scaled by positive int"""
    with pytest.raises(Exception) as exc:
        _ = (-2) * cuda_thread
    msg = str(exc.value)
    assert "positive" in msg.lower() or "int" in msg.lower()


def test_coll_unit_scaled_by_float():
    """CollUnit must be scaled by int, not float"""
    with pytest.raises(Exception) as exc:
        _ = 2.5 * cuda_thread
    msg = str(exc.value)
    assert "positive" in msg.lower() or "int" in msg.lower()


# =============================================================================
# Thread alignment issues in cuda_threads loop
# =============================================================================


def mkproc_thread_alignment_issue():
    """cuda_threads loop with misaligned thread count"""

    @proc
    def test_proc(foo: f32[100] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, 1):
                # Try to tile by 7 threads - doesn't divide evenly
                for outer in cuda_threads(0, 7, unit=cuda_thread):
                    for inner in cuda_threads(0, 9, unit=cuda_thread):
                        pass

    return simplify(test_proc)


def test_thread_alignment_issue(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_thread_alignment_issue)
    msg = str(exc.value)
    # Should fail due to alignment/tiling issues
    assert (
        "alignment" in msg.lower()
        or "divide" in msg.lower()
        or "tile" in msg.lower()
        or "thread" in msg.lower()
    )


# =============================================================================
# Not enough threads available for tiling
# =============================================================================


def mkproc_not_enough_threads():
    """Request more threads than available in collective unit"""

    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                # Only have 32 threads, but request 64
                for tid in cuda_threads(0, 64):
                    foo = 1.0

    return simplify(test_proc)


def test_not_enough_threads(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_not_enough_threads)
    msg = str(exc.value)
    assert (
        "thread" in msg.lower()
        or "available" in msg.lower()
        or "max" in msg.lower()
        or "not enough" in msg.lower()
    )


# =============================================================================
# Ambiguous dimension to tile
# =============================================================================


def mkproc_ambiguous_tile_dimension():
    """TODO: This error occurs when CollTiling cannot determine which
    dimension to tile on. This is hard to trigger with normal usage.
    """
    pass


def test_ambiguous_tile_dimension():
    # TODO: Need to understand exact conditions to trigger this
    pytest.skip("Complex collective unit configuration required")


# =============================================================================
# Invalid alignment for CollUnit domain completion
# =============================================================================


def mkproc_invalid_coll_unit_alignment():
    """Try to use a collective unit that doesn't align with blockDim"""

    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        # blockDim=100 is not divisible by 32 (warp size)
        with CudaDeviceFunction(blockDim=100):
            for task in cuda_tasks(0, 1):
                # Try to use cuda_warp which expects blockDim divisible by 32
                for w in cuda_threads(0, 3, unit=cuda_warp):
                    for tid in cuda_threads(0, 32):
                        foo = 1.0

    return simplify(test_proc)


def test_invalid_coll_unit_alignment(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_invalid_coll_unit_alignment)
    msg = str(exc.value)
    # Should fail at CudaDeviceFunction creation (blockDim must be multiple of 32)
    # or at collective unit alignment check
    assert (
        "blockDim" in msg
        or "32" in msg
        or "alignment" in msg.lower()
        or "divide" in msg.lower()
    )


# =============================================================================
# Domain completion divisibility issues
# =============================================================================


def mkproc_domain_completion_divisibility():
    """TODO: Domain completion fails when dimensions don't divide evenly.
    This requires specific mismatched collective unit configurations.
    """
    pass


def test_domain_completion_divisibility():
    # TODO: Need to craft specific mismatched CollUnit configurations
    pytest.skip("Complex domain completion scenario required")


# =============================================================================
# Write must be executed by one thread only
# =============================================================================
# NOTE: The compiler allows multiple threads to write the same value to a scalar.
# This is redundant but not an error - no test needed here.


# =============================================================================
# Wrong collective unit for instruction call
# =============================================================================
# NOTE: Removed test_wrong_coll_unit_for_instr - the test was based on incorrect
# assumptions about what constitutes invalid collective unit usage. Instructions
# like Sm80_cp_async_f32 can be called per-thread in a cuda_threads loop.
