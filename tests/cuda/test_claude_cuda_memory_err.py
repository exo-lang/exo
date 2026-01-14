"""Tests for error conditions in spork/cuda_memory.py

These tests cover error paths related to CUDA memory allocation.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *


# =============================================================================
# Cannot allocate scalar CudaGmemLinear
# =============================================================================


def mkproc_scalar_gmem_alloc():
    """CudaGmemLinear does not support scalar allocation"""
    # David: feel free to remove this test if the restriction is fixed.

    @proc
    def test_proc():
        scalar_gmem: f32 @ CudaGmemLinear
        scalar_gmem = 1.0

    return simplify(test_proc)


def test_scalar_gmem_alloc(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_scalar_gmem_alloc)
    msg = str(exc.value)
    assert "scalar" in msg.lower() or "CudaGmemLinear" in msg


# =============================================================================
# CudaRmemPacked32 doesn't support certain types
# =============================================================================


def mkproc_rmem_packed32_unsupported_type():
    """CudaRmemPacked32 only supports f32, i32, f16, bf16"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                # i8 is not in allowed_types for CudaRmemPacked32
                packed: i8[4] @ CudaRmemPacked32
                for tid in cuda_threads(0, 1):
                    # Must actually use the variable to prevent dead code elimination
                    packed[0] = 0

    return simplify(test_proc)


def test_rmem_packed32_unsupported_type(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_rmem_packed32_unsupported_type)
    # The TypeError may be wrapped, so check the full exception chain
    msg = str(exc.value)
    assert "CudaRmemPacked32" in msg and "i8" in msg


# =============================================================================
# CudaGridConstant requires constant shapes
# =============================================================================


def mkproc_grid_constant_dynamic_shape(use_dynamic=True):
    """CudaGridConstant requires shapes to be compile-time constants"""

    if use_dynamic:

        @proc
        def test_proc(N: size, dst: f32[16] @ CudaGmemLinear):
            # Local allocation with dynamic shape should fail
            gc: f32[N] @ CudaGridConstant
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 1):
                        # Read from gc (CudaGridConstant is read-only on device)
                        dst[0] = gc[0]

        return simplify(test_proc)
    else:

        @proc
        def test_proc(dst: f32[16] @ CudaGmemLinear):
            # Local allocation with static shape should work
            gc: f32[16] @ CudaGridConstant
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 1):
                        # Read from gc (CudaGridConstant is read-only on device)
                        dst[0] = gc[0]

        return simplify(test_proc)


def test_grid_constant_dynamic_shape_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_grid_constant_dynamic_shape, use_dynamic=True)
    msg = str(exc.value)
    assert (
        "CudaGridConstant" in msg or "constant" in msg.lower() or "shape" in msg.lower()
    )


def test_grid_constant_static_shape_positive(compiler):
    # Static shape should work
    compiler.cuda_cpu_test(mkproc_grid_constant_dynamic_shape, use_dynamic=False)


# =============================================================================
# CudaSmem allocation with non-constant shape
# =============================================================================


def mkproc_smem_non_const_shape(use_dynamic=True):
    """SMEM allocations require constant shapes"""

    if use_dynamic:

        @proc
        def test_proc(N: size):
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    smem: f32[N] @ CudaSmemLinear
                    for tid in cuda_threads(0, 32):
                        smem[tid] = 1.0

        return simplify(test_proc)
    else:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    smem: f32[64] @ CudaSmemLinear
                    for tid in cuda_threads(0, 32):
                        smem[tid] = 1.0

        return simplify(test_proc)


def test_smem_non_const_shape_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_smem_non_const_shape, use_dynamic=True)
    msg = str(exc.value)
    # Error message should mention the non-constant dimension
    assert "N" in msg or "constant" in msg.lower() or "shape" in msg.lower()


def test_smem_const_shape_positive(compiler):
    compiler.cuda_cpu_test(mkproc_smem_non_const_shape, use_dynamic=False)
