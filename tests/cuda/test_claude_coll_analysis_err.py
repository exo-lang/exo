"""Tests for error conditions in spork/coll_analysis.py

These tests cover error paths related to CudaWarps and cuda_threads loops.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *


# =============================================================================
# cuda_threads loop outside CUDA device function
# =============================================================================


def mkproc_cuda_threads_outside_device():
    @proc
    def test_proc(foo: f32[32] @ CudaGmemLinear):
        for x in cuda_threads(0, 32):
            pass

    return simplify(test_proc)


def test_cuda_threads_outside_device(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_threads_outside_device)
    assert "cuda_threads" in str(exc.value)


# =============================================================================
# CudaWarps outside CUDA device function
# =============================================================================


def mkproc_cuda_warps_outside_device():
    @proc
    def test_proc(foo: f32[32] @ CudaGmemLinear):
        with CudaWarps(lo=0, hi=1):
            pass

    return simplify(test_proc)


def test_cuda_warps_outside_device(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_warps_outside_device)
    assert "CudaWarps" in str(exc.value)


# =============================================================================
# top-level CudaWarps must provide valid warp name
# =============================================================================


def mkproc_cuda_warps_name(warp_name="producer"):
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(
            warp_config=[CudaWarpConfig("producer", 4), CudaWarpConfig("consumer", 4)]
        ):
            for task in cuda_tasks(0, 1):
                with CudaWarps(name=warp_name):
                    for tid in cuda_threads(0, 32):
                        foo = 1.0

    return simplify(test_proc)


def test_cuda_warps_valid_name_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_cuda_warps_name, golden, warp_name="producer")


def test_cuda_warps_invalid_name_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_warps_name, warp_name="nonexistent")
    msg = str(exc.value)
    assert "nonexistent" in msg or "warp name" in msg.lower()


# =============================================================================
# named CudaWarps requires CTA not to be subdivided
# =============================================================================


def mkproc_cuda_warps_subdivided_cta(subdivide_first=True):
    if subdivide_first:

        @proc
        def test_proc(foo: f32[256] @ CudaGmemLinear):
            with CudaDeviceFunction(
                warp_config=[
                    CudaWarpConfig("producer", 4),
                    CudaWarpConfig("consumer", 4),
                ]
            ):
                for task in cuda_tasks(0, 1):
                    # First subdivide the CTA with cuda_threads
                    for outer in cuda_threads(0, 2, unit=cuda_warp):
                        # Then try to use named warps - should fail
                        with CudaWarps(name="producer"):
                            for tid in cuda_threads(0, 32):
                                foo[tid] = 1.0

    else:

        @proc
        def test_proc(foo: f32[256] @ CudaGmemLinear):
            with CudaDeviceFunction(
                warp_config=[
                    CudaWarpConfig("producer", 4),
                    CudaWarpConfig("consumer", 4),
                ]
            ):
                for task in cuda_tasks(0, 1):
                    # Use named warps at top level (no subdivision)
                    with CudaWarps(name="producer"):
                        for tid in cuda_threads(0, 128):
                            foo[tid] = 1.0

    return simplify(test_proc)


def test_cuda_warps_no_subdivide_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_cuda_warps_subdivided_cta, golden, subdivide_first=False)


def test_cuda_warps_subdivided_cta_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_warps_subdivided_cta, subdivide_first=True)
    msg = str(exc.value)
    assert "subdivided" in msg.lower() or "CudaWarps" in msg


# =============================================================================
# CudaWarps.hi out-of-range
# =============================================================================


def mkproc_cuda_warps_hi(hi_value=2):
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(
            warp_config=[CudaWarpConfig("producer", 2), CudaWarpConfig("consumer", 2)]
        ):
            for task in cuda_tasks(0, 1):
                # producer has 2 warps
                with CudaWarps(name="producer", hi=hi_value):
                    for tid in cuda_threads(0, 32 * hi_value):
                        foo = 1.0

    return simplify(test_proc)


def test_cuda_warps_hi_valid_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_cuda_warps_hi, golden, hi_value=2)


def test_cuda_warps_hi_out_of_range_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_warps_hi, hi_value=10)
    msg = str(exc.value)
    assert "out-of-range" in msg or "10" in msg or "producer" in msg


# =============================================================================
# nested CudaWarps cannot change warp name
# =============================================================================


def mkproc_nested_cuda_warps(inner_name="producer"):
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(
            warp_config=[CudaWarpConfig("producer", 4), CudaWarpConfig("consumer", 4)]
        ):
            for task in cuda_tasks(0, 1):
                with CudaWarps(name="producer"):
                    # Nested CudaWarps - same name is ok, different name fails
                    with CudaWarps(name=inner_name, lo=0, hi=2):
                        for tid in cuda_threads(0, 64):
                            foo = 1.0

    return simplify(test_proc)


def test_nested_cuda_warps_same_name_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_nested_cuda_warps, golden, inner_name="producer")


def test_nested_cuda_warps_change_name_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_nested_cuda_warps, inner_name="consumer")
    msg = str(exc.value)
    assert "nested" in msg.lower() or "change" in msg.lower() or "producer" in msg


# =============================================================================
# nested CudaWarps must define lo and hi explicitly
# =============================================================================


def mkproc_nested_cuda_warps_lo_hi(provide_hi=True):
    if provide_hi:

        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with CudaDeviceFunction(
                warp_config=[
                    CudaWarpConfig("producer", 4),
                    CudaWarpConfig("consumer", 4),
                ]
            ):
                for task in cuda_tasks(0, 1):
                    with CudaWarps(name="producer"):
                        # Nested CudaWarps with both lo and hi
                        with CudaWarps(lo=0, hi=2):
                            for tid in cuda_threads(0, 64):
                                foo = 1.0

    else:

        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with CudaDeviceFunction(
                warp_config=[
                    CudaWarpConfig("producer", 4),
                    CudaWarpConfig("consumer", 4),
                ]
            ):
                for task in cuda_tasks(0, 1):
                    with CudaWarps(name="producer"):
                        # Nested CudaWarps without hi - should fail
                        with CudaWarps(lo=0):
                            for tid in cuda_threads(0, 32):
                                foo = 1.0

    return simplify(test_proc)


def test_nested_cuda_warps_with_lo_hi_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_nested_cuda_warps_lo_hi, golden, provide_hi=True)


def test_nested_cuda_warps_missing_hi_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_nested_cuda_warps_lo_hi, provide_hi=False)
    msg = str(exc.value)
    assert "nested" in msg.lower() or "lo" in msg.lower() or "hi" in msg.lower()
