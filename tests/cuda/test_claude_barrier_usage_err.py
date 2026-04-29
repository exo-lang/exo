"""Tests for error conditions in spork/barrier_usage.py

These tests cover error paths related to barrier guarding.
"""

from __future__ import annotations

import pytest

from exo import proc, ring_buffer_by
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.stdlib.scheduling import *


# =============================================================================
# Arrive with N != 1 (barrier_usage.py line 114: "Need N = 1")
# =============================================================================


def mkproc_arrive_n_not_one(arrive_n):
    """Arrive must have N = 1"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                # Barrier distributed per warp to avoid distributed memory errors
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, arrive_n) >> bar[w]
                    Await(bar[w], cuda_in_order, 0)

    return simplify(test_proc)


def test_arrive_n_not_one_positive(compiler):
    # N = 1 is valid
    compiler.cuda_cpu_test(mkproc_arrive_n_not_one, arrive_n=1)


def test_arrive_n_not_one_negative(compiler):
    with pytest.raises(Exception) as exc:
        # N = 2 is invalid
        compiler.cuda_cpu_test(mkproc_arrive_n_not_one, arrive_n=2)
    msg = str(exc.value)
    assert "N=1" in msg and "Arrive" in msg


# =============================================================================
# Multiple Arrives with incompatible sync-tl (barrier_usage.py line 104)
# =============================================================================


def mkproc_arrive_incompatible_sync_tl(second_sync_tl):
    """Multiple Arrives to same barrier must have same sync-tl"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    # First Arrive uses cuda_in_order
                    Arrive(cuda_in_order, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, 0)
                    # Second Arrive - sync-tl determined by parameter
                    Arrive(second_sync_tl, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, 0)

    return simplify(test_proc)


def test_arrive_incompatible_sync_tl_positive(compiler):
    # Same sync-tl is valid
    compiler.cuda_cpu_test(
        mkproc_arrive_incompatible_sync_tl, second_sync_tl=cuda_in_order
    )


def test_arrive_incompatible_sync_tl_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Different sync-tl is invalid
        compiler.cuda_cpu_test(
            mkproc_arrive_incompatible_sync_tl, second_sync_tl=cuda_temporal
        )
    msg = str(exc.value)
    assert (
        "incompatible" in msg.lower() and "sync-tl" in msg.lower() and "Arrive" in msg
    )


# =============================================================================
# Multiple Awaits with incompatible sync-tl (barrier_usage.py line 152)
# =============================================================================


def mkproc_await_incompatible_sync_tl(second_sync_tl):
    """Multiple Awaits from same barrier must have same sync-tl"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, 1) >> bar[w]
                    # First Await uses cuda_in_order
                    Await(bar[w], cuda_in_order, 0)
                    Arrive(cuda_in_order, 1) >> bar[w]
                    # Second Await - sync-tl determined by parameter
                    Await(bar[w], second_sync_tl, 0)

    return simplify(test_proc)


def test_await_incompatible_sync_tl_positive(compiler):
    # Same sync-tl is valid
    compiler.cuda_cpu_test(
        mkproc_await_incompatible_sync_tl, second_sync_tl=cuda_in_order
    )


def test_await_incompatible_sync_tl_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Different sync-tl is invalid
        compiler.cuda_cpu_test(
            mkproc_await_incompatible_sync_tl, second_sync_tl=cuda_temporal
        )
    msg = str(exc.value)
    assert "incompatible" in msg.lower() and "sync-tl" in msg.lower() and "Await" in msg


# =============================================================================
# CudaMbarrier requires Await N = 0
# =============================================================================


def mkproc_mbarrier_await_n(await_n):
    """CudaMbarrier requires N = 0 for Await"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, await_n)

    return simplify(test_proc)


def test_mbarrier_await_n_positive(compiler):
    compiler.cuda_cpu_test(mkproc_mbarrier_await_n, await_n=0)


def test_mbarrier_await_n_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_mbarrier_await_n, await_n=3)
    msg = str(exc.value)
    assert "CudaMbarrier" in msg and "N = 0" in msg


# =============================================================================
# Sm80_CommitGroup requires non-negative Await N (barrier_usage.py line 161)
# =============================================================================


def mkproc_commit_group_await_n(await_n):
    """Sm80_CommitGroup requires N >= 0 for Await"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 32):
                    cg: barrier @ Sm80_CommitGroup
                    Arrive(Sm80_cp_async, 1) >> cg
                    Await(cg, cuda_in_order, await_n)

    return simplify(test_proc)


def test_commit_group_await_n_positive(compiler):
    # 0 is valid for Sm80_CommitGroup
    compiler.cuda_cpu_test(mkproc_commit_group_await_n, await_n=0)


def test_commit_group_await_n_positive_3(compiler):
    # 3 is valid for Sm80_CommitGroup
    compiler.cuda_cpu_test(mkproc_commit_group_await_n, await_n=3)


def test_commit_group_await_n_negative(compiler):
    with pytest.raises(Exception) as exc:
        # ~0 (which is -1) is invalid for Sm80_CommitGroup (requires N >= 0)
        compiler.cuda_cpu_test(mkproc_commit_group_await_n, await_n=~0)
    msg = str(exc.value)
    assert "Sm80_CommitGroup" in msg and "N >= 0" in msg


# =============================================================================
# Missing Arrive for barrier (barrier_usage.py line 324)
# =============================================================================


def mkproc_missing_arrive():
    """Barrier with Await but no Arrive"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    # Only Await, no Arrive
                    Await(bar[w], cuda_in_order, 0)

    return simplify(test_proc)


def test_missing_arrive(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_missing_arrive)
    msg = str(exc.value)
    assert "missing" in msg.lower() and "Arrive" in msg


# =============================================================================
# Missing Await for barrier (barrier_usage.py line 326)
# =============================================================================


def mkproc_missing_await():
    """Barrier with Arrive but no Await"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    # Only Arrive, no Await
                    Arrive(cuda_in_order, 1) >> bar[w]

    return simplify(test_proc)


def test_missing_await(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_missing_await)
    msg = str(exc.value)
    assert "missing" in msg.lower() and "Await" in msg


# =============================================================================
# Missing both Arrive and Await (barrier_usage.py line 320-321)
# =============================================================================


def mkproc_missing_both():
    """Barrier declared but never used"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1 @ ring_buffer_by(1)] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    pass

    return simplify(test_proc)


def test_missing_both(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_missing_both)
    msg = str(exc.value)
    assert "missing" in msg.lower() and "Arrive" in msg and "Await" in msg


# =============================================================================
# Multiple Arrives with incompatible multicasts (barrier_usage.py line 106)
# =============================================================================


def mkproc_arrive_incompatible_multicasts(same_multicast):
    """Test that multiple Arrives must have identical multicast patterns.

    With cluster configuration and 2D barrier array:
    - First Arrive uses >> bar[m, n] >> bar[m, :] (multicast along n)
    - Second Arrive must use the same pattern

    same_multicast=True: Both Arrives use same multicast pattern (valid)
    same_multicast=False: Second Arrive uses different multicast pattern (invalid)
    """
    M_CTA = 2
    N_CTA = 2

    @proc
    def test_proc():
        with CudaDeviceFunction(clusterDim=M_CTA * N_CTA, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[M_CTA, N_CTA, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                    for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                        # First Arrive with multicast along n dimension
                        (
                            Arrive(cuda_in_order, 1)
                            >> bar[m_cta, n_cta, 0]
                            >> bar[m_cta, :, 0]
                        )
                        Await(bar[m_cta, n_cta, 0], cuda_in_order, 0)
                        if same_multicast:
                            # Valid: same multicast pattern (along n)
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, n_cta, 0]
                                >> bar[m_cta, :, 0]
                            )
                        else:
                            # Invalid: different multicast pattern (along m instead of n)
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, n_cta, 0]
                                >> bar[:, n_cta, 0]
                            )
                        Await(bar[m_cta, n_cta, 0], cuda_in_order, 0)

    return simplify(test_proc)


def test_arrive_incompatible_multicasts_positive(compiler):
    # Valid: same multicast pattern on both Arrives
    compiler.cuda_cpu_test(mkproc_arrive_incompatible_multicasts, same_multicast=True)


def test_arrive_incompatible_multicasts_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: different multicast patterns
        compiler.cuda_cpu_test(
            mkproc_arrive_incompatible_multicasts, same_multicast=False
        )
    msg = str(exc.value)
    assert "incompatible" in msg.lower() and "multicast" in msg.lower()


# =============================================================================
# Await with multicast is forbidden (barrier_usage.py line 170)
# =============================================================================


def mkproc_await_multicast_forbidden(use_multicast_in_await):
    """Test that Await cannot use multicast syntax.

    Multicast (using : in barrier index) is only valid for Arrive, not Await.

    use_multicast_in_await=False: Await uses point index bar[m, n] (valid)
    use_multicast_in_await=True: Await uses interval bar[m, :] (invalid)
    """
    M_CTA = 2
    N_CTA = 2

    @proc
    def test_proc():
        with CudaDeviceFunction(clusterDim=M_CTA * N_CTA, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[M_CTA, N_CTA, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                    for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                        # Arrive with multicast
                        (
                            Arrive(cuda_in_order, 1)
                            >> bar[m_cta, n_cta, 0]
                            >> bar[m_cta, :, 0]
                        )
                        if use_multicast_in_await:
                            # Invalid: Await with multicast syntax
                            # Error: "multicast is for Arrive, not Await"
                            Await(bar[m_cta, :, 0], cuda_in_order, 0)
                        else:
                            # Valid: Await with point index
                            Await(bar[m_cta, n_cta, 0], cuda_in_order, 0)

    return simplify(test_proc)


def test_await_multicast_forbidden_positive(compiler):
    # Valid: Await with point index
    compiler.cuda_cpu_test(
        mkproc_await_multicast_forbidden, use_multicast_in_await=False
    )


def test_await_multicast_forbidden_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: Await with multicast
        compiler.cuda_cpu_test(
            mkproc_await_multicast_forbidden, use_multicast_in_await=True
        )
    msg = str(exc.value)
    # Error message: "at least one trailing barrier expression must have idx[...] be a point, not an interval"
    # or "multicast is for Arrive, not Await"
    assert ("interval" in msg.lower() or "multicast" in msg.lower()) and "Await" in msg


# =============================================================================
# Home barrier expression errors (LoopIR.py home_barrier_expr)
# =============================================================================


def mkproc_home_barrier_different_arrays(same_array):
    """Test that all barrier expressions must use the same barrier variable.

    With multicast syntax, all >> bar[...] expressions must reference the same barrier.

    same_array=True: >> bar[m, n] >> bar[m, :] (valid - same barrier)
    same_array=False: >> bar1[m, n] >> bar2[m, :] (invalid - different barriers)
    """
    M_CTA = 2
    N_CTA = 2
    device_fn = CudaDeviceFunction(clusterDim=M_CTA * N_CTA, blockDim=32)

    if same_array:

        @proc
        def test_proc():
            with device_fn:
                for task in cuda_tasks(0, 1):
                    bar: barrier[M_CTA, N_CTA, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                    for m_cta in cuda_threads(
                        0, M_CTA, unit=N_CTA * cuda_cta_in_cluster
                    ):
                        for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                            # Valid: same barrier array in both expressions
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, n_cta, 0]
                                >> bar[m_cta, :, 0]
                            )
                            Await(bar[m_cta, n_cta, 0], cuda_in_order, 0)

    else:

        @proc
        def test_proc():
            with device_fn:
                for task in cuda_tasks(0, 1):
                    bar1: barrier[M_CTA, N_CTA, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                    bar2: barrier[M_CTA, N_CTA, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                    for m_cta in cuda_threads(
                        0, M_CTA, unit=N_CTA * cuda_cta_in_cluster
                    ):
                        for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                            # Invalid: different barrier arrays
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar1[m_cta, n_cta, 0]
                                >> bar2[m_cta, :, 0]
                            )
                            Await(bar1[m_cta, n_cta, 0], cuda_in_order, 0)
                            Await(bar2[m_cta, n_cta, 0], cuda_in_order, 0)

    return simplify(test_proc)


def test_home_barrier_different_arrays_positive(compiler):
    # Valid: same barrier array in multicast
    compiler.cuda_cpu_test(mkproc_home_barrier_different_arrays, same_array=True)


def test_home_barrier_different_arrays_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: different barrier arrays in multicast
        compiler.cuda_cpu_test(mkproc_home_barrier_different_arrays, same_array=False)
    msg = str(exc.value)
    assert "different" in msg.lower() and "barrier" in msg.lower()


def mkproc_home_barrier_mismatched_points(matching_points):
    """Test that point indices at the same dimension must match across barrier expressions.

    When multiple barrier expressions provide a point for the same dimension,
    they must use the same variable.

    matching_points=True: >> bar[m, n] >> bar[m, :] (valid - m matches)
    matching_points=False: >> bar[m, :] >> bar[n, :] (invalid - m != n at index 0)
    """
    M_CTA = 2
    N_CTA = 2
    device_fn = CudaDeviceFunction(clusterDim=M_CTA * N_CTA, blockDim=32)

    @proc
    def test_proc():
        with device_fn:
            for task in cuda_tasks(0, 1):
                bar: barrier[M_CTA, N_CTA, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                    for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                        if matching_points:
                            # Valid: m_cta matches in both expressions at index 0
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, n_cta, 0]
                                >> bar[m_cta, :, 0]
                            )
                        else:
                            # Invalid: m_cta vs n_cta at index 0
                            # (both are points, but different variables)
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, :, 0]
                                >> bar[n_cta, :, 0]
                            )
                        Await(bar[m_cta, n_cta, 0], cuda_in_order, 0)

    return simplify(test_proc)


def test_home_barrier_mismatched_points_positive(compiler):
    # Valid: point indices match
    compiler.cuda_cpu_test(mkproc_home_barrier_mismatched_points, matching_points=True)


def test_home_barrier_mismatched_points_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: mismatched point variables at same index
        compiler.cuda_cpu_test(
            mkproc_home_barrier_mismatched_points, matching_points=False
        )
    msg = str(exc.value)
    assert "mismatch" in msg.lower() and "idx" in msg.lower()
