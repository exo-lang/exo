"""Tests for error conditions in spork/barrier_usage.py

These tests cover error paths related to barrier guarding.
"""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.stdlib.scheduling import *


# =============================================================================
# cannot have guarded_by when barrier mechanism has supports_guards=False
# =============================================================================


def mkproc_guarded_by_unsupported():
    """CudaCommitGroup does not support guards (supports_guards=False)"""

    @proc
    def test_proc(foo: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 32):
                    bar1: barrier @ CudaCommitGroup
                    # Try to create bar2 guarded by bar1, but CudaCommitGroup
                    # doesn't support guarding
                    # Syntax: barrier(guarded_by_name) @ BarrierMechanism
                    bar2: barrier(bar1) @ CudaCommitGroup
                    Arrive(Sm80_cp_async, 1) >> bar1
                    Await(bar1, cuda_in_order, 0)
                    Arrive(Sm80_cp_async, 1) >> bar2
                    Await(bar2, cuda_in_order, 0)

    return simplify(test_proc)


def test_guarded_by_unsupported(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_guarded_by_unsupported)
    msg = str(exc.value)
    assert "guarded_by" in msg and "supports_guards" in msg


# =============================================================================
# cannot have guarded_by when it already guards something else
# =============================================================================


def mkproc_guarded_by_already_guards():
    """Try to insert into an existing guard chain incorrectly"""

    @proc
    def test_proc(foo: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar1: barrier @ CudaMbarrier
                bar2: barrier(bar1) @ CudaMbarrier
                # bar1 already guards bar2, now trying bar3 guarded by bar1
                # which would break the chain
                bar3: barrier(bar1) @ CudaMbarrier
                for tid in cuda_threads(0, 32):
                    Arrive(cuda_in_order, 1) >> bar1
                    Await(bar1, cuda_in_order, ~1)
                    Arrive(cuda_in_order, 1) >> bar2
                    Await(bar2, cuda_in_order, ~1)

    return simplify(test_proc)


def test_guarded_by_already_guards(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_guarded_by_already_guards)
    msg = str(exc.value)
    assert "guarded_by" in msg and "guards" in msg


# =============================================================================
# cannot have guarded_by due to BarrierMechanism mismatch
# =============================================================================


def mkproc_guarded_by_mechanism_mismatch():
    """TODO: This test requires two different barrier mechanisms that both
    support guards. Currently only CudaMbarrier supports guards.
    This test is a stub until another guard-supporting barrier is added.
    """
    pass


def test_guarded_by_mechanism_mismatch():
    # TODO: Implement when there are multiple barrier mechanisms supporting guards
    # Currently only CudaMbarrier has supports_guards=True
    pytest.skip("No second barrier mechanism with supports_guards=True available")


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
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, arrive_n) >> bar[w]
                    Await(bar[w], cuda_in_order, ~1)

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
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    # First Arrive uses cuda_in_order
                    Arrive(cuda_in_order, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, ~1)
                    # Second Arrive - sync-tl determined by parameter
                    Arrive(second_sync_tl, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, ~1)

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
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, 1) >> bar[w]
                    # First Await uses cuda_in_order
                    Await(bar[w], cuda_in_order, ~1)
                    Arrive(cuda_in_order, 1) >> bar[w]
                    # Second Await - sync-tl determined by parameter
                    Await(bar[w], second_sync_tl, ~1)

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
# CudaMbarrier requires negative Await N (barrier_usage.py line 158)
# =============================================================================


def mkproc_mbarrier_await_n(await_n):
    """CudaMbarrier requires N < 0 (e.g. ~0, ~1) for Await"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, await_n)

    return simplify(test_proc)


def test_mbarrier_await_n_positive(compiler):
    # ~1 (which is -2) is valid for CudaMbarrier
    compiler.cuda_cpu_test(mkproc_mbarrier_await_n, await_n=~1)


def test_mbarrier_await_n_negative(compiler):
    with pytest.raises(Exception) as exc:
        # 0 is invalid for CudaMbarrier (requires N < 0)
        compiler.cuda_cpu_test(mkproc_mbarrier_await_n, await_n=0)
    msg = str(exc.value)
    assert "CudaMbarrier" in msg and "N < 0" in msg


# =============================================================================
# CudaCommitGroup requires non-negative Await N (barrier_usage.py line 161)
# =============================================================================


def mkproc_commit_group_await_n(await_n):
    """CudaCommitGroup requires N >= 0 for Await"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 32):
                    cg: barrier @ CudaCommitGroup
                    Arrive(Sm80_cp_async, 1) >> cg
                    Await(cg, cuda_in_order, await_n)

    return simplify(test_proc)


def test_commit_group_await_n_positive(compiler):
    # 0 is valid for CudaCommitGroup
    compiler.cuda_cpu_test(mkproc_commit_group_await_n, await_n=0)


def test_commit_group_await_n_negative(compiler):
    with pytest.raises(Exception) as exc:
        # ~0 (which is -1) is invalid for CudaCommitGroup (requires N >= 0)
        compiler.cuda_cpu_test(mkproc_commit_group_await_n, await_n=~0)
    msg = str(exc.value)
    assert "CudaCommitGroup" in msg and "N >= 0" in msg


# =============================================================================
# Missing Arrive for barrier (barrier_usage.py line 324)
# =============================================================================


def mkproc_missing_arrive():
    """Barrier with Await but no Arrive"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    # Only Await, no Arrive
                    Await(bar[w], cuda_in_order, ~1)

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
                bar: barrier[1] @ CudaMbarrier
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
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    pass

    return simplify(test_proc)


def test_missing_both(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_missing_both)
    msg = str(exc.value)
    assert "missing" in msg.lower() and "Arrive" in msg and "Await" in msg


# =============================================================================
# Uniform Await N requirement for CudaMbarrier (barrier_usage.py line 167)
# =============================================================================


def mkproc_mbarrier_non_uniform_await_n(second_await_n):
    """CudaMbarrier requires all Awaits to have same N (uniform_await_N trait)"""

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    Arrive(cuda_in_order, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, ~1)  # First Await with N = ~1
                    Arrive(cuda_in_order, 1) >> bar[w]
                    Await(bar[w], cuda_in_order, second_await_n)  # Second Await

    return simplify(test_proc)


def test_mbarrier_uniform_await_n_positive(compiler):
    # Same N values are valid
    compiler.cuda_cpu_test(mkproc_mbarrier_non_uniform_await_n, second_await_n=~1)


def test_mbarrier_non_uniform_await_n_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Different N values are invalid for CudaMbarrier
        compiler.cuda_cpu_test(mkproc_mbarrier_non_uniform_await_n, second_await_n=~2)
    msg = str(exc.value)
    assert "incompatible" in msg.lower() and "N" in msg and "uniform" in msg.lower()


# =============================================================================
# Multiple Arrives with incompatible multicasts (barrier_usage.py line 106)
# =============================================================================


def test_arrive_incompatible_multicasts():
    # TODO: Requires multicast syntax (>> bar[...] >> bar[..., :])
    # which is used in cluster scenarios
    pytest.skip("Multicast test requires cluster configuration")


# =============================================================================
# Await with multicast is forbidden (barrier_usage.py line 170)
# =============================================================================


def test_await_multicast_forbidden():
    # TODO: Await does not support multicast syntax
    # Error: "multicast is for Arrive, not Await"
    pytest.skip("Await multicast test requires multicast syntax understanding")


# =============================================================================
# Guarding order violations (barrier_usage.py check_guarding function)
# These errors occur when using guarded_by barriers incorrectly
# =============================================================================


def test_guarding_arrive_before_await():
    # TODO: barrier_usage.py line 418-421
    # "expect {get_await_str()} before {s}"
    # When await_first is True but Arrive comes before Await
    pytest.skip("Guarding order test requires guarded_by pattern")


def test_guarding_await_before_arrive():
    # TODO: barrier_usage.py line 428-433
    # "expect {get_arrive_str()} before {s}"
    # When await_first is False but Await comes before Arrive
    pytest.skip("Guarding order test requires guarded_by pattern")


def test_guarding_sync_inside_seq_loop():
    # TODO: barrier_usage.py line 405-414
    # "forbidden here when Await->Arrive sees usage outside"
    # Sync inside sequential loop when guarding requires ordering
    pytest.skip("Guarding order test requires guarded_by pattern")


def test_guarding_unmatched_await():
    # TODO: barrier_usage.py line 445-448
    # "without subsequent {get_arrive_str()} in body"
    pytest.skip("Guarding order test requires guarded_by pattern")


def test_guarding_unmatched_arrive():
    # TODO: barrier_usage.py line 449-452
    # "without subsequent {get_await_str()} in body"
    pytest.skip("Guarding order test requires guarded_by pattern")
