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
                bar: barrier[M_CTA, N_CTA] @ CudaMbarrier
                for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                    for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                        # First Arrive with multicast along n dimension
                        Arrive(cuda_in_order, 1) >> bar[m_cta, n_cta] >> bar[m_cta, :]
                        Await(bar[m_cta, n_cta], cuda_in_order, ~0)
                        if same_multicast:
                            # Valid: same multicast pattern (along n)
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, n_cta]
                                >> bar[m_cta, :]
                            )
                        else:
                            # Invalid: different multicast pattern (along m instead of n)
                            (
                                Arrive(cuda_in_order, 1)
                                >> bar[m_cta, n_cta]
                                >> bar[:, n_cta]
                            )
                        Await(bar[m_cta, n_cta], cuda_in_order, ~0)

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
                bar: barrier[M_CTA, N_CTA] @ CudaMbarrier
                for m_cta in cuda_threads(0, M_CTA, unit=N_CTA * cuda_cta_in_cluster):
                    for n_cta in cuda_threads(0, N_CTA, unit=cuda_cta_in_cluster):
                        # Arrive with multicast
                        Arrive(cuda_in_order, 1) >> bar[m_cta, n_cta] >> bar[m_cta, :]
                        if use_multicast_in_await:
                            # Invalid: Await with multicast syntax
                            # Error: "multicast is for Arrive, not Await"
                            Await(bar[m_cta, :], cuda_in_order, ~0)
                        else:
                            # Valid: Await with point index
                            Await(bar[m_cta, n_cta], cuda_in_order, ~0)

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
# Guarding order violations (barrier_usage.py check_guarding function)
# These errors occur when using guarded_by barriers incorrectly
# =============================================================================


def mkproc_guarding_arrive_before_await(correct_order):
    """Test guarding order: when await_first=True, Arrive must come after Await.

    With bar2 guarded_by bar1, they form a guard cycle:
    - bar1.guarded_by = bar2, bar2.guarded_by = bar1
    - For bar1: need Await(bar2) before Arrive >> bar1
    - For bar2: need Await(bar1) before Arrive >> bar2

    correct_order=True: Await(bar2) -> Arrive >> bar1 -> Await(bar1) -> Arrive >> bar2 (valid)
    correct_order=False: Arrive >> bar1 first (invalid - expect Await before Arrive)
    """

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar1: barrier[1] @ CudaMbarrier
                bar2: barrier(bar1)[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    if correct_order:
                        # Valid: await bar2's guard (bar1) first, then arrive to bar2
                        # But actually we need to check bar1's check_guarding too
                        # For bar1: await bar1.guarded_by (bar2), then arrive bar1
                        Await(bar2[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar1[w]
                        Await(bar1[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar2[w]
                    else:
                        # Invalid: Arrive >> bar1 without prior Await(bar2)
                        # Error: "expect Await(bar2, ...) before ..."
                        Arrive(cuda_in_order, 1) >> bar1[w]
                        Await(bar2[w], cuda_in_order, ~1)
                        Await(bar1[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar2[w]

    return simplify(test_proc)


def test_guarding_arrive_before_await_positive(compiler):
    # Valid: Await before Arrive in await-first pattern
    compiler.cuda_cpu_test(mkproc_guarding_arrive_before_await, correct_order=True)


def test_guarding_arrive_before_await_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: Arrive before Await in await-first pattern
        compiler.cuda_cpu_test(mkproc_guarding_arrive_before_await, correct_order=False)
    msg = str(exc.value)
    assert "expect" in msg.lower() and "Await" in msg and "before" in msg.lower()


def mkproc_guarding_await_before_arrive(correct_order):
    """Test guarding order: when await_first=False (arrive-first), Await must come after Arrive.

    For a self-guarding barrier (bar guards itself):
    - If first statement is Arrive, await_first=False
    - Valid pattern: Arrive -> Await -> Arrive -> Await
    - Invalid: Arrive -> Await -> Await (second Await without prior Arrive)

    correct_order=True: Arrive -> Await -> Arrive -> Await (valid)
    correct_order=False: Arrive -> Await -> Await (invalid - expect Arrive before second Await)
    """

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    if correct_order:
                        # Valid: arrive-first pattern with proper alternation
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Await(bar[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Await(bar[w], cuda_in_order, ~1)
                    else:
                        # Invalid: two Awaits in a row after establishing arrive-first
                        # Error: "expect Arrive(...) >> bar before ..."
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Await(bar[w], cuda_in_order, ~1)
                        Await(bar[w], cuda_in_order, ~1)  # Error here!

    return simplify(test_proc)


def test_guarding_await_before_arrive_positive(compiler):
    # Valid: arrive-first pattern with proper alternation
    compiler.cuda_cpu_test(mkproc_guarding_await_before_arrive, correct_order=True)


def test_guarding_await_before_arrive_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: two Awaits without intervening Arrive
        compiler.cuda_cpu_test(mkproc_guarding_await_before_arrive, correct_order=False)
    msg = str(exc.value)
    assert "expect" in msg.lower() and "Arrive" in msg and "before" in msg.lower()


def mkproc_guarding_sync_inside_seq_loop(sync_inside_loop):
    """Test that sync statements are forbidden inside seq loops when there's unmatched sync outside.

    For guard pair (bar1 guards bar2, bar2 guarded_by bar1):
    - Start with Await(bar2) outside the loop (creates unmatched_await)
    - If we then enter a seq loop and try to Arrive >> bar1, that's forbidden
    - The Arrive must be outside the loop to match the Await

    sync_inside_loop=False: sync outside loop (valid)
    sync_inside_loop=True: sync inside seq loop while unmatched sync outside (invalid)
    """

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar1: barrier[1] @ CudaMbarrier
                bar2: barrier(bar1)[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    if sync_inside_loop:
                        # Invalid: Await outside, then Arrive inside seq loop
                        Await(bar2[w], cuda_in_order, ~1)
                        for i in seq(0, 2):
                            # Error: "forbidden here when Await->Arrive sees usage outside"
                            Arrive(cuda_in_order, 1) >> bar1[w]
                            Await(bar1[w], cuda_in_order, ~1)
                            Arrive(cuda_in_order, 1) >> bar2[w]
                            Await(bar2[w], cuda_in_order, ~1)
                    else:
                        # Valid: all syncs outside seq loops
                        Await(bar2[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar1[w]
                        Await(bar1[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar2[w]

    return simplify(test_proc)


def test_guarding_sync_inside_seq_loop_positive(compiler):
    # Valid: syncs outside seq loops
    compiler.cuda_cpu_test(mkproc_guarding_sync_inside_seq_loop, sync_inside_loop=False)


def test_guarding_sync_inside_seq_loop_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: sync inside seq loop when there's unmatched sync outside
        compiler.cuda_cpu_test(
            mkproc_guarding_sync_inside_seq_loop, sync_inside_loop=True
        )
    msg = str(exc.value)
    assert "forbidden" in msg.lower() and "seq" in msg.lower()


def mkproc_guarding_unmatched_await(has_matching_arrive):
    """Test that Await must have a subsequent Arrive in the body.

    For guard pair (bar1 guards bar2):
    - With await-first pattern, we do Await(bar2) then Arrive >> bar1
    - If we only do Await(bar2) without the Arrive >> bar1, error

    has_matching_arrive=True: Await(bar2) followed by Arrive >> bar1 (valid)
    has_matching_arrive=False: Await(bar2) alone (invalid - missing subsequent Arrive)
    """

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar1: barrier[1] @ CudaMbarrier
                bar2: barrier(bar1)[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    if has_matching_arrive:
                        # Valid: Await followed by matching Arrive
                        Await(bar2[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar1[w]
                        Await(bar1[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar2[w]
                    else:
                        # Invalid: Await without subsequent Arrive in body
                        # Error: "... without subsequent Arrive(...) >> bar1 in body"
                        Await(bar2[w], cuda_in_order, ~1)
                        # Missing: Arrive(cuda_in_order, 1) >> bar1[w]
                        # We still need SOME valid Arrive/Await to avoid "missing Arrive"
                        # but the unmatched Await(bar2) causes the error
                        Arrive(cuda_in_order, 1) >> bar2[w]
                        Await(bar1[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar1[w]

    return simplify(test_proc)


def test_guarding_unmatched_await_positive(compiler):
    # Valid: Await followed by matching Arrive
    compiler.cuda_cpu_test(mkproc_guarding_unmatched_await, has_matching_arrive=True)


def test_guarding_unmatched_await_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: Await without subsequent matching Arrive
        compiler.cuda_cpu_test(
            mkproc_guarding_unmatched_await, has_matching_arrive=False
        )
    msg = str(exc.value)
    # Either "without subsequent" or the expect error should trigger
    assert ("without subsequent" in msg.lower() or "expect" in msg.lower()) and (
        "Arrive" in msg or "Await" in msg
    )


def mkproc_guarding_unmatched_arrive(proper_alternation):
    """Test that Arrive must have a subsequent Await in the body (guarding check).

    For a self-guarding barrier with arrive-first pattern:
    - Valid: Arrive -> Await -> Arrive -> Await (proper alternation)
    - Invalid: Arrive -> Arrive -> Await (two Arrives in a row triggers guarding error)

    proper_alternation=True: Arrive -> Await -> Arrive -> Await (valid)
    proper_alternation=False: Arrive -> Arrive -> Await (invalid - expect Await before second Arrive)
    """

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[1] @ CudaMbarrier
                for w in cuda_threads(0, 1, unit=cuda_warp):
                    if proper_alternation:
                        # Valid: Arrive followed by matching Await, then repeat
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Await(bar[w], cuda_in_order, ~1)
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Await(bar[w], cuda_in_order, ~1)
                    else:
                        # Invalid: Two Arrives in a row
                        # Error: "expect Await(bar, ...) before ..."
                        Arrive(cuda_in_order, 1) >> bar[w]
                        Arrive(cuda_in_order, 1) >> bar[w]  # Error here!
                        Await(bar[w], cuda_in_order, ~1)
                        Await(bar[w], cuda_in_order, ~1)

    return simplify(test_proc)


def test_guarding_unmatched_arrive_positive(compiler):
    # Valid: Arrive followed by matching Await with proper alternation
    compiler.cuda_cpu_test(mkproc_guarding_unmatched_arrive, proper_alternation=True)


def test_guarding_unmatched_arrive_negative(compiler):
    with pytest.raises(Exception) as exc:
        # Invalid: Two Arrives in a row (no Await between them)
        compiler.cuda_cpu_test(
            mkproc_guarding_unmatched_arrive, proper_alternation=False
        )
    msg = str(exc.value)
    # Error should be "expect Await(...) before ..."
    assert "expect" in msg.lower() and "Await" in msg and "before" in msg.lower()
