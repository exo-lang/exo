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
