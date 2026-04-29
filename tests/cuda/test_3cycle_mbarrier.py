# fmt: off
# !!!!!! See WARNING for why fmt: off in this file !!!!!!
from __future__ import annotations

import inspect
import pytest

from exo import proc, ring_buffer_by
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *

from exo.spork import excut
from exo.core.LoopIR import get_global_debug_log


# This will be used to test the 3 guard cycle case and guarding
# failures due to thread mismatches.
# Note, I think Claude generated some other barrier guard tests,
# but not involving thread mismatch issues.
def mkproc_3cycle_mbarriers(
    A_delay,
    B_delay,
    C_delay,
    wrong_box=False,
    wrong_offset=False,
    wrong_thread_pitch=False,
    wrong_duplicate_await=False,
):
    warp_config = [
        CudaWarpConfig("A", 2),
        CudaWarpConfig("B", 2),
        CudaWarpConfig("C", 8),
    ]

    if wrong_offset:
        test_2, test_4 = 0, 2
    elif wrong_box:
        test_2, test_4 = 2, 3
    else:
        test_2, test_4 = 2, 4

    if wrong_thread_pitch:
        test_warpgroup = 2 * cuda_warp
        test_2, test_4 = 0, 1
    else:
        test_warpgroup = cuda_warpgroup

    if wrong_duplicate_await:
        test_0, test_1 = 1, 2
    else:
        test_0, test_1 = 0, 1

    n_iters = 25
    depth = A_delay + B_delay + C_delay


    # WARNING, the goldens for get_remarks_golden() are sensitive
    # to line numbers of the proc source code, hence this assert.
    # Update the assert and the goldens if you have to.
    # In general, testing debug logging is super unpleasant.
    # However, this is critical for enhancing the UX for Exo,
    # so I want at least some minimal test coverage for it.
    assert inspect.currentframe().f_lineno == 65

    @proc
    def proc_3cycle_mbarriers():
        with CudaDeviceFunction(warp_config=warp_config):
            for task in cuda_tasks(0, 1):
                AtoB: barrier[2, (n_iters + B_delay) @ ring_buffer_by(depth)] @ CudaMbarrierPreArrive(B_delay)
                BtoC: barrier[2, (n_iters + C_delay) @ ring_buffer_by(depth)] @ CudaMbarrierPreArrive(C_delay)
                CtoA: barrier[2, (n_iters + A_delay) @ ring_buffer_by(depth)] @ CudaMbarrierPreArrive(A_delay)
                for n in seq(0, n_iters):
                    with CudaWarps(name="B"):
                        for w in cuda_threads(0, 2, unit=cuda_warp):
                            Await(AtoB[w, n], cuda_in_order, 0)
                            Arrive(cuda_in_order) >> BtoC[w, n + C_delay]
                    with CudaWarps(name="C"):
                        for wg in cuda_threads(0, 2, unit=test_warpgroup):
                            with CudaWarps(test_2, test_4):
                                Await(BtoC[wg, n], cuda_in_order, 0)
                        for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                            with CudaWarps(2, 4):
                                Arrive(cuda_in_order) >> CtoA[wg, n + A_delay]
                    with CudaWarps(name="A"):
                        # Single thread is valid!
                        if n == 0:
                            for tid in cuda_threads(0, 2, unit=cuda_thread):
                                Await(CtoA[tid, n], cuda_in_order, 0)
                                Arrive(cuda_in_order) >> AtoB[tid, n + B_delay]
                        else:
                            with CudaWarps(test_0, test_1):
                                for tid in cuda_threads(0, 2, unit=cuda_thread):
                                    Await(CtoA[tid, n], cuda_in_order, 0)
                                    Arrive(cuda_in_order) >> AtoB[tid, n + B_delay]


    proc_3cycle_mbarriers.sync_check()

    return proc_3cycle_mbarriers


def get_remarks_golden(compiler, proc_name):
    # Fragile test utility we rarely use to get the contents
    # of the debug logs that exocc would have written out.
    # Anytime we add more logging, the golden will change.
    # We just want to force some logging code coverage.
    debug_log = get_global_debug_log()
    debug_log.write_all_impl()
    with open(str(compiler.workdir / "debug" / f"{proc_name}-analysis.py")) as f:
        return f.read()


def mkref_3cycle_mbarriers(
    xrg: excut.ExcutReferenceGenerator,
    A_delay,
    B_delay,
    C_delay,
):
    xrg.begin_cuda()
    mbarriers = xrg.new_varname("mbarriers")
    depth = A_delay + B_delay + C_delay

    def Await(bar_name, n, i):
        assert i < 2
        if bar_name == "AtoB":
            delay = B_delay
            var_idx = 0
        elif bar_name == "BtoC":
            delay = C_delay
            var_idx = 1
        elif bar_name == "CtoA":
            delay = A_delay
            var_idx = 2
        else:
            assert 0

        if n >= delay:
            r = (n - delay) % depth
            parity = ((n - delay) // depth) & 1
            var = mbarriers[var_idx, i, r]
            xrg("mbarrier.test_wait.parity.acquire.cta.shared::cta.b64", var, parity)

    for task in xrg.stride_blockIdx(1):
        # Thread 0 sets up mbarriers
        with xrg.permuted():
            for r in range(0, depth):
                for i in range(0, 2):
                    # AtoB
                    xrg("mbarrier.init.shared::cta.b64", mbarriers[0, i, r], 1)
                    # BtoC
                    xrg("mbarrier.init.shared::cta.b64", mbarriers[1, i, r], 32)
                    # CtoA
                    xrg("mbarrier.init.shared::cta.b64", mbarriers[2, i, r], 64)
        for n in range(0, 25):
            # with CudaWarps("B")
            for w in xrg.stride_threadIdx(2, stride=32, offset=64):
                for lane in xrg.stride_threadIdx(32):
                    Await("AtoB", n, w)
                    xrg("mbarrier.arrive.shared::cta.b64", mbarriers[1, w, n % depth])
            # with CudaWarps("C")
            for wg in xrg.stride_threadIdx(2, stride=128, offset=128):
                # with CudaWarps(2, 4)
                for lane in xrg.stride_threadIdx(64, stride=1, offset=64):
                    Await("BtoC", n, wg)
                    xrg("mbarrier.arrive.shared::cta.b64", mbarriers[2, wg, n % depth])
            # with CudaWarps("A")
            for tid in xrg.stride_threadIdx(2, stride=1, offset=0):
                Await("CtoA", n, tid)
                xrg("mbarrier.arrive.shared::cta.b64", mbarriers[0, tid, n % depth])
    xrg.end_cuda()


def test_3cycle_mbarriers_excut(compiler_Sm80):
    compiler_Sm80.excut_test(
        mkproc_3cycle_mbarriers,
        mkref_3cycle_mbarriers,
        A_delay=1,
        B_delay=2,
        C_delay=3,
    )


def test_3cycle_mbarriers_positive(compiler, golden):
    compiler.cuda_cpu_test(
        mkproc_3cycle_mbarriers, A_delay=1, B_delay=2, C_delay=3, golden=golden
    )


def test_3cycle_mbarriers_wrong_box(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers, A_delay=1, B_delay=2, C_delay=3, wrong_box=True
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "HAZARD" in msg


def test_3cycle_mbarriers_wrong_offset(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers, A_delay=1, B_delay=2, C_delay=3, wrong_offset=True
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "HAZARD" in msg


def test_3cycle_mbarriers_wrong_thread_pitch(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers,
            A_delay=1,
            B_delay=2,
            C_delay=3,
            wrong_thread_pitch=True,
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "HAZARD" in msg


def test_3cycle_mbarriers_wrong_duplicate_await(compiler, golden):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(
            mkproc_3cycle_mbarriers,
            A_delay=1,
            B_delay=2,
            C_delay=3,
            wrong_duplicate_await=True,
        )

    assert golden == get_remarks_golden(compiler, "proc_3cycle_mbarriers")

    msg = str(exc.value)
    assert "HAZARD" in msg
