import pytest

from exo import *

from exo.core.LoopIR import get_global_debug_log
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *

from exo.platforms.Sm90.tk_gemm_util import (
    handwrite_gemm,
    GemmTestBug,
    GemmConfig,
    L_divisor,
    M_divisor,
    N_divisor,
    K_cluster_divisor,
)

from dataclasses import replace  # Name conflict with exo.stdlib.scheduling...

from typing import List, Dict, Set, Tuple, Optional, Union


base_config = GemmConfig(ncta_M=2, A_major="row", B_major="col", C_major="row")


def sync_check_helper(
    compiler,
    gemm: Procedure,
    exception_text="",
    error_remarks: List[str] = [],
    *,
    L=1,
    M=960,
    N=800,
    K_cluster=720,
    K_split,
):
    """Run sync_check on gemm proc.

    If an empty exception_text is given, expect no sync error.
    Otherwise, expect a sync error, and expect each str
    in error_remarks to appear as a substring of the debug log written
    and exception_text to be a substring of the Exception's included text.

    """

    def run_check():
        gemm.sync_check(L=L, M=M, N=N, K_cluster=K_cluster, K_split=K_split)

    if not exception_text:
        run_check()
    else:
        with pytest.raises(Exception) as exc:
            run_check()
        assert exception_text in str(exc.value)

        proc_name_with_sizes = "-".join(
            [str(gemm.name())] + [str(sz) for sz in (L, M, N, K_split, K_cluster)]
        )

        debug_log = get_global_debug_log()
        debug_log.write_all_impl()
        with open(
            str(compiler.workdir / "debug" / f"{proc_name_with_sizes}-sync-error.py")
        ) as f:
            actual_remarks = f.read()
            for remark in error_remarks:
                assert remark in actual_remarks


def test_ping_pong_positive(compiler):
    config = replace(base_config, ping_pong=True)
    gemm = handwrite_gemm(config)
    sync_check_helper(compiler, gemm, K_split=1)


def test_split_k_positive(compiler):
    config = replace(base_config, enable_split_k=True)
    gemm = handwrite_gemm(config)
    sync_check_helper(compiler, gemm, K_split=2)


def test_coop_positive(compiler):
    config = base_config
    gemm = handwrite_gemm(config)
    sync_check_helper(compiler, gemm, K_split=1)


def test_wrong_K_split(compiler):
    """Use non-split-K kernel as split-K: there's a WAW hazard on the C output.

    The fact that we're allowed to write the non-split-K kernel like this at
    all is an artifact of how sync_check works only with fixed sizes for
    now. If we have static checking, it should be flagged unsafe
    (unless we assume K_split=1 statically).

    """
    config = base_config
    gemm = handwrite_gemm(config)
    exc = "WAW HAZARD"
    error_remarks = ["WAW HAZARD @ C[", "task_k = 1"]
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=2)


def test_coop_wrong_wgmma_cg(compiler):
    """Use N=2 instead of N=1 for wgmma commit group wait.

    This should cause a hazard as there will be too much latency in the
    consumer -> producer syncs.

    For the cooperative kernel, this will manifest as a write-after-read
    hazard, when the producer tries to write to the 0th ring buffer slot
    again, when iter_k=ring_depth.

    """
    ring_depth = 4
    config = replace(base_config, bug=GemmTestBug.wrong_wgmma_cg, ring_depth=ring_depth)
    gemm = handwrite_gemm(config)
    exc = "WAR HAZARD"
    error_remarks = ["WAR HAZARD @ A_smem[0, 0, 0, ", f"iter_k = {ring_depth}"]
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=1)


def test_ping_pong_wrong_wgmma_cg(compiler):
    """Use N=2 instead of N=1 for wgmma commit group wait.

    This should cause a hazard as there will be too much latency in the
    consumer -> producer syncs.

    For the ping-pong kernel, this will manifest as a
    free_missing_arrive error. When the Arrive >> war happens,
    the VisRecord for the SMEM is not actually syncs-with the
    Arrive (because the commit group was faulty and didn't wait
    for the wgmma). Eventually the abstract machine will detect
    that this VisRecord was supposed to have a pending await,
    but did not.

    """
    ring_depth = 4
    config = replace(
        base_config,
        bug=GemmTestBug.wrong_wgmma_cg,
        ring_depth=ring_depth,
        ping_pong=True,
    )
    gemm = handwrite_gemm(config)
    exc = "free_missing_arrive HAZARD"
    error_remarks = ["free_missing_arrive HAZARD @ free(B_smem["]  # A_smem OK too
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=1)


def test_coop_missing_cluster_sync_before_epilogue(compiler):
    config = replace(
        base_config, bug=GemmTestBug.coop_missing_cluster_sync_before_epilogue
    )
    gemm = handwrite_gemm(config)
    exc = "free_missing_sync HAZARD"
    error_remarks = ["free_missing_sync HAZARD @ free(B_smem["]  # A_smem OK too
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=1)


def test_coop_missing_cluster_sync_after_epilogue(compiler):
    config = replace(
        base_config, bug=GemmTestBug.coop_missing_cluster_sync_after_epilogue
    )
    gemm = handwrite_gemm(config)
    exc = "free_missing_sync HAZARD"
    error_remarks = ["free_missing_sync HAZARD @ free(C_smem["]
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=1)


def add_cta_sync_before_epilogue(gemm: Procedure, config: GemmConfig):
    """Given the GEMM proc was created with a missing cluster sync before
    the epilogue, modify it so a sync is placed before the epilogue,
    but done per CTA, not per cluster.  This is insufficient since
    free'ing SMEM requires a cluster-wide sync (due to the
    multicasting), unless the clusterDim=1 in the first place.

    """

    assert not config.ping_pong
    assert config.bug == GemmTestBug.coop_missing_cluster_sync_before_epilogue
    C_smem = gemm.find_alloc_or_arg("C_smem")
    gemm = insert_fence(gemm, C_smem.before(), cuda_in_order, cuda_in_order)
    fence = gemm.forward(C_smem).prev()
    gemm = add_loop(gemm, fence, "cta_m", config.ncta_M)
    gemm = add_loop(gemm, fence, "cta_n", config.ncta_N)
    fence = gemm.forward(fence)
    gemm = set_loop_mode(gemm, fence.parent(), CudaThreads(unit=cuda_cta_in_cluster))
    gemm = set_loop_mode(
        gemm,
        fence.parent().parent(),
        CudaThreads(unit=config.ncta_N * cuda_cta_in_cluster),
    )
    print(gemm)
    return gemm


def test_coop_cta_sync_before_epilogue_positive(compiler):
    config = replace(
        base_config,
        ncta_M=1,
        ncta_N=1,
        bug=GemmTestBug.coop_missing_cluster_sync_before_epilogue,
    )
    gemm = handwrite_gemm(config)
    gemm = add_cta_sync_before_epilogue(gemm, config)
    sync_check_helper(compiler, gemm, K_split=1)


def test_coop_cta_sync_before_epilogue_negative(compiler):
    config = replace(
        base_config,
        ncta_M=2,
        ncta_N=2,
        bug=GemmTestBug.coop_missing_cluster_sync_before_epilogue,
    )
    gemm = handwrite_gemm(config)
    gemm = add_cta_sync_before_epilogue(gemm, config)
    exc = "free_missing_sync HAZARD"
    error_remarks = ["free_missing_sync HAZARD @ free(B_smem["]  # A_smem OK too
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=1)


def test_coop_missing_await_raw(compiler):
    config = replace(base_config, bug=GemmTestBug.missing_await_raw)
    gemm = handwrite_gemm(config)
    exc = "RAW HAZARD"
    error_remarks = [
        "RAW HAZARD @ A_smem[0, 0, 1, ",
        "cta_m = 0",
        "cta_n = 0",
        "iter_k = 1",
    ]
    sync_check_helper(compiler, gemm, exc, error_remarks, K_split=1)
