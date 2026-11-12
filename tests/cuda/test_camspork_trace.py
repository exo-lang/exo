"""PROTOTYPE camspork sync env trace tests (endgame_plan/plan_camspork_test.md).

Each test pairs an Exo proc (traced through sync_check/camspork) with a hand-written
reference (camspork_ref.SyncEnvRef), excut-style: thread IDs are computed by hand.
Tests never leave an access without a later sync (the trace only samples sync events).
"""

from __future__ import annotations

import itertools
from typing import Iterable, List

import pytest

from exo import proc, instr, ring_buffer_by
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import simplify, unroll_loop
from exo.spork import timelines
from exo.spork.timelines import (
    cuda_in_order_instr,
    Sm80_cp_async_instr,
    tma_to_gmem_async_instr,
)
from exo.spork.camspork_ref import (
    AccessQuals,
    ElemIdx,
    RingGuard,
    StrElemName,
    StrVarName,
    SyncEnvRef,
    access_quals,
    check_both_fail,
    check_trace,
)

IN_ORDER = timelines.cuda_in_order.get_timeline_set_bits()

# Proc variables named in references (str(Sym) of the procs' variables below).
x = StrVarName("x")
g = StrVarName("g")
smem = StrVarName("smem")
cg = StrVarName("cg")
cs = StrVarName("cs")
bar = StrVarName("bar")
tb = StrVarName("tb")
war = StrVarName("war")
raw = StrVarName("raw")


def smem_quals(**kw) -> AccessQuals:
    return access_quals(CudaSmemLinear, cuda_in_order_instr, **kw)


def gmem_quals(**kw) -> AccessQuals:
    return access_quals(CudaGmemLinear, cuda_in_order_instr, **kw)


def free_smem(ref: SyncEnvRef, var: StrVarName, task=0):
    # SMEM free: every cluster thread must be synchronized with prior accesses.
    ref.free(
        var,
        check_tids=ref.task_threads(task),
        precondition_bits=smem_quals().precondition_bits,
    )


def assert_no_errors(errors):
    assert not errors, "\n".join(errors)


# =============================================================================
# Named warps + non-trivial numeric range (the tricky CudaWarps case)
# =============================================================================


def mkproc_named_numeric_warps(wrong_lo=False):
    # A: warps [0, 2), B: warps [2, 6); 192 threads.
    lo = 0 if wrong_lo else 1

    @proc
    def proc_trace():
        with CudaDeviceFunction(
            warp_config=[CudaWarpConfig("A", 2), CudaWarpConfig("B", 4)]
        ):
            for task in cuda_tasks(0, 1):
                x: f32[2, 32] @ CudaSmemLinear
                # Name restriction + non-trivial numeric range: B warps [1, 3)
                with CudaWarps(lo, 3, name="B"):
                    for w in cuda_threads(0, 2, unit=cuda_warp):
                        for t in cuda_threads(0, 32):
                            x[w, t] = 0
                Fence(cuda_in_order, cuda_in_order)
                with CudaWarps(name="A"):
                    for w in cuda_threads(0, 2, unit=cuda_warp):
                        for t in cuda_threads(0, 32):
                            x[w, t] += 1
                Fence(cuda_in_order, cuda_in_order)

    return proc_trace


def mkref_named_numeric_warps():
    ref = SyncEnvRef([x[0, 0], x[1, 31]], threads_per_task=192)
    task = ref.task_threads()

    def task_body():
        ref.alloc(x, (2, 32))
        for w in range(2):
            for t in range(32):
                ref.mutate(
                    x[w, t], [(3 + w) * 32 + t], quals=smem_quals(convergent=True)
                )
        ref.fence(task, IN_ORDER, IN_ORDER)
        for w in range(2):
            for t in range(32):
                ref.mutate(x[w, t], [w * 32 + t], quals=smem_quals(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, x)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_named_numeric_warps_positive(compiler, tmp_path):
    p = mkproc_named_numeric_warps()
    assert_no_errors(check_trace(p, mkref_named_numeric_warps(), tmp_path))


def test_named_numeric_warps_negative(compiler, tmp_path):
    """Lowering bug stand-in: proc uses B warps [0, 2), reference expects [1, 3)."""
    p = mkproc_named_numeric_warps(wrong_lo=True)
    errors = check_trace(p, mkref_named_numeric_warps(), tmp_path)
    assert errors
    assert "x[0, 0] mutate" in errors[0]


# =============================================================================
# GMEM argument live across task-end and device-end JoinThreads + implicit Fences
# =============================================================================


def mkproc_gmem_join():
    @proc
    def proc_gmem_join(g: f32[2, 32] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, 1):
                for w in cuda_threads(0, 2, unit=cuda_warp):
                    for t in cuda_threads(0, 32):
                        g[w, t] = 0

    return proc_gmem_join


def test_gmem_join_threads(compiler, tmp_path):
    ref = SyncEnvRef([g[0, 0], g[1, 31]], threads_per_task=64)
    ref.alloc(g, (2, 32))

    def task_body():
        for w in range(2):
            for t in range(32):
                ref.mutate(g[w, t], [w * 32 + t], quals=gmem_quals(convergent=True))

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_gmem_join(), ref, tmp_path))


# =============================================================================
# Strided (non-contiguous) thread collectives
# =============================================================================


def mkproc_strided():
    @proc
    def proc_strided():
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                x: f32[4, 32] @ CudaSmemLinear
                # Iteration w2 is executed by warp 2 * w2 (unit = 2 warps).
                for w2 in cuda_threads(0, 4, unit=2 * cuda_warp):
                    for t in cuda_threads(0, 32):
                        x[w2, t] = 0
                Fence(cuda_in_order, cuda_in_order)
                with CudaWarps(1, 2):
                    for t in cuda_threads(0, 32):
                        x[0, t] += 1
                Fence(cuda_in_order, cuda_in_order)

    return proc_strided


def test_strided_collective(compiler, tmp_path):
    ref = SyncEnvRef([x[1, 0], x[3, 31], x[0, 5]], threads_per_task=256)
    task = ref.task_threads()

    def task_body():
        ref.alloc(x, (4, 32))
        for w2 in range(4):
            for t in range(32):
                ref.mutate(x[w2, t], [w2 * 64 + t], quals=smem_quals(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        for t in range(32):
            ref.mutate(x[0, t], [32 + t], quals=smem_quals(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, x)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_strided(), ref, tmp_path))


# =============================================================================
# Async instr (out-of-order qual-tl) + Sm80_CommitGroup Arrive/Await
# =============================================================================


def mkproc_cp_async():
    @proc
    def proc_cp_async(src: f32[32, 4] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                x: f32[32, 4] @ CudaSmemLinear
                for tid in cuda_threads(0, 32):
                    cg: barrier @ Sm80_CommitGroup
                    Sm80_cp_async_f32(x[tid, 0:4], src[tid, 0:4], size=4)
                    Arrive(Sm80_cp_async, 1) >> cg
                    Await(cg, cuda_in_order, 0)
                    x[tid, 0] += 1
                Fence(cuda_in_order, cuda_in_order)

    return proc_cp_async


def test_cp_async_commit_group(compiler, tmp_path):
    ref = SyncEnvRef([x[0, 0], x[0, 3], x[31, 1], cg[()]], threads_per_task=32)
    task = ref.task_threads()
    cp_async = access_quals(CudaSmemLinear, Sm80_cp_async_instr)
    L1 = timelines.Sm80_cp_async.get_timeline_set_bits()

    def task_body():
        ref.alloc(x, (32, 4))
        for tid in range(32):
            ref.alloc_barrier(cg, ())
            # cp.async dst: out-of-order initial qual-tl; TAG 1 since the
            # commit-group Arrive is executed per thread. (src is a read-only
            # GMEM arg: not sync-checked.)
            ref.mutate(x[tid, 0], [tid], extent=(1, 4), tag=1, quals=cp_async)
            ref.arrive([tid], L1, cg[()])
            ref.await_([tid], IN_ORDER, cg[()], 0)
            ref.mutate(x[tid, 0], [tid], quals=smem_quals(convergent=True))
            ref.free_barrier(cg)
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, x)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_cp_async(), ref, tmp_path))


# =============================================================================
# CudaClusterSync Arrive/Await (clusterDim=2)
# =============================================================================


def mkproc_cluster_sync():
    @proc
    def proc_cluster_sync(g: f32[2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=64):
            for task in cuda_tasks(0, 1):
                cs: barrier @ CudaClusterSync
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for t in cuda_threads(0, 1):
                        g[cta] = 0
                Arrive(cuda_in_order, 1) >> cs
                Await(cs, cuda_in_order, 0)
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    with CudaWarps(1, 2):
                        for t in cuda_threads(0, 1):
                            g[1 - cta] += 1

    return proc_cluster_sync


def test_cluster_sync(compiler, tmp_path):
    # Threads are numbered by CTA rank, then thread index in the CTA.
    ref = SyncEnvRef([g[0], g[1], cs[()]], threads_per_task=128)
    task = ref.task_threads()
    ref.alloc(g, (2,))

    def task_body():
        ref.alloc_barrier(cs, (), expected=1, one_shot_await=True)
        for cta in range(2):
            ref.mutate(g[cta], [cta * 64], quals=gmem_quals(convergent=True))
        ref.arrive(task, IN_ORDER, cs[()])
        ref.await_(task, IN_ORDER, cs[()], 0)
        for cta in range(2):
            ref.mutate(g[1 - cta], [cta * 64 + 32], quals=gmem_quals(convergent=True))
        ref.free_barrier(cs)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_cluster_sync(), ref, tmp_path))


# =============================================================================
# Windows: AssignmentRecord copy-on-write
# =============================================================================


@instr
class trace_write_4x4:
    def behavior(dst: [f32][4, 4] @ CudaSmemLinear):
        for i in seq(0, 4):
            for j in seq(0, 4):
                dst[i, j] = 0.0

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        return ["// trace_write_4x4"]


@instr
class trace_write_2x2:
    def behavior(dst: [f32][2, 2] @ CudaSmemLinear):
        for i in seq(0, 2):
            for j in seq(0, 2):
                dst[i, j] = 0.0

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        return ["// trace_write_2x2"]


@instr
class trace_read_4x4:
    def behavior(src: [f32][4, 4] @ CudaSmemLinear):
        pass

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        return ["// trace_read_4x4"]


@instr
class trace_read_2x2:
    def behavior(src: [f32][2, 2] @ CudaSmemLinear):
        pass

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        return ["// trace_read_2x2"]


WINDOW_PROBES = [x[0, 0], x[0, 3], x[2, 1], x[3, 3]]


def mkproc_big_then_small():
    @proc
    def proc_big_then_small():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                x: f32[4, 4] @ CudaSmemLinear
                for t in cuda_threads(0, 1):
                    trace_write_4x4(x[0:4, 0:4])
                Fence(cuda_in_order, cuda_in_order)
                for i in cuda_threads(0, 2, unit=2 * cuda_thread):
                    for j in cuda_threads(0, 2):
                        trace_read_2x2(x[2 * i : 2 * i + 2, 2 * j : 2 * j + 2])
                Fence(cuda_in_order, cuda_in_order)

    return proc_big_then_small


def test_window_big_then_small(compiler, tmp_path):
    """One big window write shares one AssignmentRecord; small reads must de-converge it."""
    ref = SyncEnvRef(WINDOW_PROBES, threads_per_task=32)
    task = ref.task_threads()

    def task_body():
        ref.alloc(x, (4, 4))
        ref.mutate(x[0, 0], [0], extent=(4, 4), quals=smem_quals())
        ref.fence(task, IN_ORDER, IN_ORDER)
        for i in range(2):
            for j in range(2):
                ref.read(
                    x[2 * i, 2 * j], [2 * i + j], extent=(2, 2), quals=smem_quals()
                )
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, x)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_big_then_small(), ref, tmp_path))


def mkproc_small_then_big():
    @proc
    def proc_small_then_big():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                x: f32[4, 4] @ CudaSmemLinear
                for i in cuda_threads(0, 2, unit=2 * cuda_thread):
                    for j in cuda_threads(0, 2):
                        trace_write_2x2(x[2 * i : 2 * i + 2, 2 * j : 2 * j + 2])
                Fence(cuda_in_order, cuda_in_order)
                for t in cuda_threads(0, 1):
                    trace_read_4x4(x[0:4, 0:4])
                Fence(cuda_in_order, cuda_in_order)

    return proc_small_then_big


def test_window_small_then_big(compiler, tmp_path):
    """Distinct per-block state must not be merged by one big window read."""
    ref = SyncEnvRef(WINDOW_PROBES, threads_per_task=32)
    task = ref.task_threads()

    def task_body():
        ref.alloc(x, (4, 4))
        for i in range(2):
            for j in range(2):
                ref.mutate(
                    x[2 * i, 2 * j], [2 * i + j], extent=(2, 2), quals=smem_quals()
                )
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.read(x[0, 0], [0], extent=(4, 4), quals=smem_quals())
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, x)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_small_then_big(), ref, tmp_path))


def mkproc_overlapping():
    @proc
    def proc_overlapping():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                x: f32[4, 4] @ CudaSmemLinear
                for t in cuda_threads(0, 1):
                    trace_write_2x2(x[0:2, 0:2])
                    trace_write_2x2(x[2:4, 2:4])
                Fence(cuda_in_order, cuda_in_order)
                for i in cuda_threads(0, 2, unit=cuda_thread):
                    # Thread 0 reads x[1:3, 1:3]; thread 1 reads x[0:2, 2:4].
                    trace_read_2x2(x[1 - i : 3 - i, 1 + i : 3 + i])
                Fence(cuda_in_order, cuda_in_order)
                for t in cuda_threads(0, 1):
                    trace_write_2x2(x[1:3, 1:3])
                Fence(cuda_in_order, cuda_in_order)
                for t in cuda_threads(0, 1):
                    trace_read_2x2(x[0:2, 0:2])
                Fence(cuda_in_order, cuda_in_order)

    return proc_overlapping


def mkref_overlapping():
    ref = SyncEnvRef(
        [x[1, 1], x[0, 0], x[2, 2], x[1, 2], x[0, 3]],
        threads_per_task=32,
    )
    task = ref.task_threads()

    def task_body():
        ref.alloc(x, (4, 4))
        ref.mutate(x[0, 0], [0], extent=(2, 2), quals=smem_quals())
        ref.mutate(x[2, 2], [0], extent=(2, 2), quals=smem_quals())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.read(x[1, 1], [0], extent=(2, 2), quals=smem_quals())
        ref.read(x[0, 2], [1], extent=(2, 2), quals=smem_quals())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.mutate(x[1, 1], [0], extent=(2, 2), quals=smem_quals())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.read(x[0, 0], [0], extent=(2, 2), quals=smem_quals())
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, x)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_window_overlapping(compiler, tmp_path):
    """Partially overlapping windows, written, read, rewritten."""
    assert_no_errors(check_trace(mkproc_overlapping(), mkref_overlapping(), tmp_path))


# =============================================================================
# Negative: both camspork and the reference must reject (trace not inspected)
# =============================================================================


def mkproc_missing_fence():
    @proc
    def proc_missing_fence():
        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, 1):
                x: f32[2, 32] @ CudaSmemLinear
                for w in cuda_threads(0, 2, unit=cuda_warp):
                    for t in cuda_threads(0, 32):
                        x[w, t] = 0
                # Missing Fence: warp 1 overwrites what warp 0 wrote.
                for w in cuda_threads(0, 2, unit=cuda_warp):
                    for t in cuda_threads(0, 32):
                        x[1 - w, t] += 1
                Fence(cuda_in_order, cuda_in_order)

    return proc_missing_fence


def test_missing_fence_negative(compiler):
    def mkref():
        ref = SyncEnvRef([x[0, 0]], threads_per_task=64)

        def task_body():
            ref.alloc(x, (2, 32))
            for w in range(2):
                for t in range(32):
                    ref.mutate(x[w, t], [w * 32 + t], quals=smem_quals(convergent=True))
            for w in range(2):
                for t in range(32):
                    ref.mutate(
                        x[1 - w, t], [w * 32 + t], quals=smem_quals(convergent=True)
                    )

        ref.cuda_device_function(lambda: ref.device_task(0, task_body))

    check_both_fail(mkproc_missing_fence(), mkref)


# =============================================================================
# Cross-task WAR formerly hidden by camspork discarding prior reads upon a mutate
#
# Each task: generic read of g[0, 0], Fence(generic -> async), TMA reduce (atomic)
# into g. No hazard within a task. With 2 tasks, task 1's TMA reduce races task 0's
# read (WAR); camspork used to miss this because task 0's own reduce discarded
# task 0's read record. If every task reads, task 1's read already fails (RAW vs
# task 0's reduce), which masks the WAR; so the WAR-only case has only task 0 read.
# =============================================================================


def mkproc_read_then_reduce(n_tasks, only_task0_reads):
    rd_hi = 1 if only_task0_reads else n_tasks

    # fmt: off
    @proc
    def read_then_reduce(g: f32[8, 8] @ CudaGmemLinear):
        g_tmap = g[:, :] @ Sm90_tensorMap(0, 8, 8)
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, n_tasks):
                smem: f32[8, 8] @ CudaSmemLinear
                for m in cuda_threads(0, 8):
                    for k in seq(0, 8):
                        smem[m, k] = 0
                if task < rd_hi:
                    for t in cuda_threads(0, 1):
                        r: f32 @ CudaRmem
                        r = g[0, 0]
                Fence(cuda_in_order, cuda_generic_and_async_proxy)
                with CudaWarps(0, 1):
                    Sm90_tma_reduce_add_2d(
                        g_tmap[0:8, 0:8], smem[:, :],
                        dst=f32, src=f32, size0=8, size1=8, smem_box=(8, 8), swizzle=0,
                    )
                    cg: barrier @ Sm90_TmaCommitGroup
                    Arrive(tma_to_gmem_async, 1) >> cg
                    Await(cg, cuda_in_order, 0)
    # fmt: on
    return simplify(read_then_reduce)


def mkref_read_then_reduce(n_tasks, only_task0_reads):
    rd_hi = 1 if only_task0_reads else n_tasks
    ref = SyncEnvRef(
        [g[0, 0], g[7, 7], smem[0, 0], cg[()]],
        threads_per_task=32,
        task_shape=(n_tasks,),
    )
    generic_and_async = timelines.cuda_generic_and_async_proxy.get_timeline_set_bits()
    tma_L1 = timelines.tma_to_gmem_async.get_timeline_set_bits()
    # TMA operands: out-of-order, non-convergent; TAG 32 since the commit-group
    # Arrive witnessing them is executed by one warp. dst is an atomic reduce.
    tma_dst = access_quals(CudaGmemLinear, tma_to_gmem_async_instr)
    tma_src = access_quals(CudaSmemLinear, tma_to_gmem_async_instr)
    atomic = timelines.tma_to_gmem_atomic_qual.as_bit()
    ref.alloc(g, (8, 8))

    def task_body(task):
        base = ref.task_threads(task)[0][0]
        tt = ref.task_threads(task)
        ref.alloc(smem, (8, 8))
        for m in range(8):
            for k in range(8):
                ref.mutate(smem[m, k], [base + m], quals=smem_quals(convergent=True))
        if task < rd_hi:
            # r = g[0, 0] (r is CudaRmem, not probed: not modeled)
            ref.read(g[0, 0], [base], quals=gmem_quals(convergent=True))
        ref.fence(tt, IN_ORDER, generic_and_async)
        ref.mutate(
            g[0, 0], tt, extent=(8, 8), tag=32, atomic_bits=atomic, quals=tma_dst
        )
        ref.read(smem[0, 0], tt, extent=(8, 8), tag=32, quals=tma_src)
        ref.alloc_barrier(cg, ())
        ref.arrive(tt, tma_L1, cg[()])
        ref.await_(tt, IN_ORDER, cg[()], 0)
        ref.free_barrier(cg)
        free_smem(ref, smem, task)

    def device_body():
        for task in range(n_tasks):
            ref.device_task(task, lambda: task_body(task), depth=1)

    ref.cuda_device_function(device_body)
    return ref


def test_read_then_reduce_1_task(compiler, tmp_path):
    errors = check_trace(
        mkproc_read_then_reduce(1, False), mkref_read_then_reduce(1, False), tmp_path
    )
    assert_no_errors(errors)


def test_read_then_reduce_2_tasks_all_read_negative(compiler):
    """Both reject, but via task 1's read (RAW vs task 0's reduce), masking the WAR."""
    check_both_fail(
        mkproc_read_then_reduce(2, False), lambda: mkref_read_then_reduce(2, False)
    )


def test_read_then_reduce_2_tasks_negative(compiler):
    """Only task 0 reads: task 1's TMA reduce must be rejected (WAR vs task 0's read)."""
    check_both_fail(
        mkproc_read_then_reduce(2, True), lambda: mkref_read_then_reduce(2, True)
    )


# =============================================================================
# mbarrier literal arrive counts (endgame_plan/plan_multicast_barrier_counts.md)
#
# Counts are observed through the PendingAwaits that Arrives put on GMEM records,
# and through whether Awaits augment those records.
# Every mbarrier here has extent == ring depth, so no ring slot is reused
# (ring reuse: see "Managed ring buffers" below).
# =============================================================================


def barrier_elem_names(var: StrVarName, idxs: Iterable[ElemIdx]) -> List[StrElemName]:
    return [var[idx] for idx in idxs]


def mkproc_mbarrier_multicast_2cta(arrive_warps):
    # clusterDim=2, blockDim=128; warps [0, arrive_warps) of each CTA arrive.
    @proc
    def proc_mbarrier_multicast_2cta(g: f32[2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=128):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    with CudaWarps(0, arrive_warps):
                        for t in cuda_threads(0, 1):
                            g[cta] = 0
                        Arrive(cuda_in_order, 1) >> bar[cta, 0] >> bar[:, 0]
                # Awaits in a later loop nest: the trace serializes the CTA loop.
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(bar[cta, 0], cuda_in_order, 0)
                    for t in cuda_threads(0, 1):
                        g[1 - cta] += 1

    return simplify(proc_mbarrier_multicast_2cta)


def mkref_mbarrier_multicast_2cta(arrive_warps):
    T = 32 * arrive_warps  # threads per Arrive
    E = 2 * T  # mbarrier.init count: T threads x 2 CTAs
    ref = SyncEnvRef([g[0], g[1]], threads_per_task=256)
    ref.alloc(g, (2,))

    def task_body():
        ref.alloc_mbarrier(bar, (2, 1), expected=E, ring_dim=1, ring_depth=1)
        for cta in range(2):
            base = cta * 128
            ref.mutate(g[cta], [base], quals=gmem_quals(convergent=True))
            # Each target's own count: CTA 1's record gets z = T on both barriers.
            ref.arrive(
                ref.tids(base, base + T),
                IN_ORDER,
                barrier_elem_names(bar, [(0, 0), (1, 0)]),
            )
        for cta in range(2):
            base = cta * 128
            ref.await_(ref.tids(base, base + 128), IN_ORDER, bar[cta, 0])
            ref.mutate(g[1 - cta], [base], quals=gmem_quals(convergent=True))
        ref.free_barrier(bar)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


@pytest.mark.parametrize("arrive_warps", [1, 2])
def test_mbarrier_multicast_2cta(compiler, tmp_path, arrive_warps):
    errors = check_trace(
        mkproc_mbarrier_multicast_2cta(arrive_warps),
        mkref_mbarrier_multicast_2cta(arrive_warps),
        tmp_path,
    )
    assert_no_errors(errors)


def mkproc_mbarrier_multicast_4cta():
    # clusterDim=4 as (m, n) = 2 x 2; multicast along n only (group of 2 CTAs).
    @proc
    def proc_mbarrier_multicast_4cta(g: f32[2, 2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=4, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 2, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                    for n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        for t in cuda_threads(0, 1):
                            g[m, n] = 0
                        Arrive(cuda_in_order, 1) >> bar[m, n, 0] >> bar[m, :, 0]
                for m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                    for n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        Await(bar[m, n, 0], cuda_in_order, 0)
                        for t in cuda_threads(0, 1):
                            g[m, 1 - n] += 1

    return simplify(proc_mbarrier_multicast_4cta)


def test_mbarrier_multicast_4cta(compiler, tmp_path):
    E = 2 * 32  # 32 threads x 2 CTAs per multicast group
    probes = [g[m, n] for m in range(2) for n in range(2)]
    ref = SyncEnvRef(probes, threads_per_task=128)
    ref.alloc(g, (2, 2))

    def task_body():
        ref.alloc_mbarrier(bar, (2, 2, 1), expected=E, ring_dim=2, ring_depth=1)
        for m in range(2):
            for n in range(2):
                base = (2 * m + n) * 32
                ref.mutate(g[m, n], [base], quals=gmem_quals(convergent=True))
                # Row m only: bar[1 - m, ...] counts are untouched.
                ref.arrive(
                    ref.tids(base, base + 32),
                    IN_ORDER,
                    barrier_elem_names(bar, [(m, 0, 0), (m, 1, 0)]),
                )
        for m in range(2):
            for n in range(2):
                base = (2 * m + n) * 32
                ref.await_(ref.tids(base, base + 32), IN_ORDER, bar[m, n, 0])
                ref.mutate(g[m, 1 - n], [base], quals=gmem_quals(convergent=True))
        ref.free_barrier(bar)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_mbarrier_multicast_4cta(), ref, tmp_path))


def mkproc_mbarrier_pre_arrive():
    # bar[:, 0] is pre-arrived: Awaiting it needs no Arrive (count starts at E).
    @proc
    def proc_mbarrier_pre_arrive(g: f32[2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 2 @ ring_buffer_by(2)] @ CudaMbarrierPreArrive(1)
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(bar[cta, 0], cuda_in_order, 0)
                    for t in cuda_threads(0, 1):
                        g[cta] = 0
                    Arrive(cuda_in_order, 1) >> bar[cta, 1] >> bar[:, 1]
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(bar[cta, 1], cuda_in_order, 0)
                    for t in cuda_threads(0, 1):
                        g[1 - cta] += 1

    return simplify(proc_mbarrier_pre_arrive)


def test_mbarrier_pre_arrive(compiler, tmp_path):
    E = 2 * 32
    ref = SyncEnvRef([g[0], g[1]], threads_per_task=64)
    ref.alloc(g, (2,))

    def task_body():
        # bar[:, 0] pre-arrived at allocation (no trace event).
        ref.alloc_mbarrier(
            bar, (2, 2), expected=E, ring_dim=1, ring_depth=2, pre_arrive=1
        )
        for cta in range(2):
            base = cta * 32
            ref.await_(ref.tids(base, base + 32), IN_ORDER, bar[cta, 0])
            ref.mutate(g[cta], [base], quals=gmem_quals(convergent=True))
            ref.arrive(
                ref.tids(base, base + 32),
                IN_ORDER,
                barrier_elem_names(bar, [(0, 1), (1, 1)]),
            )
        for cta in range(2):
            base = cta * 32
            ref.await_(ref.tids(base, base + 32), IN_ORDER, bar[cta, 1])
            ref.mutate(g[1 - cta], [base], quals=gmem_quals(convergent=True))
        ref.free_barrier(bar)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_mbarrier_pre_arrive(), ref, tmp_path))


# Negative: an Await in the same CTA loop body as its multicast Arrive.
# Valid CUDA, but the trace runs CTA 0's body before CTA 1's Arrive, so the
# strict Await sees 32 of 64 arrives ("No forward progress"; documented conservatism).


def mkproc_mbarrier_same_body_await():
    @proc
    def proc_mbarrier_same_body_await(g: f32[2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for t in cuda_threads(0, 1):
                        g[cta] = 0
                    Arrive(cuda_in_order, 1) >> bar[cta, 0] >> bar[:, 0]
                    Await(bar[cta, 0], cuda_in_order, 0)

    return simplify(proc_mbarrier_same_body_await)


def mkref_mbarrier_same_body_await():
    ref = SyncEnvRef([g[0]], threads_per_task=64)
    ref.alloc(g, (2,))

    def task_body():
        ref.alloc_mbarrier(bar, (2, 1), expected=64, ring_dim=1, ring_depth=1)
        for cta in range(2):
            base = cta * 32
            ref.mutate(g[cta], [base], quals=gmem_quals(convergent=True))
            ref.arrive(
                ref.tids(base, base + 32),
                IN_ORDER,
                barrier_elem_names(bar, [(0, 0), (1, 0)]),
            )
            ref.await_(ref.tids(base, base + 32), IN_ORDER, bar[cta, 0])
        ref.free_barrier(bar)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_mbarrier_same_body_await_negative(compiler):
    check_both_fail(mkproc_mbarrier_same_body_await(), mkref_mbarrier_same_body_await)


# Negative: two Arrives on the same (ring depth 1) element: 128 > 64 arrives.


def mkproc_mbarrier_too_many_arrives():
    @proc
    def proc_mbarrier_too_many_arrives(g: f32[2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for t in cuda_threads(0, 1):
                        g[cta] = 0
                    Arrive(cuda_in_order, 1) >> bar[cta, 0] >> bar[:, 0]
                    Arrive(cuda_in_order, 1) >> bar[cta, 0] >> bar[:, 0]
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(bar[cta, 0], cuda_in_order, 0)

    return simplify(proc_mbarrier_too_many_arrives)


def mkref_mbarrier_too_many_arrives():
    ref = SyncEnvRef([g[0]], threads_per_task=64)
    ref.alloc(g, (2,))

    def task_body():
        ref.alloc_mbarrier(bar, (2, 1), expected=64, ring_dim=1, ring_depth=1)
        for cta in range(2):
            base = cta * 32
            ref.mutate(g[cta], [base], quals=gmem_quals(convergent=True))
            for _ in range(2):
                ref.arrive(
                    ref.tids(base, base + 32),
                    IN_ORDER,
                    barrier_elem_names(bar, [(0, 0), (1, 0)]),
                )

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_mbarrier_too_many_arrives_negative(compiler):
    check_both_fail(mkproc_mbarrier_too_many_arrives(), mkref_mbarrier_too_many_arrives)


# The tests below were rejected statically by the multicast convergence check
# (removed; plan_multicast_barrier_counts.md). Literal counts + strict Await
# now decide them dynamically.


def mkproc_mbarrier_seq_in_cta_loop():
    # seq loop between the multicast iterator's cuda_threads loop and the Arrive.
    @proc
    def proc_mbarrier_seq_in_cta_loop(g: f32[2, 2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 2 @ ring_buffer_by(2)] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for k in seq(0, 2):
                        for t in cuda_threads(0, 1):
                            g[cta, k] = 0
                        Arrive(cuda_in_order, 1) >> bar[cta, k] >> bar[:, k]
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for k in seq(0, 2):
                        Await(bar[cta, k], cuda_in_order, 0)
                        for t in cuda_threads(0, 1):
                            g[1 - cta, k] += 1

    return simplify(proc_mbarrier_seq_in_cta_loop)


def test_mbarrier_seq_in_cta_loop(compiler, tmp_path):
    probes = [g[c, k] for c in range(2) for k in range(2)]
    ref = SyncEnvRef(probes, threads_per_task=64)
    ref.alloc(g, (2, 2))

    def task_body():
        ref.alloc_mbarrier(bar, (2, 2), expected=64, ring_dim=1, ring_depth=2)
        for cta in range(2):
            base = cta * 32
            for k in range(2):
                ref.mutate(g[cta, k], [base], quals=gmem_quals(convergent=True))
                ref.arrive(
                    ref.tids(base, base + 32),
                    IN_ORDER,
                    barrier_elem_names(bar, [(0, k), (1, k)]),
                )
        for cta in range(2):
            base = cta * 32
            for k in range(2):
                ref.await_(ref.tids(base, base + 32), IN_ORDER, bar[cta, k])
                ref.mutate(g[1 - cta, k], [base], quals=gmem_quals(convergent=True))
        ref.free_barrier(bar)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_mbarrier_seq_in_cta_loop(), ref, tmp_path))


def mkproc_mbarrier_if_around_arrive():
    # Invalid: only CTA 0 arrives, so CTA 1's arrives on bar[:, 0] never happen.
    @proc
    def proc_mbarrier_if_around_arrive(g: f32[2] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, 1 @ ring_buffer_by(1)] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    for t in cuda_threads(0, 1):
                        g[cta] = 0
                    if cta == 0:
                        Arrive(cuda_in_order, 1) >> bar[cta, 0] >> bar[:, 0]
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    Await(bar[cta, 0], cuda_in_order, 0)

    return simplify(proc_mbarrier_if_around_arrive)


def mkref_mbarrier_if_around_arrive():
    ref = SyncEnvRef([g[0]], threads_per_task=64)
    ref.alloc(g, (2,))

    def task_body():
        ref.alloc_mbarrier(bar, (2, 1), expected=64, ring_dim=1, ring_depth=1)
        for cta in range(2):
            ref.mutate(g[cta], [cta * 32], quals=gmem_quals(convergent=True))
        ref.arrive(ref.tids(0, 32), IN_ORDER, barrier_elem_names(bar, [(0, 0), (1, 0)]))
        ref.await_(ref.tids(0, 32), IN_ORDER, bar[0, 0])  # 32 of 64

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_mbarrier_if_around_arrive_negative(compiler):
    check_both_fail(mkproc_mbarrier_if_around_arrive(), mkref_mbarrier_if_around_arrive)


def mkproc_mbarrier_turnstile(lag):
    # CTA 1 at iteration k waits (turnstile tb) for CTA 0 to reach iteration k + lag.
    # lag=2 deadlocks in CUDA: CTA 0 blocks at Await(bar[0, 1]) waiting for CTA 1's
    # iteration-1 Arrive, while CTA 1 waits in iteration 0 for CTA 0's iteration 2.
    # lag=1 is valid CUDA, but the serialized trace can't show it (conservative).
    # Formerly both passed sync_check if the convergence check was disabled.
    K = 4
    KL = K + lag

    @proc
    def proc_mbarrier_turnstile():
        with CudaDeviceFunction(clusterDim=2, blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[2, K @ ring_buffer_by(K)] @ CudaMbarrier
                tb: barrier[2, KL @ ring_buffer_by(KL)] @ CudaMbarrier
                for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                    if cta == 1:
                        for j in seq(0, lag):
                            Arrive(cuda_in_order, 1) >> tb[cta, j] >> tb[:, j]
                    for k in seq(0, K):
                        Arrive(cuda_in_order, 1) >> bar[cta, k] >> bar[:, k]
                        if cta == 0:
                            Arrive(cuda_in_order, 1) >> tb[cta, k] >> tb[:, k]
                        else:
                            (
                                Arrive(cuda_in_order, 1)
                                >> tb[cta, k + lag]
                                >> tb[:, k + lag]
                            )
                            Await(tb[cta, k + lag], cuda_in_order, 0)
                        Await(bar[cta, k], cuda_in_order, 0)
                    if cta == 0:
                        for j in seq(K, KL):
                            Arrive(cuda_in_order, 1) >> tb[cta, j] >> tb[:, j]

    return simplify(proc_mbarrier_turnstile)


def mkref_mbarrier_turnstile(lag):
    K = 4
    ref = SyncEnvRef([], threads_per_task=64)

    def task_body():
        ref.alloc_mbarrier(bar, (2, K), expected=64, ring_dim=1, ring_depth=K)
        ref.alloc_mbarrier(
            tb, (2, K + lag), expected=64, ring_dim=1, ring_depth=K + lag
        )
        # CTA 0's whole body runs first in the trace.
        tids = ref.tids(0, 32)
        k = 0
        ref.arrive(tids, IN_ORDER, barrier_elem_names(bar, [(0, k), (1, k)]))
        ref.arrive(tids, IN_ORDER, barrier_elem_names(tb, [(0, k), (1, k)]))
        ref.await_(tids, IN_ORDER, bar[0, k])  # 32 of 64: no forward progress

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


@pytest.mark.parametrize("lag", [1, 2])
def test_mbarrier_turnstile_negative(compiler, lag):
    check_both_fail(
        mkproc_mbarrier_turnstile(lag), lambda: mkref_mbarrier_turnstile(lag)
    )


# =============================================================================
# Managed ring buffers (amspork3 SmemAppendix.tex: allocTrDetail / freeTrDetail)
#
# Ring slots are reused here (extent > ring depth), so the alloc records' pending
# awaits on the previous phase / guard are visible in the trace. freeTrDetail is only
# observable through pass/fail of the free check.
# =============================================================================

MB = timelines.cuda_mbarrier_qual.as_bit()


def mkproc_mbarrier_ring(arrive_warps):
    # Self-guarded mbarrier ring: K phases on D physical mbarriers.
    # Warps [0, arrive_warps) Arrive; warps 0 and 1 Await separately. With
    # arrive_warps=1, warp 1's Await of phase k is never followed by its own Arrive
    # on phase k + D, so freeing the mbarrier fails (free_missing_arrive).
    K, D = 4, 2

    @proc
    def proc_mbarrier_ring():
        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, 1):
                bar: barrier[K @ ring_buffer_by(D)] @ CudaMbarrier
                for k in seq(0, K):
                    with CudaWarps(0, arrive_warps):
                        Arrive(cuda_in_order, 1) >> bar[k]
                    with CudaWarps(0, 1):
                        Await(bar[k], cuda_in_order, 0)
                    with CudaWarps(1, 2):
                        Await(bar[k], cuda_in_order, 0)

    return simplify(proc_mbarrier_ring)


def mkref_mbarrier_ring(arrive_warps):
    K, D = 4, 2
    ref = SyncEnvRef([bar[k] for k in range(K)], threads_per_task=64)
    task = ref.task_threads()

    def task_body():
        ref.alloc_mbarrier(
            bar, (K,), expected=32 * arrive_warps, ring_dim=0, ring_depth=D
        )
        for k in range(K):
            ref.arrive(ref.tids(0, 32 * arrive_warps), IN_ORDER, bar[k])
            ref.await_(ref.tids(0, 32), IN_ORDER, bar[k])
            ref.await_(ref.tids(32, 64), IN_ORDER, bar[k])
        ref.free_barrier(bar)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_mbarrier_ring(compiler, tmp_path):
    assert_no_errors(
        check_trace(mkproc_mbarrier_ring(2), mkref_mbarrier_ring(2), tmp_path)
    )


def test_mbarrier_ring_free_negative(compiler):
    check_both_fail(mkproc_mbarrier_ring(1), lambda: mkref_mbarrier_ring(1))


def mkproc_mbarrier_ring_missing_await():
    # Arrive on phase D before anyone Awaited phase 0 of the same physical mbarrier.
    K, D = 4, 2

    @proc
    def proc_mbarrier_ring_missing_await():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                bar: barrier[K @ ring_buffer_by(D)] @ CudaMbarrier
                for k in seq(0, K):
                    Arrive(cuda_in_order, 1) >> bar[k]
                for k in seq(0, K):
                    Await(bar[k], cuda_in_order, 0)

    return simplify(proc_mbarrier_ring_missing_await)


def mkref_mbarrier_ring_missing_await():
    K, D = 4, 2
    ref = SyncEnvRef([], threads_per_task=32)

    def task_body():
        ref.alloc_mbarrier(bar, (K,), expected=32, ring_dim=0, ring_depth=D)
        for k in range(K):
            ref.arrive(ref.tids(0, 32), IN_ORDER, bar[k])

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_mbarrier_ring_missing_await_negative(compiler):
    check_both_fail(
        mkproc_mbarrier_ring_missing_await(), mkref_mbarrier_ring_missing_await
    )


def mkproc_smem_ring(early_war_arrive=False):
    # SMEM data ring buffer x (K slots, depth D) guarded by war (K + D phases,
    # first D pre-arrived), as in the Sm90 GEMMs. Producer warp 0 writes x[k];
    # consumer warp 1 copies it to g. early_war_arrive: the consumer releases the
    # slot (Arrive on war[k + D]) before reading it, so freeing x fails.
    K, D = 4, 2
    late_war_arrive = not early_war_arrive

    @proc
    def proc_smem_ring(g: f32[32] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=64):
            for task in cuda_tasks(0, 1):
                war: barrier[(K + D) @ ring_buffer_by(D)] @ CudaMbarrierPreArrive(D)
                raw: barrier[K @ ring_buffer_by(D)] @ CudaMbarrier
                x: f32[K @ ring_buffer_by(D), 32].ring_guarded_by(war) @ CudaSmemLinear
                for k in seq(0, K):
                    with CudaWarps(0, 1):
                        Await(war[k], cuda_in_order, 0)
                        for t in cuda_threads(0, 32):
                            x[k, t] = 0
                        Arrive(cuda_in_order, 1) >> raw[k]
                    with CudaWarps(1, 2):
                        Await(raw[k], cuda_in_order, 0)
                        if early_war_arrive:
                            Arrive(cuda_in_order, 1) >> war[k + D]
                        for t in cuda_threads(0, 32):
                            g[t] = x[k, t]
                        if late_war_arrive:
                            Arrive(cuda_in_order, 1) >> war[k + D]

    return simplify(proc_smem_ring)


def mkref_smem_ring(early_war_arrive=False, probes=()):
    K, D = 4, 2
    ref = SyncEnvRef(probes, threads_per_task=64)
    ref.alloc(g, (32,))
    producer, consumer = ref.tids(0, 32), ref.tids(32, 64)

    def task_body():
        ref.alloc_mbarrier(
            war, (K + D,), expected=32, ring_dim=0, ring_depth=D, pre_arrive=D
        )
        ref.alloc_mbarrier(raw, (K,), expected=32, ring_dim=0, ring_depth=D)
        ring = RingGuard(war, ring_dim=0, ring_depth=D)
        ref.alloc(x, (K, 32), ring=ring, mask=smem_quals().mask)
        for k in range(K):
            ref.await_(producer, IN_ORDER, war[k])
            for t in range(32):
                ref.mutate(x[k, t], [t], quals=smem_quals(convergent=True))
            ref.arrive(producer, IN_ORDER, raw[k])
            ref.await_(consumer, IN_ORDER, raw[k])
            if early_war_arrive:
                ref.arrive(consumer, IN_ORDER, war[k + D])
            for t in range(32):
                ref.read(x[k, t], [32 + t], quals=smem_quals(convergent=True))
                ref.mutate(g[t], [32 + t], quals=gmem_quals(convergent=True))
            if not early_war_arrive:
                ref.arrive(consumer, IN_ORDER, war[k + D])
        ref.free(x)
        ref.free_barrier(raw)
        ref.free_barrier(war)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    return ref


def test_smem_ring(compiler, tmp_path):
    probes = [x[0, 0], x[2, 5], war[2], raw[3], g[0]]
    assert_no_errors(
        check_trace(mkproc_smem_ring(), mkref_smem_ring(probes=probes), tmp_path)
    )


def test_smem_ring_early_war_arrive_negative(compiler):
    check_both_fail(
        mkproc_smem_ring(early_war_arrive=True),
        lambda: mkref_smem_ring(early_war_arrive=True),
    )


def mkproc_smem_ring_multicast_2cta():
    # As mkproc_smem_ring, per CTA of a 2-CTA cluster, with the war and raw Arrives
    # multicast to both CTAs (like the GEMMs' A tile): guards(x[c, k]) = {war[0, k], war[1, k]}.
    # raw must multicast too: otherwise CTA 0's consumer would Arrive on CTA 1's next
    # war phase with no ordering after CTA 1's producer Awaited the current one (a real race).
    # Producers and consumers are in separate CTA loops so the serialized trace makes progress.
    # Blind spot: gNext's multicast expansion can't be tested, since an Arrive always
    # covers the whole multicast group (all of gNext arrives, or none of it).
    K, D = 3, 2

    @proc
    def proc_smem_ring_multicast_2cta(g: f32[2, 32] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=2, blockDim=64):
            for task in cuda_tasks(0, 1):
                war: barrier[2, (K + D) @ ring_buffer_by(D)] @ CudaMbarrierPreArrive(D)
                raw: barrier[2, K @ ring_buffer_by(D)] @ CudaMbarrier
                x: (
                    f32[2, K @ ring_buffer_by(D), 32].ring_guarded_by(war)
                    @ CudaSmemLinear
                )
                for k in seq(0, K):
                    for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        with CudaWarps(0, 1):
                            Await(war[cta, k], cuda_in_order, 0)
                            for t in cuda_threads(0, 32):
                                x[cta, k, t] = 0
                            Arrive(cuda_in_order, 1) >> raw[cta, k] >> raw[:, k]
                    for cta in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                        with CudaWarps(1, 2):
                            Await(raw[cta, k], cuda_in_order, 0)
                            for t in cuda_threads(0, 32):
                                g[cta, t] = x[cta, k, t]
                            Arrive(cuda_in_order, 1) >> war[cta, k + D] >> war[:, k + D]

    return simplify(proc_smem_ring_multicast_2cta)


def test_smem_ring_multicast_2cta(compiler, tmp_path):
    K, D = 3, 2
    probes = [x[0, 0, 0], x[1, 2, 7], war[0, 2], war[1, 4]]
    ref = SyncEnvRef(probes, threads_per_task=128)
    ref.alloc(g, (2, 32))

    def task_body():
        ref.alloc_mbarrier(
            war, (2, K + D), expected=64, ring_dim=1, ring_depth=D, pre_arrive=D
        )
        ref.alloc_mbarrier(raw, (2, K), expected=64, ring_dim=1, ring_depth=D)
        # `>> war[cta, k + D] >> war[:, k + D]`
        ring = RingGuard(
            war,
            ring_dim=1,
            ring_depth=D,
            guard_multicast=((False, False), (True, False)),
        )
        ref.alloc(x, (2, K, 32), ring=ring, mask=smem_quals().mask)
        for k in range(K):
            for cta in range(2):
                base = cta * 64
                producer = ref.tids(base, base + 32)
                ref.await_(producer, IN_ORDER, war[cta, k])
                for t in range(32):
                    ref.mutate(
                        x[cta, k, t], [base + t], quals=smem_quals(convergent=True)
                    )
                ref.arrive(
                    producer, IN_ORDER, barrier_elem_names(raw, [(0, k), (1, k)])
                )
            for cta in range(2):
                base = cta * 64
                consumer = ref.tids(base + 32, base + 64)
                ref.await_(consumer, IN_ORDER, raw[cta, k])
                for t in range(32):
                    ref.read(
                        x[cta, k, t],
                        [base + 32 + t],
                        quals=smem_quals(convergent=True),
                    )
                    ref.mutate(
                        g[cta, t],
                        [base + 32 + t],
                        quals=gmem_quals(convergent=True),
                    )
                ref.arrive(
                    consumer,
                    IN_ORDER,
                    barrier_elem_names(war, [(0, k + D), (1, k + D)]),
                )
        ref.free(x)
        ref.free_barrier(raw)
        ref.free_barrier(war)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_smem_ring_multicast_2cta(), ref, tmp_path))


def mkproc_smem_ring_cross_4cta():
    # 2 x 2 cluster (m, n) with the GEMMs' cross-shaped multicast:
    # `>> war[m, n, k] >> war[m, :, k] >> war[:, n, k]` reaches 3 of the 4 CTAs, so
    # guards(x[m, n, k]) has 3 elements (the union of a row and a column, not the
    # whole 2 x 2 rectangle). raw multicasts the same way (see mkproc_smem_ring_multicast_2cta).
    K, D = 3, 2

    @proc
    def proc_smem_ring_cross_4cta(g: f32[2, 2, 32] @ CudaGmemLinear):
        with CudaDeviceFunction(clusterDim=4, blockDim=64):
            for task in cuda_tasks(0, 1):
                war: barrier[2, 2, (K + D) @ ring_buffer_by(D)] @ CudaMbarrierPreArrive(
                    D
                )
                raw: barrier[2, 2, K @ ring_buffer_by(D)] @ CudaMbarrier
                x: (
                    f32[2, 2, K @ ring_buffer_by(D), 32].ring_guarded_by(war)
                    @ CudaSmemLinear
                )
                for k in seq(0, K):
                    for m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                        for n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                            with CudaWarps(0, 1):
                                Await(war[m, n, k], cuda_in_order, 0)
                                for t in cuda_threads(0, 32):
                                    x[m, n, k, t] = 0
                                (
                                    Arrive(cuda_in_order, 1)
                                    >> raw[m, n, k]
                                    >> raw[m, :, k]
                                    >> raw[:, n, k]
                                )
                    for m in cuda_threads(0, 2, unit=2 * cuda_cta_in_cluster):
                        for n in cuda_threads(0, 2, unit=cuda_cta_in_cluster):
                            with CudaWarps(1, 2):
                                Await(raw[m, n, k], cuda_in_order, 0)
                                for t in cuda_threads(0, 32):
                                    g[m, n, t] = x[m, n, k, t]
                                (
                                    Arrive(cuda_in_order, 1)
                                    >> war[m, n, k + D]
                                    >> war[m, :, k + D]
                                    >> war[:, n, k + D]
                                )

    return simplify(proc_smem_ring_cross_4cta)


def test_smem_ring_cross_4cta(compiler, tmp_path):
    K, D = 3, 2
    E = 3 * 32  # each mbarrier gets Arrives from the 3 CTAs of its cross
    probes = [x[1, 1, 2, 7], x[0, 1, 0, 0], war[0, 0, 2]]
    ref = SyncEnvRef(probes, threads_per_task=256)
    ref.alloc(g, (2, 2, 32))

    def cross_guard_idxs(m, n, k):
        return [(m, n, k), (m, 1 - n, k), (1 - m, n, k)]

    def task_body():
        ref.alloc_mbarrier(
            war, (2, 2, K + D), expected=E, ring_dim=2, ring_depth=D, pre_arrive=D
        )
        ref.alloc_mbarrier(raw, (2, 2, K), expected=E, ring_dim=2, ring_depth=D)
        # `>> war[m, n, k] >> war[m, :, k] >> war[:, n, k]`
        mc = ((False, False, False), (False, True, False), (True, False, False))
        ring = RingGuard(war, ring_dim=2, ring_depth=D, guard_multicast=mc)
        ref.alloc(x, (2, 2, K, 32), ring=ring, mask=smem_quals().mask)
        for k in range(K):
            for m in range(2):
                for n in range(2):
                    base = (2 * m + n) * 64
                    producer = ref.tids(base, base + 32)
                    ref.await_(producer, IN_ORDER, war[m, n, k])
                    for t in range(32):
                        ref.mutate(
                            x[m, n, k, t],
                            [base + t],
                            quals=smem_quals(convergent=True),
                        )
                    ref.arrive(
                        producer,
                        IN_ORDER,
                        barrier_elem_names(raw, cross_guard_idxs(m, n, k)),
                    )
            for m in range(2):
                for n in range(2):
                    base = (2 * m + n) * 64
                    consumer = ref.tids(base + 32, base + 64)
                    ref.await_(consumer, IN_ORDER, raw[m, n, k])
                    for t in range(32):
                        ref.read(
                            x[m, n, k, t],
                            [base + 32 + t],
                            quals=smem_quals(convergent=True),
                        )
                        ref.mutate(
                            g[m, n, t],
                            [base + 32 + t],
                            quals=gmem_quals(convergent=True),
                        )
                    ref.arrive(
                        consumer,
                        IN_ORDER,
                        barrier_elem_names(war, cross_guard_idxs(m, n, k + D)),
                    )
        ref.free(x)
        ref.free_barrier(raw)
        ref.free_barrier(war)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    assert_no_errors(check_trace(mkproc_smem_ring_cross_4cta(), ref, tmp_path))


# =============================================================================
# Multi-dimensional cuda_tasks: camspork's task_index is the lexicographic position
# of the task in the loop nest (SyncEnvRef task_shape).
# =============================================================================

TASKS_2D = (2, 3)


def mkproc_tasks_2d():
    M, N = TASKS_2D

    @proc
    def proc_tasks_2d(g: f32[M, N, 32] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for tm in cuda_tasks(0, M):
                for tn in cuda_tasks(0, N):
                    for t in cuda_threads(0, 32):
                        g[tm, tn, t] = 0
                    Fence(cuda_in_order, cuda_in_order)

    return proc_tasks_2d


def mkref_tasks_2d(task_order):
    """task_order: the reference's assumed linearization (to show the trace checks it)."""
    M, N = TASKS_2D
    probes = [g[0, 2, 0], g[1, 0, 31], g[1, 2, 5]]
    ref = SyncEnvRef(probes, threads_per_task=32, task_shape=task_order((M, N)))
    ref.alloc(g, (M, N, 32))

    def task_body(tm, tn):
        task = task_order((tm, tn))
        base = ref.task_threads(task)[0][0]
        for t in range(32):
            ref.mutate(g[tm, tn, t], [base + t], quals=gmem_quals(convergent=True))
        ref.fence(ref.task_threads(task), IN_ORDER, IN_ORDER)

    def device_body():
        for tm, tn in itertools.product(range(M), range(N)):
            ref.device_task(task_order((tm, tn)), lambda: task_body(tm, tn), depth=1)

    ref.cuda_device_function(device_body)
    return ref


def test_tasks_2d(compiler, tmp_path):
    ref = mkref_tasks_2d(lambda idx: idx)
    assert_no_errors(check_trace(mkproc_tasks_2d(), ref, tmp_path))


def test_tasks_2d_transposed_negative(compiler, tmp_path):
    """Reference assumes column-major task IDs: the trace must disagree."""
    ref = mkref_tasks_2d(lambda idx: idx[::-1])
    errors = check_trace(mkproc_tasks_2d(), ref, tmp_path)
    assert errors
    assert "threads" in errors[0]


# =============================================================================
# Probing one of several same-named variables (var_cursors): unrolling duplicates
# the x alloc, so the name x alone is ambiguous.
# =============================================================================


def mkproc_unrolled_x():
    @proc
    def proc_unrolled_x(g: f32[2, 32] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for i in seq(0, 2):
                    x: f32[32] @ CudaSmemLinear
                    for t in cuda_threads(0, 32):
                        x[t] = 0
                    Fence(cuda_in_order, cuda_in_order)
                    for t in cuda_threads(0, 32):
                        g[i, t] = x[t]
                    Fence(cuda_in_order, cuda_in_order)

    return unroll_loop(proc_unrolled_x, "i")


def test_probe_cursor_unrolled(compiler, tmp_path):
    """Trace only the second unrolled copy of x (the first is unprobed, so the
    reference may give it any name)."""
    p = mkproc_unrolled_x()
    x_copy0 = StrVarName("x_copy0")
    ref = SyncEnvRef([x[0], x[31], g[1, 0]], threads_per_task=32)
    task = ref.task_threads()
    ref.alloc(g, (2, 32))

    def task_body():
        for i, var in enumerate([x_copy0, x]):
            ref.alloc(var, (32,))
            for t in range(32):
                ref.mutate(var[t], [t], quals=smem_quals(convergent=True))
            ref.fence(task, IN_ORDER, IN_ORDER)
            for t in range(32):
                ref.read(var[t], [t], quals=smem_quals(convergent=True))
                ref.mutate(g[i, t], [t], quals=gmem_quals(convergent=True))
            ref.fence(task, IN_ORDER, IN_ORDER)
            free_smem(ref, var)

    ref.cuda_device_function(lambda: ref.device_task(0, task_body))
    x_alloc = p.find("x : _ #1")
    assert_no_errors(check_trace(p, ref, tmp_path, var_cursors={x: x_alloc}))


def test_probe_cursor_ambiguous_name(compiler, tmp_path):
    """Without var_cursors, the unrolled name x is ambiguous."""
    ref = SyncEnvRef([x[0]], threads_per_task=32)
    with pytest.raises(AssertionError, match="ambiguous"):
        check_trace(mkproc_unrolled_x(), ref, tmp_path)


def test_probe_cursor_name_mismatch(compiler, tmp_path):
    """var_cursors must map a name to a cursor of a variable with that name
    (here, x to the argument g): the trace only names variables by str(Sym)."""
    p = mkproc_unrolled_x()
    ref = SyncEnvRef([x[0]], threads_per_task=32)
    with pytest.raises(AssertionError, match="maps to a cursor of variable g"):
        check_trace(p, ref, tmp_path, var_cursors={x: p.args()[0]})
