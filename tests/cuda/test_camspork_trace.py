"""PROTOTYPE camspork sync env trace tests (endgame_plan/plan_camspork_test.md).

Each test pairs an Exo proc (traced through sync_check/camspork) with a hand-written
reference (camspork_ref.SyncEnvRef), excut-style: thread IDs are computed by hand.
Scope for now: no mbarriers, managed ring buffers, atomics, or multi-dimensional tasks.
Tests never leave an access without a later sync (the trace only samples sync events).
"""

from __future__ import annotations

from exo import proc, instr
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import simplify
from exo.spork import timelines
from exo.spork.timelines import (
    cuda_in_order_instr,
    Sm80_cp_async_instr,
    tma_to_gmem_async_instr,
)
from exo.spork.camspork_ref import (
    SyncEnvRef,
    check_both_fail,
    check_trace,
    qual_info,
)

IN_ORDER = timelines.cuda_in_order.get_timeline_set_bits()


def smem(**kw):
    return qual_info(CudaSmemLinear, cuda_in_order_instr, **kw)


def gmem(**kw):
    return qual_info(CudaGmemLinear, cuda_in_order_instr, **kw)


def free_smem(ref, var, task=0):
    # SMEM free: every cluster thread must be synchronized with prior accesses.
    ref.free(
        var,
        check_tids=ref.task_threads(task),
        precondition_bits=smem()["precondition_bits"],
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
    ref = SyncEnvRef([("x", (0, 0)), ("x", (1, 31))], threads_per_task=192)
    task = ref.task_threads()

    def task_body():
        ref.alloc("x", (2, 32))
        for w in range(2):
            for t in range(32):
                ref.mutate("x", (w, t), [(3 + w) * 32 + t], **smem(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        for w in range(2):
            for t in range(32):
                ref.mutate("x", (w, t), [w * 32 + t], **smem(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, "x")

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
    ref = SyncEnvRef([("g", (0, 0)), ("g", (1, 31))], threads_per_task=64)
    ref.alloc("g", (2, 32))

    def task_body():
        for w in range(2):
            for t in range(32):
                ref.mutate("g", (w, t), [w * 32 + t], **gmem(convergent=True))

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
    ref = SyncEnvRef(
        [("x", (1, 0)), ("x", (3, 31)), ("x", (0, 5))], threads_per_task=256
    )
    task = ref.task_threads()

    def task_body():
        ref.alloc("x", (4, 32))
        for w2 in range(4):
            for t in range(32):
                ref.mutate("x", (w2, t), [w2 * 64 + t], **smem(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        for t in range(32):
            ref.mutate("x", (0, t), [32 + t], **smem(convergent=True))
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, "x")

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
    ref = SyncEnvRef(
        [("x", (0, 0)), ("x", (0, 3)), ("x", (31, 1)), ("cg", ())], threads_per_task=32
    )
    task = ref.task_threads()
    cp_async = qual_info(CudaSmemLinear, Sm80_cp_async_instr)
    L1 = timelines.Sm80_cp_async.get_timeline_set_bits()

    def task_body():
        ref.alloc("x", (32, 4))
        for tid in range(32):
            ref.alloc_barrier("cg", ())
            # cp.async dst: out-of-order initial qual-tl; TAG 1 since the
            # commit-group Arrive is executed per thread. (src is a read-only
            # GMEM arg: not sync-checked.)
            ref.mutate("x", (tid, 0), [tid], extent=(1, 4), tag=1, **cp_async)
            ref.arrive([tid], L1, "cg")
            ref.await_([tid], IN_ORDER, "cg", 0)
            ref.mutate("x", (tid, 0), [tid], **smem(convergent=True))
            ref.free_barrier("cg")
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, "x")

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
    ref = SyncEnvRef([("g", (0,)), ("g", (1,)), ("cs", ())], threads_per_task=128)
    task = ref.task_threads()
    ref.alloc("g", (2,))

    def task_body():
        ref.alloc_barrier("cs", (), one_shot_arrive=True, one_shot_await=True)
        for cta in range(2):
            ref.mutate("g", (cta,), [cta * 64], **gmem(convergent=True))
        ref.arrive(task, IN_ORDER, "cs")
        ref.await_(task, IN_ORDER, "cs", 0)
        for cta in range(2):
            ref.mutate("g", (1 - cta,), [cta * 64 + 32], **gmem(convergent=True))
        ref.free_barrier("cs")

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


WINDOW_PROBES = [("x", (0, 0)), ("x", (0, 3)), ("x", (2, 1)), ("x", (3, 3))]


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
        ref.alloc("x", (4, 4))
        ref.mutate("x", (0, 0), [0], extent=(4, 4), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        for i in range(2):
            for j in range(2):
                ref.read("x", (2 * i, 2 * j), [2 * i + j], extent=(2, 2), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, "x")

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
        ref.alloc("x", (4, 4))
        for i in range(2):
            for j in range(2):
                ref.mutate("x", (2 * i, 2 * j), [2 * i + j], extent=(2, 2), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.read("x", (0, 0), [0], extent=(4, 4), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, "x")

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
        [("x", (1, 1)), ("x", (0, 0)), ("x", (2, 2)), ("x", (1, 2)), ("x", (0, 3))],
        threads_per_task=32,
    )
    task = ref.task_threads()

    def task_body():
        ref.alloc("x", (4, 4))
        ref.mutate("x", (0, 0), [0], extent=(2, 2), **smem())
        ref.mutate("x", (2, 2), [0], extent=(2, 2), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.read("x", (1, 1), [0], extent=(2, 2), **smem())
        ref.read("x", (0, 2), [1], extent=(2, 2), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.mutate("x", (1, 1), [0], extent=(2, 2), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        ref.read("x", (0, 0), [0], extent=(2, 2), **smem())
        ref.fence(task, IN_ORDER, IN_ORDER)
        free_smem(ref, "x")

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
        ref = SyncEnvRef([("x", (0, 0))], threads_per_task=64)

        def task_body():
            ref.alloc("x", (2, 32))
            for w in range(2):
                for t in range(32):
                    ref.mutate("x", (w, t), [w * 32 + t], **smem(convergent=True))
            for w in range(2):
                for t in range(32):
                    ref.mutate("x", (1 - w, t), [w * 32 + t], **smem(convergent=True))

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
        [("g", (0, 0)), ("g", (7, 7)), ("smem", (0, 0)), ("cg", ())],
        threads_per_task=32,
    )
    generic_and_async = timelines.cuda_generic_and_async_proxy.get_timeline_set_bits()
    tma_L1 = timelines.tma_to_gmem_async.get_timeline_set_bits()
    # TMA operands: out-of-order, non-convergent; TAG 32 since the commit-group
    # Arrive witnessing them is executed by one warp. dst is an atomic reduce.
    tma_dst = qual_info(CudaGmemLinear, tma_to_gmem_async_instr)
    tma_src = qual_info(CudaSmemLinear, tma_to_gmem_async_instr)
    atomic = timelines.tma_to_gmem_atomic_qual.as_bit()
    ref.alloc("g", (8, 8))

    def task_body(task):
        base = task * 32
        tt = ref.task_threads(task)
        ref.alloc("smem", (8, 8))
        for m in range(8):
            for k in range(8):
                ref.mutate("smem", (m, k), [base + m], **smem(convergent=True))
        if task < rd_hi:
            # r = g[0, 0] (r is CudaRmem, not probed: not modeled)
            ref.read("g", (0, 0), [base], **gmem(convergent=True))
        ref.fence(tt, IN_ORDER, generic_and_async)
        ref.mutate(
            "g", (0, 0), tt, extent=(8, 8), tag=32, atomic_bits=atomic, **tma_dst
        )
        ref.read("smem", (0, 0), tt, extent=(8, 8), tag=32, **tma_src)
        ref.alloc_barrier("cg", ())
        ref.arrive(tt, tma_L1, "cg")
        ref.await_(tt, IN_ORDER, "cg", 0)
        ref.free_barrier("cg")
        free_smem(ref, "smem", task)

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
