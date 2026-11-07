"""PROTOTYPE camspork sync env trace tests (endgame_plan/plan_sketch_camspork_test.md)."""

from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.spork import timelines
from exo.spork.timelines import Qual_tl, cuda_in_order_instr
from exo.spork.camspork_ref import (
    SyncEnvRef,
    compare_traces,
    dump_reference_trace,
    parse_camspork_trace,
)


def smem_access_info():
    q = CudaSmemLinear.qual_tl_dict[cuda_in_order_instr]
    return dict(
        initial_bit=q.initial_bit,
        precondition_bits=q.precondition_bits,
        mask=CudaSmemLinear.make_qual_tl_mask(),
        convergent=True,  # Assign/Reduce are always convergent accesses
    )


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


def ref_device_function(ref, probes, body):
    """Implicit syncs of CudaDeviceFunction (sync_check.py lowering)."""
    cuda_bits = timelines.cuda_stream_sync.get_timeline_set_bits()
    cpu_bit = timelines.cpu_in_order_qual.as_bit()
    ref.sync(
        "Fence",
        ref.cpu,
        probes,
        lambda: ref.fence(ref.cpu, cpu_bit | cuda_bits, cuda_bits),
    )
    body()
    ref.sync("JoinThreads", ref.cpu, probes, lambda: ref.join_threads(ref.cpu))
    ref.sync("Fence", ref.cpu, probes, lambda: ref.fence(ref.cpu, cuda_bits, cuda_bits))


def mkref_named_numeric_warps(probes):
    ref = SyncEnvRef(threads_per_task=192)
    info = smem_access_info()
    bits = timelines.cuda_in_order.get_timeline_set_bits()
    task_threads = ref.tids(0, 192)

    def task_body():
        ref.alloc("x", (2, 32))
        for w in range(2):
            for t in range(32):
                ref.access(True, "x", (w, t), [(3 + w) * 32 + t], **info)
        ref.sync(
            "Fence", task_threads, probes, lambda: ref.fence(task_threads, bits, bits)
        )
        for w in range(2):
            for t in range(32):
                ref.access(True, "x", (w, t), [w * 32 + t], **info)
        ref.sync(
            "Fence", task_threads, probes, lambda: ref.fence(task_threads, bits, bits)
        )
        ref.free("x")
        # End of device task
        ref.sync(
            "JoinThreads", task_threads, probes, lambda: ref.join_threads(task_threads)
        )

    ref_device_function(ref, probes, task_body)
    return ref


PROBES = [("x", (0, 0)), ("x", (1, 31))]


def run_trace_test(compiler, p, ref, tmp_path):
    actual_path = str(tmp_path / "camspork_trace.jsonl")
    p.sync_check(_camspork_trace=(actual_path, PROBES))
    actual = parse_camspork_trace(actual_path)
    qual_names = [str(q) for q in Qual_tl.get_all()]
    dump_reference_trace(str(tmp_path / "reference_trace.txt"), ref.trace, qual_names)
    return compare_traces(actual, ref.trace, qual_names)


def test_named_numeric_warps_positive(compiler, tmp_path):
    p = mkproc_named_numeric_warps()
    errors = run_trace_test(compiler, p, mkref_named_numeric_warps(PROBES), tmp_path)
    assert not errors, "\n".join(errors)


def test_named_numeric_warps_negative(compiler, tmp_path):
    """Lowering bug stand-in: proc uses B warps [0, 2), reference expects [1, 3)."""
    p = mkproc_named_numeric_warps(wrong_lo=True)
    errors = run_trace_test(compiler, p, mkref_named_numeric_warps(PROBES), tmp_path)
    assert errors
    assert "x[0, 0] mutate" in errors[0]
