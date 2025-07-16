from __future__ import annotations

import numpy as np
import pytest

from exo import proc
from exo.core.LoopIR import T
from exo.platforms.cuda import *
from exo.stdlib.scheduling import *

from exo.spork import excut


@instr
class excut_trace_3index:
    def behavior(i0: index, i1: index, i2: index):
        pass

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        action_id = excut.excut_c_str_id("excut_trace_3index")
        return [
            f"exo_excutLog.log_action({action_id}, 0, __LINE__);",
            f"exo_excutLog.log_u32_arg({args.i0.index()});",
            f"exo_excutLog.log_u32_arg({args.i1.index()});",
            f"exo_excutLog.log_u32_arg({args.i2.index()});",
        ]


@proc
def proc_simple_coll_units():
    with CudaDeviceFunction(blockDim=128):
        for task in cuda_tasks(0, 3):
            for w in cuda_threads(0, 4, unit=cuda_warp):
                for t in cuda_threads(0, 32, unit=cuda_thread):
                    excut_trace_3index(0, w, t)
            for w in cuda_threads(0, 3, unit=cuda_warp):
                for t in cuda_threads(0, 16, unit=cuda_thread):
                    excut_trace_3index(1, w, t)
            for t in cuda_threads(0, 66, unit=cuda_thread):
                excut_trace_3index(2, 0, t)


def mkref_simple_coll_units(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(3):
        for w in xrg.stride_threadIdx(4, stride=32):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 0, w, t)
        for w in xrg.stride_threadIdx(3, stride=32):
            for t in xrg.stride_threadIdx(16):
                xrg("excut_trace_3index", 1, w, t)
        for t in xrg.stride_threadIdx(66):
            xrg("excut_trace_3index", 2, 0, t)
    xrg.end_cuda()


def invoke_test(p, mkref, compiler, golden):
    cu = compiler.cuda_test_context(p, sm=80, excut=golden is None)
    if golden is None:
        cu(None)
        cu.excut_concordance(mkref)
    else:
        cu.compare_golden(golden)


def test_simple_coll_units_excut(compiler):
    invoke_test(proc_simple_coll_units, mkref_simple_coll_units, compiler, None)


def test_simple_coll_units_golden(compiler, golden):
    invoke_test(proc_simple_coll_units, mkref_simple_coll_units, compiler, golden)


@proc
def proc_odd_threads():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 3):
            for x in cuda_threads(0, 10, unit=23 * cuda_thread):
                for y in cuda_threads(0, 23):
                    excut_trace_3index(0, x, y)
                for y in cuda_threads(0, 20):
                    excut_trace_3index(1, x, y)


def mkref_odd_threads(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(3):
        for x in xrg.stride_threadIdx(10, stride=23):
            for y in xrg.stride_threadIdx(23):
                xrg("excut_trace_3index", 0, x, y)
            for y in xrg.stride_threadIdx(20):
                xrg("excut_trace_3index", 1, x, y)
    xrg.end_cuda()


def test_odd_threads_excut(compiler):
    invoke_test(proc_odd_threads, mkref_odd_threads, compiler, None)


def test_odd_threads_golden(compiler, golden):
    invoke_test(proc_odd_threads, mkref_odd_threads, compiler, golden)


@proc
def proc_simple_CudaWarps():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 3):
            for w in cuda_threads(0, 3, unit=cuda_warp):
                for t in cuda_threads(0, 32, unit=cuda_thread):
                    excut_trace_3index(0, w, t)
            with CudaWarps(2, 5):
                for w in cuda_threads(0, 3, unit=cuda_warp):
                    for t in cuda_threads(0, 32, unit=cuda_thread):
                        excut_trace_3index(1, w, t)
                for w2 in cuda_threads(0, 2, unit=cuda_warp):
                    for t in cuda_threads(0, 32, unit=cuda_thread):
                        excut_trace_3index(2, w2, t)
                for t in cuda_threads(0, 39, unit=cuda_thread):
                    excut_trace_3index(3, 0, t)


def mkref_simple_CudaWarps(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(3):
        for w in xrg.stride_threadIdx(3, stride=32):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 0, w, t)
        for w in xrg.stride_threadIdx(3, stride=32, offset=64):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 1, w, t)
        for w in xrg.stride_threadIdx(2, stride=32, offset=64):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 2, w, t)
        for t in xrg.stride_threadIdx(39, stride=1, offset=64):
            xrg("excut_trace_3index", 3, 0, t)
    xrg.end_cuda()


def test_simple_CudaWarps_excut(compiler):
    invoke_test(proc_simple_CudaWarps, mkref_simple_CudaWarps, compiler, None)


def test_simple_CudaWarps_golden(compiler, golden):
    invoke_test(proc_simple_CudaWarps, mkref_simple_CudaWarps, compiler, golden)


@proc
def proc_CudaWarps_in_wg():
    with CudaDeviceFunction(blockDim=512):
        for task in cuda_tasks(0, 3):
            for wg in cuda_threads(0, 4, unit=cuda_warpgroup):
                for t in cuda_threads(0, 128, unit=cuda_thread):
                    excut_trace_3index(0, wg, t)
                with CudaWarps(2, 4):
                    for t in cuda_threads(0, 64):
                        excut_trace_3index(1, wg, t)
                    # This should stack with CudaWarps(2, 4),
                    # i.e., be like CudaWarps(3, 4)
                    with CudaWarps(1, 2):
                        for t in cuda_threads(0, 32):
                            excut_trace_3index(2, wg, t)
            with CudaWarps(8, 16):
                for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                    with CudaWarps(0, 2):
                        for t in cuda_threads(0, 64):
                            excut_trace_3index(3, wg, t)
                    with CudaWarps(2, 4):
                        for t in cuda_threads(0, 64):
                            excut_trace_3index(4, wg, t)
                        for t in cuda_threads(0, 42):
                            excut_trace_3index(5, wg, t)


def mkref_CudaWarps_in_wg(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(3):
        for wg in xrg.stride_threadIdx(4, stride=128):
            for t in xrg.stride_threadIdx(128):
                xrg("excut_trace_3index", 0, wg, t)
            for t in xrg.stride_threadIdx(64, offset=64):
                xrg("excut_trace_3index", 1, wg, t)
            for t in xrg.stride_threadIdx(32, offset=96):
                xrg("excut_trace_3index", 2, wg, t)
        for wg in xrg.stride_threadIdx(2, stride=128, offset=256):
            for t in xrg.stride_threadIdx(64):
                xrg("excut_trace_3index", 3, wg, t)
            for t in xrg.stride_threadIdx(64, offset=64):
                xrg("excut_trace_3index", 4, wg, t)
            for t in xrg.stride_threadIdx(42, offset=64):
                xrg("excut_trace_3index", 5, wg, t)
    xrg.end_cuda()


def test_CudaWarps_in_wg_excut(compiler):
    invoke_test(proc_CudaWarps_in_wg, mkref_CudaWarps_in_wg, compiler, None)


def test_CudaWarps_in_wg_golden(compiler, golden):
    invoke_test(proc_CudaWarps_in_wg, mkref_CudaWarps_in_wg, compiler, golden)


@proc
def proc_simple_named_warps():
    with CudaDeviceFunction(
        warp_config=[CudaWarpConfig("abc", 16), CudaWarpConfig("xyz", 4)]
    ):
        for task in cuda_tasks(0, 3):
            for t in cuda_threads(0, 320, unit=cuda_thread):
                excut_trace_3index(0, 0, t)
            for t in cuda_threads(0, 300, unit=cuda_thread):
                excut_trace_3index(1, 0, t)
            with CudaWarps(name="abc"):
                for w in cuda_threads(0, 16, unit=cuda_warp):
                    for t in cuda_threads(0, 32):
                        excut_trace_3index(2, w, t)
                with CudaWarps(2, 4, name="abc"):
                    for t in cuda_threads(0, 42):
                        excut_trace_3index(3, 0, t)
            with CudaWarps(name="xyz"):
                for w in cuda_threads(0, 3, unit=cuda_warp):
                    for t in cuda_threads(0, 10):
                        excut_trace_3index(4, w, t)
                with CudaWarps(1, 3, name="xyz"):
                    for w in cuda_threads(0, 2, unit=cuda_warp):
                        for t in cuda_threads(0, 32):
                            excut_trace_3index(5, w, t)
                    with CudaWarps(1, 2):
                        for t in cuda_threads(0, 32):
                            excut_trace_3index(6, 0, t)


def mkref_simple_named_warps(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(3):
        for t in xrg.stride_threadIdx(320):
            xrg("excut_trace_3index", 0, 0, t)
        for t in xrg.stride_threadIdx(300):
            xrg("excut_trace_3index", 1, 0, t)
        for w in xrg.stride_threadIdx(16, stride=32):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 2, w, t)
        for t in xrg.stride_threadIdx(42, offset=64):
            xrg("excut_trace_3index", 3, 0, t)
        for w in xrg.stride_threadIdx(3, stride=32, offset=16 * 32):
            for t in xrg.stride_threadIdx(10):
                xrg("excut_trace_3index", 4, w, t)
        for w in xrg.stride_threadIdx(2, stride=32, offset=17 * 32):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 5, w, t)
        for t in xrg.stride_threadIdx(32, offset=18 * 32):
            xrg("excut_trace_3index", 6, 0, t)
    xrg.end_cuda()


def test_simple_named_warps_excut(compiler):
    invoke_test(proc_simple_named_warps, mkref_simple_named_warps, compiler, None)


def test_simple_named_warps_golden(compiler, golden):
    invoke_test(proc_simple_named_warps, mkref_simple_named_warps, compiler, golden)


cuda_1_3 = CollUnit((3,), (1,), "cuda_1_3")
cuda_1_8 = CollUnit((8,), (1,), "cuda_1_8")
cuda_64_128 = CollUnit((128,), (64,), "cuda_64_128")


@proc
def proc_strange_domain():
    with CudaDeviceFunction(blockDim=384):
        for task in cuda_tasks(0, 3):
            # unit = 1 thread per 8
            # x = threadIdx.x % 8
            for x in cuda_threads(0, 8, unit=cuda_1_8):
                # y = threadIdx.x / 128
                for y in cuda_threads(0, 3, unit=128 * cuda_thread):
                    # z = (threadIdx.x / 8) % 16
                    for z in cuda_threads(0, 16, unit=cuda_thread):
                        excut_trace_3index(x, y, z)

            for t in cuda_threads(0, 384):
                excut_trace_3index(t, t, t)

            # x = threadIdx.x / 48
            for x in cuda_threads(0, 8, unit=48 * cuda_thread):
                # y = threadIdx.x % 3; mask out if y >= 2
                for y in cuda_threads(0, 2, unit=cuda_1_3):
                    # z = (threadIdx.x / 3) % 16
                    for z in cuda_threads(0, 16, unit=cuda_thread):
                        excut_trace_3index(x, y, z)

            # unit = 2 warps per 4
            # x = (threadIdx.x / 64) % 2
            for x in cuda_threads(0, 2, unit=cuda_64_128):
                for y in cuda_threads(0, 3, unit=2 * cuda_warp):
                    for z in cuda_threads(0, 64):
                        excut_trace_3index(x, y, z)
                    with CudaWarps(1, 2):
                        for z in cuda_threads(0, 32):
                            excut_trace_3index(x, y, z)


def mkref_strange_domain(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for blockIdx in xrg.stride_blockIdx(3):
        for threadIdx in xrg.stride_threadIdx(384):
            x = threadIdx % 8
            y = threadIdx // 128
            z = (threadIdx // 8) % 16
            xrg("excut_trace_3index", x, y, z)
            xrg("excut_trace_3index", threadIdx, threadIdx, threadIdx)
            x = threadIdx // 48
            y = threadIdx % 3
            z = (threadIdx // 3) % 16
            if y < 2:
                xrg("excut_trace_3index", x, y, z)
            x = (threadIdx // 64) % 2
            y = threadIdx // 128
            z = threadIdx % 64
            xrg("excut_trace_3index", x, y, z)
            if z >= 32:
                xrg("excut_trace_3index", x, y, z - 32)
    xrg.end_cuda()


def test_strange_domain_excut(compiler):
    invoke_test(proc_strange_domain, mkref_strange_domain, compiler, None)


def test_strange_domain_golden(compiler, golden):
    invoke_test(proc_strange_domain, mkref_strange_domain, compiler, None)


def mkproc_scalar_write(unit):
    @proc
    def proc_scalar_write(dram: i32[1] @ DRAM, src: i32 @ CudaGridConstant):
        gmem: i32[1] @ CudaGmemLinear
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for t in cuda_threads(0, 1, unit=unit):
                    gmem[0] = src
        cudaMemcpyAsync_dtoh_1i32(1, dram[:], gmem[:])

    return proc_scalar_write


def test_scalar_write_positive(compiler):
    cu = compiler.cuda_test_context(mkproc_scalar_write(cuda_thread), sm=80, excut=True)
    src = np.ndarray(shape=(1,), dtype=np.int32)
    dst = np.ndarray(shape=(1,), dtype=np.int32)
    src[0] = 1337
    cu(None, dst, src)
    assert dst[0] == src[0]


def test_scalar_write_negative(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(mkproc_scalar_write(cuda_warp), sm=80)
    assert "gmem[0] = src" in str(exc.value)
