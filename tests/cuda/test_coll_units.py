from __future__ import annotations

import numpy as np
import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
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
    return cu


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


def mkproc_named_warps(
    wrong_name=False,
    missing_name=False,
    change_name=False,
    missing_lo=False,
    missing_hi=False,
    out_of_range=False,
):
    abc = None if missing_name else "wrong_name" if wrong_name else "abc"
    xyz = "abc" if change_name else "xyz"
    _2 = None if missing_lo else 2
    _10 = None if missing_hi else 10
    _16 = 17 if out_of_range else 16

    @proc
    def proc_named_warps():
        with CudaDeviceFunction(
            warp_config=[CudaWarpConfig("abc", 16), CudaWarpConfig("xyz", 4)]
        ):
            for task in cuda_tasks(0, 3):
                for t in cuda_threads(0, 320, unit=cuda_thread):
                    excut_trace_3index(0, 0, t)
                for t in cuda_threads(0, 300, unit=cuda_thread):
                    excut_trace_3index(1, 0, t)
                with CudaWarps(0, _16, name=abc):
                    for w in cuda_threads(0, 16, unit=cuda_warp):
                        for t in cuda_threads(0, 32):
                            excut_trace_3index(2, w, t)
                    with CudaWarps(_2, _10, name=abc):
                        for w2 in cuda_threads(0, 4, unit=2 * cuda_warp):
                            with CudaWarps(1, 2):
                                for t in cuda_threads(0, 17):
                                    excut_trace_3index(3, w2, t)
                with CudaWarps(name="xyz"):
                    for w in cuda_threads(0, 3, unit=cuda_warp):
                        for t in cuda_threads(0, 10):
                            excut_trace_3index(4, w, t)
                    with CudaWarps(1, 3, name=xyz):
                        for w in cuda_threads(0, 2, unit=cuda_warp):
                            for t in cuda_threads(0, 32):
                                excut_trace_3index(5, w, t)
                        with CudaWarps(1, 2):
                            for t in cuda_threads(0, 32):
                                excut_trace_3index(6, 0, t)
                with CudaWarps(1, 4, name="xyz"):
                    for t in cuda_threads(0, 32):
                        excut_trace_3index(7, 0, t)

    return proc_named_warps


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
        for w2 in xrg.stride_threadIdx(4, stride=64, offset=2 * 32):
            for t in xrg.stride_threadIdx(17, offset=32):
                xrg("excut_trace_3index", 3, w2, t)
        for w in xrg.stride_threadIdx(3, stride=32, offset=16 * 32):
            for t in xrg.stride_threadIdx(10):
                xrg("excut_trace_3index", 4, w, t)
        for w in xrg.stride_threadIdx(2, stride=32, offset=17 * 32):
            for t in xrg.stride_threadIdx(32):
                xrg("excut_trace_3index", 5, w, t)
        for t in xrg.stride_threadIdx(32, offset=18 * 32):
            xrg("excut_trace_3index", 6, 0, t)
        for t in xrg.stride_threadIdx(32, offset=17 * 32):
            xrg("excut_trace_3index", 7, 0, t)
    xrg.end_cuda()


def test_simple_named_warps_excut(compiler):
    invoke_test(mkproc_named_warps(), mkref_simple_named_warps, compiler, None)


def test_simple_named_warps_golden(compiler, golden):
    invoke_test(mkproc_named_warps(), mkref_simple_named_warps, compiler, golden)


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
    cu = compiler.cuda_test_context(mkproc_scalar_write(cuda_thread), sm=80)
    src = np.ndarray(shape=(1,), dtype=np.int32)
    dst = np.ndarray(shape=(1,), dtype=np.int32)
    src[0] = 1337
    cu(None, dst, src)
    assert dst[0] == src[0]


def test_scalar_write_negative(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(mkproc_scalar_write(cuda_warp), sm=80)
    assert "gmem[0] = src" in str(exc.value)


def test_wrong_warp_name(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_named_warps(wrong_name=True), sm=80)
    assert "wrong_name" in str(exc.value)


def test_missing_warp_name(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_named_warps(missing_name=True), sm=80)
    assert "None" in str(exc.value)


def test_change_warp_name(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_named_warps(change_name=True), sm=80)
    assert "cannot change warp name" in str(exc.value)


def test_missing_CudaWarps_lo(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_named_warps(missing_lo=True), sm=80)
    assert " lo " in str(exc.value)


def test_missing_CudaWarps_lo(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_named_warps(missing_hi=True), sm=80)
    assert " hi " in str(exc.value)


def test_CudaWarps_out_of_range(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_named_warps(out_of_range=True), sm=80)
    assert "17" in str(exc.value)


def mkproc_CudaWarps_in_loop(unit):
    warp_config = [CudaWarpConfig("abc", 16), CudaWarpConfig("xyz", 4)]

    @proc
    def proc_CudaWarps_in_loop():
        with CudaDeviceFunction(clusterDim=2, warp_config=warp_config):
            for task in cuda_tasks(0, 1):
                # The CudaAsync and seq-loop should have no effect on with CudaWarps.
                with CudaAsync(Sm80_cp_async_instr):
                    for seq_i in seq(0, 5):
                        # This loop will mess up the CudaWarps if the unit is
                        # sub-CTA, but not if it's inter-CTA.
                        for par_i in cuda_threads(0, 2, unit=unit):
                            with CudaWarps(name="abc"):
                                pass

    return proc_CudaWarps_in_loop


def test_CudaWarps_in_loop_positive(compiler):
    cu = compiler.cuda_test_context(
        mkproc_CudaWarps_in_loop(cuda_cta_in_cluster), sm="90a"
    )


def test_CudaWarps_in_loop_negative(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(
            mkproc_CudaWarps_in_loop(cuda_warpgroup), sm="90a"
        )
    assert "cuda_threads loop" in str(exc.value)


def mkproc_warp_instr(warp=cuda_warp):
    @proc
    def proc_warp_instr(dst: f32[16, 8] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                d: f32[16, 8] @ Sm80_RmemMatrixD(16, 8)
                for i in cuda_threads(0, 1, unit=warp):
                    Sm80_mma_store_d_tf32(dst[:, :], d[:, :])

    return proc_warp_instr


def test_warp_instr_positive(compiler):
    cu = compiler.cuda_test_context(mkproc_warp_instr(cuda_warp), sm=80)


def test_warp_instr_negative(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(mkproc_warp_instr(cuda_thread), sm=80)
    assert "32" in str(exc.value)


@instr
class excut_warpgroup_instr:
    def behavior(i: index):
        pass

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_warpgroup

    def codegen(self, args):
        action_id = excut.excut_c_str_id("excut_warpgroup_instr")
        return [
            f"exo_excutLog.log_action({action_id}, 0, __LINE__);",
            f"exo_excutLog.log_u32_arg({args.i.index()});",
        ]


def mkproc_warpgroup_align(abc_warps=8, xyz_warps=24):
    warp_config = [CudaWarpConfig("abc", abc_warps), CudaWarpConfig("xyz", xyz_warps)]

    @proc
    def proc_warpgroup_align():
        with CudaDeviceFunction(warp_config=warp_config):
            for task in cuda_tasks(0, 1):
                with CudaWarps(name="xyz"):
                    for wg in cuda_threads(0, 2, unit=cuda_warpgroup):
                        excut_warpgroup_instr(wg)

    return proc_warpgroup_align


def mkref_warpgroup_align(xrg: excut.ExcutReferenceGenerator):
    abc_warps = 8
    xrg.begin_cuda()
    for wg in xrg.stride_threadIdx(2, stride=128, offset=abc_warps * 32):
        for t in xrg.stride_threadIdx(128):
            xrg("excut_warpgroup_instr", wg)
    xrg.end_cuda()


def test_warpgroup_align_positive(compiler):
    invoke_test(mkproc_warpgroup_align(8, 24), mkref_warpgroup_align, compiler, None)


def test_warpgroup_align_negative(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(mkproc_warpgroup_align(7, 25), sm=80)
    assert "alignment" in str(exc.value)


def mkproc_warpgroup_shape(unit0, unit1):
    @proc
    def proc_warpgroup_shape():
        with CudaDeviceFunction(blockDim=512):
            for task in cuda_tasks(0, 1):
                for i0 in cuda_threads(0, 2, unit=unit0):
                    for i1 in cuda_threads(0, 2, unit=unit1):
                        excut_warpgroup_instr(0)

    return proc_warpgroup_shape


def test_warpgroup_shape_positive(compiler):
    p = mkproc_warpgroup_shape(256 * cuda_thread, 128 * cuda_thread)
    cu = compiler.cuda_test_context(p, sm=80)


def test_warpgroup_shape_negative(compiler):
    # Weird test.
    # unit1 (the inner CollUnit) has 128 threads, but in the wrong shape
    # (so it's not actually a warpgroup).
    unit0 = CollUnit((4, 128), (4, 64), "unit1")
    unit1 = CollUnit((4, 128), (2, 64), "unit1")
    p = mkproc_warpgroup_shape(unit0, unit1)
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(p, sm=80)
    assert "shape (2, 64)" in str(exc.value)


def mkproc_cuda_threads_bounds(lo=0, variable_bounds=False):
    @proc
    def proc_cuda_threads_bounds(N: size, dst: i32[N] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, N):  # This is OK
                for seq in seq(0, N):  # This is OK
                    if variable_bounds:
                        for test_iter in cuda_threads(lo, N):
                            dst[test_iter] = 0
                    else:
                        for test_iter in cuda_threads(lo, 200 + 56):
                            if test_iter < N:
                                dst[test_iter] = 0

    return simplify(proc_cuda_threads_bounds)


def test_cuda_threads_bounds_postive(compiler):
    cu = compiler.cuda_test_context(mkproc_cuda_threads_bounds(), sm=80)


def test_cuda_threads_bounds_wrong_lo(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(mkproc_cuda_threads_bounds(lo=137), sm=80)
    assert "test_iter" in str(exc.value)


def test_cuda_threads_bounds_variable(compiler):
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(
            mkproc_cuda_threads_bounds(variable_bounds=True), sm=80
        )
    assert "test_iter" in str(exc.value)


# fmt: off


def test_invalid_index_expression(compiler):
    @proc
    def seq_fail():
        # TeX: version seq_fail 1
        # TeX: begin seq_fail
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                vals: f32[16, 16, 16, 4] @ CudaRmem
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color line *
                        #   b
                        for s in seq(0, 16):
                            # Expecting tiling chain 256 $\mapsto$ ... $\mapsto$ 1
                            # TeX: color line *
          #                                    b                            gggggggggggggggggg
          # Failure: non-cuda_threads iterator s consumed when we only have m: $256\mapsto 16$
                            # TeX: color line *
                            #    g  b  v
                            vals[m, s, n, 0] = 0
                            # Remedy: reorder s and n
                            # TeX: color line *
                            #    g  v  b            gggggggggggggggggg  vvvvvvvvvvvvvvvv
                            vals[m, n, s, 0] = 0  # m: $256\mapsto 16$, n: $16\mapsto 1$
        # TeX: end seq_fail

    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(seq_fail, sm=80)
    assert "cuda_threads" in str(exc.value)


def test_mismatched_CollIndexExpr(compiler):
    @proc
    def mismatched():
        # TeX: version mismatched 1
        # TeX: begin mismatched[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                vals: f32[16, 4] @ CudaRmem
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=4 * cuda_thread):# m = threadIdx.x / 4
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 4, unit=cuda_thread):# n = threadIdx.x % 4
                        # TeX: color line *
                        #    g  v         ggggggggggggggggg  vvvvvvvvvvvvvvv
                        vals[m, n] = 0  # m: $256\mapsto 4$, n: $4\mapsto 1$
                # TeX: color line *
                #            rrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
                # Deduction: threadIdx.x / 4 (m), threadIdx.x % 4 (n)
                with CudaWarps(1, 3):
                    for m in cuda_threads(0, 16, unit=4 * cuda_thread):# m = (threadIdx.x - 32) / 4
                        for n in cuda_threads(0, 4, unit=cuda_thread): # n = threadIdx.x % 4
                            # TeX: color line *
                            #    g  v         ggggggggggggggggg  vvvvvvvvvvvvvvv
                            vals[m, n] = 0  # m: $256\mapsto 4$, n: $4\mapsto 1$
                    # TeX: color line *
                    #                       rrrrrrrrrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
                    # Mismatched deduction: (threadIdx.x - 32) / 4 (m), threadIdx.x % 4 (n)
                # TeX: color line *
                #   y
                for t in cuda_threads(0, 16, unit=cuda_thread):# t = threadIdx.x
                    # TeX: color line *
                    #   b
                    for s in seq(0, 4):
                        # TeX: color line *
                        #    y  b         yyyyyyyyyyyyyyyyy   b
                        vals[t, s] = 0  # t: $256\mapsto 1$;  s not distributed
                # TeX: color line *
                #                       rrrrrrrrrrr  y
                # Mismatched deduction: threadIdx.x (t)  [1 dims != 2 dims]
        # TeX: end mismatched[0]

    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(mismatched, sm=80)
    assert "(threadIdx.x - 32) / 4" in str(exc.value)
    assert "threadIdx.x / 4" in str(exc.value)


def test_matched_CollIndexExpr(compiler):
    @proc
    def matched():
        # TeX: version matched 1
        # TeX: begin matched[0]
        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                vals: f32[16, 8] @ CudaRmem
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=8 * cuda_thread):# m = threadIdx.x / 8
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 8, unit=cuda_thread):# n = threadIdx.x % 8
                        # TeX: color line *
                        #    g  v         ggggggggggggggggg  vvvvvvvvvvvvvvv
                        vals[m, n] = 0  # m: $128\mapsto 8$, n: $8\mapsto 1$
                # TeX: color line *
                #            rrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
                # Deduction: threadIdx.x / 8 (m), threadIdx.x % 8 (n)
                #
                # TeX: color remark! matched[0]
                #                                                            rrrrrrrrrrrrr
                # The names of the variables do not matter; only the deduced CollIndexExpr
                for a in cuda_threads(0, 16, unit=8 * cuda_thread):# a = threadIdx.x / 8
                    for b in cuda_threads(0, 8, unit=cuda_thread):# b = threadIdx.x % 8
                        vals[a, b] = 0
                # TeX: color line *
                #            rrrrrrrrrrrrrrr      rrrrrrrrrrrrrrr
                # Deduction: threadIdx.x / 8 (a), threadIdx.x % 8 (b)
                #
                # TeX: remark! matched[0]
                # We can also transpose the loops and have it still work.
                # TeX: color remark matched[0]
                #                                                               rrrrrrrrrrrrr
                # The tiling chain is different, but it works since the deduced CollIndexExpr
                # tuple matches. This example requires a custom collective unit (every 8th thread).
                # n = threadIdx.x % 8
                # TeX: color line *
                #   v
                for n in cuda_threads(0, 8, unit=CollUnit((8,), (1,), "one_thread_per_8", None)):
                                               # CollUnit(domain, box, __repr__, scaled_dim_idx)
                    # TeX: color line *
                    #   g
                    for m in cuda_threads(0, 16, unit=cuda_thread):  # m = threadIdx.x / 8
                        # TeX: color line *
                        #    g  v         vvvvvvvvvvvvvvvvvv  gggggggggggggggg
                        vals[m, n] = 0  # n: $128\mapsto 16$, m: $16\mapsto 1$ (not same order as indices)
                # TeX: color line *
                #            rrrrrrrrrrrrrrr ggg  rrrrrrrrrrrrrrr vvv
                # Deduction: threadIdx.x / 8 (m), threadIdx.x % 8 (n)
        # TeX: end matched[0]
    cu = compiler.cuda_test_context(matched, sm=80)


def test_broken_chain(compiler):
    @proc
    def broken_chain():
        # TeX: version broken_chain 3
        # TeX: begin broken_chain[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                # TeX: color line *
                #                                 rrrrrrrrrrrrrrrrrrrrrr
                vals: f32[16, 8, 2] @ CudaRmem  # $t_a = 256$; $t_n = 1$
                # TeX: color line *
                #   y                                                   yyyyyyyyyyyyyyyyyyy
                for b in cuda_threads(0, 2, unit=128 * cuda_thread):  # b: $256\mapsto 128$
                    # TeX: color line *
                    #   g                                                  ggggggggggggggggg
                    for m in cuda_threads(0, 16, unit=8 * cuda_thread):  # m: $128\mapsto 8$
                        # TeX: color line *
                        #   v                                             vvvvvvvvvvvvvvv
                        for n in cuda_threads(0, 8, unit=cuda_thread):  # n: $8\mapsto 1$
                            # TeX: color line *
                            #   b
                            for s in seq(0, 2):
                                # TeX: color line *
                # ggggggggggggggggg     vvvvvvvvvvvvvvv                                   rrrrrrrrrrrrrr
                # m: $128\mapsto 8$ and n: $8\mapsto 1$ is insufficient to reach the goal $256\mapsto 1$
                # TeX: color line *
                #                                                  b
                # Distributed memory analysis fails upon consuming s (non-cuda_threads iter)
                                # TeX: color line *
                                #    g  v  b
                                vals[m, n, s] = 0
                # TeX: color line *
                #                                                                                      y
                # This rule enforces that two different thread collectives (in this case, differing by b value)
                # can't access the same distributed shard.
        # TeX: end broken_chain[0]
    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(broken_chain, sm=80)
    assert "128" in str(exc.value)


def test_warp_example(compiler):
    @proc
    def warp_example():
        # TeX: begin warp_example[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                # TeX: color line *
                #      rrrr                                         rrrrrrrrrrrrrrrrrrrrrrr
                D: f32[2, 4, 6, 16, 8] @ Sm80_RmemMatrixD(16, 8)  # $t_a = 256$, $t_n = 32$
                # TeX: color line *
                #            rrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrrrrrrrrrr
                # Deduction: threadIdx.x / 128, threadIdx.x % 128 / 32
                # TeX: color line *
                #   gg
                for mw in cuda_threads(0, 2, unit=128 * cuda_thread):# mw = threadIdx.x / 128
                    # TeX: color line *
                    #   vv
                    for nw in cuda_threads(0, 4, unit=32 * cuda_thread):
                        # nw = threadIdx.x % 128 / 32
                        # TeX: color line *
                        #   b
                        for s in seq(0, 6):
                            # TeX: color line *
                            #                      gg  vv  b           gggggggggggggggggggg  vvvvvvvvvvvvvvvvvvv
                            Sm80_mma_zero_d_tf32(D[mw, nw, s, :, :]) # mw: $256\mapsto 128$, nw: $128\mapsto 32$
                    # TeX: color line *
                    #                                   b                                        rrrrrrrrrr
                    # Indexing by seq-for iter variable s is OK as we already reached the target $t_n$ = 32
    # TeX: end warp_example[0]

    cu = compiler.cuda_test_context(warp_example, sm=80)


@proc
def chain_0():
    # TeX: begin chain[0]
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 1):
            # TeX: color line *
            #         rrrrr                   rrrrrrrrrrrrrrrrrrrrrr
            vals: f32[16, 8, 2] @ CudaRmem  # $t_a = 256$; $t_n = 1$
            # TeX: color line *
            #            rrrrrrrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrr
            # Deduction: (threadIdx.x - 64) / 8, threadIdx.x % 8
            # tile = (256,), box = (256,), offset = (0,)
            with CudaWarps(2, 6):  # Offset by 2*32 = 64 threads; box = (6-2)*32 = 128 threads
                # TeX: color line *
                #        gggggg
                # tile = (256,), box = (128,), offset=(64,)
                # TeX: color line *
                #   g                                                  ggggggggggggggggg
                for m in cuda_threads(0, 16, unit=8 * cuda_thread):  # m: $256\mapsto 8$
                    # TeX: color line *
                    #        gggg
                    # tile = (8,), box = (8,), offset = (0,)
                    # TeX: color line *
                    #   v                                             vvvvvvvvvvvvvvv
                    for n in cuda_threads(0, 8, unit=cuda_thread):  # n: $8\mapsto 1$
                        # tile = (1,), box = (1,), offset = (0,)
                        # TeX: color line *
                        #   b
                        for s in seq(0, 2):
                            # TeX: color line *
                            #    g  v  b         ggggggggggggggggg  vvvvvvvvvvvvvvv
                            vals[m, n, s] = 0  # m: $256\mapsto 8$, n: $8\mapsto 1$
                            excut_trace_3index(m, n, s)
# TeX: end chain[0]


def mkref_chain_0(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for threadIdx in xrg.stride_threadIdx(128, offset=64):
        m = threadIdx // 8
        n = threadIdx % 8
        for s in range(2):
            xrg("excut_trace_3index", m, n, s)
    xrg.end_cuda()


def test_chain_0_excut(compiler):
    invoke_test(chain_0, mkref_chain_0, compiler, None)


def test_chain_0_golden(compiler, golden):
    cu = invoke_test(chain_0, mkref_chain_0, compiler, golden)
    # vals should be deduced to be an array
    assert "vals[s] = 0" in cu.fn.get_source_by_ext("cuh")


@proc
def chain_1():
    # TeX: begin chain[1]
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 1):
            # TeX: color line *
            #         rrrrrrrr                rrrrrrrrrrrrrrrrrrrrrr
            vals: f32[16, 8, 2] @ CudaRmem  # $t_a = 256$; $t_n = 1$
            # TeX: color line *
            #            rrrrrrrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrrr
            # Deduction: threadIdx.x % 128 / 8, threadIdx.x % 8, threadIdx.x / 128
            # TeX: color line *
            #   y                                                   yyyyyyyyyyyyyyyyyyy
            for b in cuda_threads(0, 2, unit=128 * cuda_thread):  # b: $256\mapsto 128$
                # TeX: color line *
                #   g                                                  ggggggggggggggggg
                for m in cuda_threads(0, 16, unit=8 * cuda_thread):  # m: $128\mapsto 8$
                    # TeX: color line *
                    #   v                                             vvvvvvvvvvvvvvv
                    for n in cuda_threads(0, 8, unit=cuda_thread):  # n: $8\mapsto 1$
                        # TeX: color line *
                        #    g  v  y         yyyyyyyyyyyyyyyyyyy  ggggggggggggggggg  vvvvvvvvvvvvvvv
                        vals[m, n, b] = 0  # b: $256\mapsto 128$, m: $128\mapsto 8$, n: $8\mapsto 1$
                        excut_trace_3index(m, n, b)


def mkref_chain_1(xrg: excut.ExcutReferenceGenerator):
    xrg.begin_cuda()
    for threadIdx in xrg.stride_threadIdx(256):
        m = (threadIdx // 8) % 16
        n = threadIdx % 8
        b = threadIdx // 128
        xrg("excut_trace_3index", m, n, b)
    xrg.end_cuda()


def test_chain_1_excut(compiler):
    invoke_test(chain_1, mkref_chain_1, compiler, None)


def test_chain_1_golden(compiler, golden):
    cu = invoke_test(chain_1, mkref_chain_1, compiler, golden)
    # vals should be deduced to be a scalar
    assert "vals = 0" in cu.fn.get_source_by_ext("cuh")


def test_repeated_index(compiler):
    @proc
    def repeated_index():
        # TeX: begin repeated[0]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                # TeX: color line *
                #                                   rrrrrrrrrrrrrrrrrrrrrr
                vals: f32[16, 16, 16] @ CudaRmem  # $t_a = 256$, $t_n = 1$
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):
                        # TeX: color line *
                        #    g  g  v         gggggggggggggggggg  gggggggggggggggggg
                        vals[m, m, n] = 0  # m: $256\mapsto 16$, m: $256\mapsto 16$
                # TeX: color line *
                #                                       ggggggggggg                            rrrrrrrrr
                # Fail: we encounter another index with $t_0 = 256$ before we reach the target $t_n = 1$

    with pytest.raises(Exception) as exc:
        cu = compiler.cuda_test_context(repeated_index, sm=80)
    assert "repeated" in str(exc.value)
# TeX: end repeated[0]


def test_repeated_index_fixed(compiler):
    @proc
    def repeated_index():
        # TeX: begin repeated[1]
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                # TeX: color line *
                #         rrrrrr  yy                rrrrrrrrrrrrrrrrrrrrrr
                vals: f32[16, 16, 16] @ CudaRmem  # $t_a = 256$, $t_n = 1$
                # TeX: color line *
                #            rrrrrrrrrrrrrrrr  rrrrrrrrrrrrrrrr
                # Deduction: threadIdx.x % 16, threadIdx.x / 16
                # TeX: color line *
                #   g
                for m in cuda_threads(0, 16, unit=16 * cuda_thread):# m = threadIdx. / 16
                    # TeX: color line *
                    #   v
                    for n in cuda_threads(0, 16, unit=cuda_thread):# n = threadIdx.x % 16
                        # TeX: color line *
                        #    v  g            gggggggggggggggggg  vvvvvvvvvvvvvvvv
                        vals[n, m, m] = 0  # m: $256\mapsto 16$, n: $16\mapsto 1$
                # TeX: color line *
                #                                                                             rrrrrrrrr
                # Second m not deduced as distributed idx since we already reached the target $t_n = 1$
    cu = compiler.cuda_test_context(repeated_index, sm=80)
    cuh = cu.fn.get_source_by_ext("cuh")
    assert "float vals[16]" in cuh
    assert "vals[exo_16thr_m] = 0" in cuh
