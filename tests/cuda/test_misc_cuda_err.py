from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *


TestTensorMap = Sm90_tensorMap(128, 256, 128)


def mkproc_cuda_tasks(
    missing_cuda_tasks=False, extra_stmt_before=False, extra_stmt_after=False
):
    have_cuda_tasks = not missing_cuda_tasks

    @proc
    def test_proc(ptr: f32[4] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            if extra_stmt_before:
                ptr[0] = 100
            if have_cuda_tasks:
                for a in cuda_tasks(0, 4):
                    for t in cuda_threads(0, 1):
                        ptr[a] = 3
                if extra_stmt_after:
                    for t in cuda_threads(0, 1):
                        ptr[0] = 100

    return simplify(test_proc)


def test_missing_cuda_tasks(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_tasks, missing_cuda_tasks=True)
    assert "cuda_tasks" in str(exc.value)


def test_alone_cuda_tasks_0(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_tasks, extra_stmt_before=True)
    assert "cuda_tasks" in str(exc.value)


def test_alone_cuda_tasks_1(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_cuda_tasks, extra_stmt_after=True)
    assert "cuda_tasks" in str(exc.value)


def mkproc_duplicate_cuda_tasks_name():
    @proc
    def duplicate_cuda_tasks_name():
        with CudaDeviceFunction(blockDim=256):
            for a in cuda_tasks(0, 4):
                for a in cuda_tasks(0, 8):
                    pass

    return duplicate_cuda_tasks_name


def test_duplicate_cuda_tasks_name(compiler):
    # Feel free to fix this limitation if you care.
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_duplicate_cuda_tasks_name)


def mkproc_invalid_par():
    @proc
    def proc_invalid_par():
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 4):
                for n in par(0, 8):
                    pass

    return proc_invalid_par


def test_invalid_par(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_invalid_par)
    assert "unexpected loop mode par" in str(exc.value)


def mkproc_invalid_cuda_tasks():
    @proc
    def proc_invalid_cuda_tasks():
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 4):
                for n in cuda_threads(0, 8):
                    for task2 in cuda_tasks(0, 4):
                        pass

    return proc_invalid_cuda_tasks


def test_invalid_cuda_tasks(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_invalid_cuda_tasks)
    assert "cuda_tasks" in str(exc.value)


def mkproc_invalid_cuda_threads():
    @proc
    def proc_invalid_cuda_threads(foo: i32[4] @ DRAM):
        for x in cuda_threads(0, 4):
            foo[x] = 10

    return proc_invalid_cuda_threads


def test_invalid_cuda_threads(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_invalid_cuda_threads)
    assert "unexpected loop mode cuda_threads in x loop" in str(exc.value)


def mkproc_grid_constant_window(is_window=False, is_window_stmt=False):
    @proc
    def test_proc(gc: [f32][8] @ CudaGridConstant, x: f32 @ CudaGmemLinear):
        gc_copy: f32 @ CudaGridConstant
        gc_copy = gc[1]
        gc_win = gc[1:]
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 1):
                    if is_window_stmt:
                        x = gc_win[1]
                    if is_window:
                        x = gc[1]
                    x = gc_copy

    return simplify(test_proc)


def test_grid_constant_window_positive(compiler):
    compiler.cuda_cpu_test(mkproc_grid_constant_window)


def test_grid_constant_window_negative_0(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_grid_constant_window, is_window=True)
    assert "parameter gc cannot be a window"


def test_grid_constant_window_negative_1(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_grid_constant_window, is_window_stmt=True)
    assert "parameter gc_win cannot be a window"


def mkproc_non_const_smem():
    @proc
    def proc_non_const_smem(not_a_constant: size):
        assert not_a_constant >= 1
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 3):
                smem: f32[not_a_constant, 256] @ CudaSmemLinear
                for tid in cuda_threads(0, 256):
                    smem[0, tid] = 10

    return proc_non_const_smem


def test_non_const_smem(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_non_const_smem)
    assert "[not_a_constant, 256]" in str(exc.value)


def mkproc_CUtensorMap_wrong_mem():
    @proc
    def proc_CUtensorMap_wrong_mem(M: size, N: size, C: f32[M, N] @ DRAM):
        C_t = C[:256, :128] @ TestTensorMap

    return proc_CUtensorMap_wrong_mem


def test_CUtensorMap_wrong_mem(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_CUtensorMap_wrong_mem, sm="90a")
    assert "CudaGmemLinear" in str(exc.value)


def mkproc_no_trailing_barrier():
    # fmt: off
    @proc
    def proc_no_trailing_barrier(gmem: f32[128] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                smem: f32[128] @ CudaSmemLinear
                bar: barrier @ CudaMbarrier
                Await(bar, cuda_in_order, 1)
                for tid in cuda_threads(0, 32):
                    with CudaAsync(Sm80_cp_async):
                        Sm80_cp_async_f32(
                            smem[4 * tid : 4 * tid + 4],
                            gmem[4 * tid : 4 * tid + 4],
                            size=4,
                        ) >> bar
                Arrive(Sm80_cp_async, 1) >> bar
    return proc_no_trailing_barrier
    # fmt: on


def test_no_trailing_barrier(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_no_trailing_barrier)
    assert "does not take trailing barrier expression" in str(exc.value)


def mkproc_wrong_CudaAsync():
    @proc
    def proc_wrong_CudaAsync():
        with CudaAsync(Sm80_cp_async_instr):
            pass

    return proc_wrong_CudaAsync


def test_wrong_CudaAsync(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_wrong_CudaAsync)
    assert (
        "CudaAsync(Sm80_cp_async_instr) requires instr-tl cuda_in_order_instr"
        in str(exc.value)
    )


def mkproc_wrong_CudaDeviceFunction():
    @proc
    def proc_wrong_CudaDeviceFunction(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=1024):
            for task in cuda_tasks(0, 1):
                with CudaDeviceFunction(blockDim=256):
                    for taskB in cuda_tasks(0, 1):
                        for tid in cuda_threads(0, 1):
                            foo = 10

    return proc_wrong_CudaDeviceFunction


def test_wrong_CudaDeviceFunction(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_wrong_CudaDeviceFunction)
    assert "requires instr-tl cpu_in_order_instr" in str(exc.value)


def test_write_CudaGridConstant(compiler):
    @proc
    def test_proc(foo: f32 @ DRAM, gmem: f32 @ CudaGmemLinear):
        gc: f32 @ CudaGridConstant
        gc = foo
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 1):
                    gc = gmem

    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(lambda: test_proc)
    assert (
        "CudaGridConstant does not allow mutable access in a scope with instr-tl cuda_in_order_instr"
        in str(exc.value)
    )


def test_reduce_wrong_instr_tl(compiler):
    @proc
    def test_proc(gmem: f32 @ CudaGmemLinear):
        gmem += 5

    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(lambda: test_proc)
    assert (
        "CudaGmemLinear does not allow any access in a scope with instr-tl cpu_in_order_instr"
        in str(exc.value)
    )


def test_read_wrong_instr_tl(compiler):
    @proc
    def test_proc(gmem: f32 @ CudaGmemLinear):
        local: f32 @ DRAM
        local = gmem

    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(lambda: test_proc)
    assert (
        "CudaGmemLinear does not allow reads in a scope with instr-tl cpu_in_order_instr"
        in str(exc.value)
    )


def mkproc_alloc_in_cuda(test_mem):
    @proc
    def test_proc(dst: f32[256] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                src: f32[256] @ test_mem
                for tid in cuda_threads(0, 256):
                    src[tid] = 19
                    dst[tid] = src[tid]

    return test_proc


def test_smem_in_cuda(compiler):
    compiler.cuda_cpu_test(mkproc_alloc_in_cuda, test_mem=CudaSmemLinear)


def test_gmem_in_cuda(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_alloc_in_cuda, test_mem=CudaGmemLinear)
    assert (
        "CudaGmemLinear cannot be allocated in a scope with instr-tl cuda_in_order_instr"
        in str(exc.value)
    )


def test_window_instr_tl(compiler):
    @proc
    def test_proc(M: size, N: size, gmem: f32[M, N] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                tensor_map = gmem[:, :] @ Sm90_tensorMap(128, 128, 256)

    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(lambda: test_proc, sm="90a")
    assert (
        "Sm90_tensorMap(128, 128, 256) cannot be constructed in a scope with instr-tl cuda_in_order_instr"
        in str(exc.value)
    )


def test_call_instr_tl(compiler):
    @proc
    def test_proc(gmem: f32[32] @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=128):
            for task in cuda_tasks(0, 1):
                smem: f32[32] @ CudaSmemLinear
                for tid in cuda_threads(0, 32):
                    Sm80_cp_async_f32(smem[tid : tid + 1], gmem[tid : tid + 1], size=1)

    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(lambda: test_proc, sm=80)
    assert "requires instr-tl Sm80_cp_async_instr" in str(exc.value)


def test_pyparser_unexpected_shift():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            x >> y

    assert ">>" in str(exc.value)


def test_pyparser_unexpected_plus():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            x * y

    assert ">>" in str(exc.value)


def test_pyparser_fence_trailing_barrier_exprs():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Fence(cuda_in_order, cuda_in_order) >> bar

    assert "Fence" in str(exc.value)
    assert ">>" in str(exc.value)


def test_pyparser_await_trailing_barrier_exprs():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Await(bar, cuda_in_order, 0) >> bar

    assert "Await" in str(exc.value)
    assert ">>" in str(exc.value)


def test_pyparser_multiple_with_item():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: i32, y: i32):
            with CudaDeviceFunction(blockDim=32) as a, CudaAsync(wgmma_async) as b:
                for task in cuda_tasks(0, 1):
                    pass

    assert "1 withitem" in str(exc.value)


def test_pyparser_unknown_loop_mode():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(z: size, t: f32[z]):
            for i in bogus_loop_name(0, z):
                t[i] = 1

    assert "bogus_loop_name" in str(exc.value)


def test_pyparser_bad_loop_mode_arg():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(z: size, t: f32[z]):
            for i in seq(0, z, bogus_kwarg=1000):
                t[i] = 1


@instr
class bogus_test_instr:
    def behavior():
        pass

    def instance(self):
        self.coll_unit = cuda_thread
        self.instr_tl = cuda_in_order_instr
        self.instr_format = ["// bogus_test_instr"]


def test_pyparser_too_many_BarrierExpr():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 32):
                        bar: barrier @ CudaMbarrier
                        Arrive(cuda_in_order, 1) >> bar
                        bogus_test_instr() >> bar >> bar
                        Await(bar, cuda_in_order, 0)

    assert "bogus_test_instr cannot have more than 1 trailing barrier expr" in str(
        exc.value
    )


def test_pyparser_suggest_sync_tl():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    Fence(Sm80_cp_async_instr, cuda_in_order)

    assert "Sm80_cp_async?" in str(exc.value)


def test_pyparser_not_sync_tl():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    Fence("xyzzy", cuda_in_order)

    assert "xyzzy" in str(exc.value)


def test_pyparser_Arrive_kwarg():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1, bogus_kwarg=19) >> bar
                    Await(bar, cuda_in_order, 0)

    assert "bogus_kwarg" in str(exc.value)


def test_pyparser_Arrive_wrong_arg_count():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1, 19) >> bar
                    Await(bar, cuda_in_order, 0)

    assert "Arrive expects 2 arguments" in str(exc.value)


def test_pyparser_Await_wrong_arg_count():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1) >> bar
                    Await(bar, cuda_in_order)

    assert "Await expects 3 arguments" in str(exc.value)


def test_pyparser_Fence_wrong_arg_count():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    Fence()

    assert "Fence expects 2 arguments" in str(exc.value)


def test_pyparser_Await_N_not_int():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc():
            with CudaDeviceFunction(blockDim=32):
                for task in cuda_tasks(0, 1):
                    bar: barrier @ CudaMbarrier
                    Arrive(cuda_in_order, 1) >> bar
                    Await(bar, cuda_in_order, 0.75)

    assert "Await" in str(exc.value)
    assert "0.75" in str(exc.value)


def test_pyparser_unexpected_BarrierType():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(x: f32 @ CudaMbarrier):
            pass

    msg = str(exc.value)
    assert "CudaMbarrier" in msg


def test_typecheck_barrier_type():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(xyzzy: f32):
            Arrive(cuda_in_order, 1) >> xyzzy

    msg = str(exc.value)
    assert "requires barrier type" in msg
    assert "xyzzy: f32" in msg


def test_typecheck_barrier_indices():
    with pytest.raises(Exception) as exc:

        @proc
        def test_proc(xyzzy: f32):
            with CudaDeviceFunction(clusterDim=8, blockDim=256):
                for task in cuda_tasks(0, 4):
                    bars: barrier[2, 4] @ CudaMbarrier
                    for m in cuda_threads(0, 2, unit=4 * cuda_cta_in_cluster):
                        Arrive(cuda_in_order, 1) >> bars[m]

    msg = str(exc.value)
    assert "expected 2 indices" in msg


def test_typecheck_unexpected_SpecialWindow():
    with pytest.raises(Exception) as exc:

        @proc
        def bar(t: [f32][4, 8]):
            pass

        @proc
        def foo(t: f32[8, 8]):
            bar(t[4:, :] @ TestTensorMap)

    msg = str(exc.value)
    assert "Can only create SpecialWindow as part of WindowStmt" in msg


def test_CudaDeviceFunction_blockDim_valid():
    for b in (-64, 0, 0.5, 1):
        with pytest.raises(Exception) as exc:
            CudaDeviceFunction(blockDim=b)
        msg = str(exc.value)
        assert f"blockDim={b} must be a positive multiple of 32" in msg


def test_CudaDeviceFunction_duplicate_warp_name():
    warp_config = [
        CudaWarpConfig("foobar", 4),
        CudaWarpConfig("okay", 4),
        CudaWarpConfig("foobar", 8),
    ]
    with pytest.raises(Exception) as exc:
        CudaDeviceFunction(warp_config=warp_config)
    msg = str(exc.value)
    assert "warp name 'foobar'" in msg


def test_CudaDeviceFunction_setmaxnreg_multiple_128():
    warp_config = [
        CudaWarpConfig("foobar", 4, setmaxnreg_inc=232),
        CudaWarpConfig("okay", 6, setmaxnreg_dec=40),
    ]
    with pytest.raises(Exception) as exc:
        CudaDeviceFunction(warp_config=warp_config)
    msg = str(exc.value)
    assert "128" in msg
    assert "setmaxnreg" in msg
    assert "blockDim=320" in msg


def test_CudaDeviceFunction_setmaxnreg_uniform_128():
    warp_config = [
        CudaWarpConfig("foobar", 5, setmaxnreg_inc=232),
        CudaWarpConfig("okay", 7, setmaxnreg_dec=40),
    ]
    with pytest.raises(Exception) as exc:
        CudaDeviceFunction(warp_config=warp_config)
    msg = str(exc.value)
    assert "128" in msg
    assert "uniform" in msg
    assert "setmaxnreg" in msg


def test_CudaDeviceFunction_setmaxnreg_valid():
    for nreg in (16, 100, 320, 232):
        warp_config = [
            CudaWarpConfig("foobar", 8, setmaxnreg_inc=232),
            CudaWarpConfig("okay", 4, setmaxnreg_dec=nreg),
        ]
        with pytest.raises(Exception) as exc:
            CudaDeviceFunction(warp_config=warp_config)
        msg = str(exc.value)
        if nreg == 232:
            assert "232 used both for setmaxnreg.inc and setmaxnreg.dec" in msg
        else:
            assert "setmaxnreg must be a multiple of 8 in [24, 256]" in msg


def test_CudaDeviceFunction_bad_warp_name():
    for name in ("hello world", "***"):
        with pytest.raises(Exception) as exc:
            CudaDeviceFunction(warp_config=[CudaWarpConfig(name, 4)])
        msg = str(exc.value)
        assert "C identifier" in msg
        assert repr(name) in msg
