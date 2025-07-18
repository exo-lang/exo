from __future__ import annotations

import pytest

from exo import proc
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *
from exo.platforms.Sm90 import *
from exo.stdlib.scheduling import *


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
        compiler.cuda_test_context(mkproc_cuda_tasks(missing_cuda_tasks=True), sm=80)
    assert "cuda_tasks" in str(exc.value)


def test_alone_cuda_tasks_0(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_cuda_tasks(extra_stmt_before=True), sm=80)
    assert "cuda_tasks" in str(exc.value)


def test_alone_cuda_tasks_1(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_cuda_tasks(extra_stmt_after=True), sm=80)
    assert "cuda_tasks" in str(exc.value)


@proc
def duplicate_cuda_tasks_name():
    with CudaDeviceFunction(blockDim=256):
        for a in cuda_tasks(0, 4):
            for a in cuda_tasks(0, 8):
                pass


def test_duplicate_cuda_tasks_name(compiler):
    # Feel free to fix this limitation if you care.
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(duplicate_cuda_tasks_name, sm=80)


@proc
def proc_invalid_par():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 4):
            for n in par(0, 8):
                pass


def test_invalid_par(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_invalid_par, sm=80)
    assert "unexpected loop mode par" in str(exc.value)


@proc
def proc_invalid_cuda_tasks():
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 4):
            for n in cuda_threads(0, 8):
                for task2 in cuda_tasks(0, 4):
                    pass


def test_invalid_cuda_tasks(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_invalid_cuda_tasks, sm=80)
    assert "cuda_tasks" in str(exc.value)


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
    compiler.cuda_test_context(mkproc_grid_constant_window(), sm=80)


def test_grid_constant_window_negative_0(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_grid_constant_window(is_window=True), sm=80)
    assert "parameter gc cannot be a window"


def test_grid_constant_window_negative_1(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(
            mkproc_grid_constant_window(is_window_stmt=True), sm=80
        )
    assert "parameter gc_win cannot be a window"


@proc
def proc_non_const_smem(not_a_constant: size):
    assert not_a_constant >= 1
    with CudaDeviceFunction(blockDim=256):
        for task in cuda_tasks(0, 3):
            smem: f32[not_a_constant, 256] @ CudaSmemLinear
            for tid in cuda_threads(0, 256):
                smem[0, tid] = 10


def test_non_const_smem(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_non_const_smem, sm=80)
    assert "[not_a_constant, 256]" in str(exc.value)


@proc
def proc_CUtensorMap_wrong_mem(M: size, N: size, C: f32[M, N] @ DRAM):
    C_t = C[:256, :128] @ Sm90_tensorMap(128, 256, 128)


def test_CUtensorMap_wrong_mem(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_CUtensorMap_wrong_mem, sm="90a")
    assert "CudaGmemLinear" in str(exc.value)


@proc
def proc_no_trailing_barrier(gmem: f32[128] @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=32):
        for task in cuda_tasks(0, 1):
            smem: f32[128] @ CudaSmemLinear
            bar: barrier @ CudaMbarrier
            Await(bar, cuda_in_order, 1)
            for tid in cuda_threads(0, 32):
                with CudaAsync(Sm80_cp_async):
                    (
                        Sm80_cp_async_f32(
                            smem[4 * tid : 4 * tid + 4],
                            gmem[4 * tid : 4 * tid + 4],
                            size=4,
                        )
                        >> bar
                    )
            Arrive(Sm80_cp_async, 1) >> bar


def test_no_trailing_barrier(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_no_trailing_barrier, sm=80)
    assert "does not take trailing barrier expression" in str(exc.value)


@proc
def proc_wrong_CudaAsync():
    with CudaAsync(Sm80_cp_async_instr):
        pass


def test_wrong_CudaAsync(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_wrong_CudaAsync, sm=80)
    assert (
        "CudaAsync(Sm80_cp_async_instr) requires instr-tl cuda_in_order_instr"
        in str(exc.value)
    )


@proc
def proc_wrong_CudaDeviceFunction(foo: f32 @ CudaGmemLinear):
    with CudaDeviceFunction(blockDim=1024):
        for task in cuda_tasks(0, 1):
            with CudaDeviceFunction(blockDim=256):
                for taskB in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 1):
                        foo = 10


def test_wrong_CudaDeviceFunction(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(proc_wrong_CudaDeviceFunction, sm=80)
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
        compiler.cuda_test_context(test_proc, sm=80)
    assert (
        "CudaGridConstant does not allow mutable access in a scope with instr-tl cuda_in_order_instr"
        in str(exc.value)
    )


def test_reduce_wrong_instr_tl(compiler):
    @proc
    def test_proc(gmem: f32 @ CudaGmemLinear):
        gmem += 5

    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(test_proc, sm=80)
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
        compiler.cuda_test_context(test_proc, sm=80)
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
    compiler.cuda_test_context(mkproc_alloc_in_cuda(CudaSmemLinear), sm=80)


def test_gmem_in_cuda(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_test_context(mkproc_alloc_in_cuda(CudaGmemLinear), sm=80)
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
        compiler.cuda_test_context(test_proc, sm="90a")
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
        compiler.cuda_test_context(test_proc, sm=80)
    assert "requires instr-tl Sm80_cp_async_instr" in str(exc.value)
