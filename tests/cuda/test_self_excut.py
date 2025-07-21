"""Self-tests for excut

Test that the excut tracing utility and concordance checks are
themselves working.  This is to ensure that other test cases that rely
on excut don't incorrectly pass due to excut bugs.

"""

from __future__ import annotations
from dataclasses import dataclass
import functools
import pytest
import sys

from exo import *
from exo.stdlib.scheduling import *
from exo.platforms.cuda import *
from exo.platforms.Sm80 import *

from exo.spork import excut


@dataclass(slots=True)
class excut_trace_smem_str_base:
    def behavior(smem: [f32][4] @ CudaSmemLinear):
        pass

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        action_id = excut.excut_c_str_id("excut_tracer")
        tracer_id = excut.excut_c_str_id(self.tracer_str)
        # excut_tracer(tracer_str, smem_ptr)
        return [
            f"exo_excutLog.log_action({action_id}, 0, __LINE__);",
            f"exo_excutLog.log_u32_arg(exo_smemU32({args.smem.index_ptr(0)}));",
            f"exo_excutLog.log_str_id_arg({tracer_id});",
        ]


@instr
class excut_tracer_foo(excut_trace_smem_str_base):
    tracer_str = "foo"


@instr
class excut_tracer_bar(excut_trace_smem_str_base):
    tracer_str = "bar"


@proc
def Sm80_test_proc():
    gmem: f32[1024] @ CudaGmemLinear
    for i in seq(1, 3):
        with CudaDeviceFunction(blockDim=64):
            for cta in cuda_tasks(0, i * 2):
                smem: f32[256] @ CudaSmemLinear
                for tid in cuda_threads(0, 64):
                    with CudaAsync(Sm80_cp_async_instr):
                        # cp.async.cg.shared.global
                        Sm80_cp_async_f32(
                            smem[4 * tid : 4 * tid + 4],
                            gmem[256 * cta + 4 * tid : 256 * cta + 4 * tid + 4],
                            size=4,
                        )
                # barrier.cta.sync 0
                Fence(cuda_in_order, cuda_in_order)

                # excut_tracer foo/bar
                for tid in cuda_threads(0, 64):
                    if tid < cta:
                        excut_tracer_foo(smem[4 * tid : 4 * tid + 4])
                    else:
                        excut_tracer_bar(smem[4 * tid : 4 * tid + 4])


def test_excut_bootstrap(compiler_Sm80):
    cu = compiler_Sm80.cuda_test_context(Sm80_test_proc, excut=True)
    old_buffer_size = cu._saved_excut_buffer_size
    assert isinstance(old_buffer_size, int)
    cu.set_excut_buffer_size(0)  # Test the retry-on-out-of-memory functionality
    cu(None)
    cu.set_excut_buffer_size(old_buffer_size)

    trace_actions = cu.trace_actions

    # Manually check the trace actions are correct.
    # This test is not future-proof to minute changes to excut outputs, so all
    # other tests use the concordance utility ... but we don't trust it yet for
    # the bootstrapping test.
    assert len(trace_actions) == 2 + 192 * 6
    malloc = trace_actions[0]
    free = trace_actions[-1]

    # cudaMallocAsync(size, stream, out ptr), cudaFreeAsync(ptr, stream)
    assert malloc.action_name == "cudaMallocAsync"
    assert len(malloc.args) == 3
    gmem_base, gmem_size, stream = malloc.args
    assert gmem_size == 1024 * 4
    assert isinstance(gmem_base, int)
    assert malloc.device_name == "cpu"
    assert free.action_name == "cudaFreeAsync"
    assert len(free.args) == 2
    assert free.args[0] == gmem_base

    # Actions are sorted by CUDA kernel launch, then blockIdx, then threadIdx.
    trace_idx = 1
    for i in range(1, 3):
        for blockIdx in range(2 * i):
            smem_base = None
            for threadIdx in range(64):
                linear_tid = blockIdx * 64 + threadIdx
                thread_actions = trace_actions[trace_idx : trace_idx + 3]
                trace_idx += 3
                for action in thread_actions:
                    assert action.device_name == "cuda"
                    assert action.blockIdx == blockIdx
                    assert action.threadIdx == threadIdx

                cp_async, cta_sync, tracer = thread_actions
                assert cp_async.action_name == "cp.async.cg.shared.global"
                assert len(cp_async.args) == 3
                smem_dst, gmem_src, n_bytes = cp_async.args
                if threadIdx == 0:
                    smem_base = smem_dst

                assert smem_dst == smem_base + threadIdx * 16
                assert gmem_src == gmem_base + threadIdx * 16 + blockIdx * 1024
                assert n_bytes == 16

                assert cta_sync.action_name == "barrier.cta.sync"
                assert len(cta_sync.args) == 1
                assert cta_sync.args[0] == 0

                assert tracer.action_name == "excut_tracer"
                assert len(tracer.args) == 2
                assert tracer.args[0] == smem_dst
                assert tracer.args[1] == ("foo" if threadIdx < blockIdx else "bar")


def mkref_test_simple_reference(
    xrg: excut.ExcutReferenceGenerator,
    wrong_blockIdx=False,
    wrong_threadIdx=False,
    skip_all_cp_async=False,
    skip_some_cp_async=False,
    wrong_int_arg=False,
    wrong_deduction=False,
    num_frees=1,
    cuda_launches=2,
    wrong_str_arg=False,
    wrong_device_name=False,
    test_sink=False,
    too_many_args=False,
    too_few_args=False,
    type_mismatch=False,
    reverse_permutation=False,
    wrong_permutation=False,
    allow_permutation=False,
    free_ptr_var=False,
    alt_free_gmem_ptr=False,
    alt_free_gmem_offset=0,
):
    gmem_ptr = xrg.new_varname("gmem_ptr")

    if wrong_device_name:
        xrg.begin_cuda()
        for threadIdx in xrg.stride_threadIdx(1):
            xrg("cudaMallocAsync", gmem_ptr, 4096, 0)
        xrg.end_cuda()
    else:
        xrg("cudaMallocAsync", gmem_ptr, 4096, 0)

    for i in range(1, cuda_launches + 1):
        xrg.begin_cuda()
        for blockIdx in xrg.stride_blockIdx(2 * i, stride=2 if wrong_blockIdx else 1):
            smem_base = xrg.new_varname(f"smem_base_{i}_{blockIdx}")
            for threadIdx in xrg.stride_threadIdx(
                64, stride=2 if wrong_threadIdx else 1
            ):
                smem_dst = smem_base + 16 * threadIdx
                if wrong_deduction:
                    gmem_src = gmem_ptr
                else:
                    gmem_src = gmem_ptr + 16 * threadIdx + 1024 * blockIdx
                str_arg = "foo" if threadIdx < blockIdx or wrong_str_arg else "bar"
                n_bytes = excut.sink if test_sink else 0x1337 if wrong_int_arg else 16
                if type_mismatch:
                    barrier_args = ("barrier.cta.sync", "0")
                elif too_few_args:
                    barrier_args = ("barrier.cta.sync",)
                elif too_many_args:
                    barrier_args = ("barrier.cta.sync", 0, 0)
                else:
                    barrier_args = ("barrier.cta.sync", 0)

                arg_tups = []
                if skip_all_cp_async or (skip_some_cp_async and threadIdx < 10):
                    pass
                else:
                    arg_tups.append(
                        ("cp.async.cg.shared.global", smem_dst, gmem_src, n_bytes)
                    )
                arg_tups.append(barrier_args)
                arg_tups.append(("excut_tracer", smem_dst, str_arg))
                if wrong_permutation:
                    arg_tups[1] = arg_tups[0]
                if reverse_permutation:
                    arg_tups = arg_tups[::-1]
                if allow_permutation:
                    arg_tups = arg_tups[::-1]
                    with xrg.permuted():
                        for tup in arg_tups:
                            xrg(*tup)
                else:
                    for tup in arg_tups:
                        xrg(*tup)
        xrg.end_cuda()

    if free_ptr_var:
        free_ptr = xrg.new_varname("free_ptr")
    elif alt_free_gmem_ptr:
        free_ptr = gmem_ptr[1] + alt_free_gmem_offset
    else:
        free_ptr = gmem_ptr
    for i in range(num_frees):
        xrg("cudaFreeAsync", free_ptr, 0)


def impl_test_trace(mkref, cu, error_substr, **kwargs):
    mkref = functools.partial(mkref, **kwargs)

    if error_substr:
        with pytest.raises(excut.ExcutConcordanceError) as exc:
            cu.excut_concordance(mkref, f"excut_ref_{error_substr}.json")
        # Note, we paste a lot of context in subsequent lines of the message
        # so we only scan line 1 to avoid undermining the test.
        assert error_substr in str(exc.value).split("\n")[0]
    else:
        return cu.excut_concordance(mkref)


def test_simple_reference(compiler_Sm80):
    """Simple diagnoses of mismatches between trace and reference actions

    There are also a few cases of acceptable deviations.
    """
    cu = compiler_Sm80.cuda_test_context(Sm80_test_proc, excut=True)
    cu(None)
    impl_test = functools.partial(impl_test_trace, mkref_test_simple_reference)

    # Note: each of the following is logically a separate test case,
    # but we merge them all together to avoid wasting a ton of time
    # compiling the same CUDA code.
    impl_test(cu, None)
    impl_test(cu, "blockIdx", wrong_blockIdx=True)
    impl_test(cu, "threadIdx", wrong_threadIdx=True)
    impl_test(cu, "device_name", wrong_device_name=True)
    impl_test(cu, None, test_sink=True)
    impl_test(cu, "0 != 1", too_few_args=True)
    impl_test(cu, "2 != 1", too_many_args=True)
    impl_test(cu, "'0' != 0x0", type_mismatch=True)

    # If the reference trace has no action of a certain name, we
    # should ignore the trace having this action logged (filter out
    # these trace actions)
    impl_test(cu, None, skip_all_cp_async=True)
    impl_test(cu, None, num_frees=0)

    # If only some cp.async are missing, then we need to diagnose the
    # trace missing cp.async
    impl_test(cu, "cp.async.cg.shared.global", skip_some_cp_async=True)

    # Check diagnosing incorrect number of (non-filtered) trace actions vs reference actions.
    impl_test(cu, "No reference action left", num_frees=2)
    impl_test(cu, "No reference action left", num_frees=0, cuda_launches=1)
    impl_test(cu, "No trace action left", num_frees=0, cuda_launches=3)

    # Check diagnosing incorrect arguments.
    # This error should be preferred over the above category of error.
    # The wrongness of num_frees/cuda_launches is a red herring.
    impl_test(cu, "0x1337", wrong_int_arg=True)
    impl_test(cu, "gmem_ptr", wrong_deduction=True)
    impl_test(cu, "gmem_ptr", wrong_deduction=True, num_frees=0, cuda_launches=1)
    impl_test(cu, "gmem_ptr", wrong_deduction=True, num_frees=0, cuda_launches=3)
    impl_test(cu, "foo", wrong_str_arg=True)

    # Permutation testing
    impl_test(cu, "action_name", allow_permutation=False, reverse_permutation=True)
    impl_test(cu, None, allow_permutation=True, reverse_permutation=True)
    impl_test(cu, "action_name", allow_permutation=True, wrong_permutation=True)

    # Variable deduction: should allow gmem_ptr and free_ptr to be deduced the same
    impl_test(cu, None, free_ptr_var=True)
    gmem_ptr = cu.get_var("gmem_ptr")
    free_ptr = cu.get_var("free_ptr")
    assert gmem_ptr(cu) == free_ptr(cu)
    assert (gmem_ptr + 100)(cu) == gmem_ptr(cu) + 100

    # Variable deduction: should forbid gmem_ptr and gmem_ptr[1] to be deduced the same
    impl_test(cu, "gmem_ptr[1]", alt_free_gmem_ptr=True)

    # Variable deduction: deduced gmem_ptr == gmem_ptr[1] + 888 (weird test)
    impl_test(cu, None, alt_free_gmem_ptr=True, alt_free_gmem_offset=888)
    gmem_ptr = cu.get_var("gmem_ptr")
    gmem_ptr1 = cu.get_var("gmem_ptr")[(1,)]
    assert gmem_ptr(cu) == gmem_ptr1(cu) + 888


@instr
class excut_trace_init_smem:
    def behavior(smem: [i32][1] @ CudaSmemLinear, value: i32 @ CudaRmem):
        smem[0] = value

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        action_id = excut.excut_c_str_id("excut_trace_init_smem")
        return [
            f"{args.smem.index()} = {args.value.index()};",
            f"exo_excutLog.log_action({action_id}, 0, __LINE__);",
            f"exo_excutLog.log_u32_arg(exo_smemU32({args.smem.index_ptr(0)}));",
            f"exo_excutLog.log_u32_arg({args.value.index()});",
        ]


@instr
class excut_trace_log_smem:
    def behavior(smem: [i32][1] @ CudaSmemLinear):
        pass

    def instance(self):
        self.instr_tl = cuda_in_order_instr
        self.coll_unit = cuda_thread

    def codegen(self, args):
        action_id = excut.excut_c_str_id("excut_trace_log_smem")
        return [
            f"exo_excutLog.log_action({action_id}, 0, __LINE__);",
            f"exo_excutLog.log_u32_arg(exo_smemU32({args.smem.index_ptr(0)}));",
            f"exo_excutLog.log_u32_arg({args.smem.index(0)});",
        ]


"""Trickier test cases:

Test the interaction between variable deductions and permutations.
This is a harder case in excut, because we need to not "eagerly"
deduce variable values in permutations, which could cause unfair
test failures later.

"""


def mkproc_advanced(test_idx_1, test_idx_2):
    @proc
    def cuda_proc():
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                smem: i32[4] @ CudaSmemLinear
                for tid in cuda_threads(0, 1):
                    _0: i32 @ CudaRmem
                    _1: i32 @ CudaRmem
                    _2: i32 @ CudaRmem
                    _0 = 0
                    _1 = 1
                    _2 = 2

                    # Part 0
                    excut_trace_init_smem(smem[0:1], _0)

                    # Part 1
                    excut_trace_init_smem(smem[0:1], _1)
                    excut_trace_init_smem(smem[1:2], _1)
                    excut_trace_init_smem(smem[2:3], _1)
                    excut_trace_init_smem(smem[3:4], _2)

                    # Part 2
                    excut_trace_log_smem(smem[test_idx_1 : test_idx_1 + 1])
                    excut_trace_log_smem(smem[test_idx_2 : test_idx_2 + 1])

                    # Part 3
                    # "Former group"
                    excut_trace_init_smem(smem[0:1], _1)
                    excut_trace_init_smem(smem[1:2], _1)
                    # "Latter group"
                    excut_trace_init_smem(smem[2:3], _1)
                    excut_trace_init_smem(smem[3:4], _2)

    return rename(cuda_proc, f"cuda_proc_{test_idx_1}_{test_idx_2}")


def mkref_advanced(
    xrg: excut.ExcutReferenceGenerator,
    multiple_2s=False,
    wrong_place_2=False,
    too_many=False,
    not_enough=False,
):
    xrg.begin_cuda()
    for threadIdx in xrg.stride_threadIdx(1):
        smem = xrg.new_varname("smem")
        A = xrg.new_varname("A")
        B = xrg.new_varname("B")
        C = xrg.new_varname("C")
        D = xrg.new_varname("D")

        # Part 0
        xrg("excut_trace_init_smem", smem, 0)

        # Part 1
        with xrg.permuted():
            xrg("excut_trace_init_smem", A, 1)
            xrg("excut_trace_init_smem", D, 2)
            xrg("excut_trace_init_smem", B, 1)
            xrg("excut_trace_init_smem", C, 2 if multiple_2s else 1)

        # Part 2
        xrg("excut_trace_log_smem", A, 1)
        xrg("excut_trace_log_smem", D, 2)

        # Part 3
        with xrg.permuted():
            # "Former group"
            xrg("excut_trace_init_smem", A, 1)
            xrg("excut_trace_init_smem", B, 1)
            if wrong_place_2:
                xrg("excut_trace_init_smem", D, 2)
        if not not_enough:
            # "Latter group"
            with xrg.permuted():
                xrg("excut_trace_init_smem", C, 1)
                if not wrong_place_2:
                    xrg("excut_trace_init_smem", D, 2)
            if too_many:
                xrg("excut_trace_init_smem", D, 2)
    xrg.end_cuda()


def impl_test_advanced_A(compiler_Sm80, test_idx_1):
    cu_proc = mkproc_advanced(test_idx_1, 3)
    cu = compiler_Sm80.cuda_test_context(cu_proc, excut=True)
    cu(None)
    mkref = mkref_advanced

    # In part 2, we deduce that A = &smem[test_idx_1]
    #
    # In part 3, in the former group, we have smem[0] = 1 and smem[1] = 1.
    # We will deduce one of them as A = 1 and the other as B = 1.
    # Hence, the test should fail if test_idx_1 == 2.
    result = impl_test_trace(mkref, cu, "mismatch" if test_idx_1 == 2 else None)
    if result:
        xrg, deductions = result
        A = xrg.get_var("A")
        smem = xrg.get_var("smem")
        assert smem(deductions) + 4 * test_idx_1 == A(deductions)

    # Also check this is robust against too many / not enough reference actions.
    # As in the simpler tests, the more specific error (for test_idx_1 = 2) is
    # prioritized over the generic action count mismatch error.
    impl_test_trace(
        mkref,
        cu,
        "mismatch" if test_idx_1 == 2 else "No trace action left",
        too_many=True,
    )
    impl_test_trace(
        mkref,
        cu,
        "mismatch" if test_idx_1 == 2 else "No reference action left",
        not_enough=True,
    )


def test_advanced_A0(compiler_Sm80):
    impl_test_advanced_A(compiler_Sm80, 0)


def test_advanced_A1(compiler_Sm80):
    impl_test_advanced_A(compiler_Sm80, 1)


def test_advanced_A2(compiler_Sm80):
    impl_test_advanced_A(compiler_Sm80, 2)


def test_advanced_multiple_2s(compiler_Sm80):
    cu_proc = mkproc_advanced(1, 3)
    cu = compiler_Sm80.cuda_test_context(cu_proc, excut=True)
    cu(None)
    mkref = mkref_advanced

    # Since both C=2 and D=2 in the reference actions, one of them will fail
    # to match, as the trace will have only 1 of the 4 values set to 2.
    return impl_test_trace(mkref, cu, "Already matched", multiple_2s=True)


def test_advanced_wrong_place_2(compiler_Sm80):
    cu_proc = mkproc_advanced(1, 3)
    cu = compiler_Sm80.cuda_test_context(cu_proc, excut=True)
    cu(None)
    mkref = mkref_advanced

    # Part 3 has 2 groups of permutations
    # The D=2 action is in the latter group, and shouldn't be able to match with
    # the D=2 in the trace which is in the former group.
    return impl_test_trace(mkref, cu, "D !=", wrong_place_2=True)


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


def mkproc_self_excut_CudaWarps(trace_lo, trace_hi, ref_lo, ref_hi):
    n_threads = 32 * (trace_hi - trace_lo)
    blockDim = 32 * trace_hi

    @proc
    def test_proc():
        with CudaDeviceFunction(blockDim=blockDim):
            for x in cuda_tasks(0, 3):
                for y in cuda_tasks(0, 2):
                    with CudaWarps(trace_lo, trace_hi):
                        for z in cuda_threads(0, n_threads):
                            excut_trace_3index(x, y, z)

    return test_proc


def mkref_self_excut_CudaWarps(
    xrg: excut.ExcutReferenceGenerator, trace_lo, trace_hi, ref_lo, ref_hi
):
    n_threads = 32 * (ref_hi - ref_lo)
    xrg.begin_cuda()
    for x in xrg.stride_blockIdx(3, stride=2):
        for y in xrg.stride_blockIdx(2):
            for z in xrg.stride_threadIdx(n_threads, offset=ref_lo * 32):
                xrg("excut_trace_3index", x, y, z)
    xrg.end_cuda()


def test_self_excut_CudaWarps_positive(compiler_Sm80):
    cu = compiler_Sm80.excut_test(
        mkproc_self_excut_CudaWarps,
        mkref_self_excut_CudaWarps,
        trace_lo=2,
        trace_hi=10,
        ref_lo=2,
        ref_hi=10,
    )


def test_self_excut_CudaWarps_negative(compiler_Sm80):
    """Mismatched, real kernel logs using warps [1, 9) but reference expects [2, 10)

    This is really not that deep a test, but it does use the
    excut_test helper function, which otherwise isn't tested in a
    negative test case here.  This is crucial for guarding against
    mistakes in the excut_test helper.

    """
    with pytest.raises(Exception) as exc:
        compiler_Sm80.excut_test(
            mkproc_self_excut_CudaWarps,
            mkref_self_excut_CudaWarps,
            trace_lo=1,
            trace_hi=9,
            ref_lo=2,
            ref_hi=10,
        )
    msg = str(exc.value)
    assert "threadIdx" in msg
    assert "32 != 64" in msg or "64 != 32" in msg
