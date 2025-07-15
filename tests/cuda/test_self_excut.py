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
            f"exo_excutLog.log_u32_arg(exo_smemU32(&{args.smem.index(0)}));",
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


def test_excut_bootstrap(compiler):
    cu = compiler.cuda_test_context(Sm80_test_proc, excut=True)
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
):
    gmem_ptr = xrg.new_varname("gmem_ptr")

    if wrong_device_name:
        for threadIdx in xrg.stride_threadIdx(1):
            xrg("cudaMallocAsync", gmem_ptr, 4096, 0)
    else:
        xrg("cudaMallocAsync", gmem_ptr, 4096, 0)

    for i in range(1, cuda_launches + 1):
        for blockIdx in xrg.stride_blockIdx(2 * i, 2 if wrong_blockIdx else 1):
            smem_base = xrg.new_varname(f"smem_base_{i}_{blockIdx}")
            for threadIdx in xrg.stride_threadIdx(64, 2 if wrong_threadIdx else 1):
                smem_dst = smem_base + 16 * threadIdx
                if wrong_deduction:
                    gmem_src = gmem_ptr
                else:
                    gmem_src = gmem_ptr + 16 * threadIdx + 1024 * blockIdx
                str_arg = "foo" if threadIdx < blockIdx or wrong_str_arg else "bar"
                n_bytes = None if test_sink else 0x1337 if wrong_int_arg else 16
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

    for i in range(num_frees):
        xrg("cudaFreeAsync", gmem_ptr, 0)


def impl_test_trace(mkref, cu, error_substr, **kwargs):
    mkref = functools.partial(mkref, **kwargs)

    if error_substr:
        with pytest.raises(excut.ExcutConcordanceError) as exc:
            cu.excut_concordance(mkref, f"excut_ref_{error_substr}.json")
        # Note, we paste a lot of context in subsequent lines of the message
        # so we only scan line 1 to avoid undermining the test.
        assert error_substr in str(exc.value).split("\n")[0]
    else:
        cu.excut_concordance(mkref)


def test_simple_reference(compiler):
    """Simple diagnoses of mismatches between trace and reference actions

    There are also a few cases of acceptable deviations.
    """
    cu = compiler.cuda_test_context(Sm80_test_proc, excut=True)
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
