"""Self-tests for excut

Test that the excut tracing utility and concordance checks are
themselves working.  This is to ensure that other test cases that rely
on excut don't incorrectly pass due to excut bugs.

"""

from __future__ import annotations
from dataclasses import dataclass

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
    xtc = compiler.excut_test_context(Sm80_test_proc)
    assert isinstance(xtc._saved_buffer_size, int)
    xtc.set_buffer_size(0)  # Test the retry-on-out-of-memory functionality
    xtc(None)

    trace_actions = xtc.trace_actions

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
    gmem_size, stream, gmem_base = malloc.args
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
