# Memory, instructions, and actor kinds specific to CUDA sm_90 and sm_90a (H100)
# Everything exported by this module starts with Sm90_
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.actor_kinds import (
    sig_tma_to_smem,
    sig_tma_to_gmem,
    sig_wgmma_rmem_a,
    sig_wgmma_rmem_d,
    sig_wgmma_smem,
    tma_to_smem_async,
    tma_to_gmem_async,
    wgmma_async,
    wgmma_async_smem,
    wgmma_fence_1,
    wgmma_fence_2,
    cuda_async_proxy,
    cuda_async_proxy_wgmma,
    cuda_generic_and_async_proxy,
)

__all__ = [
    "sig_tma_to_smem",
    "sig_tma_to_gmem",
    "sig_wgmma_rmem_a",
    "sig_wgmma_rmem_d",
    "sig_wgmma_smem",
    "tma_to_smem_async",
    "tma_to_gmem_async",
    "wgmma_async",
    "wgmma_async_smem",
    "wgmma_fence_1",
    "wgmma_fence_2",
    "cuda_async_proxy",
    "cuda_async_proxy_wgmma",
    "cuda_generic_and_async_proxy",
]

# We use these but don't put them in __all__
from ..API import instr
from ..core.memory import memwin_template
from ..spork.cuda_memory import *
from ..spork.coll_algebra import cuda_warp, cuda_warpgroup
