# Memory, instructions, instr-tl, sync-tl specific to CUDA sm_100 and sm_100a (Blackwell)
from __future__ import annotations

# Currently we import from the exo.spork directory,
# which users shouldn't import directly.
from ..spork.timelines import (
    cuda_in_order_instr,
    tcgen05_mma_instr,
    tcgen05_cp_instr,
    tcgen05_shift_instr,
    tcgen05_ld_instr,
    tcgen05_st_instr,
    cuda_in_order,
    cuda_generic_and_async_proxy,
    cuda_tmem_qual_tl_dict,
)

__all__ = [
    "cuda_in_order_instr",
    "tcgen05_mma_instr",
    "tcgen05_cp_instr",
    "tcgen05_shift_instr",
    "tcgen05_ld_instr",
    "tcgen05_st_instr",
    "cuda_in_order",
    "cuda_generic_and_async_proxy",
    "cuda_tmem_qual_tl_dict",
]

# We use these but don't put them in __all__
from math import prod
from ..API import (
    instr,
    memwin_template,
    window_encoder,
    window_indexer,
    Memory,
    MemGlobalC,
    MemIncludeC,
    SpecialWindow,
    WindowFeatures,
    WindowEncoder,
    WindowIndexer,
    ScalarInfo,
    UtilInjector,
    CIR_Wrapper,
    InstrInfo,
    InstrArgs,
    AtomicityInfo,
)
from ..spork.cuda_memory import (
    CudaBasicDeviceVisible,
    CudaSmemLinear,
    CudaSmemAtomicity16B,
)
from ..spork.coll_algebra import (
    cuda_warp,
    cuda_warpgroup,
    cuda_warp_in_cluster,
    cuda_warp_in_cluster_strided,
    cuda_warp_in_cluster,
    cuda_cta_in_cluster,
)

from .cuda import InlinePtxGen, simple_ptx_c_lines
from .Sm90 import Sm90_SmemSwizzled
