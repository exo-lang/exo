from dataclasses import dataclass
from typing import Dict, List, Type, Optional
from .memory import (
    DRAM,
    MemWin,
    AllocableMemWin,
    Memory,
    SpecialWindow,
    BarrierMechanism,
)
from ..spork.coll_algebra import CollUnit
from ..spork.timelines import Instr_tl, Qual_tl


@dataclass(slots=True)
class AtomicityInfo:
    qual_tl_list: List[Qual_tl]


@dataclass(slots=True)
class AccessInfo:
    mem: Type[MemWin] = DRAM
    out_of_order: bool = None
    access_by_owner_only: Optional[bool] = False  # See distributed_coll_units

    # Set automatically, don't edit.
    const: bool = False
    write_only: bool = False

    # UNSTABLE: for experiments on figuring out how to model TMA's
    # zero-pad behavior (reading GMEM) and implied predication (writing GMEM)
    # when out-of-bounds coordinates are given.
    allow_out_of_bounds: bool = False

    # For warp shuffles and TMA: identifies that the first
    # len(distributed_coll_units) dimensions of the window parameter
    # are expected to be distributed dimensions, as if accessed with
    #
    # def my_instr(..., param : [T][sz0, sz1, ...]):
    #     for i0 in cuda_threads(0, sz0, unit=distributed_coll_units[0]):
    #         for i1 in cuda_threads(0, sz1, unit=distributed_coll_units[1]):
    #             # ...
    #             param[i0, i1, ... ]
    #
    # access_by_owner_only=True is poorly-maintained, may be removed.
    # If false, each element is modelled as accessed by the calling thread collective Tc.
    # If true, each element is modelled as-if it were accessed in the above loop
    # literally, i.e. with To being the owner of the shard, the access is by Tc \cap To.
    distributed_coll_units: List[CollUnit] = ()

    atomicity: Optional[AtomicityInfo] = None


@dataclass(init=False, slots=True)
class InstrInfo:
    # instr_format split by lines; strongly suggest InstrInfo.codegen(...) instead.
    instr_format: Optional[List[str]]

    # Duplicate utils or includes will be removed during codegen,
    # even if the duplicates are from different instructions.
    c_utils: List[str]  # Added into .c files before compiled procs
    c_includes: List[str]  # Included into .c files before utils and procs
    cu_utils: List[str]  # Added into .cuh files in exo_CudaUtil namespace
    cu_includes: List[str]  # Included into .cuh files before utils and procs

    coll_unit: CollUnit
    instr_tl: Instr_tl

    # For each formal parameter x, the InstrInfo.instance function must
    # initialize the attributes of access_info[str(x)]: AccessInfo.
    access_info: Dict[str, AccessInfo]

    # The instr expects a trailing barrier expr iff barrier_mechanism is not None.
    # barrier_coll_units is akin to AccessInfo.distributed_coll_units.
    # The barrier must be allocated @ barrier_mechanism and
    # have dim len(barrier_coll_units).
    barrier_mechanism: Optional[Type[BarrierMechanism]]
    barrier_coll_units: List[CollUnit]

    # For internal use
    _tparam_dict: dict
    _formatted_tparam_kwargs: str
