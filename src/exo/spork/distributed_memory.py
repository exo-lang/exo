from dataclasses import dataclass
from math import prod
from typing import Optional, List, Type, Dict, Tuple

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR
from .coll_algebra import (
    CollTiling,
    CollUnit,
    CollIndexExpr,
    CollParam,
)
from .barrier_usage import BarrierUsage
from .loop_modes import _CodegenPar


@dataclass(slots=True, init=False)
class ThreadIter:
    """Information for an iter variable from a for-threads parallel loop (cuda_threads)"""

    codegen_par: _CodegenPar
    coll_index_expr: CollIndexExpr
    coll_tiling: CollTiling
    parent_tile_num_threads: int
    child_tile_num_threads: int
    thread_pitch: int

    def __init__(self, coll_tiling: CollTiling):
        self.codegen_par = _CodegenPar(
            coll_tiling.codegen_expr.codegen(),
            (coll_tiling.codegen_lo, coll_tiling.codegen_hi),
        )
        self.coll_index_expr = coll_tiling.tile_expr
        self.coll_tiling = coll_tiling
        assert isinstance(coll_tiling.parent, CollTiling)
        self.parent_tile_num_threads = coll_tiling.parent.tile_num_threads()
        self.child_tile_num_threads = coll_tiling.tile_num_threads()
        self.thread_pitch = coll_tiling.thread_pitch

    def cname(self, name):
        """Mangling convention for generated C variables"""
        N = self.child_tile_num_threads
        return f"exo_{N}thr_{name}"


@dataclass(slots=True)
class DistributedAllocState(object):
    # Some GPU allocations are "distributed", when the collective unit
    # (e.g. CTA) that allocated a tensor doesn't match the "native unit"
    # of the memory type (e.g. thread for a register; warp for a wmma tile).
    #
    # Some of the leading dimensions of the tensor will be deduced to be
    # "distributed", i.e., correspond to a thread index rather than a
    # (CUDA syntactic) array index. e.g. if the CTA size is 512, something like
    #
    # foo : f32[32,16,4] @ CudaRmem  # Access with 2D grid of 32 x 16 threads
    #
    # may lower to `float foo[4]` since the first 2 dimensions are distributed.
    #
    # We deduce this from the usage of the memory, and enforce that each thread
    # only accesses its own index. TODO explain all this tiling stuff better.
    #
    # In the rewrite phase, we will strip the leading len(first_distributed_iters)-many
    # indices from all uses of the memory ... this is just a hack for
    # code lowering; ignore that this changes the real meaning of the LoopIR.

    # Set upon inspecting the first read/write of the allocation
    # Subsequent uses check that the usage pattern matches the recorded
    # first_distributed_iters exactly.
    first_usage_stmt: Optional[LoopIR.stmt]
    first_distributed_iters: List[Sym]

    # Within an indexing expression name[i0,i1,...],
    # 0 <= iN < distributed_extents[N] for all uses
    # (although iN need to range all the way to distributed_extents[N]).
    distributed_extents: List[int]

    # CollTiling at the point of the Exo object code allocation
    alloc_coll_tiling: CollTiling

    # Target native unit; we want
    optional_native_unit: Optional[CollUnit]

    # Deduced suballocation. 1:1 mapping between thread collectives
    # in the leaf CollTiling, and distributed slices.
    # (ignoring unused indices/thread collectives)
    # Should match the optional_native_unit, if supplied;
    # see DistributedIdxFsm.check_native_unit
    leaf_coll_tiling: Optional[CollTiling]

    # Only used for barrier types
    Arrive: Optional[CollTiling]
    Await: Optional[CollTiling]
    ReverseArrive: Optional[CollTiling]
    ReverseAwait: Optional[CollTiling]

    def __init__(self, alloc_coll_tiling, optional_native_unit):
        assert isinstance(alloc_coll_tiling, CollTiling)
        if optional_native_unit is not None:
            assert isinstance(optional_native_unit, CollUnit)
        self.first_usage_stmt = None
        self.first_distributed_iters = []
        self.distributed_extents = []
        self.alloc_coll_tiling = alloc_coll_tiling
        self.optional_native_unit = optional_native_unit
        self.leaf_coll_tiling = None
        self.Arrive = None
        self.Await = None
        self.ReverseArrive = None
        self.ReverseAwait = None

    def n_distributed_dims(self):
        return len(self.first_distributed_iters)

    @staticmethod
    def from_fence(s: LoopIR.SyncStmt, coll_tiling: CollTiling):
        assert not s.sync_type.is_split()
        result = DistributedAllocState(coll_tiling, None)
        result.first_usage_stmt = s
        result.Arrive = coll_tiling
        result.Await = coll_tiling
        return result

    def codegen_slices_to_root(
        self,
        hi_thread_pitch: int,
        thread_iters: Dict[Sym, ThreadIter],
        distributed_iters: Optional[List[Sym]] = None,
    ):
        """Function needed to codegen mbarriers and mbarrier-like objects.

        Ignoring clusters, we need to generate a unique index for each
        logically-separate mbarrier object to put in shared memory.
        This is based on the explicit distributed indices, plus thread
        iterators between the point-of-allocation of the barrier and
        the CollTiling root. e.g.

        for i0 in cuda_threads(0, 2, unit=cuda_warpgroup):
            bar : barrier @ CudaMbarrier
            for i1 in cuda_threads(0, 4, unit=cuda_warp):
                Arrive(cuda_classic, bar[i1], 1)

        We need a total of 8 mbarriers for all i0 x i1 combinations, i0
        being the implicit to-root index and i1 the explicit distributed index.

        This needs to ignore tiling in the CTA-in-cluster dimension, so we
        ignore iterators that have a thread pitch >= hi_thread_pitch.
        (intended usage hi_thread_pitch=blockDim, but I generalize this here)

        If distributed_iters is given, return C++ index expr: str
        Else, return the total number of slices: int.

        """
        count = 1
        prods = []

        def handle_idx(nm, ext):
            nonlocal count
            info = thread_iters[nm]
            if 0 < info.thread_pitch < hi_thread_pitch:
                assert ext >= 1
                if ext > 1:
                    cname = info.cname(nm.name())
                    if count == 1:
                        prods.append(cname)
                    else:
                        prods.append(f"{count}*{cname}")
                    count *= ext

        tmp_iters = (
            self.first_distributed_iters
            if distributed_iters is None
            else distributed_iters
        )
        assert len(tmp_iters) == len(self.distributed_extents)
        for nm, ext in zip(reversed(tmp_iters), reversed(self.distributed_extents)):
            handle_idx(nm, ext)
        coll_tiling = self.alloc_coll_tiling
        while coll_tiling is not None:
            if coll_tiling.tile_count != 1:
                handle_idx(coll_tiling.iter, coll_tiling.tile_count)
            coll_tiling = coll_tiling.parent

        if distributed_iters is None:
            return count
        else:
            return " + ".join(prods) if prods else "0"


@dataclass(slots=True)  # convenient to auto-define repr for debugging
class DistributedIdxFsm:
    """State-machine like object for analyzing distributed memory indexing

    Inspect indices of a read/write (rw_node.idx) one by one with consume_idx.
    Uninspected indices aren't parsed, so we don't enforce requirements on them.

    """

    # LoopIR stmt that contains the indexing
    context_stmt: LoopIR.stmt

    # CollTiling at the point of allocation
    alloc_coll_tiling: CollTiling

    # Target collective unit for holding each slice (optional termination
    # condition; if not provided, is_done() must not be called).
    optional_native_unit: Optional[CollUnit]
    optional_native_num_threads: Optional[int]

    # CollTiling and loop iterator of the loop level at which the distributed
    # slices of the allocation are individually allocated at.
    leaf_coll_tiling: CollTiling
    leaf_iter: Optional[Sym]

    # Environments from compiler
    loop_mode_name: str  # Expected LoopMode for thread iterators
    thread_iters: Dict[Sym, ThreadIter]
    coll_env: Dict[CollParam, int]

    # Iterators parsed in order as distributed indices
    distributed_iters: List[Sym]
    distributed_extents: List[int]

    # Parsed iterators: parent_num_tile_threads -> (Sym, child_num_tile_threads)
    t0_iter_t1: Dict[int, Tuple[Sym, int]]

    # Progress of deduced tiling
    cur_num_threads: int

    def __init__(
        self,
        context_stmt: LoopIR.stmt,
        state: DistributedAllocState,
        loop_mode_name: str,
        thread_iters: Dict[Sym, ThreadIter],
        coll_env: Dict[CollParam, int],
    ):
        self.context_stmt = context_stmt
        self.alloc_coll_tiling = state.alloc_coll_tiling
        if state.optional_native_unit is None:
            self.optional_native_unit = None
            self.optional_native_num_threads = None
        else:
            assert isinstance(state.optional_native_unit, CollUnit)
            self.optional_native_unit = state.optional_native_unit
            self.optional_native_num_threads = state.optional_native_unit.int_threads(
                coll_env
            )
        self.leaf_coll_tiling = state.alloc_coll_tiling  # will be updated
        self.leaf_iter = None
        self.loop_mode_name = loop_mode_name
        self.thread_iters = thread_iters
        self.coll_env = coll_env
        self.distributed_iters = []
        self.distributed_extents = []
        self.t0_iter_t1 = {}
        self.cur_num_threads = state.alloc_coll_tiling.tile_num_threads()

    def consume_idx(self, node: LoopIR.stmt, typ: LoopIR.type, i: int):
        """Process node.idx[i] as the next distributed index"""
        idx_e = node.idx[i]
        if isinstance(idx_e, LoopIR.Read):
            iter_sym = idx_e.name
        elif isinstance(idx_e, LoopIR.Point) and isinstance(idx_e.pt, LoopIR.Read):
            iter_sym = idx_e.pt.name
        else:
            self.bad_idx(node, f"expected single variable name, not {idx_e}")
        iter_info: ThreadIter
        iter_info = self.thread_iters.get(iter_sym)
        if iter_info is None:
            self.bad_idx(node, f"`{iter_sym}` not from {self.loop_mode_name} loop")

        shape = typ.shape()
        if shape:  # TODO remove this backdoor. Shape check should be mandatory
            const_extent = None
            if isinstance(e := shape[i], LoopIR.Const):
                const_extent = e.val
            tile_count = iter_info.coll_tiling.tile_count
            if tile_count != const_extent:
                self.bad_idx(
                    node,
                    f"`{iter_sym}` range [0, {tile_count}] mismatches extent `{shape[i]}` in {typ}",
                )

        # Note we use the num_tile_threads, not num_box_threads, throughout
        # this analysis, because we care about dividing the "ownership" of
        # slices, so warp specialization (box, offset) doesn't matter.

        t0 = iter_info.parent_tile_num_threads
        t1 = iter_info.child_tile_num_threads
        if t0 != t1:
            if t0 in self.t0_iter_t1:
                self.bad_idx(
                    node, f"unexpected (repeated?) index {idx_e} (duplicate t0={t0})"
                )
            self.t0_iter_t1[t0] = (iter_sym, t1)
        if t1 < self.leaf_coll_tiling.tile_num_threads():
            self.leaf_coll_tiling = iter_info.coll_tiling
            self.leaf_iter = iter_sym
        if (n := self.optional_native_num_threads) is not None:
            if t1 < n:
                self.bad_idx(
                    node,
                    f'Iterator {iter_sym} yields thread collectives of {t1} threads; "overshot" native target {n} threads',
                )

        hi = iter_info.coll_tiling.tile_count
        assert isinstance(hi, int)
        self.distributed_iters.append(iter_sym)
        self.distributed_extents.append(hi)

        # Each index variable subdivides a CollTiling, translating
        # a parent_num_tile_threads -> child_num_tile_threads.
        # Search for a valid chain from alloc_num_threads -> native_num_threads
        #
        # We have to do this now, not separately, so cur_num_threads is updated
        # and we don't try to parse non-distributed dims and give false errors.
        while entry := self.t0_iter_t1.get(self.cur_num_threads):
            iter_sym, t1 = entry
            self.cur_num_threads = entry[1]

    def is_done(self, node):
        """True if we should not call consume_idx() again.

        This is the case when the tiling successfully suballocated
        slices to thread collectives with the stored native unit, but
        not the converse ... still have to call check_native_unit()"""
        n = self.optional_native_num_threads
        assert n is not None
        return self.cur_num_threads <= n

    def check_native_unit(self, node):
        """Check that the leaf tiling matches the stored native unit"""
        assert self.optional_native_unit is not None
        if msg := self.leaf_coll_tiling.unit_mismatch(
            self.optional_native_unit, self.coll_env
        ):
            _iter = self.leaf_iter
            if _iter is None:
                self.bad_idx(
                    node, f"Wrong collective unit at point of allocation: {msg}"
                )
            else:
                self.bad_idx(
                    node,
                    f"Tried to allocate under {_iter} loop; wrong collective unit: {msg}",
                )

    def check_store_state(self, node, state: DistributedAllocState):
        """Update the allocation state with analysis results

        If this distributed memory analysis is not the first for the
        allocation, we check that the usage pattern is compatible with
        that of the first usage.

        We could have stored `state` in the constructor, but I want to
        make the mutation more explicit at the call site.

        """

        def format_iters(iters):
            return "[" + ", ".join(str(n) for n in iters) + "]"

        if state.first_usage_stmt is None:
            state.first_usage_stmt = self.context_stmt
            state.first_distributed_iters = self.distributed_iters
            state.distributed_extents = self.distributed_extents
            state.leaf_coll_tiling = self.leaf_coll_tiling
            return

        first_stmt = state.first_usage_stmt
        first_iters = state.first_distributed_iters
        cur_iters = self.distributed_iters
        if len(first_iters) != len(cur_iters):
            d1 = format_iters(first_iters)
            d2 = format_iters(cur_iters)
            self.bad_idx(
                node,
                f"Deduced {len(cur_iters)} distributed dims {d2}; mismatches {d1} deduced at {first_stmt.srcinfo} ({first_stmt})",
            )

        for i1, i2 in zip(first_iters, cur_iters):
            c1 = self.thread_iters[i1].coll_index_expr
            c2 = self.thread_iters[i2].coll_index_expr
            if not c1.equiv_index(c2):
                d1 = format_iters(first_iters)
                d2 = format_iters(cur_iters)
                raise ValueError(
                    f"Mismatched distributed dims {node.name}{d1} and {node.name}{d2}:\n"
                    f"{i1}={c1.codegen()} != {i2}={c2.codegen()}\n"
                    f"Usage 1: {first_stmt} : {first_stmt.srcinfo}\n"
                    f"Usage 2: {self.context_stmt} : {self.context_stmt.srcinfo}"
                )

        assert len(self.distributed_extents) == len(state.distributed_extents)
        for i, v in enumerate(self.distributed_extents):
            state.distributed_extents[i] = max(state.distributed_extents[i], v)

    def inspect_arrive_await(
        self,
        sync: LoopIR.SyncStmt,
        coll_tiling: CollTiling,
        barrier_usage: BarrierUsage,
        state: DistributedAllocState,
    ):
        """Subsequent to check_store_state, for non-Fence SyncStmts,
        we additionally check requirements for the collective tiling

        * Tile size of usage matches the leaf tiling.
        * Equivalent CollTiling for all syncs of a certain kind
          (Arrive, Await, ReverseArrive, ReverseAwait)
        * If the barrier type has a pairing requirement, additionally,
          check equivalent CollTilings for paired Arrive/Await;
          Arrive/ReverseAwait, or ReverseArrive/Await.

        """
        assert isinstance(sync, LoopIR.SyncStmt)
        sync_type = sync.sync_type
        assert sync_type.is_split()
        # Update state.Arrive, state.Await,
        # state.ReverseArrive, or state.ReverseAwait
        leaf_T = state.leaf_coll_tiling.tile_num_threads()
        sync_T = coll_tiling.tile_num_threads()
        if leaf_T != sync_T:
            bar = f"{sync.name}[" + ", ".join(str(n) for n in sync.idx) + "]"
            raise ValueError(
                f"{sync.srcinfo}: {sync} executed with tile size {sync_T} threads; mismatches {leaf_T} threads deduced from {bar} (i.e. multiple thread collectives share the same index; missing indices)?"
            )
        # Get state.Arrive, state.Await, state.ReverseArrive, or state.ReverseAwait
        attr = sync_type.fname()
        old_coll_tiling = getattr(state, attr)

        # CollTilings that need to be equivalent
        to_check: List[Tuple[CollTiling, str]] = []

        if old_coll_tiling is not None:
            # Will check equivalence with previous stmt of same sync type
            to_check.append((old_coll_tiling, attr))
        else:
            # Save new state.Arrive, state.Await, state.ReverseArrive, or state.ReverseAwait
            setattr(state, attr, coll_tiling)

        if barrier_usage.barrier_type.traits().requires_pairing:
            # Will check equivalence with previous stmt of paired sync type
            attr = barrier_usage.paired_fname(sync_type)
            # Get state.Arrive, state.Await, state.ReverseArrive, or state.ReverseAwait
            old_coll_tiling = getattr(state, attr)
            if old_coll_tiling is not None:
                to_check.append((old_coll_tiling, attr))

        # Check equivalence
        for old_coll_tiling, fname in to_check:
            domain0 = old_coll_tiling.full_domain
            tile0 = old_coll_tiling.tile
            box0 = old_coll_tiling.box
            offset0 = old_coll_tiling.offset
            domain1 = coll_tiling.full_domain
            tile1 = coll_tiling.tile
            box1 = coll_tiling.box
            offset1 = coll_tiling.offset
            if (
                domain0 != domain1
                or box0 != box1
                or tile0 != tile1
                or offset0 != offset1
            ):
                raise ValueError(
                    f"{sync.srcinfo}: {sync} has inconsistent collective tiling with previous {fname}\n"
                    f"Saw tile={tile0}, box={box0}, offset={offset0}\n"
                    f"Saw tile={tile1}, box={box1}, offset={offset1}"
                )

    def bad_idx(self, node, msg):
        iter_text = "".join(
            f"\n{nm} {t0}->{t1}" for (t0, (nm, t1)) in self.t0_iter_t1.items()
        )
        if (n := self.optional_native_num_threads) is not None:
            native_suffix = f" to {n}"
        else:
            native_suffix = ""
        alloc_num_threads = self.alloc_coll_tiling.tile_num_threads()
        raise ValueError(
            f"{node.srcinfo}: Distributed memory analysis "
            f"(from {alloc_num_threads} threads{native_suffix}) "
            f"for {node.name} failed: {msg}\n(at {self.context_stmt}) "
            f"inspected iters:{iter_text or ' <none>'}"
        )
