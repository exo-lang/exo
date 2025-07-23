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
    clusterDim_param,
    blockDim_param,
    DomainCompletionOp,
)
from .barrier_usage import BarrierUsage
from .loop_modes import _CodegenPar
from .sync_types import SyncType
from .barrier_usage import SyncInfo
from .base_with_context import is_if_holding_with


@dataclass(slots=True, init=False)
class ThreadIter:
    """Information for an iter variable from a for-threads parallel loop (cuda_threads)"""

    codegen_par: _CodegenPar
    coll_index_expr: CollIndexExpr
    coll_tiling: CollTiling
    parent_tile_num_threads: int
    child_tile_num_threads: int
    thread_pitch: int

    def __init__(self, coll_tiling: CollTiling, comment: Optional[str] = None):
        self.codegen_par = _CodegenPar(
            coll_tiling.codegen_expr.codegen(),
            comment,
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

    # Deduced compile-time const shape of the array of distributed shards.
    # e.g. f32[8, 4, j] with 2 distributed dims gives [8, 4].
    # This is possible redundant (i.e. maybe we should just require the LoopIR
    # type of the indexed variable to be passed each time).
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

    # Information for Arrive/Await statements, split by usage
    # of front (+name) and back (-name) queue barrier array.
    # Fence() stmts are decomposed as an front_arrive + front_await
    # Index using constants in SyncType or get_{front|back}_{arrive|await}.
    sync_coll_tilings: List[Optional[CollTiling]]

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
        self.sync_coll_tilings = [None] * SyncType.n_info_idx

    def n_distributed_dims(self):
        return len(self.first_distributed_iters)

    def get_front_arrive(self) -> Optional[CollTiling]:
        return self.sync_coll_tilings[SyncType.front_arrive_idx]

    def get_front_await(self) -> Optional[CollTiling]:
        return self.sync_coll_tilings[SyncType.front_await_idx]

    def get_back_arrive(self) -> Optional[CollTiling]:
        return self.sync_coll_tilings[SyncType.back_arrive_idx]

    def get_back_await(self) -> Optional[CollTiling]:
        return self.sync_coll_tilings[SyncType.back_await_idx]

    @staticmethod
    def from_fence(s: LoopIR.SyncStmt, coll_tiling: CollTiling):
        assert not s.sync_type.is_split()
        result = DistributedAllocState(coll_tiling, None)
        result.first_usage_stmt = s
        result.sync_coll_tilings[SyncType.front_arrive_idx] = coll_tiling
        result.sync_coll_tilings[SyncType.front_await_idx] = coll_tiling
        return result

    def codegen_slices_to_root(
        self,
        hi_thread_pitch: int,
        thread_iters: Dict[Sym, ThreadIter],
        distributed_iters: Optional[List[Optional[Sym]]] = None,
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
                Arrive(cuda_classic, 1) >> bar[i1]

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
            if nm is not None:
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

    def cta_xor_list(
        self, blockDim: int, thread_iters: Dict[Sym, ThreadIter], sync_info: SyncInfo
    ) -> List[int]:
        """Intended for Arrive statements for mbarriers in distributed shared memory.

        Compile the arrive statement with the given
        SyncStmt.multicasts() value (list of multicast flags)
        to a series of arrives on the CTAs with ranks
        [(blockIdx.x % clusterDim) ^ m for m in cta_xor_list(..)]

        """
        stmt = sync_info.stmts[0]
        multicasts = sync_info.multicasts
        mask_bits = 0
        iterators: List[Sym] = self.first_distributed_iters
        for multicast_flags in multicasts:
            assert len(multicast_flags) == len(iterators)
            tmp_bits = 1
            for flag, sym in zip(multicast_flags, iterators):
                if flag:
                    info = thread_iters[sym]
                    thread_pitch = info.thread_pitch
                    cta_count = info.coll_tiling.tile_count
                    if cta_count >= 2:
                        if thread_pitch % blockDim != 0:
                            raise ValueError(
                                f"{stmt.srcinfo}: {sym} thread_pitch {thread_pitch} not divisible by blockDim ({blockDim}); cannot be multicast (in {stmt})"
                            )
                        cta_pitch = thread_pitch // blockDim
                        new_bits = 0
                        for n in range(cta_count):
                            new_bits |= tmp_bits << (n * cta_pitch)
                        tmp_bits = new_bits
            mask_bits |= tmp_bits
        xor_list = [
            bit_index
            for bit_index in range(mask_bits.bit_length())
            if ((mask_bits >> bit_index) & 1)
        ]
        # Limitation: excut tests require this for now as of 2025-07-22
        assert xor_list[0] == 0
        return xor_list

    def codegen_cta_mask(
        self, blockDim: int, thread_iters: Dict[Sym, ThreadIter], e: LoopIR.BarrierExpr
    ) -> str:
        """Translate BarrierExpr to CTA mask"""
        assert isinstance(e, LoopIR.BarrierExpr)
        base_num = 1
        shift_mask = 0
        iterators: List[Sym] = self.first_distributed_iters
        flags = e.multicast_flags()
        assert len(iterators) == len(flags)
        for multicast, sym in zip(flags, iterators):
            info = thread_iters[sym]
            thread_pitch = info.thread_pitch
            if thread_pitch < blockDim:
                # thread_pitch = 0: [0, 1] interval has no effect on CTA mask
                # 0 < thread_pitch < blockDim: non-CTA index has no effect on CTA
                continue
            cta_count = info.coll_tiling.tile_count
            cta_pitch = thread_pitch // blockDim
            assert cta_pitch * blockDim == thread_pitch
            assert cta_count >= 2

            # CUDA model fundamentally assumes power-of-2 CTA counts
            cta_count_log2 = cta_count.bit_length() - 1
            cta_pitch_log2 = cta_pitch.bit_length() - 1
            assert cta_count == 1 << cta_count_log2
            assert cta_pitch == 1 << cta_pitch_log2

            if multicast:
                tmp = 1
                for i in range(1, cta_count):
                    tmp = (tmp << cta_pitch) | 1
                base_num *= tmp
            else:
                shift_mask |= ((1 << cta_count_log2) - 1) << cta_pitch_log2

        # Imagine arranging the CTAs into an N-dimensional cuboid, where N
        # is the number of array indices corresponding to CTA-in-cluster
        # dimensions. Then base_num is the mask corresponding to the sub-cuboid
        # of CTAs that the 0th CTA multicasts to, and the shift is needed to
        # reposition the sub-cuboid to get the target CTAs for this CTA.
        if shift_mask == 0:
            return f"uint16_t({hex(base_num)})"
        else:
            return f"uint16_t({hex(base_num)} << (blockIdx.x & {hex(shift_mask)}))"


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
    alloc_box_num_threads: int

    # For analyzing intervals passed to instrs.
    # The CollTiling is initially that of the caller, and we create new
    # ones progressively, tiled by callee_coll_units[callee_unit_idx++]
    # for each distributed interval passed to the instr.
    callee_coll_tiling: CollTiling
    callee_coll_units: List[CollUnit]
    callee_unit_idx: int

    def __init__(
        self,
        context_stmt: LoopIR.stmt,
        state: DistributedAllocState,
        loop_mode_name: str,
        thread_iters: Dict[Sym, ThreadIter],  # May be modified
        coll_env: Dict[CollParam, int],
        callee_coll_tiling: CollTiling,
        callee_coll_units: List[CollUnit],
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
        self.thread_iters = thread_iters  # must NOT be a copy
        self.coll_env = coll_env
        self.distributed_iters = []
        self.distributed_extents = []
        self.t0_iter_t1 = {}
        self.cur_num_threads = state.alloc_coll_tiling.tile_num_threads()
        self.alloc_box_num_threads = state.alloc_coll_tiling.box_num_threads()
        self.callee_coll_tiling = callee_coll_tiling
        self.callee_coll_units = callee_coll_units
        self.callee_unit_idx = 0

    def consume_idx(self, node, typ: LoopIR.type, i: int):
        """Process node.idx[i] as the next distributed index"""
        shape = typ.shape()
        const_extent = None
        if i < len(shape) and isinstance(e := shape[i], LoopIR.Const):
            const_extent = e.val

        idx_e = node.idx[i]
        if isinstance(idx_e, LoopIR.Read):
            iter_sym = idx_e.name
        elif isinstance(idx_e, LoopIR.Point) and isinstance(idx_e.pt, LoopIR.Read):
            iter_sym = idx_e.pt.name
        elif isinstance(idx_e, LoopIR.Interval):
            if len(self.callee_coll_units) <= self.callee_unit_idx:
                self.bad_idx(node, f"expected point, not interval {idx_e}")
            if (
                not isinstance(idx_e.lo, LoopIR.Const)
                or idx_e.lo.val != 0
                or not isinstance(idx_e.hi, LoopIR.Const)
                or idx_e.hi.val != const_extent
            ):
                self.bad_idx(node, f"expected 0:{const_extent}, not {idx_e}")
            iter_sym = Sym(f"CALLEE_DISTRIBUTED_IDX_{self.callee_unit_idx}")
            unit = self.callee_coll_units[self.callee_unit_idx]
            self.callee_coll_tiling = self.callee_coll_tiling.tiled(
                iter_sym, unit, const_extent, self.coll_env
            )
            self.callee_unit_idx += 1
            # HACK: writing state of new synthetic Sym to thread_iters.
            # This may actually be used later, since it goes into
            # distributed_iters which could be really confusing.
            # Hence the note on how thread_iters may be modified.
            self.thread_iters[iter_sym] = ThreadIter(self.callee_coll_tiling)
        else:
            self.bad_idx(node, f"expected single variable name, not {idx_e}")
        iter_info: ThreadIter
        iter_info = self.thread_iters.get(iter_sym)
        if iter_info is None:
            self.bad_idx(node, f"`{iter_sym}` not from {self.loop_mode_name} loop")

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
                    node, f"unexpected (repeated?) index {iter_sym} (duplicate t0={t0})"
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

        return (const_extent, iter_sym)  # Internal use by consume_SyncStmt_idx

    def consume_SyncStmt_idx(
        self,
        stmt_stack: List[LoopIR.stmt],
        sync_stmt: LoopIR.SyncStmt,
        typ: LoopIR.type,
        i: int,
    ):
        """Process sync_stmt.barriers[n].idx[i] for all n"""
        assert typ.is_barrier()

        home_barrier = sync_stmt.home_barrier_expr()

        # Analyze the point
        const_extent, iter_sym = self.consume_idx(home_barrier, typ, i)

        # Range check for intervals
        any_multicast = False
        for e in sync_stmt.barriers:
            idx = e.idx[i]
            if isinstance(idx, LoopIR.Interval):
                any_multicast = True
                lo, hi = idx.lo, idx.hi
                if not isinstance(lo, LoopIR.Const) or lo.val != 0:
                    self.bad_idx(e, f"Expected idx[{i}] to be 0:{const_extent}")
                if not isinstance(hi, LoopIR.Const) or hi.val != const_extent:
                    self.bad_idx(e, f"Expected idx[{i}] to be 0:{const_extent}")

        # Check convergence requirement for multicasted iterators.
        # Go from root-to-leaf of AST, and complain if there are seq-for or
        # there are if-else (not with) between the loop that defines the
        # multicast iterator and the SyncStmt.
        if any_multicast:
            iter_sym_loop = None
            for s in stmt_stack:
                if iter_sym_loop is None:
                    if isinstance(s, LoopIR.For) and s.iter == iter_sym:
                        iter_sym_loop = s
                else:
                    # Now within the loop found, start enforcing
                    sus = None
                    if is_if_holding_with(s, LoopIR):
                        pass
                    elif isinstance(s, LoopIR.If):
                        sus = f"if {s.cond}"
                    elif isinstance(s, LoopIR.For) and not s.loop_mode.is_par():
                        sus = f"for {s.iter} in {s.loop_mode.format_loop_cond(s.lo, s.hi)}"
                    if sus:
                        raise ValueError(
                            f"{sync_stmt.srcinfo}: multicasted {iter_sym} fails "
                            f"convergence requirement due to `{sus}` at "
                            f"{s.srcinfo} (SyncStmt: {sync_stmt})"
                        )

    def is_done(self, node):
        """True if we should not call consume_idx() again.

        This is the case when the tiling successfully suballocated
        slices to thread collectives with the stored native unit, but
        not the converse ... still have to call check_native_unit()"""
        n = self.optional_native_num_threads
        assert n is not None
        # TODO: explain this min(...) in coll_algebra.pdf
        return min(self.cur_num_threads, self.alloc_box_num_threads) <= n

    def check_native_unit(self, node):
        """Check that the leaf tiling matches the stored native unit"""
        unit = self.optional_native_unit
        assert unit is not None
        _iter = self.leaf_iter
        is_distributed = _iter is not None
        ignore_box = self.alloc_box_num_threads != self.optional_native_num_threads
        if msg := self.leaf_coll_tiling.unit_mismatch(
            unit,
            self.coll_env,
            ignore_box=ignore_box,
        ):
            if is_distributed:
                self.bad_idx(
                    node,
                    f"Tried to allocate under {_iter} loop; wrong collective unit: {msg}",
                )
            else:
                self.bad_idx(
                    node, f"Wrong collective unit at point of allocation: {msg}"
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

        assert self.distributed_extents == state.distributed_extents

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
        * Equivalent CollTiling for same action on same queue barrier array.
          (+Arrive, +Await, -Arrive, -Await) +/- meaning front/back.
        * If the barrier type has a pairing requirement, additionally,
          check equivalent CollTilings for paired +Arrive/+Await;
          +Arrive/-Await, or -Arrive/+Await.

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
        # Get CollTiling for Arrive >> +name, Await(+name), Arrive >> -name, or Await(-name)
        name = sync.barriers[0].name
        back = sync.barriers[0].back
        info_idx = sync_type.info_idx(back)
        old_coll_tiling = state.sync_coll_tilings[info_idx]

        # CollTilings that need to be equivalent
        to_check: List[Tuple[CollTiling, str]] = []

        if old_coll_tiling is not None:
            # Will check equivalence with previous stmt of same sync type
            f_text = str(sync)
            to_check.append((old_coll_tiling, f_text))
        else:
            # Save new state
            state.sync_coll_tilings[info_idx] = coll_tiling

        if barrier_usage.barrier_type.traits().requires_pairing:
            # Will check equivalence with previous stmt of paired sync type
            paired_back = back ^ barrier_usage.has_back_array()
            sign = "-" if paired_back else "+"
            paired_info_idx = sync_type.info_idx(paired_back, swap=True)
            if sync_type.is_arrive():  # Arrive paired with Await
                f_text = f"Await({sign}{name}, ...)"
            else:
                f_text = f"Arrive(...) >> {sign}{name}"
            if old_coll_tiling := state.sync_coll_tilings[paired_info_idx]:
                to_check.append((old_coll_tiling, f_text))

        # Check equivalence (the code here only checks issues that wouldn't be
        # flagged by the primary distributed memory deduction, i.e., issues
        # related to masked-out threads, so we check box, offset).
        for old_coll_tiling, f_text in to_check:
            old_completion = DomainCompletionOp(
                old_coll_tiling.full_domain, coll_tiling.full_domain, False
            )
            new_completion = DomainCompletionOp(
                coll_tiling.full_domain, old_coll_tiling.full_domain, False
            )
            box0 = old_completion.new_size(old_coll_tiling.box)
            offset0 = old_completion.new_offset(old_coll_tiling.offset)
            box1 = new_completion.new_size(coll_tiling.box)
            offset1 = new_completion.new_offset(coll_tiling.offset)
            if box0 != box1 or offset0 != offset1:
                raise ValueError(
                    f"{sync.srcinfo}: {sync} has inconsistent collective tiling with previous {f_text}\n"
                    f"Saw box={box0}, offset={offset0}\n"
                    f"Saw box={box1}, offset={offset1}"
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
