from dataclasses import dataclass
from math import prod
from typing import Optional, List, Type, Dict, Tuple, Callable

from ..core.prelude import Sym
from ..core.LoopIR import LoopIR, T
from .coll_algebra import (
    CollCodegen,
    CollTiling,
    CollUnit,
    CollIndexExpr,
    CollParam,
    CollDimOp,
    clusterDim_param,
    blockDim_param,
    DomainCompletionOp,
    CollTilingError,
)
from .barrier_usage import BarrierUsage, SyncInfo
from .loop_modes import _CodegenPar
from .sync_types import SyncType
from .base_with_context import is_if_holding_with


@dataclass(slots=True, init=False)
class ThreadIter:
    """Information for an iter variable from a for-threads parallel loop (cuda_threads), or rewritten CudaWarps"""

    codegen_par: _CodegenPar
    coll_index_expr: CollIndexExpr
    coll_tiling: CollTiling
    tile_count: int
    thread_pitch: int
    mangle: bool
    iter: Sym

    def __init__(
        self,
        coll_tiling: CollTiling,
        comment: Optional[str] = None,
        warp_name_filter: Optional[str] = None,
        mangle: bool = True,
        # Messy special-case args for handling 2-step CudaWarps compile.
        prior_am_dim_idx=None,
        prior_am_offset=0,
        prior_am_box=None,
    ):
        codegen: CollCodegen = coll_tiling.get_codegen()
        dim_idx = codegen.dim_idx
        if dim_idx != prior_am_dim_idx:
            # CudaWarps should apply to the same domain always.
            assert dim_idx is None or prior_am_dim_idx is None
        if dim_idx is not None:
            am_box = codegen.box
        elif prior_am_dim_idx is not None:
            assert prior_am_box > 0
            am_box = prior_am_box
        else:
            am_box = -1

        self.codegen_par = _CodegenPar(
            codegen.codegen_expr.codegen(),
            comment,
            (codegen.codegen_static_lo, codegen.codegen_static_hi),
            warp_name_filter,
            coll_tiling.get_domain(),
            prior_am_dim_idx or dim_idx,
            prior_am_offset + codegen.offset,
            am_box,
        )

        self.coll_index_expr = codegen.codegen_expr
        self.coll_tiling = coll_tiling
        self.tile_count = codegen.tile_count
        self.thread_pitch = codegen.thread_pitch
        self.mangle = mangle
        self.iter = coll_tiling.get_codegen().iter

    def cname(self, name):
        """Mangling convention for generated C variables"""
        return f"exo_{self.thread_pitch}thr_{name}"


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
    first_usage_coll_tiling: Optional[CollTiling]

    # Alloc or Fence stmt that allocates the variable + the variable's type.
    alloc_stmt: LoopIR.Alloc | LoopIR.SyncStmt
    alloc_type: LoopIR.type

    # CollTiling at the point of the Exo object code allocation
    alloc_coll_tiling: CollTiling

    # Target native unit; we want to have one distributed shard resident
    # in each active native-unit-shaped thread collective.
    # If not specified, we want to have one distributed shard resident
    # in each thread collective at the usage site (i.e. no sharing).
    optional_native_unit: Optional[CollUnit]

    # Information for Arrive/Await statements, split by usage.
    # Fence() stmts are decomposed as an arrive + await
    arrive_coll_tiling: Optional[CollTiling]
    await_coll_tiling: Optional[CollTiling]

    def __init__(
        self,
        alloc_stmt: LoopIR.stmt,
        coll_tiling: CollTiling,
        optional_native_unit: Optional[CollUnit],
        env: Dict[CollParam, int],
    ):
        assert isinstance(coll_tiling, CollTiling)
        if optional_native_unit is not None:
            assert isinstance(optional_native_unit, CollUnit)
            assert not optional_native_unit.agnostic, optional_native_unit
            tmp = coll_tiling
            tmp = tmp.unit_completion(optional_native_unit, env)
            box = tmp.get_box()
            expected_box = tmp.get_expected_box()
            assert len(box) == len(expected_box)
            for i, expect_c in enumerate(expected_box):
                assert (
                    expect_c is not None
                ), "shouldn't happen for non-agnostic unit and partial_prepend=1"
                if expect_c > 1 and box[i] != expect_c:
                    raise CollTilingError(
                        f"Missing threads to match {optional_native_unit}\n"
                        f"domain={tmp.get_domain()}, box={box}; expected box={expected_box} (wrong @ box[{i}])"
                    )
            self.alloc_coll_tiling = tmp
        else:
            self.alloc_coll_tiling = coll_tiling
        self.first_usage_stmt = None
        self.first_distributed_iters = []
        self.first_usage_coll_tiling = None
        self.alloc_stmt = alloc_stmt
        if isinstance(alloc_stmt, LoopIR.Alloc):
            self.alloc_type = alloc_stmt.type
        else:
            assert isinstance(alloc_stmt, LoopIR.SyncStmt)
            self.alloc_type = T.barrier
        self.optional_native_unit = optional_native_unit
        self.arrive_coll_tiling = None
        self.await_coll_tiling = None

    def n_distributed_dims(self):
        return len(self.first_distributed_iters)

    def get_arrive(self) -> Optional[CollTiling]:
        return self.arrive_coll_tiling

    def get_await(self) -> Optional[CollTiling]:
        return self.await_coll_tiling

    def get_distributed_extents(self) -> Tuple[int]:
        return tuple(
            self.get_const_shape_coord(i) for i in range(0, self.n_distributed_dims())
        )

    def get_const_shape_coord(self, i) -> int:
        t = self.alloc_type
        e = t.shape()[i]
        if isinstance(e, LoopIR.Const):
            return int(e.val)
        raise ValueError(
            f"{self.alloc_stmt.srcinfo}: distributed memory deduction failed for {t}\n"
            f"shape[{i}] must be constant."
        )

    @staticmethod
    def from_fence(s: LoopIR.SyncStmt, coll_tiling: CollTiling):
        assert not s.sync_type.is_split()
        result = DistributedAllocState(s, coll_tiling, None, None)
        result.first_usage_stmt = s
        result.arrive_coll_tiling = coll_tiling
        result.await_coll_tiling = coll_tiling
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

        for i0 in cuda_threads(0, 2, unit=cuda_warpgroup):  # implicit
            bar : barrier[4] @ CudaMbarrier
            for i1 in cuda_threads(0, 4, unit=cuda_warp):  # explicit
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

        # Handle explicit indices (given in index expression)
        tmp_iters = (
            self.first_distributed_iters
            if distributed_iters is None
            else distributed_iters
        )
        distributed_extents = self.get_distributed_extents()
        assert len(tmp_iters) == len(distributed_extents)
        for nm, ext in zip(reversed(tmp_iters), reversed(distributed_extents)):
            if nm is not None:
                handle_idx(nm, ext)

        # Handle implicit indices; relevant cuda_threads iterators
        # from the allocation point up to the root of the CudaDeviceFunction.
        for op in self.alloc_coll_tiling.get_dim_ops():
            op: CollDimOp
            if op.tile_count > 1:
                handle_idx(op.iter, op.tile_count)

        # Return either typed result, as specified by the docstring.
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
        [(cluster_ctarank % clusterDim) ^ m for m in cta_xor_list(..)]

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
                    cta_count = info.tile_count
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
            cta_count = info.tile_count
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

    # LoopIR node that contains the idx to parse, and the LoopIR stmt that
    # contains it (itself, if not idx_node is not an expr).
    idx_node: LoopIR.stmt | LoopIR.expr
    context_stmt: LoopIR.stmt

    # CollTiling at the point of allocation
    alloc_coll_tiling: CollTiling

    # CollTiling at use site (completed for native_unit, if applicable);
    # further tiled by any callee_coll_units provided.
    usage_coll_tiling: CollTiling

    # Distributed iterators that we expect to see; initially true
    # and set to false when found (i.e. not needed anymore).
    # The iterators are then pushed in order to the distributed_iters list.
    distributed_iters_needed: Dict[Sym, bool]
    distributed_iters: List[Sym]

    # Synthetic iterators used to model the internals of an instr.
    # callee_distributed_idx tracks the progress through this list;
    # we substitute one iterator for each window interval found.
    callee_distributed_iters: List[Sym]
    callee_distributed_idx: int

    # Environments from compiler
    loop_mode_name: str  # Expected LoopMode for thread iterators
    thread_iters: Dict[Sym, ThreadIter]

    def __init__(
        self,
        idx_node: LoopIR.stmt | LoopIR.expr,
        context_stmt: LoopIR.stmt,
        state: DistributedAllocState,
        loop_mode_name: str,
        thread_iters: Dict[Sym, ThreadIter],  # May be modified
        coll_env: Dict[CollParam, int],
        coll_tiling_here: CollTiling,
        callee_coll_units: List[CollUnit],
    ):
        distributed_iters_needed = {}
        callee_distributed_iters = []
        assert isinstance(context_stmt, LoopIR.stmt)
        self.context_stmt = context_stmt
        self.idx_node = idx_node
        self.loop_mode_name = loop_mode_name
        self.thread_iters = thread_iters
        alloc_coll_tiling = state.alloc_coll_tiling
        self.alloc_coll_tiling = alloc_coll_tiling
        self.distributed_iters_needed = distributed_iters_needed
        self.callee_distributed_iters = callee_distributed_iters
        self.callee_distributed_idx = 0
        self.distributed_iters = []
        self.usage_coll_tiling = coll_tiling_here  # changed later

        # Complete the collective tiling for the given native unit, if supplied.
        tiling = coll_tiling_here
        native_unit = state.optional_native_unit
        if native_unit is not None:
            tiling = tiling.unit_completion(native_unit, coll_env)

        # If the usage is as a parameter of an instr where the instruction
        # expects multiple shards, we need to tile the usage_coll_tiling
        # based on what's going on inside the instr, and save a "synthetic"
        # ThreadIter used to identify this internal parallelization.
        assert len(callee_coll_units) <= 1, "manually check this works"
        idx_i = -1
        for unit_i, unit in enumerate(callee_coll_units):
            # Search for the next interval in idx_node.idx to match with
            # a callee distributed dimension.
            idx = idx_node.idx
            while True:
                idx_i += 1
                assert idx_i < len(idx), "Should have been caught by typecheck?"
                if not isinstance(e := idx[idx_i], LoopIR.Interval):
                    continue
                break
            const_extent = state.get_const_shape_coord(idx_i)
            lo = e.lo
            hi = e.hi
            if (
                not isinstance(lo, LoopIR.Const)
                or lo.val != 0
                or not isinstance(hi, LoopIR.Const)
                or hi.val != const_extent
            ):
                self.bad_idx(
                    idx_node, f"expected idx[{idx_i}]=0:{const_extent}, not {e}"
                )
            # Tile usage_coll_tiling using a new "synthetic" iterator variable.
            # Store information about this inside thread_iters, as specified.
            _iter = Sym(f"_{unit_i}_CALLEE_DISTRIBUTED")
            callee_distributed_iters.append(_iter)
            tiling = tiling.tiled(_iter, unit, const_extent, coll_env)
            thread_iters[_iter] = ThreadIter(tiling)

        # If a native unit is provided, we need to make sure that the CollTiling
        # is sufficiently subdivided to match. This is the complement to
        # inside DistributedAllocState where we checked full dimensions instead
        # of subdivided (here it's okay if those dimensions are now partial).
        if native_unit is not None:
            box = tiling.get_box()
            expected_box = tiling.get_expected_box()
            assert len(box) == len(expected_box)
            for i, expect_c in enumerate(expected_box):
                assert (
                    expect_c is not None
                ), "shouldn't happen for non-agnostic unit and partial_prepend=1"
                if expect_c == 1 and box[i] != 1:
                    raise CollTilingError(
                        f"Missing subdivision on dims[{i}] to match {native_unit}\n"
                        f"domain={tiling.get_domain()}, box={box}; expected box={expected_box}"
                    )

        # Take a census of all distributed iterator indices we expect to see.
        # If no native unit, this is all non-trivial (tile_count > 1) iterators
        # defined in parallel-for loops between the allocation point and usage point
        # (such that no two thread collectives share a shard).
        # If a native unit exists, we expect the same except only
        # on subdivided coll dimensions.
        alloc_tree_depth = alloc_coll_tiling.get_tree_depth()
        dim_ops = (
            tiling.get_dim_ops() if native_unit is None else tiling.get_subdiv_dim_ops()
        )
        for op in dim_ops:
            op: CollDimOp
            if op.tree_depth > alloc_tree_depth and op.tile_count > 1:
                distributed_iters_needed[op.iter] = True

        self.usage_coll_tiling = tiling

    def consume_idx(self, state: DistributedAllocState, i: int):
        """Process idx_node.idx[i] as the next distributed index

        Note, this function + is_done() exists for mostly historical reasons;
        we can rewrite to just have the consume_idx loop handled internally,
        as long as consume_SyncStmt_idx gets replaced too.

        """
        node = self.idx_node
        e = node.idx[i]
        if isinstance(e, LoopIR.Point) and isinstance(e.pt, LoopIR.Read):
            iter_sym = e.pt.name
        elif isinstance(e, LoopIR.Interval):
            iters = self.callee_distributed_iters
            callee_i = self.callee_distributed_idx
            if callee_i < len(iters):
                # Substitute the next callee-internal iterators.
                iter_sym = iters[callee_i]
                self.callee_distributed_idx = 1 + callee_i
            else:
                self.bad_idx(node, f"Expected single variable name, not interval {e}")
        elif isinstance(e, LoopIR.Read):
            iter_sym = e.name
        else:
            self.bad_idx(node, f"Expected single variable name, not {e}")

        thread_iter = self.thread_iters.get(iter_sym)
        thread_iter: ThreadIter
        if thread_iter is None:
            self.bad_idx(
                node, f"Expected {self.loop_mode_name}-loop iterator, not {iter_sym}"
            )
        if thread_iter.thread_pitch == 0:
            # Do-nothing iterator, not inspected.
            assert iter_sym not in self.distributed_iters_needed
        else:
            # Mark the iterator as found (forbid duplicates)
            needed_dict = self.distributed_iters_needed
            if iter_sym not in needed_dict:
                self.bad_idx(
                    node, f"{iter_sym} is not an expected distributed iterator"
                )
            if not needed_dict[iter_sym]:
                self.bad_idx(node, f"{iter_sym} repeated unexectedly")
            needed_dict[iter_sym] = False

        # Record the distributed iterator
        self.distributed_iters.append(iter_sym)

        # Check that the range (tile_count) of the distributed iterator matches
        # the underlying tensor being accessed. This is stricter than boundscheck.
        const_extent = state.get_const_shape_coord(i)
        if thread_iter.tile_count != const_extent:
            self.bad_idx(
                node,
                f"{iter_sym}.tile_count = {thread_iter.tile_count}; must be {const_extent} to match underlying tensor",
            )

        return (const_extent, iter_sym)  # Internal use by consume_SyncStmt_idx

    def consume_SyncStmt_idx(
        self,
        state: DistributedAllocState,
        stmt_stack: List[LoopIR.stmt],
        sync_stmt: LoopIR.SyncStmt,
        typ: LoopIR.type,
        i: int,
    ):
        """Process sync_stmt.barriers[n].idx[i] for all n

        Assumes that the DistributedIdxFsm was initialized with
        idx_node=s.home_barrier_expr().

        """
        const_extent, iter_sym = self.consume_idx(state, i)

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

    def is_done(self):
        """True if we should not call consume_idx() again."""
        return not any(self.distributed_iters_needed.values())

    def check_store_state(self, state: DistributedAllocState) -> bool:
        """Update the allocation state with analysis results

        If this distributed memory analysis is not the first for the
        allocation, we check that the usage pattern is compatible with
        that of the first usage.
        Returns whether this is the first usage seen.

        We could have stored `state` in the constructor, but I want to
        make the mutation more explicit at the call site.

        """
        missing = [
            _iter for _iter, needed in self.distributed_iters_needed.items() if needed
        ]
        if missing:
            missing.sort()
            missing_str = ", ".join(str(_iter) for _iter in missing)
            self.bad_idx(self.idx_node, "Missing: " + missing_str)

        if state.first_usage_stmt is None:
            state.first_usage_stmt = self.context_stmt
            state.first_distributed_iters = self.distributed_iters
            state.first_usage_coll_tiling = self.usage_coll_tiling
            return True

        first_distributed_iters = state.first_distributed_iters
        first_usage_coll_tiling = state.first_usage_coll_tiling
        second_distributed_iters = self.distributed_iters
        second_usage_coll_tiling = self.usage_coll_tiling
        thread_iters = self.thread_iters
        msg = None

        for sym1, sym2 in zip(first_distributed_iters, second_distributed_iters):
            info1 = thread_iters[sym1]
            info2 = thread_iters[sym2]
            if info1.thread_pitch != info2.thread_pitch:
                msg = (
                    f"{sym1}.thread_pitch ({info1.thread_pitch}) != "
                    f"{sym2}.thread_pitch ({info2.thread_pitch})"
                )
                break

        if msg is None:
            if len(first_distributed_iters) != len(second_distributed_iters):
                msg = (
                    "Different number of distributed dimensions deduced; "
                    "possible reason, used iterator with tile_count=1 [(0, 1)-loop]"
                )
        if msg is None:
            if state.optional_native_unit is not None:
                msg = first_usage_coll_tiling.base_mismatch(
                    second_usage_coll_tiling, subdiv_only=True
                )

        if msg is not None:
            s1 = state.first_usage_stmt
            s2 = self.context_stmt

            lines = [f"Distributed memory deduction for {self.idx_node.name} failed"]
            lines.append(str(msg))
            lines.append(f"First usage: {s1} @ {s1.srcinfo}")
            txt = ", ".join(str(sym1) for sym1 in first_distributed_iters)
            lines.append(f"First distributed iterators: [{txt}]")
            for sym1 in first_distributed_iters:
                info = thread_iters[sym1]
                lines.append(
                    f"  {sym1} = {info.coll_index_expr.codegen()}; thread_pitch={info.thread_pitch}"
                )
            lines.append(f"Second usage: {s2} @ {s2.srcinfo}")
            txt = ", ".join(str(sym2) for sym2 in second_distributed_iters)
            for sym2 in second_distributed_iters:
                info = thread_iters[sym2]
                lines.append(
                    f"  {sym2} = {info.coll_index_expr.codegen()}; thread_pitch={info.thread_pitch}"
                )
            raise ValueError("\n".join(lines))
        return False

    def inspect_arrive_await(
        self,
        sync: LoopIR.SyncStmt,
        coll_tiling: CollTiling,
        get_barrier_usage: Callable[[Sym], BarrierUsage],
        get_state: Callable[[Sym], Optional[DistributedAllocState]],
    ):
        """Subsequent to check_store_state, for non-Fence SyncStmts,
        we additionally check requirements for the collective tiling

        * Equivalent CollTiling for same action on same queue barrier array.
          action = Arrive/Await
        * If the barrier type has a guarding requirement, additionally,
          check equivalent CollTilings for matched Arrive/Await.
        * If the barrier type requires the same threads for Arrive/Await,
          check equivalent CollTilings for same-barrier Arrive/Await.

        """
        assert isinstance(sync, LoopIR.SyncStmt)
        nm = sync.barriers[0].name
        barrier_usage = get_barrier_usage(nm)
        state = get_state(nm)
        assert isinstance(barrier_usage, BarrierUsage)
        assert isinstance(state, DistributedAllocState)

        # We will update state.arrive_coll_tiling or state.await_coll_tiling
        sync_type = sync.sync_type
        assert sync_type.is_split()
        # Get CollTiling for Arrive >> name or Await(name)
        name = sync.barriers[0].name
        is_await = sync_type.is_await()
        old_coll_tiling = (
            state.await_coll_tiling if is_await else state.arrive_coll_tiling
        )

        # CollTilings that need to be equivalent
        to_check: List[Tuple[CollTiling, str]] = []

        if old_coll_tiling is not None:
            # Will check equivalence with previous stmt of same sync type
            f_text = str(sync)
            to_check.append((old_coll_tiling, f_text))
        else:
            # Save new state
            if is_await:
                state.await_coll_tiling = coll_tiling
            else:
                state.arrive_coll_tiling = coll_tiling

        if barrier_usage.barrier_mechanism.traits().requires_guarding:
            # Will check equivalence with previous stmt of matched sync type
            if sync_type.is_arrive():
                guarded_by = barrier_usage.guarded_by
                f_text = f"Await({guarded_by}, ...) [guarded_by]"
                other_state = get_state(guarded_by)
                if other_state is not None:
                    other_coll_tiling = other_state.await_coll_tiling
            else:
                guards = barrier_usage.guards
                f_text = f"Arrive(...) >> {guards} [guards]"
                other_state = get_state(guards)
                if other_state is not None:
                    other_coll_tiling = other_state.arrive_coll_tiling
            if other_coll_tiling is not None:
                to_check.append((other_coll_tiling, f_text))

        if not barrier_usage.barrier_mechanism.traits().different_arrive_await_threads:
            # Will check equivalence between Arrive/Await.
            if sync_type.is_arrive():
                f_text = f"Await({nm}) [different_arrive_await_threads=False]"
                other_coll_tiling = state.await_coll_tiling
            else:
                f_text = f"Arrive(...) >> {nm} [different_arrive_await_threads=False]"
                other_coll_tiling = state.arrive_coll_tiling
            if other_coll_tiling is not None:
                to_check.append((other_coll_tiling, f_text))

        # Check equivalence; subdiv_only=False case is stricter, checking perfect
        # equality of all executing thread sets. (We still rely on the rest of
        # distributed memory analysis to reason that the thread sets are assigned
        # to barrier array elements consistently).
        for old_coll_tiling, f_text in to_check:
            if msg := old_coll_tiling.base_mismatch(coll_tiling, subdiv_only=False):
                raise ValueError(
                    f"{sync.srcinfo}: {sync} has inconsistent collective tiling with previous {f_text}: {msg}"
                )

    def bad_idx(self, node, msg):
        distributed_iters = list(self.distributed_iters_needed)
        distributed_iters.sort()
        txt = ", ".join(str(sym) for sym in distributed_iters)
        raise ValueError(
            f"{node.srcinfo}: Distributed memory deduction "
            f"for {node.name} failed:\n{msg}\n"
            f"(at {self.context_stmt}, searching for iterators [{txt}]) "
        )
