from __future__ import annotations

from math import prod
from typing import List, Optional, Tuple

from ..core.prelude import Sym, SrcInfo, extclass

from .collectives import CollectiveUnit, SpecializeCollective, cuda_cluster, cuda_block
from .loop_modes import LoopMode, CudaClusters, CudaBlocks
from .actor_kinds import ActorKind, cpu, cuda_sync
from .async_config import BaseAsyncConfig, CudaDeviceFunction, CudaAsync

from ..core.LoopIR import LoopIR, LoopIR_Do, GetReads, GetWrites


class ParallelForRecord(object):
    __slots__ = ["exo_iter", "c_iter", "c_range", "static_range"]

    exo_iter: str  # Iteration variable name in Exo code
    c_iter: str  # Iteration variable name in C code
    # [c_range[0], c_range[1]) -- iteration bounds as compiled C code fragments
    c_range: Tuple[str, str]
    # [static_range[0], static_range[1]) -- statically analyzed maximal bounds
    static_range: Tuple[int, int]

    def __init__(self, exo_iter, c_iter, c_range, static_range):
        self.exo_iter = exo_iter
        self.c_iter = c_iter
        self.c_range = c_range
        self.static_range = static_range

    def __repr__(self):
        return f"ParallelForRecord(exo_iter={self.exo_iter}, c_iter={self.c_iter}, c_range={self.c_range}, static_range={self.static_range})"


class ForRecord(object):
    __slots__ = ["loop_mode"]

    loop_mode: LoopMode

    def __init__(self, loop_mode):
        self.loop_mode = loop_mode

    def __repr__(self):
        return f"ForRecord({self.loop_mode})"


class CollectiveScope(object):
    __slots__ = [
        "parent",
        "root_node",
        "relative_thread_range",
        "collective_unit",
        "loop_records",
        "partial",
    ]

    parent: Optional[CollectiveScope]
    root_node: (LoopIR.For, LoopIR.If)
    relative_thread_range: Tuple[int, int]
    collective_unit: CollectiveUnit
    loop_records: List[ParallelForRecord]
    partial: bool

    def __init__(self, parent, root_node, relative_thread_range, collective_unit):
        assert parent is None or isinstance(parent, CollectiveScope)
        assert isinstance(root_node, (LoopIR.For, LoopIR.If))

        self.parent = parent
        self.root_node = root_node
        self.relative_thread_range = relative_thread_range
        self.collective_unit = collective_unit
        self.loop_records = []
        self.partial = True

    def static_shape(self):
        return [record.static_range for record in self.loop_records]

    def defined_by_str(self):
        # TODO change when LoopIR with hack is fixed
        if isinstance(self.root_node, LoopIR.If):
            return f"defined by SpecializeCollective @ {self.root_node.srcinfo}"
        else:
            return f"defined by {self.root_node.iter} loop @ {self.root_node.srcinfo}"

    def __repr__(self):
        return f"<CollectiveScope {type(self.root_node)} {str(self.collective_unit)} threads:{self.relative_thread_range}>"


# This is where most of the backend logic for CUDA should go.
class SporkEnv(object):
    _base_kernel_name: str
    _future_lines: List
    _parallel_for_stack: Optional[ParallelForRecord]
    _async_config_stack: List[BaseAsyncConfig]
    _collective_scope: CollectiveScope
    _clusterDim: Optional[int]
    _blockDim: int
    _grid_str: Optional[str]

    __slots__ = {
        "_base_kernel_name": "Prefix of name of the CUDA kernel being generated",
        "_future_lines": "List of str lines, or objects that can be converted to str TODO",
        "_parallel_for_stack": "Info pushed for each parallel for",
        "_async_config_stack": "Info pushed for each async block",
        "_collective_scope": "Current collective scope information",
        "_clusterDim": "Cuda blocks per cuda cluster",
        "_blockDim": "Cuda threads per cuda block",
        "_grid_str": "Compiled expression for grid size",
    }

    def __init__(self, base_kernel_name: str, async_stmt: LoopIR.If):
        assert isinstance(async_stmt, LoopIR.If)
        assert isinstance(async_stmt.cond, LoopIR.Const)
        assert isinstance(async_stmt.cond.val, CudaDeviceFunction)
        config = async_stmt.cond.val

        self._base_kernel_name = base_kernel_name
        self._future_lines = []
        self._parallel_for_stack = []
        self._async_config_stack = [config]
        self._collective_scope = None
        self._clusterDim = config.clusterDim
        self._blockDim = config.blockDim

        if len(async_stmt.body) == 1 and isinstance(async_stmt.body[0], LoopIR.For):
            for_stmt = async_stmt.body[0]
        else:
            # TODO fix arbitrary restriction
            raise ValueError(
                "{async_stmt.srcinfo}: must have a single parallel for as body"
            )

        loop_mode = for_stmt.loop_mode
        if isinstance(loop_mode, CudaClusters):
            pass
        elif isinstance(loop_mode, CudaBlocks):
            pass
        else:
            raise ValueError(
                f"{for_stmt.srcinfo}: CUDA kernel must be defined by a `for cuda_clusters` or `for cuda_blocks` loop"
            )

    def __bool__(self):
        return True

    def get_device_name(self):
        return "cuda"

    def add_line(self, line: str):
        assert isinstance(line, str)
        self._future_lines.append(line)

    def get_async_config(self):
        assert self._async_config_stack
        return self._async_config_stack[-1]

    def push_async(self, config: BaseAsyncConfig):
        assert isinstance(
            config, CudaAsync
        ), "compiler error: incorrect async_config should have been detected earlier"
        self._async_config_stack.append(config)

    def pop_async(self):
        self._async_config_stack.pop()

    def push_for(
        self,
        s: LoopIR.For,
        c_iter: str,
        c_range,
        sym_range,
        tabs,
    ) -> bool:
        emit_loop = True
        if s.loop_mode.is_par:
            static_range = (
                sym_range[0],
                None if sym_range[1] is None else sym_range[1] + 1,
            )
            parallel_for_record = ParallelForRecord(
                s.iter, c_iter, c_range, static_range
            )
            self._parallel_for_stack.append(parallel_for_record)
            self._push_collective_scope_update(s, parallel_for_record)
            emit_loop = False
            if not self._collective_scope.partial:
                self._compile_collective_scope_iter_vars(tabs)
        return emit_loop

    def pop_for(self, s: LoopIR.For):
        if s.loop_mode.is_par:
            assert self._parallel_for_stack
            self._parallel_for_stack.pop()
            self._pop_collective_scope_update(s)

    # TODO change when LoopIR with hack is fixed
    def push_specialize_collective(self, s: LoopIR.If) -> str:
        spec = s.cond.val
        assert isinstance(spec, SpecializeCollective)
        self._push_collective_scope_update(s)
        current_collective_scope, parent_collective_scope = (
            self._collective_scope,
            self._collective_scope.parent,
        )
        assert parent_collective_scope is not None
        current_collective_unit = current_collective_scope.collective_unit
        parent_collective_unit = parent_collective_scope.collective_unit
        assert parent_collective_unit.contains(current_collective_unit)
        assert isinstance(current_collective_unit.thread_count, int)

        if parent_collective_unit == cuda_block:
            modulo_str = ""
        else:
            assert isinstance(parent_collective_unit.thread_count, int)
            modulo_str = f" % {parent_collective_unit.thread_count}u"
        if current_collective_unit.thread_count == 1:
            collective_index_str = f"threadIdx.x{modulo_str}"
        else:
            collective_index_str = (
                f"(threadIdx.x / {current_collective_unit.thread_count}u){modulo_str}"
            )

        assert spec.lo >= 0
        assert spec.hi > spec.lo
        if spec.lo == 0:
            return f"{collective_index_str} < {spec.hi}u"
        else:
            return f"{collective_index_str} >= {spec.lo}u && {collective_index_str} < {spec.hi}u"

    # TODO change when LoopIR with hack is fixed
    def pop_specialize_collective(self, s: LoopIR.If):
        assert self._collective_scope.root_node is s
        self._collective_scope = self._collective_scope.parent

    def get_kernel_body(self):
        return "\n".join(["{"] + self._future_lines + ["}"])

    def get_kernel_prototype_launch(
        self, s: LoopIR.For, cpu_varname_env, cpu_envtyp, format_ctype
    ):
        scanner = KernelArgsScanner(cpu_varname_env, cpu_envtyp)
        scanner.do_s(s)
        ctypes_varnames = scanner.get_ctypes_varnames(format_ctype)
        name = self._base_kernel_name + "CU"
        prototype = (
            f"__global__ void {name}("
            + ", ".join(t + " " + n for t, n in ctypes_varnames)
            + ")"
        )
        grid = self._grid_str
        assert grid is not None, "called too early"
        launch = (
            f"if ({grid} > 0) {name}<<<{grid}, {self._blockDim}>>>("
            + ", ".join(n for t, n in ctypes_varnames)
            + ");"
        )
        return prototype, launch

    def _push_collective_scope_update(self, s, parallel_for_record=None):
        parent_collective_scope = self._collective_scope
        if parent_collective_scope is not None:
            parent_collective_unit = parent_collective_scope.collective_unit
            parent_collective_threads = self._collective_unit_thread_count(
                parent_collective_unit
            )

        # TODO change when LoopIR with hack is fixed
        if isinstance(s, LoopIR.If):
            assert parallel_for_record is None
            spec = s.cond.val
            assert isinstance(spec, SpecializeCollective)
            new_collective_unit = spec.unit
            if new_collective_unit.thread_count is None:
                # Currently can't specialize blocks within clusters. Assume
                # collective specialization is implemented with threadIdx for now.
                raise ValueError(
                    "f{s.srcinfo}: Can't specialize collectives of unit type {str(new_collective_unit)}"
                )
            collective_thread_count = new_collective_unit.thread_count
            relative_thread_range = (
                spec.lo * collective_thread_count,
                spec.hi * collective_thread_count,
            )
        elif isinstance(s, LoopIR.For):
            assert isinstance(parallel_for_record, ParallelForRecord)
            loop_mode = s.loop_mode
            assert loop_mode.is_par
            new_collective_unit = loop_mode.collective_unit()
            collective_thread_count = self._collective_unit_thread_count(
                new_collective_unit
            )
            # This will be used if no collective specialization was defined.
            # Opportunistically use all parallelism available in parent
            # collective.  This is meaningless when there is no parent
            # collective scope, but we fill in "good enough" values sufficient
            # to avoid errors downstream.
            if parent_collective_scope is None:
                relative_thread_range = (0, collective_thread_count)
            else:
                relative_thread_range = (0, parent_collective_threads)
        else:
            assert 0

        # Check that this parallel for or SpecializeCollective is valid
        # to open a new collective scope in the existing collective scope.
        # Part 1: unit checking (e.g. can't nest cuda blocks in cuda threads)
        if parent_collective_scope is not None:
            if parent_collective_scope.partial:
                # TODO change when LoopIR with hack is fixed
                if isinstance(s, LoopIR.If):
                    raise ValueError(
                        f"{s.srcinfo}: Can't specialize in partial collective scope (must nest inside parallel for)"
                    )
                else:
                    # Is this reachable? Shouldn't have been detected
                    # as partial if the nested for loop's collecitve
                    # unit type doesn't match?
                    assert isinstance(s, LoopIR.For)
                    assert s.loop_mode.is_par
                    if s.loop_mode.collective_unit() != parent_collective_unit:
                        raise ValueError(
                            f"{s.srcinfo}: {s.iter} loop uses wrong collective unit;"
                            f" match {parent_collective_unit} {parent_collective_scope.defined_by_str()}"
                        )

            elif not parent_collective_unit.contains(
                new_collective_unit
            ):  # and not partial

                # TODO change when LoopIR with hack is fixed
                if isinstance(s, LoopIR.If):
                    ValueError(
                        f"{s.srcinfo}: Can't specialize {new_collective_unit} in {parent_collective_unit} scope"
                    )
                if isinstance(s, LoopIR.For):
                    note = ""
                    if new_collective_unit == parent_collective_unit:
                        # This can improperly trip if bugs are introduced to
                        # the code for "detect multidimensional iteration
                        # space for collective scope"
                        note = "(to define a multidimensional iteration space, this loop must be the only statement in the body of the parent loop) "
                    raise ValueError(
                        f"{s.srcinfo}: {note}Can't nest {new_collective_unit} loop"
                        f" ({s.iter}) in {parent_collective_unit} scope,"
                        f" {parent_collective_scope.defined_by_str()}"
                    )

        defines_new_collective_scope = (
            parent_collective_scope is None or not parent_collective_scope.partial
        )

        # Part 2: size checking, must fit in parent collective's threads
        if defines_new_collective_scope and parent_collective_scope is not None:
            max_thread_index = relative_thread_range[1] - 1
            fail_reason = None
            if max_thread_index >= parent_collective_threads:
                fail_reason = f"{spec} maximum relative thread index {max_thread_index} requested overflows"
            if parent_collective_threads % collective_thread_count != 0:
                fail_reason = f"{str(new_collective_unit)} ({collective_thread_count} threads) does not divide"

            if fail_reason:
                raise ValueError(
                    f"{s.srcinfo}:"
                    f" {fail_reason} {parent_collective_threads} threads"
                    f" available from a single parent {str(parent_collective_unit)},"
                    f" {parent_collective_scope.defined_by_str()}"
                )

        # Define new collective scope and/or increase the dimension of the
        # loop iteration space of the collective scope
        if defines_new_collective_scope:
            self._collective_scope = CollectiveScope(
                parent_collective_scope, s, relative_thread_range, new_collective_unit
            )
        if parallel_for_record is not None:
            self._collective_scope.loop_records.append(parallel_for_record)
            # Detect multidimensional iteration space for collective scope
            self._collective_scope.partial = False
            if len(s.body) == 1:
                body_stmt = s.body[0]
                if isinstance(body_stmt, LoopIR.For):
                    loop_mode = body_stmt.loop_mode
                    self._collective_scope.partial = (
                        loop_mode.is_par
                        and loop_mode.collective_unit() == new_collective_unit
                    )

    def _pop_collective_scope_update(self, s):
        assert isinstance(self._collective_scope, CollectiveScope)

        # TODO change when LoopIR with hack is fixed
        if isinstance(s, LoopIR.If):
            assert self._collective_scope.root_node is s
            self._collective_scope = self._collective_scope.parent
        elif isinstance(s, LoopIR.For):
            assert s.loop_mode.is_par
            loop_record = self._collective_scope.loop_records.pop()
            assert loop_record.exo_iter == s.iter
            self._collective_scope.partial = True
            if self._collective_scope.root_node is s:
                self._collective_scope = self._collective_scope.parent
        else:
            assert 0

    def _compile_collective_scope_iter_vars(self, tabs):
        collective_scope = self._collective_scope
        assert isinstance(collective_scope, CollectiveScope)
        assert not collective_scope.partial
        assert collective_scope.loop_records

        lo_tid, hi_tid = collective_scope.relative_thread_range
        collective_unit = collective_scope.collective_unit
        collective_threads = self._collective_unit_thread_count(collective_unit)
        if collective_scope.parent is None:
            # Top-level cuda_clusters or cuda_blocks loop defines correctly-sized
            # kernel launch so we can treat it like it has infinite threads.
            finite_threads = False
        else:
            finite_threads = True
            num_threads = hi_tid - lo_tid
            num_collectives = num_threads // collective_threads

        # Dimensionality of multidimensional iteration space
        dim = len(collective_scope.loop_records)

        # Error message helper
        def kvetch(reason):
            parts = []
            parts.append(f"{collective_scope.root_node.srcinfo}:")
            parts.append(f"invalid parallel {collective_scope.collective_unit} loop:")
            parts.append(reason)
            parts.append("\nITERATION VARS:")
            for record in collective_scope.loop_records:
                s_lo, s_hi = record.static_range
                if s_lo is None:
                    s_lo = "?"
                if s_hi is None:
                    s_hi = "?"
                parts.append(f"{record.exo_iter}:({s_lo},{s_hi})")
            raise ValueError(" ".join(parts))

        # Check number of collectives used is correct, based on static bounds
        # of loop iteration variables.
        if finite_threads:
            collectives_needed = 1
            for record in collective_scope.loop_records:
                s_lo, s_hi = record.static_range
                if s_lo is None or s_hi is None:
                    kvetch(f"could not deduce static bounds on {record.exo_iter}")
                collectives_needed *= s_hi - s_lo

            if collectives_needed > num_collectives:
                kvetch(
                    f"{collectives_needed} collective lanes"
                    f" ({collective_scope.collective_unit}) needed"
                    f" but only {num_collectives} collective lanes"
                    f" ({num_threads} threads) of parent"
                    f" {collective_scope.parent.collective_unit} available"
                )

        # Number the collectives allocated to the parallel iteration space
        # starting from 0 (the start from 0 thing is non-trivial when we do
        # collective specialization, e.g. assigning warps 4-7 to do something)
        assert lo_tid % collective_threads == 0
        collective_offset_str = (
            "" if lo_tid == 0 else f" - {lo_tid // collective_threads}u"
        )
        if collective_unit == cuda_cluster:
            collective_index = (
                f"(blockIdx.x / {self._clusterDim}u{collective_offset_str})"
            )
        elif collective_unit == cuda_block:
            if collective_offset_str:
                collective_index = f"(blockIdx.x{collective_offset_str})"
            else:
                collective_index = "blockIdx.x"
        else:
            assert isinstance(collective_unit.thread_count, int)
            if collective_unit.thread_count != 1 or collective_offset_str:
                collective_index = f"(threadIdx.x / {collective_unit.thread_count}{collective_offset_str})"
            else:
                collective_index = "threadIdx.x"

        def dim_size_in_parens(record):
            if record.c_range[0] == "0":
                return "(" + record.c_range[1] + ")"
            else:
                return f"(({record.c_range[1]}) - ({record.c_range[0]}))"

        # Generate code for "iteration" variables.
        # A "for loop" becomes an if statement, masking out collectives assigned
        # to outside the multidimensional iteration space.
        # Inner loop varies fastest wrt collective index, similar to sequential code.
        for i_dim, record in enumerate(collective_scope.loop_records):
            var = record.c_iter
            modulus = f"unsigned{dim_size_in_parens(record)}"
            if i_dim == dim - 1:
                div_stride = ""
            else:
                div_stride = f" / unsigned({'*'.join(dim_size_in_parens(r) for r in collective_scope.loop_records[i_dim+1:])})"
            expr = f"int(({collective_index}{div_stride}) % {modulus})"
            if record.c_range[0] != "0":
                expr = f"{expr} + {record.c_range[0]}"
            dim_tabs = tabs[2 * (dim - i_dim - 1) :]
            if (
                i_dim != 0
                and record.c_range[0] == str(record.static_range[0])
                and record.c_range[1] == str(record.static_range[1])
            ):
                # Omit loop condition if we know the static and dynamic ranges
                # match, except the outermost dim, where we may still need to
                # guard excess threads.
                cond = "1"
            else:
                cond = f"{record.c_range[0]} <= {var} && {var} < {record.c_range[1]}"
            self.add_line(f"{dim_tabs}if (auto {var} = {expr}; {cond}) {{")

        # If this is the top-level parallel loop nest, we also need to generate
        # the string for configuring the number of blocks to launch.
        if collective_scope.parent is None:
            self._grid_str = " * ".join(
                dim_size_in_parens(record) for record in collective_scope.loop_records
            )
            if collective_unit == cuda_cluster:
                self._grid_str = f"self._clusterDim * {self._grid_str}"
            else:
                assert collective_unit == cuda_block

    def _collective_unit_thread_count(self, collective_unit):
        if (threads := collective_unit.thread_count) is not None:
            return threads
        elif collective_unit == cuda_cluster:
            # We check and return 0 on invalid use of cuda_clusters to not
            # interfere with higher quality error messages defined elsewhere.
            blocks = self._clusterDim
            return self._blockDim * (0 if blocks is None else blocks)
        elif collective_unit == cuda_block:
            return self._blockDim
        else:
            assert 0


class KernelArgsScanner(LoopIR_Do):
    __slots__ = ["cpu_varname_env", "cpu_envtyp", "cpu_to_gpu_mut"]

    cpu_varname_env: dict[Sym, str]
    cpu_envtyp: dict[Sym, LoopIR.type]
    cpu_to_gpu_mut: dict[Sym, bool]

    def __init__(self, cpu_varname_env, cpu_envtyp):
        self.cpu_varname_env = cpu_varname_env
        self.cpu_envtyp = cpu_envtyp
        self.cpu_to_gpu_mut = {}

    def do_s(self, s):
        self.do_indexing(s)
        if write_info := GetWrites.scan_s(s):
            sym, _ = write_info
            if sym in self.cpu_varname_env:
                self.cpu_to_gpu_mut[sym] = True
        super().do_s(s)

    def do_e(self, e):
        self.do_indexing(e)
        if read_info := GetReads.scan_e(e):
            sym, _ = read_info
            if sym in self.cpu_varname_env:
                self.cpu_to_gpu_mut.setdefault(sym, False)
        super().do_e(e)

    def do_indexing(self, node):
        """Check node for tensor indexing.

        We need to scan not just explicit variable reads/writes in Exo
        object code, but also size variables that will be used only in
        the generated C code for computing indices into tensors

        """
        try:
            sym = node.name
        except AttributeError:
            return
        typ = self.cpu_envtyp.get(sym)
        if isinstance(typ, LoopIR.Tensor):
            for e in typ.hi:
                self.do_e(e)

    def get_ctypes_varnames(self, format_ctype):
        sym_mut = sorted(self.cpu_to_gpu_mut.items(), key=lambda tup: tup[0]._id)
        return [
            (
                format_ctype(self.cpu_envtyp[sym], self.cpu_to_gpu_mut[sym]),
                self.cpu_varname_env[sym],
            )
            for sym, _ in sym_mut
        ]
