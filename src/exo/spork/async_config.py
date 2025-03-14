from typing import Optional
from warnings import warn

from .actor_kinds import ActorKind
from . import actor_kinds
from .base_with_context import BaseWithContext, is_if_holding_with
from ..core.LoopIR import LoopIR, LoopIR_Rewrite
from ..core.memory import DRAM, Memory, SpecialWindow


class BaseAsyncConfig(BaseWithContext):
    """Base class for a configuration of an async block.

    For example, the derived CudaDeviceFunction configures a block of
    code to be interpreted as code lowered to a CUDA device function.

    At a minimum, the derived class must specify the ActorKind that
    the child statements execute with, the name of the device
    (compiler backend), and parent_actor_kind.

    """

    __slots__ = []

    def get_actor_kind(self):
        raise NotImplementedError()

    def get_device_name(self):
        raise NotImplementedError()

    def parent_actor_kind(self) -> ActorKind:
        """Controls allowed nesting of async blocks in other async blocks.

        The async block must appear in a code block with the given actor kind
        """
        raise NotImplementedError()


class CudaDeviceFunction(BaseAsyncConfig):
    __slots__ = ["blockDim", "clusterDim", "blocks_per_sm"]

    def __init__(self, blockDim: int, clusterDim: int = 1, blocks_per_sm: int = 1):
        assert isinstance(blockDim, int) and blockDim > 0
        assert isinstance(clusterDim, int) and clusterDim > 0
        self.blockDim = blockDim
        self.clusterDim = clusterDim
        self.blocks_per_sm = blocks_per_sm

    def get_actor_kind(self):
        return actor_kinds.cuda_classic  # Synchronous (non-async) CUDA instr

    def get_device_name(self):
        return "cuda"

    def parent_actor_kind(self):
        return actor_kinds.cpu

    def __repr__(self):
        return f"CudaDeviceFunction({self.blockDim}, {self.clusterDim}, {self.blocks_per_sm})"

    def __eq__(self, other):
        return (
            type(other) == CudaDeviceFunction
            and self.blockDim == other.blockDim
            and self.clusterDim == other.clusterDim
            and self.blocks_per_sm == other.blocks_per_sm
        )


class CudaAsync(BaseAsyncConfig):
    __slots__ = ["_actor_kind"]

    def __init__(self, actor_kind):
        assert actor_kind in actor_kinds.cuda_async_actor_kinds
        self._actor_kind = actor_kind

    def get_actor_kind(self):
        return self._actor_kind

    def get_device_name(self):
        return "cuda"

    def parent_actor_kind(self):
        return actor_kinds.cuda_classic

    def __repr__(self):
        return f"CudaAsync({self._actor_kind})"

    def __eq__(self, other):
        return type(other) == CudaAsync and self._actor_kind == other._actor_kind


class ActorKindAnalysis(LoopIR_Rewrite):
    __slots__ = ["actor_kind", "sym_memwin"]

    def __init__(self):
        self.actor_kind = actor_kinds.cpu  # Currently inspected scope's actor kind
        self.sym_memwin = dict()  # Sym -> MemWin type

    def map_s(self, s):
        old_actor_kind = self.actor_kind

        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, BaseAsyncConfig):
                needed = ctx.parent_actor_kind()
                if needed != self.actor_kind:
                    raise ValueError(
                        f"{s.srcinfo}: {ctx.__class__.__name__} "
                        f"requires actor kind {needed}; actor kind "
                        f"in scope is actually {self.actor_kind}"
                    )
                self.actor_kind = ctx.get_actor_kind()
        else:
            self.inspect_s(s)

        super().map_s(s)
        self.actor_kind = old_actor_kind

    def map_e(self, e):
        self.inspect_e(e)
        super().map_e(e)

    def inspect_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if not s.type.is_numeric():
                return

            memwin = self.sym_memwin[s.name]
            perm = memwin.actor_kind_permission(self.actor_kind, is_instr=False)
            if "w" in perm:
                assert "r" in perm, "Not supported: write without read permission"
            else:
                self.warn_weird_letters(memwin, perm)
                action = "mutable access" if "r" in perm else "any access"
                raise TypeError(
                    f"{s.srcinfo}: {s.name} (memory type "
                    f"{memwin.name()}) does not allow {action} in a "
                    f"scope with actor kind {self.actor_kind}"
                )
        elif isinstance(s, LoopIR.Alloc):
            mem = s.mem or DRAM
            self.sym_memwin[s.name] = mem
            assert issubclass(mem, Memory)
            perm = mem.actor_kind_permission(self.actor_kind, is_instr=False)
            if "c" not in perm:
                self.warn_weird_letters(mem, perm)
                raise TypeError(
                    f"{s.srcinfo}: {s.name} (memory type "
                    f"{mem.name()}) cannot be allocated in a scope "
                    f"with actor kind {self.actor_kind}"
                )
        elif isinstance(s, LoopIR.WindowStmt):
            special_window = s.special_window
            self.sym_memwin[s.name] = special_window or self.sym_memwin[s.rhs.name]

            if not special_window:
                return

            assert issubclass(special_window, SpecialWindow)
            perm = special_window.actor_kind_permission(self.actor_kind, is_instr=False)
            if "c" not in perm:
                self.warn_weird_letters(special_window, perm)
                raise TypeError(
                    f"{s.srcinfo}: a special window {s.name} "
                    f"of type {special_window.name()} cannot be "
                    f"constructed in a scope with actor kind "
                    f"{self.actor_kind}"
                )
        elif isinstance(s, LoopIR.Call):
            callee = s.f
            if callee.instr is None:
                if self.actor_kind != actor_kinds.cpu:
                    # We currently assume the top-level actor kind of all
                    # non-instr procs is cpu ... so must be called by CPU.
                    raise TypeError(
                        f"{s.srcinfo}: non-instr proc must be called "
                        f"by the CPU (scope has actor kind "
                        f"{self.actor_kind})"
                    )
            else:
                pass
                # TODO

    def inspect_e(self, e):
        if isinstance(e, LoopIR.Read) and e.type.is_numeric():
            memwin = self.sym_memwin[e.name]
            perm = memwin.actor_kind_permission(self.actor_kind, is_instr=False)
            if "r" not in perm:
                assert "w" not in perm, "Not supported: write without read permission"
                raise TypeError(
                    f"{e.srcinfo}: {e.name} (memory type "
                    f"{memwin.name()}) does not allow reads in a "
                    f"scope with actor kind {self.actor_kind}"
                )

    def run(self, proc):
        for arg in proc.args:
            memwin = arg.mem or DRAM
            self.sym_memwin[arg.name] = memwin
        return super().apply_proc(proc)

    def warn_weird_letters(self, memwin, perm):
        for c in perm:
            if c not in "rwc":
                warn(f"{memwin.name()}.actor_kind_permission gave unknown letter {c!r}")
