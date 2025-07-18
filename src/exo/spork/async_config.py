from copy import deepcopy
from typing import Optional, List, Dict
from warnings import warn

from .timelines import (
    Instr_tl,
    Sync_tl,
    cpu_in_order_instr,
    cuda_in_order_instr,
    cuda_async_instr_tl,
)
from .base_with_context import BaseWithContext, is_if_holding_with
from .cuda_warp_config import CudaWarpConfig, WarpLayoutInfo
from ..core.LoopIR import LoopIR, LoopIR_Rewrite
from ..core.memory import DRAM, Memory, SpecialWindow, AllocableMemWin


class BaseAsyncConfig(BaseWithContext):
    """Base class for a configuration of an async block.

    For example, the derived CudaDeviceFunction configures a block of
    code to be interpreted as code lowered to a CUDA device function.

    At a minimum, the derived class must specify the instr-tl that
    the child statements execute with, the name of the device
    (compiler backend), and parent_instr_tl.

    """

    __slots__ = []

    def get_instr_tl(self):
        raise NotImplementedError()

    def get_device_name(self):
        raise NotImplementedError()

    def parent_instr_tl(self) -> Instr_tl:
        """Controls allowed nesting of async blocks in other async blocks.

        The async block must appear in a code block with the given instr-tl
        """
        raise NotImplementedError()


class CudaDeviceFunction(BaseAsyncConfig):
    __slots__ = [
        "blockDim",
        "clusterDim",
        "blocks_per_sm",
        "_warp_config_arg",
        "named_warps",
        "setmaxnreg_is_inc",
    ]

    blockDim: int
    clusterDim: int
    blocks_per_sm: int
    _warp_config_arg: object  # passed through to repr
    named_warps: Dict[str, WarpLayoutInfo]
    # Census of CudaWarpConfig setmaxnreg requests, and whether that register
    # count corresponds to setmaxnreg.inc (or setmaxnreg.dec)
    setmaxnreg_is_inc: Dict[int, bool]

    def __init__(
        self,
        blockDim: Optional[int] = None,
        clusterDim: int = 1,
        blocks_per_sm: int = 1,
        warp_config: Optional[List[CudaWarpConfig]] = None,
    ):
        assert isinstance(clusterDim, int) and clusterDim > 0
        self.clusterDim = clusterDim
        assert isinstance(blocks_per_sm, int) and blocks_per_sm > 0
        self.blocks_per_sm = blocks_per_sm
        self._warp_config_arg = warp_config

        if blockDim is None:
            assert (
                warp_config
            ), "CudaDeviceFunction: Provide exactly one of blockDim or warp_config"
            assert all(isinstance(c, CudaWarpConfig) for c in warp_config)
            self._init_from_warp_config(warp_config)
        else:
            assert (
                not warp_config
            ), "CudaDeviceFunction: Provide exactly one of blockDim or warp_config"
            self._init_from_blockDim(blockDim)

    def get_instr_tl(self):
        return cuda_in_order_instr

    def get_device_name(self):
        return "cuda"

    def parent_instr_tl(self):
        return cpu_in_order_instr

    def __repr__(self):
        args = []
        if not self._warp_config_arg:
            args.append(f"blockDim = {self.blockDim}")
        if self.clusterDim != 1:
            args.append(f"clusterDim = {self.clusterDim}")
        if self.blocks_per_sm != 1:
            args.append(f"blocks_per_sm = {self.blocks_per_sm}")
        if self._warp_config_arg:
            args.append(f"warp_config = {self._warp_config_arg}")

        return f"CudaDeviceFunction({', '.join(args)})"

    def _init_from_blockDim(self, blockDim):
        # Warp divisibility. This is not strictly required by CUDA, but the
        # valid usage for warp-aligned / CTA-aligned stuff becomes really
        # unclear when there's a partial warp.
        if not isinstance(blockDim, int) or blockDim % 32 != 0 or blockDim <= 0:
            raise ValueError(
                f"CudaDeviceFunction: blockDim={blockDim} must be a positive multiple of 32"
            )
        self.blockDim = blockDim
        self.named_warps = {"": WarpLayoutInfo(0, blockDim // 32, "", 0)}
        self.setmaxnreg_is_inc = {}

    def _init_from_warp_config(self, warp_config):
        cnames = set()
        offset = 0
        have_setmaxnreg = False
        self.named_warps = {}
        self.setmaxnreg_is_inc = {}

        for i, w in enumerate(warp_config):
            # Convert name of CudaWarpConfig to a substring that can be
            # used as the suffix of a C identifier. Always start with
            # an underscore, unless the name is empty.
            tmp = "".join(c for c in w.name if c.isalnum() or c == "_")
            if tmp and not tmp.startswith("_"):
                tmp = "_" + tmp
            cname = tmp
            suffix = 0
            while cname in cnames:
                suffix += 1
                cname = f"{tmp}_{suffix}"
            cnames.add(cname)

            if w.name in self.named_warps:
                self._bad_warp_config(i, warp_config, f"Duplicate warp name {w.name!r}")

            is_inc = w.setmaxnreg_inc is not None
            setmaxnreg = w.setmaxnreg_inc if is_inc else w.setmaxnreg_dec
            have_setmaxnreg |= setmaxnreg is not None
            self.named_warps[w.name] = WarpLayoutInfo(
                offset, w.count, cname, setmaxnreg or 0
            )

            offset += w.count

        self.blockDim = offset * 32

        if have_setmaxnreg:
            self._init_setmaxnreg_is_inc(warp_config)

    def _init_setmaxnreg_is_inc(self, warp_config):
        if self.blockDim % 128 != 0:
            self._bad_warp_config(
                len(warp_config) - 1,
                warp_config,
                f"setmaxnreg requires multiples of 128 threads; blockDim={self.blockDim}",
            )

        offset = 0
        prev_setmaxnreg = None
        for i, w in enumerate(warp_config):
            assert w.setmaxnreg_inc is None or w.setmaxnreg_dec is None
            is_inc = w.setmaxnreg_inc is not None
            setmaxnreg = w.setmaxnreg_inc if is_inc else w.setmaxnreg_dec

            if setmaxnreg != prev_setmaxnreg and offset % 4 != 0:
                self._bad_warp_config(
                    i,
                    warp_config,
                    "setmaxnreg must be uniform within warpgroups (128 threads)",
                )
            prev_setmaxnreg = setmaxnreg

            if setmaxnreg is None:
                continue

            if setmaxnreg < 24 or setmaxnreg > 256 or setmaxnreg % 8 != 0:
                self._bad_warp_config(
                    i, warp_config, "setmaxnreg must be a multiple of 8 in [24, 256]"
                )

            if self.setmaxnreg_is_inc.get(setmaxnreg) == (not is_inc):
                self._bad_warp_config(
                    i,
                    warp_config,
                    f"regcount {setmaxnreg} used both for setmaxnreg.inc and setmaxnreg.dec",
                )

            self.setmaxnreg_is_inc[setmaxnreg] = is_inc

            offset += w.count

    def _bad_warp_config(self, i, warp_config, msg):
        lines = [f"  {w}" if i != j else f"> {w} <" for j, w in enumerate(warp_config)]
        info = "\n".join(lines)
        raise ValueError(f"CudaDeviceFunction.warp_config: {msg}:\n{info}")


class CudaAsync(BaseAsyncConfig):
    __slots__ = ["_instr_tl"]

    def __init__(self, instr_tl: Instr_tl):
        instr_tl = instr_tl.as_instr_tl()
        assert instr_tl.is_cuda_async()
        self._instr_tl = instr_tl

    def get_instr_tl(self):
        return self._instr_tl

    def get_device_name(self):
        return "cuda"

    def parent_instr_tl(self):
        return cuda_in_order_instr

    def __repr__(self):
        return f"CudaAsync({self._instr_tl})"

    def __eq__(self, other):
        return type(other) == CudaAsync and self._instr_tl == other._instr_tl


class InstrTimelineAnalysis(LoopIR_Rewrite):
    __slots__ = ["instr_tl", "instr_tl_seen", "sym_memwin", "contains_sync"]

    def __init__(self):
        self.instr_tl = cpu_in_order_instr  # Currently inspected scope's instr-tl
        self.instr_tl_seen = {cpu_in_order_instr}
        self.sym_memwin = dict()  # Sym -> MemWin type
        self.contains_sync = False

    def map_s(self, s):
        old_instr_tl = self.instr_tl

        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, BaseAsyncConfig):
                needed = ctx.parent_instr_tl()
                if needed != self.instr_tl:
                    raise ValueError(
                        f"{s.srcinfo}: {ctx!r} "
                        f"requires instr-tl {needed}; instr-tl "
                        f"in scope is actually {self.instr_tl}"
                    )
                instr_tl = ctx.get_instr_tl()
                self.instr_tl = instr_tl
                self.instr_tl_seen.add(instr_tl)
        else:
            self.inspect_s(s)

        super().map_s(s)
        self.instr_tl = old_instr_tl

    def map_e(self, e):
        self.inspect_e(e)
        super().map_e(e)

    def inspect_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if not s.type.is_numeric():
                return

            memwin = self.sym_memwin[s.name]
            perm = memwin.instr_tl_permission(self.instr_tl, is_instr=False)
            if "w" in perm:
                assert "r" in perm, "Not supported: write without read permission"
            else:
                self.warn_weird_letters(memwin, perm)
                action = "mutable access" if "r" in perm else "any access"
                raise TypeError(
                    f"{s.srcinfo}: {s.name} @ "
                    f"{memwin.name()} does not allow {action} in a "
                    f"scope with instr-tl {self.instr_tl}"
                )
        elif isinstance(s, LoopIR.SyncStmt):
            self.contains_sync = True
            if s.sync_type.is_split():
                for e in s.barriers:
                    memwin = self.sym_memwin[e.name]
                    perm = memwin.instr_tl_permission(self.instr_tl, is_instr=False)
                    if "w" in perm:
                        assert (
                            "r" in perm
                        ), "Not supported: write without read permission"
                    else:
                        self.warn_weird_letters(memwin, perm)
                        raise TypeError(
                            f"{s.srcinfo}: {e.name} (barrier type "
                            f"{memwin.name()}) does not allow SyncStmt in a "
                            f"scope with instr-tl {self.instr_tl}"
                        )
        elif isinstance(s, LoopIR.Alloc):
            self.contains_sync |= s.type.is_barrier()
            mem = s.mem or DRAM
            self.sym_memwin[s.name] = mem
            assert issubclass(mem, AllocableMemWin)
            perm = mem.instr_tl_permission(self.instr_tl, is_instr=False)
            if "c" not in perm:
                self.warn_weird_letters(mem, perm)
                raise TypeError(
                    f"{s.srcinfo}: {s.name} @ "
                    f"{mem.name()} cannot be allocated in a scope "
                    f"with instr-tl {self.instr_tl}"
                )
        elif isinstance(s, LoopIR.WindowStmt):
            special_window = s.special_window
            self.sym_memwin[s.name] = special_window or self.sym_memwin[s.rhs.name]

            if not special_window:
                return

            assert issubclass(special_window, SpecialWindow)
            perm = special_window.instr_tl_permission(self.instr_tl, is_instr=False)
            if "c" not in perm:
                self.warn_weird_letters(special_window, perm)
                raise TypeError(
                    f"{s.srcinfo}: a special window {s.name} "
                    f"of type {special_window.name()} cannot be "
                    f"constructed in a scope with instr-tl "
                    f"{self.instr_tl}"
                )
        elif isinstance(s, LoopIR.Call):
            callee = s.f
            needed = callee.proc_instr_tl()
            if self.instr_tl != needed:
                note = ""
                if needed.is_cuda_async():
                    note = f"; wrap with CudaAsync({needed})"
                raise TypeError(
                    f"{s.srcinfo}: {callee.name}() requires instr-tl "
                    f"{needed}; scope has instr-tl "
                    f"{self.instr_tl}{note}"
                )

    def inspect_e(self, e):
        if isinstance(e, LoopIR.Read) and e.type.is_numeric():
            memwin = self.sym_memwin[e.name]
            perm = memwin.instr_tl_permission(self.instr_tl, is_instr=False)
            if "r" not in perm:
                assert "w" not in perm, "Not supported: write without read permission"
                raise TypeError(
                    f"{e.srcinfo}: {e.name} @ "
                    f"{memwin.name()} does not allow reads in a "
                    f"scope with instr-tl {self.instr_tl}"
                )

    def run(self, proc):
        for arg in proc.args:
            memwin = arg.mem or DRAM
            self.sym_memwin[arg.name] = memwin
        return super().apply_proc(proc)

    def warn_weird_letters(self, memwin, perm):
        for c in perm:
            if c not in "rwc":
                warn(f"{memwin.name()}.instr_tl_permission gave unknown letter {c!r}")
