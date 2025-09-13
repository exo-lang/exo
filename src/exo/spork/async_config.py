from copy import deepcopy
from typing import Optional, List, Dict
from warnings import warn

from .timelines import (
    DeviceScope,
    Instr_tl,
    cpu_basic_device,
    cuda_basic_device,
)
from .base_with_context import BaseWithContext, is_if_holding_with
from .coll_algebra import clusterDim_param, blockDim_param, CollIndexExpr, CollTiling
from .cuda_warp_config import CudaWarpConfig, WarpLayoutInfo
from ..core.LoopIR import LoopIR, LoopIR_Rewrite
from ..core.memory import DRAM, Memory, SpecialWindow, AllocableMemWin


class BaseAsyncConfig(BaseWithContext):
    """Base class for a configuration of an async block.

    For example, the derived CudaDeviceFunction configures a block of
    code to be interpreted as code lowered to a CUDA device function.

    The derived class must specify the DeviceScope that
    the child statements execute with, and expected parent DeviceScope.

    """

    __slots__ = []

    def get_child_device(self) -> DeviceScope:
        raise NotImplementedError()

    def get_parent_device(self) -> DeviceScope:
        """Controls allowed nesting of async blocks in other async blocks.

        The async block must appear in a code block with the given device scope
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

    def get_child_device(self):
        return cuda_basic_device

    def get_parent_device(self):
        return cpu_basic_device

    def coll_env(self):
        return {clusterDim_param: self.clusterDim, blockDim_param: self.blockDim}

    def top_level_coll_tiling(self):
        # We seed the analysis of the collective units with the tiling
        # for the top-level collective (clusterDim x blockDim,
        # with redundant clusterDim removed if clusterDim = 1).
        blockDim = self.blockDim
        clusterDim = self.clusterDim
        assert clusterDim > 0 and isinstance(clusterDim, int)
        threadIdx_expr = CollIndexExpr("threadIdx.x", blockDim)
        if clusterDim == 1:
            tlc_offset = (0,)
            tlc_box = (blockDim,)
            intra_box_exprs = (threadIdx_expr,)
        else:
            tlc_offset = (0, 0)
            tlc_box = (clusterDim, blockDim)
            cta_expr = CollIndexExpr("blockIdx.x") % clusterDim
            intra_box_exprs = (cta_expr, threadIdx_expr)
        return CollTiling(
            None,  # parent
            None,  # _iter
            tlc_box,
            tlc_box,
            tlc_offset,
            tlc_box,
            intra_box_exprs,
            1,
            CollIndexExpr(0),
        )

    def __repr__(self):
        args = []
        if not self._warp_config_arg:
            args.append(f"blockDim={self.blockDim}")
        if self.clusterDim != 1:
            args.append(f"clusterDim={self.clusterDim}")
        if self.blocks_per_sm != 1:
            args.append(f"blocks_per_sm={self.blocks_per_sm}")
        if self._warp_config_arg:
            args.append(f"warp_config={self._warp_config_arg}")

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
        offset = 0
        have_setmaxnreg = False
        self.named_warps = {}
        self.setmaxnreg_is_inc = {}

        for i, w in enumerate(warp_config):
            # Convert name of CudaWarpConfig to a substring that can be
            # used as the suffix of a C identifier. Always start with
            # an underscore, unless the name is empty.
            cname = w.name
            if any(c != "_" and not c.isalnum() for c in cname):
                self._bad_warp_config(
                    i, warp_config, f"{w.name!r} needs to be a valid C identifier"
                )
            if cname:
                cname = "_" + cname

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


class DeviceScopeAnalysis(LoopIR_Rewrite):
    __slots__ = [
        "device",
        "default_instr_tl",
        "devices_seen",
        "mem_env",
        "contains_sync",
    ]

    def __init__(self):
        self.device = cpu_basic_device  # Currently inspected scope's instr-tl
        self.default_instr_tl = cpu_basic_device.get_default_instr_tl()
        self.devices_seen = {cpu_basic_device}
        self.mem_env = dict()  # Sym -> MemWin type
        self.contains_sync = False

    def map_s(self, s):
        old_device = self.device
        old_default_instr_tl = self.default_instr_tl

        if is_if_holding_with(s, LoopIR):
            ctx = s.cond.val
            if isinstance(ctx, BaseAsyncConfig):
                needed = ctx.get_parent_device()
                if needed != self.device:
                    raise ValueError(
                        f"{s.srcinfo}: {ctx!r} requires {needed}; "
                        f"device in scope is actually {self.device}"
                    )
                device = ctx.get_child_device()
                self.device = device
                self.devices_seen.add(device)
                self.default_instr_tl = device.get_default_instr_tl()
        else:
            self.inspect_s(s)

        super().map_s(s)
        self.device = old_device
        self.default_instr_tl = old_default_instr_tl

    def map_e(self, e):
        self.inspect_e(e)
        super().map_e(e)

    def inspect_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if not s.type.is_numeric():
                return

            mem = self.mem_env[s.name]
            perm = mem.device_permission(self.device, self.default_instr_tl)
            if "w" in perm:
                assert "r" in perm, "Not supported: write without read permission"
            else:
                self.warn_weird_letters(mem, perm)
                action = "mutable access" if "r" in perm else "any access"
                raise TypeError(
                    f"{s.srcinfo}: {s.name} @ "
                    f"{mem.name()} does not allow {action} in a "
                    f"scope using {self.device}"
                )
        elif isinstance(s, LoopIR.SyncStmt):
            self.contains_sync = True
            if s.sync_type.is_split():
                for e in s.barriers:
                    mem = self.mem_env[e.name]
                    perm = mem.device_permission(self.device, self.default_instr_tl)
                    if "w" in perm:
                        assert (
                            "r" in perm
                        ), "Not supported: write without read permission"
                    else:
                        self.warn_weird_letters(mem, perm)
                        raise TypeError(
                            f"{s.srcinfo}: {e.name} (barrier type "
                            f"{mem.name()}) does not allow SyncStmt in a "
                            f"scope using {self.device}"
                        )
        elif isinstance(s, LoopIR.Alloc):
            self.contains_sync |= s.type.is_barrier()
            mem = s.mem or DRAM
            self.mem_env[s.name] = mem
            assert issubclass(mem, AllocableMemWin)
            perm = mem.device_permission(self.device, self.default_instr_tl)
            if "c" not in perm:
                self.warn_weird_letters(mem, perm)
                raise TypeError(
                    f"{s.srcinfo}: {s.name} @ "
                    f"{mem.name()} cannot be allocated in a scope "
                    f"using {self.device}"
                )
        elif isinstance(s, LoopIR.WindowStmt):
            special_window = s.special_window
            self.mem_env[s.name] = special_window or self.mem_env[s.rhs.name]

            if not special_window:
                return

            assert issubclass(special_window, SpecialWindow)
            perm = special_window.device_permission(self.device, self.default_instr_tl)
            if "c" not in perm:
                self.warn_weird_letters(special_window, perm)
                raise TypeError(
                    f"{s.srcinfo}: a special window {s.name} "
                    f"@ {special_window.name()} cannot be "
                    f"constructed in a scope using {self.device}"
                )
        elif isinstance(s, LoopIR.Call):
            callee = s.f
            instr_tl = callee.proc_instr_tl()
            if not self.device.allows_instr_tl(instr_tl):
                assert isinstance(instr_tl, Instr_tl)
                raise TypeError(
                    f"{s.srcinfo}: {callee.name}() has instr-tl {instr_tl}; "
                    f"not allowed in scope using {self.device}"
                )
            assert len(s.args) == len(callee.args)
            if callee.instr:
                for caller_a, callee_a in zip(s.args, callee.args):
                    # Inspect only numeric (data) arguments, not control type arguments.
                    if not callee_a.type.is_numeric():
                        continue
                    # NB not using memory types in callee; the permissions
                    # may change due to inherintance.
                    mem = self.mem_env[caller_a.name]
                    is_const = callee.is_const_param(callee_a.name)
                    perm = mem.device_permission(self.device, instr_tl)
                    letter = "r" if is_const else "w"
                    if letter not in perm:
                        action = "mutable access" if "r" in perm else "any access"
                        raise TypeError(
                            f"{caller_a.srcinfo}: {caller_a} @ {mem.name()} "
                            f"does not allow {action} in a scope using {self.device} "
                            f"(in call to {callee.name} with instr-tl {instr_tl})"
                        )

    def inspect_e(self, e):
        if isinstance(e, LoopIR.Read) and e.type.is_numeric():
            mem = self.mem_env[e.name]
            perm = mem.device_permission(self.device, self.default_instr_tl)
            if "r" not in perm:
                assert "w" not in perm, "Not supported: write without read permission"
                raise TypeError(
                    f"{e.srcinfo}: {e.name} @ "
                    f"{mem.name()} does not allow reads in a "
                    f"scope using {self.device}"
                )

    def run(self, proc):
        for arg in proc.args:
            mem = arg.mem or DRAM
            self.mem_env[arg.name] = mem
        return super().apply_proc(proc)

    def warn_weird_letters(self, mem, perm):
        for c in perm:
            if c not in "rwc":
                warn(f"{mem.name()}.device_permission gave unknown letter {c!r}")
