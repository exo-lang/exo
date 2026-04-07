from bisect import bisect_left, insort
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Set, Optional

from ..core.memory import Memory, memwin_template
from ..core.prelude import Sym, SrcInfo

from .cuda_memory import CudaBasicSmem, SmemConfig
from .excut import InlinePtxGen, simple_ptx_c_lines, excut_c_str_id


@dataclass(slots=True)
class SmemAllocRecord:
    offset_name: str
    size: int = 0
    alignment: int = 0
    is_freed: bool = False
    # Persistent allocations are not created and destroyed, never aliased.
    # Transient allocations are created and destroyed, may be aliased.
    persistent: bool = False

    # For mbarriers
    arrive_count: int = 0
    pre_arrive_indices: Set[int] = ()


@dataclass(slots=True)
class CudaDeviceSetupInfo:
    # Total number of SMEM bytes needed
    smem_bytes: int

    # Paste into top-level CUDA device function struct.
    static_decls: List[str]

    # Paste into exo_deviceSetup.
    setup_lines: List[str]

    offset_names: Dict[Sym, str]

    def get_offset_name(self, sym: Sym):
        return self.offset_names[sym]


@dataclass(slots=True)
class Allocator:
    # Not-so-smart allocator
    # We greedily try to fit each allocation in the first gap that it fits in.
    # TODO: write a better allocator that wisely uses "holes"
    # and doesn't have blind spots from greedy allocation,
    # e.g. alloc(A), alloc(B), free(A), alloc(C), free(B), free(C)
    # where sizeof(C) > sizeof(A)
    mem_bytes: int = 0
    mem_begin_ends: List[Tuple[int, int]] = field(default_factory=list)
    sym_begin_ends: Dict[Sym, Tuple[int, int]] = field(default_factory=dict)

    def begin_suballoc(self, sym: Sym, record: SmemAllocRecord):
        assert record.size > 0, sym
        size = record.size
        alignment = record.alignment
        align_mask = alignment - 1

        tmp_begin = 0
        for i, other in enumerate(self.mem_begin_ends):
            # See if the allocation will fit before self.mem_begin_ends[i]
            begin = (tmp_begin + align_mask) & ~align_mask
            if begin + size <= other[0]:
                alloc_index = i
                break
            # Answer is no: update state for next iteration
            tmp_begin = other[1]
        else:
            # Didn't fit in any gap, we will put at the end.
            begin = (tmp_begin + align_mask) & ~align_mask
            alloc_index = len(self.mem_begin_ends)

        self.mem_begin_ends[alloc_index:alloc_index] = [(begin, begin + size)]
        end = begin + size
        self.mem_bytes = max(self.mem_bytes, end)
        self.sym_begin_ends[sym] = (begin, end)

    def end_suballoc(self, sym: Sym):
        tup = self.sym_begin_ends[sym]
        i = bisect_left(self.mem_begin_ends, tup)
        assert self.mem_begin_ends[i] == tup
        del self.mem_begin_ends[i]


@dataclass(slots=True)
class CudaDeviceSetupBuilder:
    _records: Dict[Sym, SmemAllocRecord] = field(default_factory=dict)

    # Record order that SMEM allocations were created and destroyed.
    # (_, True) means alloc, (_, False) means free.
    _smem_alloc_frees: List[Tuple[Sym, bool]] = field(default_factory=list)

    _have_proxy_fence: bool = False

    def add_mbarriers(
        self,
        sym: Sym,
        num_per_cta: int,
        arrive_count: int,
        pre_arrive_indices: List[int],
    ) -> str:
        offset_name = self.begin_smem_alloc(sym)
        self.make_persistent(sym)
        self.set_smem_alloc_size(sym, 8 * num_per_cta, 8, 8)
        assert arrive_count > 0
        self._records[sym].arrive_count = arrive_count
        self._records[sym].pre_arrive_indices = set(pre_arrive_indices)
        return offset_name

    def begin_smem_alloc(self, sym: Sym) -> str:
        """Record the start of the lifetime of an SMEM variable.

        The returned string is the "offset_name", which is the name of the C
        variable holding the offset in bytes to suballocate this SMEM variable
        from the full SMEM allocation.

        """
        assert isinstance(sym, Sym)

        n = len(self._records)
        offset_name = f"exo_smemOffset{n}_{sym}"
        record = SmemAllocRecord(offset_name)
        self._smem_alloc_frees.append((sym, True))
        assert sym not in self._records
        self._records[sym] = record
        return offset_name

    def end_smem_alloc(self, sym: Sym) -> str:
        """Record the end of the lifetime of an SMEM variable."""
        record = self._records[sym]
        assert not record.is_freed
        record.is_freed = True
        self._smem_alloc_frees.append((sym, False))
        return record.offset_name

    def make_persistent(self, sym: Sym):
        record = self._records[sym]
        record.persistent = True

    def set_smem_alloc_size(
        self,
        sym: Sym,
        size: int,
        config_alignment: int,
        opportunistic_alignment: int = SmemConfig.opportunistic_alignment,
    ):
        record = self._records[sym]
        assert size > 0
        assert record.size == 0, "duplicate set_smem_alloc_size"
        record.size = size

        # "Opportunistic alignment"
        # Force alignment to largest power-of-2 multiple of alloc size,
        # up to 128 bytes. Also consider SmemConfig.alignment.
        alignment = config_alignment
        assert alignment > 0 and 0 == (
            alignment & (alignment - 1)
        ), "SMEM alignment must be power of 2"
        while alignment < opportunistic_alignment:
            if (alignment - 1) & size:
                break
            else:
                alignment <<= 1

        record.size = size
        record.alignment = alignment

    def require_proxy_fence(self):
        self._have_proxy_fence = True

    def make_info(self, clusterDim) -> CudaDeviceSetupInfo:
        for sym, record in self._records.items():
            assert record.is_freed or record.persistent, sym

        smem_allocator = Allocator()

        # Allocate space for persistent allocations
        for sym, is_alloc in self._smem_alloc_frees:
            record = self._records[sym]
            if is_alloc and record.persistent:
                smem_allocator.begin_suballoc(sym, record)

        # Allocate space for transient allocations
        for sym, is_alloc in self._smem_alloc_frees:
            record = self._records[sym]
            if not record.persistent:
                if is_alloc:
                    smem_allocator.begin_suballoc(sym, record)
                else:
                    smem_allocator.end_suballoc(sym)

        # Now focus on generating CUDA C++ code.
        static_decls = []
        setup_lines = []
        offset_names = {}
        thread_0_active = False

        def lazy_begin_guard_thread_0():
            nonlocal thread_0_active
            if not thread_0_active:
                setup_lines.append("if (threadIdx.x == 0) {")
            thread_0_active = True

        def lazy_end_guard_thread_0():
            nonlocal thread_0_active
            if thread_0_active:
                setup_lines.append("}")
            thread_0_active = False

        # Generate static_decls (declarations of SMEM offset constants)
        # and get list of mbarriers to initialize.
        mbarrier_inits = []
        for sym, is_alloc in self._smem_alloc_frees:
            if not is_alloc:
                continue
            begin, end = smem_allocator.sym_begin_ends[sym]
            record = self._records[sym]
            static_decls.append(
                f"static constexpr unsigned {record.offset_name} = {begin};"
                f"  // {end-begin}-byte allocation"
            )
            offset_names[sym] = record.offset_name
            if record.arrive_count:
                mbarrier_inits.append((sym, record))

        # fmt: off
        # Mbarrier init code on thread 0
        if mbarrier_inits:
            lazy_begin_guard_thread_0()
        else:
            setup_lines.append("// No mbarriers used")
        for sym, record in mbarrier_inits:
            mbarrier_count = record.size // 8
            for i in range(mbarrier_count):
                ptx = InlinePtxGen("mbarrier.init.shared::cta.b64 #0#;", volatile=True)
                ptx.add_arg(f"exo_smem + {record.offset_name} + {8*i}", constraint="smem", log_as="bits")
                ptx.add_arg(record.arrive_count, constraint="n", log_as="bits")
                setup_lines.extend(ptx.as_c_lines(py_format=False, tab="    "))
                if i in record.pre_arrive_indices:
                    ptx = InlinePtxGen("mbarrier.arrive.shared::cta.b64 _, #0#;", volatile=True)
                    ptx.add_arg(f"exo_smem + {record.offset_name} + {8*i}", constraint="smem", log_as="bits")
                    ptx.add_arg(record.arrive_count, constraint="n", log_as="bits")
                    setup_lines.extend(ptx.as_c_lines(py_format=False, tab="    "))

        # Proxy fence
        if self._have_proxy_fence:
            lazy_begin_guard_thread_0()
            setup_lines.extend(simple_ptx_c_lines("fence.proxy.async", tab="  "))

        lazy_end_guard_thread_0()

        # CTA or cluster sync
        # Skip only if no mbarriers and no cluster usage.
        assert clusterDim >= 1
        if clusterDim == 1:
            if mbarrier_inits:
                setup_lines.extend(simple_ptx_c_lines("barrier.cta.sync", 0))
        else:
            setup_lines.extend(simple_ptx_c_lines("barrier.cluster.arrive.aligned"))
            setup_lines.extend(simple_ptx_c_lines("barrier.cluster.wait.aligned"))

        # fmt: on
        return CudaDeviceSetupInfo(
            smem_allocator.mem_bytes, static_decls, setup_lines, offset_names
        )
