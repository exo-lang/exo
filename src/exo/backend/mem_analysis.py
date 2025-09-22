from dataclasses import dataclass
from typing import List, Set, Dict, Type, Tuple
from ..core.prelude import Sym
from ..core.LoopIR import LoopIR

from ..core.instr_info import InstrInfo
from ..core.memory import (
    MemWin,
    AllocableMemWin,
    Memory,
    SpecialWindow,
    FreePoolTag,
    full_scope_free_pool_tag,
)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory Analysis Pass


dataclass(slots=True, init=False)


class MemoryAnalysis:
    mem_env: Dict[Sym, Type[MemWin]]
    window_alias: Dict[Sym, Sym]
    memo: Dict[int, Set[Sym]]

    def __init__(self):
        self.mem_env = {}
        self.window_alias = {}
        self.memo = {}

    def run(self, proc):
        assert isinstance(proc, LoopIR.proc)

        self.mem_env = {}

        for a in proc.args:
            if a.type.is_numeric():
                mem = a.mem
                assert issubclass(mem, MemWin)
                self.mem_env[a.name] = mem

        body = self.mem_stmts(proc.body)

        return LoopIR.proc(
            proc.name,
            proc.args,
            proc.preds,
            body,
            proc.instr,
            proc.srcinfo,
        )

    def mem_stmts(self, stmts: List[LoopIR.stmt]) -> List[LoopIR.stmt]:
        """Return a copy of stmts with each stmt checked & modified, and with Free inserted."""
        if len(stmts) == 0:
            return stmts

        def get_base_name(node):
            nm = node.name
            return self.window_alias.get(nm, nm)

        def used_e(e: LoopIR.expr) -> Set[Sym]:
            _id = id(e)
            try:
                return self.memo[_id]
            except KeyError:
                pass
            res = set()
            if isinstance(e, LoopIR.Read):
                res.add(get_base_name(e))
                for ei in e.idx:
                    res |= used_e(ei)
            elif isinstance(e, LoopIR.USub):
                res |= used_e(e.arg)
            elif isinstance(e, LoopIR.BinOp):
                res |= used_e(e.lhs)
                res |= used_e(e.rhs)
            elif isinstance(e, LoopIR.Extern):
                for ei in e.args:
                    res |= used_e(ei)
            elif isinstance(e, (LoopIR.WindowExpr, LoopIR.StrideExpr)):
                res.add(get_base_name(e))
            self.memo[_id] = res
            return res

        def used_s_tags(s: LoopIR.stmt) -> Tuple[Set[Sym], Set[FreePoolTag]]:
            """Set of variables used, and set of FreePoolTags used in allocs"""
            _id = id(s)
            used = set()
            tags = set()
            try:
                return self.memo[_id]
            except KeyError:
                pass
            if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
                used.add(get_base_name(s))
                used |= used_e(s.rhs)
            elif isinstance(s, LoopIR.WriteConfig):
                used |= used_e(s.rhs)
            elif isinstance(s, LoopIR.SyncStmt):
                for e in s.barriers:
                    used.add(get_base_name(e))
            elif isinstance(s, LoopIR.If):
                used |= used_e(s.cond)
                for b in s.body:
                    tup = used_s_tags(b)
                    used |= tup[0]
                    tags |= tup[1]
                for b in s.orelse:
                    tup = used_s_tags(b)
                    used |= tup[0]
                    tags |= tup[1]
            elif isinstance(s, LoopIR.For):
                for b in s.body:
                    tup = used_s_tags(b)
                    used |= tup[0]
                    tags |= tup[1]
            elif isinstance(s, LoopIR.Alloc):
                used.add(get_base_name(s))
                if (tag := s.mem.free_pool_tag()) is not None:
                    tags.add(tag)
            elif isinstance(s, LoopIR.Call):
                for e in s.args:
                    used |= used_e(e)
                if e := s.trailing_barrier_expr:
                    used.add(get_base_name(e))
            elif isinstance(s, LoopIR.WindowStmt):
                # mem_s handles setting up the alias.
                used |= used_e(s.rhs)
            self.memo[_id] = (used, tags)
            return used, tags

        # We put Free statements that should go after the original stmt[n]
        # at frees_after_nth[n].
        frees_after_nth = [()] * len(stmts)

        # Recurse child statements.
        stmts = [self.mem_s(s) for s in stmts]

        # Collect names of variables allocated at this level of the program.
        # Store their corresponding free stmt and FreePoolTag, except we
        # immediately
        alloc_dict = {}
        alloc_dict: Dict[Sym, Tuple[LoopIR.Free, Optional[FreePoolTag]]]
        for s in stmts:
            if isinstance(s, LoopIR.Alloc):
                nm = s.name
                free = LoopIR.Free(nm, s.type, s.mem, s.srcinfo.update(stmt_id=None))
                free_pool_tag = s.mem.free_pool_tag()
                if free_pool_tag == full_scope_free_pool_tag:
                    frees_after_nth[-1] += (free,)
                else:
                    alloc_dict[nm] = (free, free_pool_tag)

        # Go backwards and put free stmts at frees_after_nth[n].
        # Delete from alloc_dict as we add the free statements.
        free_pool_alloc_idx = {}
        for n in range(len(stmts) - 1, -1, -1):
            s = stmts[n]
            used, tags = used_s_tags(s)
            for nm in list(alloc_dict):  # list(...) needed to delete in iteration
                if nm in used:
                    free, free_pool_tag = alloc_dict[nm]
                    if free_pool_tag is None:
                        # Will insert immediately after last use
                        free_idx = n
                    else:
                        # Insert just prior to the last allocation stmt using
                        # the same free pool, or at end-of-stmts if no such alloc yet.
                        # (i.e. not found in the dict; [0 - 1] = [-1])
                        free_idx = free_pool_alloc_idx.get(free_pool_tag, 0) - 1
                    frees_after_nth[free_idx] += (free,)
                    del alloc_dict[nm]
            for free_pool_tag in tags:
                free_pool_alloc_idx[free_pool_tag] = n

        # Assemble stmts
        assert not alloc_dict
        new_body = []
        for n, s in enumerate(stmts):
            new_body.append(s)
            new_body.extend(frees_after_nth[n])
        return new_body

    def get_e_mem(self, e):
        if isinstance(e, (LoopIR.WindowExpr, LoopIR.Read)):
            # e.name not translated by window_alias:
            # we want SpecialWindow if applicable.
            return self.mem_env[e.name]
        else:
            assert False

    def mem_s(self, s: LoopIR.stmt):
        """Check correctness of s and return modified s."""
        styp = type(s)

        if (
            styp is LoopIR.Pass
            or styp is LoopIR.SyncStmt
            or styp is LoopIR.Assign
            or styp is LoopIR.Reduce
            or styp is LoopIR.WriteConfig
        ):
            return s

        elif styp is LoopIR.WindowStmt:
            rhs_mem = self.get_e_mem(s.rhs)
            self.check_window_expr(s.rhs, rhs_mem)
            lhs_mem = s.special_window or rhs_mem
            if lhs_mem != rhs_mem:
                src_mem = lhs_mem.source_memory_type()
                assert issubclass(src_mem, Memory)
                if not issubclass(rhs_mem, src_mem):
                    raise TypeError(
                        f"{s.srcinfo}: {lhs_mem.name()} expects {s.rhs} "
                        f"to be in memory {src_mem.name()} "
                        f"but it's actually in {rhs_mem.name()}"
                    )
            self.mem_env[s.name] = lhs_mem
            win_typ = s.rhs.type
            assert isinstance(win_typ, LoopIR.WindowType)
            src_name = win_typ.src_buf
            assert src_name not in self.window_alias
            self.window_alias[s.name] = src_name
            return s

        elif styp is LoopIR.Call:
            # check memory & window consistency at call boundaries
            for ca, sa in zip(s.args, s.f.args):
                if sa.type.is_numeric():
                    smem = sa.mem
                    assert issubclass(smem, MemWin)
                    cmem = self.get_e_mem(ca)
                    if not issubclass(cmem, smem):
                        raise TypeError(
                            f"{ca.srcinfo}: expected `{sa.name}` "
                            f"argument in {smem.name()} but got an "
                            f"argument in {cmem.name()}"
                        )
                if sa.type.is_win():
                    self.check_window_expr(ca, cmem)

            # Check trailing barrier expression
            bar: LoopIR.BarrierExpr = s.trailing_barrier_expr
            instr_info: InstrInfo = s.f.instr
            bar_type = None
            if instr_info is not None:
                bar_type = instr_info.barrier_type
            assert bar is None or isinstance(
                bar, LoopIR.BarrierExpr
            ), "typecheck should have caught this"
            if bar_type is None:
                if bar is not None:
                    raise TypeError(
                        f"{s.srcinfo}: {s.f.name} does not take trailing barrier expression >> {bar}"
                    )
            else:
                wrong = None
                if bar is None:
                    wrong = "<missing BarrierExpr>"
                elif not issubclass(actual_type := self.mem_env[bar.name], bar_type):
                    wrong = f">> {bar} @ {actual_type.name()}"
                if wrong:
                    raise TypeError(
                        f"{s.srcinfo}: {s.f.name} requires trailing barrier expression in {bar_type.name()}, not {wrong}"
                    )

            return s

        elif styp is LoopIR.If:
            body = self.mem_stmts(s.body)
            ebody = self.mem_stmts(s.orelse)
            return LoopIR.If(s.cond, body, ebody, s.srcinfo)
        elif styp is LoopIR.For:
            body = self.mem_stmts(s.body)
            return s.update(body=body)
        elif styp is LoopIR.Alloc:
            mem = s.mem
            assert issubclass(mem, AllocableMemWin)
            self.mem_env[s.name] = mem
            return s
        elif styp is LoopIR.Free:
            assert False, "There should not be frees inserted before mem " "analysis"
        else:
            assert False, f"bad case {styp}"

    def check_window_expr(self, e, mem):
        # Check intact packed dimensions
        scalar_info = e.type.basetype().scalar_info()
        n_packed_dims = len(mem.packed_tensor_shape(scalar_info))
        idxs = e.idx[-n_packed_dims:] if n_packed_dims else ()
        for idx in idxs:
            if not isinstance(idx, LoopIR.Interval):
                raise ValueError(
                    f"{e.srcinfo}: expected last {n_packed_dims} idx to be intervals to match {mem.name()}'s packed tensor shape (in {e})"
                )
