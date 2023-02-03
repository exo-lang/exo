import dataclasses
import re
from collections import ChainMap

from .LoopIR import (
    LoopIR,
    LoopIR_Rewrite,
    Alpha_Rename,
    LoopIR_Do,
    SubstArgs,
    T,
    lift_to_eff_expr,
)
from .LoopIR_dataflow import LoopIR_Dependencies
from .LoopIR_effects import (
    Effects as E,
    eff_filter,
    eff_bind,
    eff_null,
    get_effect_of_stmts,
)
from .effectcheck import InferEffects
from .new_eff import (
    SchedulingError,
    Check_ReorderStmts,
    Check_ReorderLoops,
    Check_FissionLoop,
    Check_DeleteConfigWrite,
    Check_ExtendEqv,
    Check_ExprEqvInContext,
    Check_BufferRW,
    Check_BufferReduceOnly,
    Check_Bounds,
    Check_IsDeadAfter,
    Check_IsIdempotent,
    Check_IsPositiveExpr,
    Check_CodeIsDead,
    Check_Aliasing,
)
from .range_analysis import AffineIndexRangeAnalysis
from .prelude import *
from .proc_eqv import get_strictest_eqv_proc
from . import internal_cursors as ic
from . import API as api


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Wrapper for LoopIR_Rewrite for scheduling directives which takes procedure cursor
# and returns Procedure object


class Cursor_Rewrite(LoopIR_Rewrite):
    def __init__(self, proc_cursor):
        self.provenance = proc_cursor.proc()
        self.orig_proc = proc_cursor
        self.proc = self.apply_proc(proc_cursor)

    def result(self, mod_config=None):
        return api.Procedure(
            self.proc, _provenance_eq_Procedure=self.provenance, _mod_config=mod_config
        )

    def map_proc(self, pc):
        p = pc._node()
        new_args = self._map_list(self.map_fnarg, p.args)
        new_preds = self.map_exprs(p.preds)
        new_body = self.map_stmts(pc.body())
        new_eff = self.map_eff(p.eff)

        if any(
            (new_args is not None, new_preds is not None, new_body is not None, new_eff)
        ):
            new_preds = new_preds or p.preds
            new_preds = [
                p for p in new_preds if not (isinstance(p, LoopIR.Const) and p.val)
            ]
            return p.update(
                args=new_args or p.args,
                preds=new_preds,
                body=new_body or p.body,
                eff=new_eff or p.eff,
            )

        return None

    def apply_stmts(self, old):
        if (new := self.map_stmts(old)) is not None:
            return new
        return [o._node() for o in old]

    def apply_s(self, old):
        if (new := self.map_s(old)) is not None:
            return new
        return [old._node()]

    def map_stmts(self, stmts):
        new_stmts = []
        needs_update = False

        for s in stmts:
            s2 = self.map_s(s)
            if s2 is None:
                new_stmts.append(s._node())
            else:
                needs_update = True
                if isinstance(s2, list):
                    new_stmts.extend(s2)
                else:
                    new_stmts.append(s2)

        if not needs_update:
            return None

        return new_stmts

    def map_s(self, sc):
        s = sc._node()
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            new_type = self.map_t(s.type)
            new_idx = self.map_exprs(s.idx)
            new_rhs = self.map_e(s.rhs)
            new_eff = self.map_eff(s.eff)
            if any((new_type, new_idx is not None, new_rhs, new_eff)):
                return [
                    s.update(
                        type=new_type or s.type,
                        idx=new_idx or s.idx,
                        rhs=new_rhs or s.rhs,
                        eff=new_eff or s.eff,
                    )
                ]
        elif isinstance(s, (LoopIR.WriteConfig, LoopIR.WindowStmt)):
            new_rhs = self.map_e(s.rhs)
            new_eff = self.map_eff(s.eff)
            if any((new_rhs, new_eff)):
                return [s.update(rhs=new_rhs or s.rhs, eff=new_eff or s.eff)]
        elif isinstance(s, LoopIR.If):
            new_cond = self.map_e(s.cond)
            new_body = self.map_stmts(sc.body())
            new_orelse = self.map_stmts(sc.orelse())
            new_eff = self.map_eff(s.eff)
            if any((new_cond, new_body is not None, new_orelse is not None, new_eff)):
                return [
                    s.update(
                        cond=new_cond or s.cond,
                        body=new_body or s.body,
                        orelse=new_orelse or s.orelse,
                        eff=new_eff or s.eff,
                    )
                ]
        elif isinstance(s, LoopIR.Seq):
            new_hi = self.map_e(s.hi)
            new_body = self.map_stmts(sc.body())
            new_eff = self.map_eff(s.eff)
            if any((new_hi, new_body is not None, new_eff)):
                return [
                    s.update(
                        hi=new_hi or s.hi, body=new_body or s.body, eff=new_eff or s.eff
                    )
                ]
        elif isinstance(s, LoopIR.Call):
            new_args = self.map_exprs(s.args)
            new_eff = self.map_eff(s.eff)
            if any((new_args is not None, new_eff)):
                return [s.update(args=new_args or s.args, eff=new_eff or s.eff)]
        elif isinstance(s, LoopIR.Alloc):
            new_type = self.map_t(s.type)
            new_eff = self.map_eff(s.eff)
            if any((new_type, new_eff)):
                return [s.update(type=new_type or s.type, eff=new_eff or s.eff)]
        elif isinstance(s, LoopIR.Pass):
            return None
        else:
            raise NotImplementedError(f"bad case {type(s)}")
        return None


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Finding Names


def name_plus_count(namestr):
    results = re.search(r"^([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$", namestr)
    if not results:
        raise TypeError(
            "expected name pattern of the form\n"
            "  ident (# integer)?\n"
            "where ident is the name of a variable "
            "and (e.g.) '#2' may optionally be attached to mean "
            "'the second occurence of that identifier"
        )

    name = results[1]
    count = int(results[3]) if results[3] else None
    return name, count


def iter_name_to_pattern(namestr):
    name, count = name_plus_count(namestr)
    if count is not None:
        count = f" #{count}"
    else:
        count = ""

    pattern = f"for {name} in _: _{count}"
    return pattern


def nested_iter_names_to_pattern(namestr, inner):
    name, count = name_plus_count(namestr)
    if count is not None:
        count = f" #{count}"
    else:
        count = ""
    assert is_valid_name(inner)

    pattern = f"for {name} in _:\n  for {inner} in _: _{count}"
    return pattern


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reorder scheduling directive


def _fixup_effects(orig_proc, p, fwd):
    p = api.Procedure(
        InferEffects(p._loopir_proc).result(), _provenance_eq_Procedure=orig_proc
    )
    fwd = ic.forward_identity(p, fwd)
    return p, fwd


# Take a conservative approach and allow stmt reordering only when they are
# writing to different buffers
# TODO: Do effectcheck's check_commutes-ish thing using SMT here
def DoReorderStmt(f_cursor, s_cursor):
    if f_cursor.next() != s_cursor:
        raise SchedulingError(
            "expected the second statement to be directly after the first"
        )
    orig_proc = f_cursor.proc()
    Check_ReorderStmts(orig_proc._loopir_proc, f_cursor._node(), s_cursor._node())
    p, fwd = s_cursor.as_block()._move(f_cursor.before())
    return _fixup_effects(orig_proc, p, fwd)


class DoPartitionLoop(LoopIR_Rewrite):
    def __init__(self, proc, loop_cursor, num):
        assert num > 0
        self.stmt = loop_cursor._node()
        self.partition_by = num
        self.proc = proc

    def map_s(self, s):
        if s is self.stmt:
            assert isinstance(s, LoopIR.Seq)

            part_by = LoopIR.Const(self.partition_by, T.int, s.srcinfo)
            new_hi = LoopIR.BinOp("-", s.hi, part_by, T.int, s.srcinfo)
            try:
                Check_IsPositiveExpr(
                    self.proc,
                    [s],
                    LoopIR.BinOp(
                        "+", new_hi, LoopIR.Const(1, T.int, s.srcinfo), T.int, s.srcinfo
                    ),
                )
            except SchedulingError:
                raise SchedulingError(
                    f"expected the new loop bound {new_hi} to be always non-negative"
                )

            loop1 = s.update(hi=part_by, eff=None)

            # all uses of the loop iteration in the second body need
            # to be offset by the partition value
            iter2 = s.iter.copy()
            iter2_node = LoopIR.Read(iter2, [], T.index, s.srcinfo)
            iter_off = LoopIR.BinOp("+", iter2_node, part_by, T.index, s.srcinfo)
            env = {s.iter: iter_off}

            body2 = SubstArgs(s.body, env).result()
            loop2 = s.update(iter=iter2, hi=new_hi, body=body2, eff=None)

            return [loop1, loop2]

        return super().map_s(s)


class DoProductLoop(Cursor_Rewrite):
    def __init__(self, proc_cursor, loop_cursor, new_name):
        self.stmt = loop_cursor._node()
        self.out_loop = self.stmt
        self.in_loop = self.out_loop.body[0]

        if len(self.out_loop.body) != 1 or not isinstance(self.in_loop, LoopIR.Seq):
            raise SchedulingError(
                f"expected loop directly inside of {self.out_loop.iter} loop"
            )

        if not isinstance(self.in_loop.hi, LoopIR.Const):
            raise SchedulingError(
                f"expected the inner loop to have a constant bound, "
                f"got {self.in_loop.hi}."
            )
        self.inside = False
        self.new_var = Sym(new_name)

        super().__init__(proc_cursor)

    def map_s(self, sc):
        s = sc._node()
        styp = type(s)
        if s is self.stmt:
            self.inside = True
            body = self.map_stmts(sc.body()[0].body())
            self.inside = False
            new_hi = LoopIR.BinOp(
                "*", self.out_loop.hi, self.in_loop.hi, T.index, s.srcinfo
            )

            return [s.update(iter=self.new_var, hi=new_hi, body=body)]

        return super().map_s(sc)

    def map_e(self, e):
        if self.inside and isinstance(e, LoopIR.Read):
            var = LoopIR.Read(self.new_var, [], T.index, e.srcinfo)
            if e.name == self.out_loop.iter:
                return LoopIR.BinOp("/", var, self.in_loop.hi, T.index, e.srcinfo)
            if e.name == self.in_loop.iter:
                return LoopIR.BinOp("%", var, self.in_loop.hi, T.index, e.srcinfo)

        return super().map_e(e)


def get_reads(e):
    if isinstance(e, LoopIR.Read):
        return sum([get_reads(e) for e in e.idx], [(e.name, e.type)])
    elif isinstance(e, LoopIR.USub):
        return get_reads(e.arg)
    elif isinstance(e, LoopIR.BinOp):
        return get_reads(e.lhs) + get_reads(e.rhs)
    elif isinstance(e, LoopIR.BuiltIn):
        return sum([get_reads(a) for a in e.args], [])
    elif isinstance(e, LoopIR.Const):
        return []
    else:
        assert False, "bad case"


class DoMergeWrites(Cursor_Rewrite):
    def __init__(self, proc_cursor, f_cursor, s_cursor):
        self.s1 = f_cursor._node()
        self.s2 = s_cursor._node()

        try:
            assert len(self.s1.idx) == len(self.s2.idx)
            for i, j in zip(self.s1.idx, self.s2.idx):
                Check_ExprEqvInContext(proc_cursor._node(), i, [self.s1], j, [self.s2])
        except SchedulingError as e:
            raise SchedulingError(
                "expected the left hand side's indices to be the same."
            ) from e

        if any(
            self.s1.name == name and self.s1.type == typ
            for name, typ in get_reads(self.s2.rhs)
        ):
            raise SchedulingError(
                "expected the right hand side of the second statement to not "
                "depend on the left hand side of the first statement."
            )

        self.new_rhs = LoopIR.BinOp(
            "+", self.s1.rhs, self.s2.rhs, self.s1.type, self.s1.srcinfo
        )

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts_c):
        stmts = [o._node() for o in stmts_c]
        if isinstance(self.s2, LoopIR.Assign):
            for i, s in enumerate(stmts):
                if s is self.s2:
                    return stmts[: i - 1] + stmts[i:]
        else:
            for i, s in enumerate(stmts):
                if s is self.s1:
                    return stmts[:i] + [s.update(rhs=self.new_rhs)] + stmts[i + 2 :]

        return super().map_stmts(stmts_c)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Split scheduling directive


class DoSplit(Cursor_Rewrite):
    def __init__(
        self, proc_cursor, loop_cursor, quot, hi, lo, tail="guard", perfect=False
    ):
        self.split_loop = loop_cursor._node()
        self.split_var = self.split_loop.iter
        self.quot = quot
        self.hi_i = Sym(hi)
        self.lo_i = Sym(lo)
        self.cut_i = Sym(lo)

        assert quot > 1

        # Tail strategies are 'cut', 'guard', and 'cut_and_guard'
        self._tail_strategy = tail
        if perfect:
            self._tail_strategy = "perfect"
        self._in_cut_tail = False

        super().__init__(proc_cursor)

    def substitute(self, srcinfo):
        cnst = lambda x: LoopIR.Const(x, T.int, srcinfo)
        rd = lambda x: LoopIR.Read(x, [], T.index, srcinfo)
        op = lambda op, lhs, rhs: LoopIR.BinOp(op, lhs, rhs, T.index, srcinfo)

        return op("+", op("*", cnst(self.quot), rd(self.hi_i)), rd(self.lo_i))

    def cut_tail_sub(self, srcinfo):
        return self._cut_tail_sub

    def map_s(self, sc):
        s = sc._node()
        styp = type(s)
        if s is self.split_loop:
            # short-hands for sanity
            def boolop(op, lhs, rhs):
                return LoopIR.BinOp(op, lhs, rhs, T.bool, s.srcinfo)

            def szop(op, lhs, rhs):
                return LoopIR.BinOp(op, lhs, rhs, lhs.type, s.srcinfo)

            def cnst(intval):
                return LoopIR.Const(intval, T.int, s.srcinfo)

            def rd(i):
                return LoopIR.Read(i, [], T.index, s.srcinfo)

            def ceildiv(lhs, rhs):
                assert isinstance(rhs, LoopIR.Const) and rhs.val > 1
                rhs_1 = cnst(rhs.val - 1)
                return szop("/", szop("+", lhs, rhs_1), rhs)

            def rng(x, hi):
                lhs = boolop("<=", cnst(0), x)
                rhs = boolop("<", x, hi)
                return boolop("and", lhs, rhs)

            def do_bind(x, hi, eff):
                cond = lift_to_eff_expr(rng(rd(x), hi))
                cond_nz = lift_to_eff_expr(boolop("<", cnst(0), hi))
                return eff_bind(x, eff, pred=cond, config_pred=cond_nz)

            # in the simple case, wrap body in a guard
            if self._tail_strategy == "guard":
                body = self.map_stmts(sc.body())
                body_eff = get_effect_of_stmts(body)
                idx_sub = self.substitute(s.srcinfo)
                cond = boolop("<", idx_sub, s.hi)
                # condition for guarded loop is applied to effects
                body_eff = eff_filter(lift_to_eff_expr(cond), body_eff)
                body = [LoopIR.If(cond, body, [], body_eff, s.srcinfo)]

                lo_rng = cnst(self.quot)
                hi_rng = ceildiv(s.hi, lo_rng)

                # pred for inner loop is: 0 <= lo <= lo_rng
                inner_eff = do_bind(self.lo_i, lo_rng, body_eff)

                return [
                    styp(
                        self.hi_i,
                        hi_rng,
                        [styp(self.lo_i, lo_rng, body, inner_eff, s.srcinfo)],
                        s.eff,
                        s.srcinfo,
                    )
                ]

            # an alternate scheme is to split the loop in two
            # by cutting off the tail into a second loop
            elif self._tail_strategy == "cut" or self._tail_strategy == "cut_and_guard":
                # if N == s.hi and Q == self.quot, then
                #   we want Ncut == (N-Q+1)/Q
                Q = cnst(self.quot)
                N = s.hi
                Ncut = szop("/", N, Q)  # floor div

                # and then for the tail loop, we want to
                # iterate from 0 to Ntail
                # where Ntail == N % Q
                Ntail = szop("%", N, Q)
                # in that loop we want the iteration variable to
                # be mapped instead to (Ncut*Q + cut_i)
                self._cut_tail_sub = szop("+", rd(self.cut_i), szop("*", Ncut, Q))

                main_body = self.map_stmts(sc.body())
                self._in_cut_tail = True
                tail_body = Alpha_Rename(self.map_stmts(sc.body())).result()
                self._in_cut_tail = False

                main_eff = get_effect_of_stmts(main_body)
                tail_eff = get_effect_of_stmts(tail_body)
                lo_eff = do_bind(self.lo_i, Q, main_eff)
                hi_eff = do_bind(self.hi_i, Ncut, lo_eff)
                tail_eff = do_bind(self.cut_i, Ntail, tail_eff)

                if self._tail_strategy == "cut_and_guard":
                    body = [styp(self.cut_i, Ntail, tail_body, tail_eff, s.srcinfo)]
                    body_eff = get_effect_of_stmts(body)
                    cond = boolop(">", Ntail, LoopIR.Const(0, T.int, s.srcinfo))
                    body_eff = eff_filter(lift_to_eff_expr(cond), body_eff)

                    loops = [
                        styp(
                            self.hi_i,
                            Ncut,
                            [styp(self.lo_i, Q, main_body, lo_eff, s.srcinfo)],
                            hi_eff,
                            s.srcinfo,
                        ),
                        LoopIR.If(cond, body, [], body_eff, s.srcinfo),
                    ]

                else:
                    loops = [
                        styp(
                            self.hi_i,
                            Ncut,
                            [styp(self.lo_i, Q, main_body, lo_eff, s.srcinfo)],
                            hi_eff,
                            s.srcinfo,
                        ),
                        styp(self.cut_i, Ntail, tail_body, tail_eff, s.srcinfo),
                    ]

                return loops

            elif self._tail_strategy == "perfect":
                if not isinstance(s.hi, LoopIR.Const):
                    raise SchedulingError(
                        f"cannot perfectly split the '{s.iter}' loop "
                        f"unless it has a constant bound"
                    )
                elif s.hi.val % self.quot != 0:
                    raise SchedulingError(
                        f"cannot perfectly split the '{s.iter}' loop "
                        f"because {self.quot} does not evenly divide "
                        f"{s.hi.val}"
                    )

                # otherwise, we're good to go
                body = self.map_stmts(sc.body())
                body_eff = get_effect_of_stmts(body)

                lo_rng = cnst(self.quot)
                hi_rng = cnst(s.hi.val // self.quot)

                # pred for inner loop is: 0 <= lo <= lo_rng
                inner_eff = do_bind(self.lo_i, lo_rng, body_eff)

                return [
                    styp(
                        self.hi_i,
                        hi_rng,
                        [styp(self.lo_i, lo_rng, body, inner_eff, s.srcinfo)],
                        s.eff,
                        s.srcinfo,
                    )
                ]

            else:
                assert False, f"bad tail strategy: {self._tail_strategy}"

        # fall-through
        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            if e.type is T.index:
                # This is a split variable, substitute it!
                if e.name is self.split_var:
                    if self._in_cut_tail:
                        return self.cut_tail_sub(e.srcinfo)
                    else:
                        return self.substitute(e.srcinfo)

        # fall-through
        return super().map_e(e)

    def map_eff_e(self, e):
        if isinstance(e, E.Var):
            if e.type is T.index:
                # This is a split variable, substitute it!
                if e.name is self.split_var:
                    if self._in_cut_tail:
                        sub = self.cut_tail_sub(e.srcinfo)
                    else:
                        sub = self.substitute(e.srcinfo)
                    return lift_to_eff_expr(sub)

        # fall-through
        return super().map_eff_e(e)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unroll scheduling directive


class DoUnroll(Cursor_Rewrite):
    def __init__(self, proc_cursor, loop_cursor):
        self.unroll_loop = loop_cursor._node()
        self.unroll_var = self.unroll_loop.iter
        self.unroll_itr = 0
        self.env = {}

        super().__init__(proc_cursor)

    def map_s(self, sc):
        s = sc._node()
        if s is self.unroll_loop:
            if not isinstance(s.hi, LoopIR.Const):
                raise SchedulingError(
                    f"expected loop '{s.iter}' to have constant bounds"
                )

            hi = s.hi.val
            if hi == 0:
                return []

            orig_body = sc.body()

            self.unroll_itr = 0

            body = Alpha_Rename(self.apply_stmts(orig_body)).result()
            for i in range(1, hi):
                self.unroll_itr = i
                body += Alpha_Rename(self.apply_stmts(orig_body)).result()

            return body

        # fall-through
        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            if e.type is T.index:
                # This is an unrolled variable, substitute it!
                if e.name is self.unroll_var:
                    return LoopIR.Const(self.unroll_itr, T.index, e.srcinfo)

        # fall-through
        return super().map_e(e)

    def map_eff_e(self, e):
        if isinstance(e, E.Var):
            if e.type is T.index:
                # This is an unrolled variable, substitute it!
                if e.name is self.unroll_var:
                    return E.Const(self.unroll_itr, T.index, e.srcinfo)

        # fall-through
        return super().map_eff_e(e)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Inline scheduling directive


class DoInline(Cursor_Rewrite):
    def __init__(self, proc_cursor, call_cursor):
        self.call_stmt = call_cursor._node()
        assert isinstance(self.call_stmt, LoopIR.Call)
        self.env = {}

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.call_stmt:
            # handle potential window expressions in call positions
            win_binds = []

            def map_bind(nm, a):
                if isinstance(a, LoopIR.WindowExpr):
                    stmt = LoopIR.WindowStmt(nm, a, eff_null(a.srcinfo), a.srcinfo)
                    win_binds.append(stmt)
                    return LoopIR.Read(nm, [], a.type, a.srcinfo)
                else:
                    return a

            # first, set-up a binding from sub-proc arguments
            # to supplied expressions at the call-site
            call_bind = {
                xd.name: map_bind(xd.name, a) for xd, a in zip(s.f.args, s.args)
            }

            # we will substitute the bindings for the call
            body = SubstArgs(s.f.body, call_bind).result()

            # note that all sub-procedure assertions must be true
            # even if not asserted, or else this call being inlined
            # wouldn't have been valid to make in the first place

            # whenever we copy code we need to alpha-rename for safety
            # the code to splice in at this point
            return Alpha_Rename(win_binds + body).result()

        # fall-through
        return super().map_s(sc)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self, e):
        return e

    def map_t(self, t):
        return t

    def map_eff(self, eff):
        return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Partial Evaluation scheduling directive


class DoPartialEval(LoopIR_Rewrite):
    def __init__(self, env):
        assert env, "Don't call _PartialEval without any substitutions"
        self.env = env

    def map_proc(self, p):
        # Validate env:
        arg_types = {x.name: x.type for x in p.args}
        for k, v in self.env.items():
            if not arg_types[k].is_indexable() and not arg_types[k].is_bool():
                raise SchedulingError(
                    "cannot partially evaluate numeric (non-index, non-bool) arguments"
                )
            if not isinstance(v, int):
                raise SchedulingError(
                    "cannot partially evaluate to a non-int, non-bool value"
                )

        p = super().map_proc(p) or p

        return p.update(args=[a for a in p.args if a.name not in self.env])

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            if e.type.is_indexable():
                assert len(e.idx) == 0
                if e.name in self.env:
                    return LoopIR.Const(self.env[e.name], T.int, e.srcinfo)
            elif e.type.is_bool():
                if e.name in self.env:
                    return LoopIR.Const(self.env[e.name], T.bool, e.srcinfo)

        return super().map_e(e)

    def map_eff_e(self, e):
        if isinstance(e, E.Var):
            if e.type.is_indexable() and e.name in self.env:
                return E.Const(self.env[e.name], T.int, e.srcinfo)
            elif e.type.is_bool() and e.name in self.env:
                return E.Const(self.env[e.name], T.bool, e.srcinfo)

        return super().map_eff_e(e)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Set Type and/or Memory Annotations scheduling directive


# This pass uses a raw name string instead of a pattern
# TODO: This op shouldn't take name, it should just take Alloc cursor...
class DoSetTypAndMem(Cursor_Rewrite):
    def __init__(self, proc_cursor, name, inst_no, basetyp=None, win=None, mem=None):
        ind = lambda x: 1 if x else 0
        assert ind(basetyp) + ind(win) + ind(mem) == 1
        self.name = name
        self.n_match = inst_no
        self.basetyp = basetyp
        self.win = win
        self.mem = mem

        super().__init__(proc_cursor)

    def check_inst(self):
        # otherwise, handle instance counting...
        if self.n_match is None:
            return True
        else:
            self.n_match = self.n_match - 1
            return self.n_match == 0

    def early_exit(self):
        return self.n_match is not None and self.n_match <= 0

    def change_precision(self, t):
        assert self.basetyp.is_real_scalar()
        if t.is_real_scalar():
            return self.basetyp
        elif isinstance(t, T.Tensor):
            assert t.type.is_real_scalar()
            return T.Tensor(t.hi, t.is_window, self.basetyp)
        else:
            assert False, "bad case"

    def change_window(self, t):
        assert isinstance(t, T.Tensor)
        assert isinstance(self.win, bool)
        return T.Tensor(t.hi, self.win, t.type)

    def map_fnarg(self, a):
        if str(a.name) != self.name:
            return a

        # otherwise, handle instance counting...
        if not self.check_inst():
            return a

        # if that passed, we definitely found the symbol being pointed at
        # So attempt the substitution
        typ = a.type
        mem = a.mem
        if self.basetyp is not None:
            if not a.type.is_numeric():
                raise SchedulingError(
                    "cannot change the precision of a " "non-numeric argument"
                )
            typ = self.change_precision(typ)
        elif self.win is not None:
            if not a.type.is_tensor_or_window():
                raise SchedulingError(
                    "cannot change windowing of a " "non-tensor/window argument"
                )
            typ = self.change_window(typ)
        else:
            assert self.mem is not None
            if not a.type.is_numeric():
                raise SchedulingError(
                    "cannot change the memory of a " "non-numeric argument"
                )
            mem = self.mem

        return LoopIR.fnarg(a.name, typ, mem, a.srcinfo)

    def map_s(self, sc):
        s = sc._node()
        if self.early_exit():
            return [s]

        if isinstance(s, LoopIR.Alloc) and str(s.name) == self.name:
            if self.check_inst():

                # if that passed, we definitely found the symbol being pointed at
                # So attempt the substitution
                typ = s.type
                assert typ.is_numeric()
                mem = s.mem
                if self.basetyp is not None:
                    typ = self.change_precision(typ)
                elif self.win is not None:
                    raise SchedulingError(
                        "cannot change an allocation to " "be or not be a window"
                    )
                else:
                    assert self.mem is not None
                    mem = self.mem

                return [LoopIR.Alloc(s.name, typ, mem, s.eff, s.srcinfo)]

        # fall-through
        return super().map_s(sc)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self, e):
        return e

    def map_t(self, t):
        return t

    def map_eff(self, eff):
        return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Call Swap scheduling directive


class DoCallSwap(Cursor_Rewrite):
    def __init__(self, proc_cursor, call_cursor, new_subproc):
        self.call_stmt = call_cursor._node()
        assert isinstance(self.call_stmt, LoopIR.Call)
        self.new_subproc = new_subproc

        super().__init__(proc_cursor)
        Check_Aliasing(self.proc)

    def mod_eq(self):
        return self.eq_mod_config

    def map_s(self, sc):
        s = sc._node()
        if s is self.call_stmt:
            old_f = s.f
            new_f = self.new_subproc
            s_new = LoopIR.Call(new_f, s.args, None, s.srcinfo)
            is_eqv, configkeys = get_strictest_eqv_proc(old_f, new_f)
            if not is_eqv:
                raise SchedulingError(
                    f"{s.srcinfo}: Cannot swap call because the two "
                    f"procedures are not equivalent"
                )
            mod_cfg = Check_ExtendEqv(self.orig_proc._node(), [s], [s_new], configkeys)
            self.eq_mod_config = mod_cfg

            return [s_new]

        # fall-through
        return super().map_s(sc)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self, e):
        return e

    def map_t(self, t):
        return t

    def map_eff(self, eff):
        return eff


class DoInlineWindow(Cursor_Rewrite):
    def __init__(self, proc_cursor, window_cursor):
        self.win_stmt = window_cursor._node()
        assert isinstance(self.win_stmt, LoopIR.WindowStmt)

        super().__init__(proc_cursor)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def calc_idx(self, idxs):
        assert len(
            [w for w in self.win_stmt.rhs.idx if isinstance(w, LoopIR.Interval)]
        ) == len(idxs)

        new_idxs = []
        win_idx = self.win_stmt.rhs.idx
        idxs = idxs.copy()  # make function non-destructive to input
        assert len(idxs) == sum([isinstance(w, LoopIR.Interval) for w in win_idx])

        def add(x, y):
            return LoopIR.BinOp("+", x, y, T.index, x.srcinfo)

        if len(idxs) > 0 and isinstance(idxs[0], LoopIR.w_access):

            def map_w(w):
                if isinstance(w, LoopIR.Point):
                    return w
                # i is from the windowing expression we're substituting into
                i = idxs.pop(0)
                if isinstance(i, LoopIR.Point):
                    return LoopIR.Point(add(i.pt, w.lo), i.srcinfo)
                else:
                    return LoopIR.Interval(add(i.lo, w.lo), add(i.hi, w.lo), i.srcinfo)

        else:

            def map_w(w):
                return w.pt if isinstance(w, LoopIR.Point) else add(idxs.pop(0), w.lo)

        return [map_w(w) for w in win_idx]

    # used to offset the stride in order to account for
    # dimensions hidden due to window-point accesses
    def calc_dim(self, dim):
        assert dim < len(
            [w for w in self.win_stmt.rhs.idx if isinstance(w, LoopIR.Interval)]
        )

        # Because our goal here is to offset `dim` in the original
        # call argument to the point indexing to the windowing expression,
        # new_dim should essencially be:
        # `dim` + "number of LoopIR.Points in the windowing expression before the `dim` number of LoopIR.Interval"
        new_dim = 0
        for w in self.win_stmt.rhs.idx:
            if isinstance(w, LoopIR.Interval):
                dim -= 1
            if dim == -1:
                return new_dim
            new_dim += 1

    def map_s(self, sc):
        s = sc._node()
        # remove the windowing statement
        if s is self.win_stmt:
            return []

        # substitute the indexing at assignment and reduction statements
        if (
            isinstance(s, (LoopIR.Assign, LoopIR.Reduce))
            and self.win_stmt.lhs == s.name
        ):
            idxs = self.calc_idx(s.idx)
            return [
                type(s)(
                    self.win_stmt.rhs.name, s.type, s.cast, idxs, s.rhs, None, s.srcinfo
                )
            ]

        return super().map_s(sc)

    def map_e(self, e):
        # etyp    = type(e)
        win_name = self.win_stmt.lhs
        buf_name = self.win_stmt.rhs.name
        win_idx = self.win_stmt.rhs.idx

        if isinstance(e, LoopIR.WindowExpr) and win_name == e.name:
            new_idxs = self.calc_idx(e.idx)

            # repair window type..
            old_typ = self.win_stmt.rhs.type
            new_type = LoopIR.WindowType(
                old_typ.src_type, old_typ.as_tensor, buf_name, new_idxs
            )

            return LoopIR.WindowExpr(
                self.win_stmt.rhs.name, new_idxs, new_type, e.srcinfo
            )

        elif isinstance(e, LoopIR.Read) and win_name == e.name:
            new_idxs = self.calc_idx(e.idx)
            return LoopIR.Read(buf_name, new_idxs, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr) and win_name == e.name:
            dim = self.calc_dim(e.dim)
            return LoopIR.StrideExpr(buf_name, dim, e.type, e.srcinfo)

        return super().map_e(e)


# TODO: Rewrite this to directly use stmt_cursor instead of after
class DoConfigWrite(Cursor_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, config, field, expr, before=False):
        assert isinstance(expr, (LoopIR.Read, LoopIR.StrideExpr, LoopIR.Const))

        self.stmt = stmt_cursor._node()
        self.config = config
        self.field = field
        self.expr = expr
        self.before = before

        self._new_cfgwrite_stmt = None

        super().__init__(proc_cursor)

        # check safety...
        mod_cfg = Check_DeleteConfigWrite(self.proc, [self._new_cfgwrite_stmt])
        self.eq_mod_config = mod_cfg

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def mod_eq(self):
        return self.eq_mod_config

    def map_stmts(self, stmts_c):
        body = []
        for i, sc in enumerate(stmts_c):
            s = sc._node()
            if s is self.stmt:
                cw_s = LoopIR.WriteConfig(
                    self.config, self.field, self.expr, None, s.srcinfo
                )
                self._new_cfgwrite_stmt = cw_s

                if self.before:
                    body += [cw_s, s]
                else:
                    body += [s, cw_s]

                # finish and exit
                body += [s._node() for s in stmts_c[i + 1 :]]
                return body

            else:
                # TODO: be smarter about None handling
                body += self.apply_s(sc)

        return body


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Bind Expression scheduling directive


class _BindConfig_AnalysisSubst(LoopIR_Rewrite):
    def __init__(self, keep_s, old_e, new_e):
        self.keep_s = keep_s
        self.old_e = old_e
        self.new_e = new_e

    def map_s(self, s):
        if s is self.keep_s:
            return [s]
        else:
            return super().map_s(s)

    def map_e(self, e):
        if e is self.old_e:
            return self.new_e
        else:
            return super().map_e(e)


class DoBindConfig(Cursor_Rewrite):
    def __init__(self, proc_cursor, config, field, expr_cursor):
        self.expr = expr_cursor._node()
        assert isinstance(self.expr, LoopIR.Read)

        self.config = config
        self.field = field
        self.found_expr = False
        self.placed_writeconfig = False
        self.sub_done = False
        self.cfg_write_s = None
        self.cfg_read_e = None

        super().__init__(proc_cursor)

        proc_analysis = _BindConfig_AnalysisSubst(
            self.cfg_write_s, self.cfg_read_e, self.expr
        ).apply_proc(self.proc)
        mod_cfg = Check_DeleteConfigWrite(proc_analysis, [self.cfg_write_s])
        self.eq_mod_config = mod_cfg

        # repair effects...
        self.proc = InferEffects(self.proc).result()
        Check_Aliasing(self.proc)

    def mod_eq(self):
        return self.eq_mod_config

    def process_block(self, block_c):
        if self.sub_done:
            return None

        new_block = []
        is_writeconfig_block = False

        modified = False

        for stmt_c in block_c:
            new_stmt = self.map_s(stmt_c)

            if self.found_expr and not self.placed_writeconfig:
                self.placed_writeconfig = True
                is_writeconfig_block = True
                wc = LoopIR.WriteConfig(
                    self.config, self.field, self.expr, None, self.expr.srcinfo
                )
                self.cfg_write_s = wc
                new_block.extend([wc])

            if new_stmt is None:
                new_block.append(stmt_c._node())
            else:
                new_block.extend(new_stmt)
                modified = True

        if is_writeconfig_block:
            self.sub_done = True

        if not modified:
            return None

        return new_block

    def map_s(self, sc):
        s = sc._node()
        if self.sub_done:
            return None  # TODO: is this right?

        # TODO: missing cases for multiple config writes. Subsequent writes are
        #   ignored.

        if isinstance(s, LoopIR.Seq):
            body = self.process_block(sc.body())
            if body:
                return [s.update(body=body)]
            return None

        if isinstance(s, LoopIR.If):
            if_then = self.process_block(sc.body())
            if_else = self.process_block(sc.orelse())
            cond = self.map_e(s.cond)
            if any((if_then, if_else, cond)):
                return [
                    s.update(
                        cond=cond or s.cond,
                        body=if_then or s.body,
                        orelse=if_else or s.orelse,
                    )
                ]

            return None

        return super().map_s(sc)

    def map_e(self, e):
        if e is self.expr and not self.sub_done:
            assert not self.found_expr
            self.found_expr = True

            self.cfg_read_e = LoopIR.ReadConfig(
                self.config, self.field, e.type, e.srcinfo
            )
            return self.cfg_read_e
        else:
            return super().map_e(e)


class DoCommuteExpr(Cursor_Rewrite):
    def __init__(self, proc_cursor, expr_cursors):
        self.exprs = [e._node() for e in expr_cursors]
        super().__init__(proc_cursor)
        self.proc = InferEffects(self.proc).result()

    def map_e(self, e):
        if e in self.exprs:
            assert isinstance(e, LoopIR.BinOp)
            return e.update(lhs=e.rhs, rhs=e.lhs)
        else:
            return super().map_e(e)


class DoBindExpr(Cursor_Rewrite):
    def __init__(self, proc_cursor, new_name, expr_cursors, cse=False):
        self.exprs = [e._node() for e in expr_cursors]
        assert all(isinstance(expr, LoopIR.expr) for expr in self.exprs)
        assert all(expr.type.is_numeric() for expr in self.exprs)
        assert self.exprs
        self.exprs = self.exprs if cse else [self.exprs[0]]

        self.new_name = Sym(new_name)
        self.expr_reads = set(sum([get_reads(e) for e in self.exprs], []))
        self.use_cse = cse
        self.found_expr = None
        self.placed_alloc = False
        self.sub_done = False
        self.found_write = False

        super().__init__(proc_cursor)

        # repair effects...
        self.proc = InferEffects(self.proc).result()
        Check_Aliasing(self.proc)

    def process_block(self, block):
        if self.sub_done:
            return block

        new_block = []
        is_alloc_block = False

        is_updated = False

        for _stmt in block:
            stmt = self.map_s(_stmt)
            if stmt is not None:
                is_updated = True
            else:
                stmt = [_stmt._node()]

            if self.found_expr and not self.placed_alloc:
                self.placed_alloc = True
                is_alloc_block = True
                alloc = LoopIR.Alloc(
                    self.new_name, T.R, None, None, self.found_expr.srcinfo
                )
                # TODO Fix Assign, probably wrong
                assign = LoopIR.Assign(
                    self.new_name,
                    T.R,
                    None,
                    [],
                    self.found_expr,
                    None,
                    self.found_expr.srcinfo,
                )
                new_block.extend([alloc, assign])

            new_block.extend(stmt)

        # If this is the block containing the new alloc, stop substituting
        if is_alloc_block:
            self.sub_done = True

        if is_updated or is_alloc_block:
            return new_block

        return None

    def map_s(self, sc):
        s = sc._node()
        if self.found_write:
            return None

        if self.sub_done:
            return super().map_s(sc)

        if isinstance(s, LoopIR.Seq):
            body = self.process_block(sc.body())
            if body is None:
                return None
            else:
                return [s.update(body=body)]

        if isinstance(s, LoopIR.If):
            # TODO: our CSE here is very conservative. It won't look for
            #  matches between the then and else branches; in other words,
            #  it is restricted to a single basic block.
            if_then = self.process_block(sc.body())
            if_else = self.process_block(sc.orelse())
            if (if_then is not None) or (if_else is not None):
                return [s.update(body=if_then or s.body, orelse=if_else or s.orelse)]
            else:
                return None

        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            e = self.exprs[0]
            new_rhs = self.map_e(s.rhs)

            # terminate CSE if the expression is written to
            if self.found_expr and self.use_cse:
                for (name, type) in self.expr_reads:
                    if s.name == name and s.type == type:
                        self.found_write = True

            if new_rhs is not None:
                return [s.update(rhs=new_rhs)]
            return None

        return super().map_s(sc)

    def map_e(self, e):
        if e in self.exprs and not self.sub_done:
            if not self.found_expr:
                # TODO: dirty hack. need real CSE-equality (i.e. modulo srcinfo)
                self.exprs = [x for x in self.exprs if str(e) == str(x)]
            self.found_expr = e
            return LoopIR.Read(self.new_name, [], e.type, e.srcinfo)
        else:
            return super().map_e(e)


# Lift if no variable dependency
class DoLiftScope(Cursor_Rewrite):
    def __init__(self, proc_cursor, if_cursor):
        self.target = if_cursor._node()
        self.target_type = (
            "if statement" if isinstance(self.target, LoopIR.If) else "for loop"
        )

        if if_cursor.parent()._node() is proc_cursor._node():
            raise SchedulingError("Cannot lift scope of top-level statement")

        assert isinstance(self.target, (LoopIR.If, LoopIR.Seq))

        super().__init__(proc_cursor)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if not isinstance(s, (LoopIR.If, LoopIR.Seq)):
            # Only ifs and loops can be interchanged
            return None

        s2 = super().map_s(sc)
        if s2:
            return s2

        if isinstance(s, LoopIR.If):
            if self.target in s.body:
                if len(s.body) > 1:
                    raise SchedulingError(
                        f"expected {self.target_type} to be directly nested in parent"
                    )

                if isinstance(self.target, LoopIR.If):
                    #                    if INNER:
                    # if OUTER:            if OUTER: A
                    #   if INNER: A        else:     C
                    #   else:     B  ~>  else:
                    # else: C              if OUTER: B
                    #                      else:     C
                    stmt_a = self.target.body
                    stmt_b = self.target.orelse
                    stmt_c = s.orelse

                    if_ac = [s.update(body=stmt_a, orelse=stmt_c)]
                    if stmt_b or stmt_c:
                        stmt_b = stmt_b or [LoopIR.Pass(None, self.target.srcinfo)]
                        if_bc = [s.update(body=stmt_b, orelse=stmt_c)]
                    else:
                        if_bc = []

                    new_if = self.target.update(body=if_ac, orelse=if_bc)
                    return [new_if]

                if isinstance(self.target, LoopIR.Seq):
                    # if OUTER:                for INNER in _:
                    #   for INNER in _: A  ~>    if OUTER: A
                    if len(s.orelse) > 0:
                        raise SchedulingError(
                            "cannot lift for loop when if has an orelse clause"
                        )

                    new_if = s.update(body=self.target.body, orelse=[])
                    new_for = self.target.update(body=[new_if])
                    return [new_for]

            if self.target in s.orelse and isinstance(self.target, LoopIR.If):
                if len(s.orelse) > 1:
                    raise SchedulingError(
                        f"expected {self.target_type} to be directly nested in parents"
                    )

                #                    if INNER:
                # if OUTER: A          if OUTER: A
                # else:                else:     B
                #   if INNER: B  ~>  else:
                #   else: C            if OUTER: A
                #                      else:     C
                stmt_a = s.body
                stmt_b = self.target.body
                stmt_c = self.target.orelse

                if_ab = [s.update(body=stmt_a, orelse=stmt_b)]
                if_ac = [s.update(body=stmt_a, orelse=stmt_c)]

                new_if = self.target.update(body=if_ab, orelse=if_ac)
                return [new_if]
        if isinstance(s, LoopIR.Seq):
            if self.target in s.body:
                if len(s.body) > 1:
                    raise SchedulingError(
                        "expected if statement to be directly nested in parents"
                    )

                if isinstance(s.body[0], LoopIR.If):
                    # for OUTER in _:      if INNER:
                    #   if INNER: A    ~>    for OUTER in _: A
                    #   else:     B        else:
                    #                        for OUTER in _: B
                    if s.iter in _FV(self.target.cond):
                        raise SchedulingError(
                            "if statement depends on iteration variable"
                        )

                    stmt_a = self.target.body
                    stmt_b = self.target.orelse

                    for_a = [s.update(body=stmt_a)]
                    for_b = [s.update(body=stmt_b)] if stmt_b else []

                    new_if = self.target.update(body=for_a, orelse=for_b)
                    return [new_if]
                if isinstance(s.body[0], LoopIR.Seq):
                    # for OUTER in _:          for INNER in _:
                    #   for INNER in _: A  ~>    for OUTER in _: A
                    Check_ReorderLoops(self.orig_proc._node(), s)

                    # TODO: This is a copy paste from old _Reorder class.
                    # Deprecate this when we deprecate effects.
                    # short-hands for sanity
                    def boolop(op, lhs, rhs):
                        return LoopIR.BinOp(op, lhs, rhs, T.bool, s.srcinfo)

                    def cnst(intval):
                        return LoopIR.Const(intval, T.int, s.srcinfo)

                    def rd(i):
                        return LoopIR.Read(i, [], T.index, s.srcinfo)

                    def rng(x, hi):
                        lhs = boolop("<=", cnst(0), x)
                        rhs = boolop("<", x, hi)
                        return boolop("and", lhs, rhs)

                    def do_bind(x, hi, eff):
                        cond = lift_to_eff_expr(rng(rd(x), hi))
                        cond_nz = boolop("<", cnst(0), hi)
                        return eff_bind(
                            x, eff, pred=cond
                        )  # TODO: , config_pred=cond_nz)

                    # this is the actual body inside both for-loops
                    body = s.body[0].body
                    body_eff = get_effect_of_stmts(body)
                    inner_eff = do_bind(s.iter, s.hi, body_eff)
                    outer_eff = do_bind(s.body[0].iter, s.body[0].hi, inner_eff)
                    return [
                        s.body[0].update(
                            body=[s.update(body=body, eff=inner_eff)], eff=outer_eff
                        )
                    ]

        return None

    def map_e(self, e):
        return None


class DoExpandDim(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, alloc_dim, indexing):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert isinstance(alloc_dim, LoopIR.expr)
        assert isinstance(indexing, LoopIR.expr)

        self.alloc_sym = self.alloc_stmt.name
        self.alloc_dim = alloc_dim
        self.indexing = indexing
        self.alloc_type = None
        self.new_alloc_stmt = False
        self.in_call_arg = False

        # size positivity check
        Check_IsPositiveExpr(proc_cursor._node(), [self.alloc_stmt], alloc_dim)

        super().__init__(proc_cursor)

        # bounds check
        Check_Bounds(self.proc, self.new_alloc_stmt, self.after_alloc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        # this chunk of code just finds the statement block
        # that comes after the allocation
        stmts = super().map_stmts(stmts)
        if stmts is not None and self.new_alloc_stmt:
            for i, s in enumerate(stmts):
                if s is self.new_alloc_stmt:
                    self.after_alloc = stmts[i + 1 :]
                    break

        return stmts

    def map_s(self, sc):
        s = sc._node()
        if s is self.alloc_stmt:
            old_typ = s.type
            new_rngs = [self.alloc_dim]

            if isinstance(old_typ, T.Tensor):
                new_rngs += old_typ.shape()

            basetyp = old_typ.basetype()
            new_typ = T.Tensor(new_rngs, False, basetyp)
            self.alloc_type = new_typ
            new_alloc = LoopIR.Alloc(s.name, new_typ, s.mem, None, s.srcinfo)
            self.new_alloc_stmt = new_alloc

            return [new_alloc]

        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)) and s.name == self.alloc_sym:
            idx = [self.indexing] + self.apply_exprs(s.idx)
            rhs = self.apply_e(s.rhs)
            return [s.update(idx=idx, rhs=rhs, eff=None)]

        if isinstance(s, LoopIR.Call):
            self.in_call_arg = True
            args = self._map_list(self.map_e, s.args) or s.args
            self.in_call_arg = False
            return [LoopIR.Call(s.f, args, None, s.srcinfo)]

        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.alloc_sym:
            if self.in_call_arg and len(e.idx) == 0:
                raise SchedulingError(
                    "TODO: Please Contact the developers to fix (i.e. add) "
                    "support for passing windows to scalar arguments"
                )
            else:
                return e.update(idx=[self.indexing] + self.apply_exprs(e.idx))

        if isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            w_idx = self._map_list(self.map_w_access, e.idx) or e.idx
            idx = [LoopIR.Point(self.indexing, e.srcinfo)] + w_idx
            return e.update(
                idx=idx, type=T.Window(self.alloc_type, e.type.as_tensor, e.name, idx)
            )

        # fall-through
        return super().map_e(e)


class DoRearrangeDim(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, permute_vector):
        self.alloc_stmt = alloc_cursor._node()
        assert isinstance(self.alloc_stmt, LoopIR.Alloc)

        # dictionary can be used to permute windows in the future...
        self.all_permute = {self.alloc_stmt.name: permute_vector}

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def should_permute(self, buf):
        return buf in self.all_permute

    def permute(self, buf, es):
        permutation = self.all_permute[buf]
        return [es[i] for i in permutation]

    def permute_single_idx(self, buf, i):
        return self.all_permute[buf].index(i)

    def check_permute_window(self, buf, idx):
        # for now just enforce a stability criteria on windowing
        # expressions w.r.t. dimension reordering
        permutation = self.all_permute[name]
        # where each index of the output window now refers to in the
        # buffer being windowed
        keep_perm = [i for i in permutation if isinstance(idx[i], LoopIR.Interval)]
        # check that these indices are monotonic
        for i, ii in zip(keep_perm[:-1], keep_perm[1:]):
            if i > ii:
                return False
        return True

    def map_s(self, sc):
        s = sc._node()
        # simply change the dimension
        if s is self.alloc_stmt:
            # construct new_hi
            new_hi = self.permute(s.name, s.type.hi)
            # construct new_type
            new_type = LoopIR.Tensor(new_hi, s.type.is_window, s.type.type)

            return [LoopIR.Alloc(s.name, new_type, s.mem, None, s.srcinfo)]

        # Adjust the use-site
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if self.should_permute(s.name):
                # shuffle
                new_idx = self.permute(s.name, s.idx)
                return [
                    type(s)(s.name, s.type, s.cast, new_idx, s.rhs, None, s.srcinfo)
                ]

        if isinstance(s, LoopIR.Call):
            # check that the arguments are not permuted buffers
            for a in s.args:
                if isinstance(a, LoopIR.Read) and self.should_permute(a.name):
                    raise SchedulingError(
                        "Cannot permute buffer '{a.name}' because it is "
                        "passed as an sub-procedure argument at {s.srcinfo}"
                    )

        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, (LoopIR.Read, LoopIR.WindowExpr)):
            if self.should_permute(e.name):
                if isinstance(e, LoopIR.WindowExpr) and not self.check_permute_window(
                    e.name, e.idx
                ):
                    raise SchedulingError(
                        f"Permuting the window expression at {e.srcinfo} "
                        f"would change the meaning of the window; "
                        f"propogating dimension rearrangement through "
                        f"windows is not currently supported"
                    )
                return type(e)(e.name, self.permute(e.name, e.idx), e.type, e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr):
            if self.should_permute(e.name):
                dim = self.permute(e.name, e.dim)
                return e.update(dim=dim)

        return super().map_e(e)


class DoDivideDim(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, dim_idx, quotient):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert isinstance(dim_idx, int)
        assert isinstance(quotient, int)

        self.alloc_sym = self.alloc_stmt.name
        self.dim_idx = dim_idx
        self.quotient = quotient

        super().__init__(proc_cursor)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def remap_idx(self, idx):
        orig_i = idx[self.dim_idx]
        srcinfo = orig_i.srcinfo
        quot = LoopIR.Const(self.quotient, T.int, srcinfo)
        hi = LoopIR.BinOp("/", orig_i, quot, orig_i.type, srcinfo)
        lo = LoopIR.BinOp("%", orig_i, quot, orig_i.type, srcinfo)
        return idx[: self.dim_idx] + [hi, lo] + idx[self.dim_idx + 1 :]

    def map_s(self, sc):
        s = sc._node()
        if s is self.alloc_stmt:
            old_typ = s.type
            old_shp = old_typ.shape()
            dim = old_shp[self.dim_idx]

            if not isinstance(dim, LoopIR.Const):
                raise SchedulingError(
                    f"Cannot divide non-literal dimension: {str(dim)}"
                )
            if not dim.val % self.quotient == 0:
                raise SchedulingError(
                    f"Cannot divide {dim.val} evenly by {self.quotient}"
                )
            denom = self.quotient
            numer = dim.val // denom
            new_shp = (
                old_shp[: self.dim_idx]
                + [
                    LoopIR.Const(numer, T.int, dim.srcinfo),
                    LoopIR.Const(denom, T.int, dim.srcinfo),
                ]
                + old_shp[self.dim_idx + 1 :]
            )
            new_typ = T.Tensor(new_shp, False, old_typ.basetype())

            return [LoopIR.Alloc(s.name, new_typ, s.mem, None, s.srcinfo)]

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)) and s.name == self.alloc_sym:
            idx = self.remap_idx(self.apply_exprs(s.idx))
            rhs = self.apply_e(s.rhs)
            return [s.update(idx=idx, rhs=rhs, eff=None)]

        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.alloc_sym:
            if not e.idx:
                raise SchedulingError(
                    f"Cannot divide {self.alloc_sym} because "
                    f"buffer is passed as an argument"
                )
            return e.update(idx=self.remap_idx(self.apply_exprs(e.idx)))
        elif isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            raise SchedulingError(
                f"Cannot divide {self.alloc_sym} because "
                f"the buffer is windowed later on"
            )

        # fall-through
        return super().map_e(e)


class DoMultiplyDim(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, hi_idx, lo_idx):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert isinstance(hi_idx, int)
        assert isinstance(lo_idx, int)

        self.alloc_sym = self.alloc_stmt.name
        self.hi_idx = hi_idx
        self.lo_idx = lo_idx
        lo_dim = self.alloc_stmt.type.shape()[lo_idx]
        if not isinstance(lo_dim, LoopIR.Const):
            raise SchedulingError(
                f"Cannot multiply with non-literal second dimension: {str(lo_dim)}"
            )
        self.lo_val = lo_dim.val

        super().__init__(proc_cursor)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def remap_idx(self, idx):
        hi = idx[self.hi_idx]
        lo = idx[self.lo_idx]
        mulval = LoopIR.Const(self.lo_val, T.int, hi.srcinfo)
        mul_hi = LoopIR.BinOp("*", mulval, hi, hi.type, hi.srcinfo)
        prod = LoopIR.BinOp("+", mul_hi, lo, T.index, hi.srcinfo)
        idx[self.hi_idx] = prod
        del idx[self.lo_idx]
        return idx

    def map_s(self, sc):
        s = sc._node()
        if s is self.alloc_stmt:
            old_typ = s.type
            shp = old_typ.shape().copy()

            hi_dim = shp[self.hi_idx]
            lo_dim = shp[self.lo_idx]
            prod = LoopIR.BinOp("*", lo_dim, hi_dim, hi_dim.type, hi_dim.srcinfo)
            shp[self.hi_idx] = prod
            del shp[self.lo_idx]

            new_typ = T.Tensor(shp, False, old_typ.basetype())

            return [LoopIR.Alloc(s.name, new_typ, s.mem, None, s.srcinfo)]

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)) and s.name == self.alloc_sym:
            return [
                s.update(
                    idx=self.remap_idx(self.apply_exprs(s.idx)),
                    rhs=self.apply_e(s.rhs),
                    eff=None,
                )
            ]

        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.alloc_sym:
            if not e.idx:
                raise SchedulingError(
                    f"Cannot multiply {self.alloc_sym} because "
                    f"buffer is passed as an argument"
                )
            return e.update(idx=self.remap_idx(self.apply_exprs(e.idx)))

        elif isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            raise SchedulingError(
                f"Cannot multiply {self.alloc_sym} because "
                f"the buffer is windowed later on"
            )

        # fall-through
        return super().map_e(e)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# *Only* lifting an allocation


class DoLiftAllocSimple(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, n_lifts):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert is_pos_int(n_lifts)

        self.n_lifts = n_lifts
        self.ctrl_ctxt = []
        self.lift_site = None

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.alloc_stmt:
            if self.n_lifts > len(self.ctrl_ctxt):
                raise SchedulingError(
                    f"specified lift level {self.n_lifts} "
                    f"is more than {len(self.ctrl_ctxt)}, "
                    f"the number of loops "
                    f"and ifs above the allocation"
                )
            if s.type.shape():
                szvars = set.union(*[_FV(sz) for sz in s.type.shape()])
                for i in self.get_ctrl_iters():
                    if i in szvars:
                        raise SchedulingError(
                            f"Cannot lift allocation statement {s} past loop "
                            f"with iteration variable {i} because "
                            f"the allocation size depends on {i}."
                        )
            self.lift_site = self.ctrl_ctxt[-self.n_lifts]

            return []

        elif isinstance(s, (LoopIR.If, LoopIR.Seq)):
            self.ctrl_ctxt.append(s)
            stmts = super().map_s(sc)
            self.ctrl_ctxt.pop()
            # TODO: it is technically possible to end up with for-loops
            # and if-statements that have empty bodies.  We should check
            # for this situation, even if it's extremely unlikely.

            if s is self.lift_site:
                new_alloc = LoopIR.Alloc(
                    self.alloc_stmt.name,
                    self.alloc_stmt.type,
                    self.alloc_stmt.mem,
                    None,
                    s.srcinfo,
                )
                stmts = [new_alloc] + stmts

            return stmts

        return super().map_s(sc)

    def get_ctrl_iters(self):
        return [
            s.iter for s in self.ctrl_ctxt[-self.n_lifts :] if isinstance(s, LoopIR.Seq)
        ]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Lift Allocation scheduling directive

# TODO: Implement autolift_alloc's logic using high-level scheduling metaprogramming and
#       delete this code
class DoLiftAlloc(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, n_lifts, mode, size, keep_dims):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert is_pos_int(n_lifts)

        if mode not in ("row", "col"):
            raise SchedulingError(f"Unknown lift mode {mode}, should be 'row' or 'col'")

        self.alloc_sym = self.alloc_stmt.name
        self.alloc_deps = LoopIR_Dependencies(
            self.alloc_sym, proc_cursor._node().body
        ).result()
        self.lift_mode = mode
        self.lift_size = size
        self.keep_dims = keep_dims

        self.n_lifts = n_lifts

        self.ctrl_ctxt = []
        self.lift_site = None

        self.lifted_stmt = None
        self.access_idxs = None
        self.alloc_type = None
        self._in_call_arg = False

        super().__init__(proc_cursor)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def idx_mode(self, access, orig):
        if self.lift_mode == "row":
            return access + orig
        elif self.lift_mode == "col":
            return orig + access
        assert False

    def map_s(self, sc):
        s = sc._node()
        if s is self.alloc_stmt:
            if self.n_lifts > len(self.ctrl_ctxt):
                raise SchedulingError(
                    f"specified lift level {self.n_lifts} "
                    "is higher than the number of loop "
                    f"{len(self.ctrl_ctxt)}"
                )
            self.lift_site = self.ctrl_ctxt[-self.n_lifts]

            # extract the ranges and variables of enclosing loops
            idxs, rngs = self.get_ctxt_itrs_and_rngs(self.n_lifts)

            # compute the lifted allocation buffer type, and
            # the new allocation statement
            new_typ = s.type
            new_rngs = []
            for r in rngs:
                if isinstance(r, LoopIR.Const):
                    assert r.val > 0, "Loop bound must be positive"
                    new_rngs.append(r)
                else:
                    new_rngs.append(
                        LoopIR.BinOp(
                            "+",
                            r,
                            LoopIR.Const(1, T.int, r.srcinfo),
                            T.index,
                            r.srcinfo,
                        )
                    )

            if isinstance(new_typ, T.Tensor):
                if self.lift_mode == "row":
                    new_rngs += new_typ.shape()
                elif self.lift_mode == "col":
                    new_rngs = new_typ.shape() + new_rngs
                else:
                    assert False

                new_typ = new_typ.basetype()

            if len(new_rngs) > 0:
                new_typ = T.Tensor(new_rngs, False, new_typ)

            # effect remains null
            self.lifted_stmt = LoopIR.Alloc(s.name, new_typ, s.mem, None, s.srcinfo)
            self.access_idxs = idxs
            self.alloc_type = new_typ

            # erase the statement from this location
            return []

        elif isinstance(s, (LoopIR.If, LoopIR.Seq)):
            # handle recursive part of pass at this statement
            self.ctrl_ctxt.append(s)
            stmts = super().map_s(sc)
            self.ctrl_ctxt.pop()

            # splice in lifted statement at the point to lift-to
            if s is self.lift_site:
                stmts = [self.lifted_stmt] + (stmts or s)

            return stmts

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            # in this case, we may need to substitute the
            # buffer name on the lhs of the assignment/reduction
            if s.name is self.alloc_sym:
                assert self.access_idxs is not None
                idx = self.idx_mode(
                    [LoopIR.Read(i, [], T.index, s.srcinfo) for i in self.access_idxs],
                    s.idx,
                )
                rhs = self.apply_e(s.rhs)
                # return allocation or reduction...
                return s.update(idx=idx, rhs=rhs, eff=None)

        elif isinstance(s, LoopIR.Call):
            # substitution in call arguments currently unsupported;
            # so setting flag here
            self._in_call_arg = True
            stmts = super().map_s(sc)
            self._in_call_arg = False
            return stmts

        # fall-through
        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.alloc_sym:
            assert self.access_idxs is not None
            if not self.access_idxs:
                return None

            # if self._in_call_arg:
            if e.type.is_real_scalar():
                idx = self.idx_mode(
                    [LoopIR.Read(i, [], T.index, e.srcinfo) for i in self.access_idxs],
                    e.idx,
                )
                return LoopIR.Read(e.name, idx, e.type, e.srcinfo)
            else:
                assert self._in_call_arg
                assert len(e.idx) == 0
                # then we need to replace this read with a
                # windowing expression
                access = [
                    LoopIR.Point(LoopIR.Read(i, [], T.index, e.srcinfo), e.srcinfo)
                    for i in self.access_idxs
                ]
                orig = [
                    LoopIR.Interval(LoopIR.Const(0, T.int, e.srcinfo), hi, e.srcinfo)
                    for hi in e.type.shape()
                ]
                idx = self.idx_mode(access, orig)
                tensor_type = (
                    e.type.as_tensor if isinstance(e.type, T.Window) else e.type
                )
                win_typ = T.Window(self.alloc_type, tensor_type, e.name, idx)
                return LoopIR.WindowExpr(e.name, idx, win_typ, e.srcinfo)

        if isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            assert self.access_idxs is not None
            if not self.access_idxs:
                return None
            # otherwise, extend windowing with accesses...

            idx = self.idx_mode(
                [
                    LoopIR.Point(LoopIR.Read(i, [], T.index, e.srcinfo), e.srcinfo)
                    for i in self.access_idxs
                ],
                e.idx,
            )
            win_typ = T.Window(self.alloc_type, e.type.as_tensor, e.name, idx)
            return LoopIR.WindowExpr(e.name, idx, win_typ, e.srcinfo)

        # fall-through
        return super().map_e(e)

    def get_ctxt_itrs_and_rngs(self, n_up):
        rngs = []
        idxs = []
        for s in self.ctrl_ctxt[-n_up:]:
            if isinstance(s, LoopIR.If):
                # if-statements do not affect allocations
                # note that this may miss opportunities to
                # shrink the allocation by being aware of
                # guards; oh well.
                continue
            elif isinstance(s, LoopIR.Seq):
                if s.iter in self.alloc_deps and self.keep_dims:
                    idxs.append(s.iter)
                    if isinstance(s.hi, LoopIR.Read):
                        assert s.hi.type.is_indexable()
                        assert len(s.hi.idx) == 0
                    elif isinstance(s.hi, LoopIR.Const):
                        assert s.hi.type == T.int
                    elif isinstance(s.hi, LoopIR.BinOp):
                        assert s.hi.type.is_indexable()
                    else:
                        assert False, "bad case"

                    if self.lift_size is not None:
                        assert isinstance(self.lift_size, int)
                        # TODO: More robust checking of self.lift_size >= s.hi
                        if isinstance(s.hi, LoopIR.Const):
                            if s.hi.val > self.lift_size:
                                raise SchedulingError(
                                    f"Lift size cannot "
                                    f"be less than for-loop bound {s.hi.val}"
                                )
                        elif isinstance(s.hi, LoopIR.BinOp) and s.hi.op == "%":
                            assert isinstance(s.hi.rhs, LoopIR.Const)
                            if s.hi.rhs.val > self.lift_size:
                                raise SchedulingError(
                                    f"Lift size cannot "
                                    f"be less than for-loop bound {s.hi}"
                                )
                        else:
                            raise NotImplementedError

                        rngs.append(LoopIR.Const(self.lift_size, T.int, s.srcinfo))
                    else:
                        rngs.append(s.hi)
            else:
                assert False, "bad case"

        return idxs, rngs


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Fissioning at a Statement scheduling directive


def check_used(variables, eff):
    for e in eff:
        if e.buffer in variables:
            return True
    return False


class _Is_Alloc_Free(LoopIR_Do):
    def __init__(self, pre, post):
        self._is_alloc_free = True
        self._alloc_var = []

        self.do_stmts(pre)

        # make sure all of _alloc_vars are not used in any of the
        # post statement
        for s in post:
            if isinstance(s, LoopIR.Reduce):  # Allow reduce
                continue
            if s.eff is None:
                continue
            if check_used(self._alloc_var, s.eff.reads):
                self._is_alloc_free = False
                break
            if check_used(self._alloc_var, s.eff.writes):
                self._is_alloc_free = False
                break
            if check_used(self._alloc_var, s.eff.reduces):
                self._is_alloc_free = False
                break

    def result(self):
        return self._is_alloc_free

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            self._alloc_var.append(s.name)

        super().do_s(s)


def _is_alloc_free(pre, post):
    return _Is_Alloc_Free(pre, post).result()


# which variable symbols are free
class _FreeVars(LoopIR_Do):
    def __init__(self, stmts):
        self._fvs = set()
        self._bound = set()

        if isinstance(stmts, LoopIR.expr):
            self.do_e(stmts)
        else:
            self.do_stmts(stmts)

    def result(self):
        return self._fvs

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if s.name not in self._bound:
                self._fvs.add(s.name)
        elif isinstance(s, LoopIR.Seq):
            self._bound.add(s.iter)
        elif isinstance(s, LoopIR.Alloc):
            self._bound.add(s.name)

        super().do_s(s)

    def do_e(self, e):
        if isinstance(e, LoopIR.Read):
            if e.name not in self._bound:
                self._fvs.add(e.name)

        super().do_e(e)


def _FV(stmts):
    return _FreeVars(stmts).result()


def _is_idempotent(stmts):
    def _stmt(s):
        styp = type(s)
        if styp is LoopIR.Reduce:
            return False
        elif styp is LoopIR.Call:
            return _is_idempotent(s.f.body)
        elif styp is LoopIR.If:
            return _is_idempotent(s.body) and _is_idempotent(s.orelse)
        elif styp is LoopIR.Seq:
            return _is_idempotent(s.body)
        else:
            return True

    return all(_stmt(s) for s in stmts)


class DoRemoveLoop(Cursor_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor):
        self.stmt = stmt_cursor._node()
        assert isinstance(self.stmt, LoopIR.stmt)
        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.stmt:
            # Check if we can remove the loop
            # Conditions are:
            # 1. Body does not depend on the loop iteration variable
            if s.iter in _FV(s.body):
                raise SchedulingError(
                    f"Cannot remove loop, {s.iter} is not " "free in the loop body."
                )

            # 2. Body is idemopotent
            Check_IsIdempotent(self.orig_proc._node(), [s])

            # 3. The loop runs at least once;
            #    If not, then place a guard around the statement
            body = Alpha_Rename(s.body).result()
            try:
                Check_IsPositiveExpr(self.orig_proc._node(), [s], s.hi)
            except SchedulingError:
                zero = LoopIR.Const(0, T.int, s.srcinfo)
                cond = LoopIR.BinOp(">", s.hi, zero, T.bool, s.srcinfo)
                body = [LoopIR.If(cond, body, [], None, s.srcinfo)]

            return body

        return super().map_s(sc)


# This is same as original FissionAfter, except that
# this does not remove loop. We have separate remove_loop
# operator for that purpose.
class DoFissionAfterSimple:
    def __init__(self, proc_cursor, stmt_cursor, n_lifts):
        self.tgt_stmt = stmt_cursor._node()
        assert isinstance(self.tgt_stmt, LoopIR.stmt)
        assert is_pos_int(n_lifts)
        self.provenance = proc_cursor.proc()
        self.orig_proc = proc_cursor._node()
        self.n_lifts = n_lifts

        self.hit_fission = False  # signal to map_stmts

        pre_body, post_body = self.map_stmts(self.orig_proc.body)
        self.proc = proc_cursor._node().update(body=pre_body + post_body, instr=None)
        self.proc = InferEffects(self.proc).result()

    def result(self):
        return api.Procedure(self.proc, _provenance_eq_Procedure=self.provenance)

    def alloc_check(self, pre, post):
        if not _is_alloc_free(pre, post):
            pre_allocs = {s.name for s in pre if isinstance(s, LoopIR.Alloc)}
            post_FV = _FV(post)
            for nm in pre_allocs:
                if nm in post_FV:
                    raise SchedulingError(
                        f"Will not fission here, because "
                        f"doing so will hide the allocation "
                        f"of {nm} from a later use site."
                    )

    # returns a pair of stmt-lists
    # for those statements occurring before and
    # after the fission point
    def map_stmts(self, stmts):
        pre_stmts = []
        post_stmts = []
        for orig_s in stmts:
            pre, post = self.map_s(orig_s)
            pre_stmts += pre
            post_stmts += post

        return pre_stmts, post_stmts

    # see map_stmts comment
    def map_s(self, s):
        if s is self.tgt_stmt:
            assert self.hit_fission == False
            self.hit_fission = True
            # none-the-less make sure we return this statement in
            # the pre-fission position
            return [s], []

        elif isinstance(s, LoopIR.If):

            # first, check if we need to split the body
            pre, post = self.map_stmts(s.body)
            if pre and post and self.n_lifts > 0:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre = LoopIR.If(s.cond, pre, [], None, s.srcinfo)
                post = LoopIR.If(s.cond, post, s.orelse, None, s.srcinfo)
                return [pre], [post]

            body = pre + post

            # if we don't, then check if we need to split the or-else
            pre, post = self.map_stmts(s.orelse)
            if pre and post and self.n_lifts > 0:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre = LoopIR.If(s.cond, body, pre, None, s.srcinfo)
                post = LoopIR.If(
                    s.cond, [LoopIR.Pass(None, s.srcinfo)], post, None, s.srcinfo
                )
                return [pre], [post]

            orelse = pre + post

            # if we neither split the body nor the or-else,
            # then we need to gather together the pre and post.
            single_stmt = LoopIR.If(s.cond, body, orelse, None, s.srcinfo)

        elif isinstance(s, LoopIR.Seq):
            styp = type(s)
            # check if we need to split the loop
            pre, post = self.map_stmts(s.body)
            if pre and post and self.n_lifts > 0:
                self.n_lifts -= 1
                self.alloc_check(pre, post)

                # we must check whether the two parts of the
                # fission can commute appropriately
                no_loop_var_pre = s.iter not in _FV(pre)
                Check_FissionLoop(self.orig_proc, s, pre, post, no_loop_var_pre)

                # we can skip the loop iteration if the
                # body doesn't depend on the loop
                # and the body is idempotent
                pre = [styp(s.iter, s.hi, pre, None, s.srcinfo)]
                pre = Alpha_Rename(pre).result()
                post = [styp(s.iter, s.hi, post, None, s.srcinfo)]
                post = Alpha_Rename(post).result()

                return pre, post

            # if we didn't split, then compose pre and post of the body
            single_stmt = styp(s.iter, s.hi, pre + post, None, s.srcinfo)

        else:
            # all other statements cannot recursively
            # contain statements, so...
            single_stmt = s

        if self.hit_fission:
            return [], [single_stmt]
        else:
            return [single_stmt], []


# TODO: Deprecate this with the one above
# structure is weird enough to skip using the Rewrite-pass super-class
class DoFissionLoops:
    def __init__(self, proc_cursor, stmt_cursor, n_lifts):
        self.tgt_stmt = stmt_cursor._node()
        assert isinstance(self.tgt_stmt, LoopIR.stmt)
        assert is_pos_int(n_lifts)
        self.provenance = proc_cursor.proc()
        self.orig_proc = proc_cursor._node()
        self.n_lifts = n_lifts

        self.hit_fission = False  # signal to map_stmts

        pre_body, post_body = self.map_stmts(self.orig_proc.body)
        self.proc = LoopIR.proc(
            name=self.orig_proc.name,
            args=self.orig_proc.args,
            preds=self.orig_proc.preds,
            body=pre_body + post_body,
            instr=None,
            eff=self.orig_proc.eff,
            srcinfo=self.orig_proc.srcinfo,
        )
        self.proc = InferEffects(self.proc).result()

    def result(self):
        return api.Procedure(self.proc, _provenance_eq_Procedure=self.provenance)

    def alloc_check(self, pre, post):
        if not _is_alloc_free(pre, post):
            raise SchedulingError(
                "Will not fission here, because "
                "an allocation might be buried "
                "in a different scope than some use-site"
            )

    # returns a pair of stmt-lists
    # for those statements occurring before and
    # after the fission point
    def map_stmts(self, stmts):
        pre_stmts = []
        post_stmts = []
        for orig_s in stmts:
            pre, post = self.map_s(orig_s)
            pre_stmts += pre
            post_stmts += post

        return pre_stmts, post_stmts

    # see map_stmts comment
    def map_s(self, s):
        if s is self.tgt_stmt:
            # assert self.hit_fission == False
            self.hit_fission = True
            # none-the-less make sure we return this statement in
            # the pre-fission position
            return [s], []

        elif isinstance(s, LoopIR.If):

            # first, check if we need to split the body
            pre, post = self.map_stmts(s.body)
            fission_body = len(pre) > 0 and len(post) > 0 and self.n_lifts > 0
            if fission_body:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre = LoopIR.If(s.cond, pre, [], None, s.srcinfo)
                post = LoopIR.If(s.cond, post, s.orelse, None, s.srcinfo)
                return [pre], [post]

            body = pre + post

            # if we don't, then check if we need to split the or-else
            pre, post = self.map_stmts(s.orelse)
            fission_orelse = len(pre) > 0 and len(post) > 0 and self.n_lifts > 0
            if fission_orelse:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre = LoopIR.If(s.cond, body, pre, None, s.srcinfo)
                post = LoopIR.If(
                    s.cond, [LoopIR.Pass(None, s.srcinfo)], post, None, s.srcinfo
                )
                return [pre], [post]

            orelse = pre + post

            # if we neither split the body nor the or-else,
            # then we need to gather together the pre and post.
            single_stmt = LoopIR.If(s.cond, body, orelse, None, s.srcinfo)

        elif isinstance(s, LoopIR.Seq):

            # check if we need to split the loop
            pre, post = self.map_stmts(s.body)
            do_fission = len(pre) > 0 and len(post) > 0 and self.n_lifts > 0
            if do_fission:
                self.n_lifts -= 1
                self.alloc_check(pre, post)

                # we can skip the loop iteration if the
                # body doesn't depend on the loop
                # and the body is idempotent
                if s.iter in _FV(pre) or not _is_idempotent(pre):
                    pre = [s.update(body=pre, eff=None)]
                    # since we are copying the binding of s.iter,
                    # we should perform an Alpha_Rename for safety
                    pre = Alpha_Rename(pre).result()
                if s.iter in _FV(post) or not _is_idempotent(post):
                    post = [s.update(body=post, eff=None)]

                return pre, post

            # if we didn't split, then compose pre and post of the body
            single_stmt = s.update(body=pre + post, eff=None)

        else:
            # all other statements cannot recursively
            # contain statements, so...
            single_stmt = s

        if self.hit_fission:
            return [], [single_stmt]
        else:
            return [single_stmt], []


class DoAddUnsafeGuard(Cursor_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, cond):
        self.stmt = stmt_cursor._node()
        self.cond = cond
        self.in_loop = False

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.stmt:
            # Check_ExprEqvInContext(self.orig_proc,
            #                       self.cond, [s],
            #                       LoopIR.Const(True, T.bool, s.srcinfo))
            s1 = Alpha_Rename([s]).result()
            return [LoopIR.If(self.cond, s1, [], None, s.srcinfo)]

        return super().map_s(sc)


class DoSpecialize(Cursor_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, conds):
        assert conds, "Must add at least one condition"
        self.stmt = stmt_cursor._node()
        self.conds = conds

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.stmt:
            else_br = Alpha_Rename([s]).result()
            for cond in reversed(self.conds):
                then_br = Alpha_Rename([s]).result()
                else_br = [LoopIR.If(cond, then_br, else_br, None, s.srcinfo)]
            return else_br

        return super().map_s(sc)


def _get_constant_bound(e):
    if isinstance(e, LoopIR.BinOp) and e.op == "%":
        return e.rhs
    raise SchedulingError(f"Could not derive constant bound on {e}")


class DoBoundAndGuard(Cursor_Rewrite):
    def __init__(self, proc_cursor, loop_cursor):
        self.loop = loop_cursor._node()
        super().__init__(proc_cursor)

    def map_s(self, sc):
        s = sc._node()
        if s == self.loop:
            assert isinstance(s, LoopIR.Seq)
            bound = _get_constant_bound(s.hi)
            guard = LoopIR.If(
                LoopIR.BinOp(
                    "<",
                    LoopIR.Read(s.iter, [], T.index, s.srcinfo),
                    s.hi,
                    T.bool,
                    s.srcinfo,
                ),
                s.body,
                [],
                None,
                s.srcinfo,
            )
            return [s.update(hi=bound, body=[guard], eff=None)]

        return super().map_s(sc)


def DoFuseLoop(proc_cursor, f_cursor, s_cursor):
    proc = proc_cursor._node()
    loop1 = f_cursor._node()
    loop2 = s_cursor._node()

    if f_cursor.next() != s_cursor:
        raise SchedulingError(
            "expected the two loops to be fused to come one right after the other"
        )

    # check if the loop bounds are equivalent
    Check_ExprEqvInContext(proc, loop1.hi, [loop1], loop2.hi, [loop2])

    x = LoopIR.Read(loop1.iter, [], T.index, loop1.srcinfo)
    y = loop2.iter
    body1 = loop1.body
    body2 = SubstArgs(loop2.body, {y: x}).result()

    proc, fwd1 = f_cursor.body()[-1].after()._insert(body2)
    proc, fwd2 = fwd1(s_cursor)._delete()
    loop = fwd2(fwd1(f_cursor))._node()

    Check_FissionLoop(proc._loopir_proc, loop, body1, body2)
    proc = InferEffects(proc._loopir_proc).result()
    return api.Procedure(proc, _provenance_eq_Procedure=proc_cursor.proc())


class DoFuseIf(Cursor_Rewrite):
    def __init__(self, proc_cursor, f_cursor, s_cursor):
        self.if1 = f_cursor._node()
        self.if2 = s_cursor._node()

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts_c):
        stmts = [s._node() for s in stmts_c]
        for i, s in enumerate(stmts):
            if s is self.if1:
                if i + 1 >= len(stmts) or stmts[i + 1] is not self.if2:
                    raise SchedulingError(
                        "expected the two if statements to be "
                        "fused to come one right after the other"
                    )

                if1, if2 = self.if1, self.if2

                # check if the loop bounds are equivalent
                Check_ExprEqvInContext(
                    self.orig_proc._node(), if1.cond, [if1], if2.cond, [if2]
                )

                cond = if1.cond
                body1 = if1.body
                body2 = if2.body
                orelse1 = if1.orelse
                orelse2 = if2.orelse
                ifstmt = LoopIR.If(
                    cond, body1 + body2, orelse1 + orelse2, None, if1.srcinfo
                )

                return stmts[:i] + [ifstmt] + stmts[i + 2 :]

        # if we reached this point, we didn't find the if statement
        return super().map_stmts(stmts)


class DoAddLoop(Cursor_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, var, hi, guard):
        self.stmt = stmt_cursor._node()
        self.var = Sym(var)
        self.hi = hi
        self.guard = guard

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.stmt:
            Check_IsIdempotent(self.orig_proc._node(), [s])
            Check_IsPositiveExpr(self.orig_proc._node(), [s], self.hi)

            sym = self.var
            hi = self.hi
            body = [s]

            if self.guard:
                rdsym = LoopIR.Read(sym, [], T.index, s.srcinfo)
                zero = LoopIR.Const(0, T.int, s.srcinfo)
                cond = LoopIR.BinOp("==", rdsym, zero, T.bool, s.srcinfo)
                body = [LoopIR.If(cond, body, [], None, s.srcinfo)]

            ir = [LoopIR.Seq(sym, hi, body, None, s.srcinfo)]
            return ir

        return super().map_s(sc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Factor out a sub-statement as a Procedure scheduling directive


def _make_closure(name, stmts, var_types):
    FVs = list(sorted(_FV(stmts)))
    info = stmts[0].srcinfo

    # work out the calling arguments (args) and sub-proc args (fnargs)
    args = []
    fnargs = []

    # first, scan over all the arguments and convert them.
    # accumulate all size symbols separately
    sizes = set()
    for v in FVs:
        typ = var_types[v]
        if typ is T.size:
            sizes.add(v)
        elif typ is T.index:
            args.append(LoopIR.Read(v, [], typ, info))
            fnargs.append(LoopIR.fnarg(v, typ, None, info))
        else:
            # add sizes (that this arg depends on) to the signature
            for sz in typ.shape():
                if isinstance(sz, Sym):
                    sizes.add(sz)
            args.append(LoopIR.Read(v, [], typ, info))
            fnargs.append(LoopIR.fnarg(v, typ, None, info))

    # now prepend all sizes to the argument list
    sizes = list(sorted(sizes))
    args = [LoopIR.Read(sz, [], T.size, info) for sz in sizes] + args
    fnargs = [LoopIR.fnarg(sz, T.size, None, info) for sz in sizes] + fnargs

    eff = None
    # TODO: raise NotImplementedError("need to figure out effect of new closure")
    closure = LoopIR.proc(name, fnargs, [], stmts, None, eff, info)

    return closure, args


class DoInsertPass(Cursor_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, before=True):
        self.stmt = stmt_cursor._node()
        self.before = before
        super().__init__(proc_cursor)

    def map_s(self, sc):
        s = sc._node()
        if s is self.stmt:
            pass_s = LoopIR.Pass(eff_null(s.srcinfo), srcinfo=s.srcinfo)
            if self.before:
                return [pass_s, s]
            else:
                return [s, pass_s]
        return super().map_s(sc)


class DoDeleteConfig(Cursor_Rewrite):
    def __init__(self, proc_cursor, config_cursor):
        self.stmt = config_cursor._node()
        self.eq_mod_config = set()
        super().__init__(proc_cursor)

    def mod_eq(self):
        return self.eq_mod_config

    def map_s(self, sc):
        s = sc._node()
        if s is self.stmt:
            mod_cfg = Check_DeleteConfigWrite(self.orig_proc._node(), [self.stmt])
            self.eq_mod_config = mod_cfg
            return []
        else:
            return super().map_s(sc)


class DoDeletePass(Cursor_Rewrite):
    def __init__(self, proc_cursor):
        super().__init__(proc_cursor)

    def map_s(self, sc):
        s = sc._node()
        if isinstance(s, LoopIR.Pass):
            return []

        elif isinstance(s, LoopIR.Seq):
            body = self.map_stmts(sc.body())
            if body is None:
                return None
            elif body == []:
                return []
            else:
                return [s.update(body=body)]

        return super().map_s(sc)


class DoExtractMethod(Cursor_Rewrite):
    def __init__(self, proc_cursor, name, stmt_cursor):
        self.match_stmt = stmt_cursor._node()
        assert isinstance(self.match_stmt, LoopIR.stmt)
        self.sub_proc_name = name
        self.new_subproc = None
        self.orig_proc = proc_cursor._node()

        self.var_types = ChainMap()

        for a in self.orig_proc.args:
            self.var_types[a.name] = a.type

        super().__init__(proc_cursor)
        Check_Aliasing(self.proc)

    def subproc(self):
        return api.Procedure(self.new_subproc)

    def push(self):
        self.var_types = self.var_types.new_child()

    def pop(self):
        self.var_types = self.var_types.parents

    def map_s(self, sc):
        s = sc._node()
        if s is self.match_stmt:
            subproc, args = _make_closure(self.sub_proc_name, [s], self.var_types)
            self.new_subproc = subproc
            return [LoopIR.Call(subproc, args, None, s.srcinfo)]
        elif isinstance(s, LoopIR.Alloc):
            self.var_types[s.name] = s.type
            return None
        elif isinstance(s, LoopIR.Seq):
            self.push()
            self.var_types[s.iter] = T.index
            body = self.map_stmts(sc.body())
            self.pop()

            if body:
                return [s.update(body=body, eff=None)]

            return None
        elif isinstance(s, LoopIR.If):
            self.push()
            body = self.map_stmts(sc.body())
            self.pop()
            self.push()
            orelse = self.map_stmts(sc.orelse())
            self.pop()

            if body or orelse:
                return [
                    s.update(body=body or s.body, orelse=orelse or s.orlse, eff=None)
                ]

            return None

        return super().map_s(sc)

    def map_e(self, e):
        return None


class _DoNormalize(Cursor_Rewrite):
    # This class operates on an idea of creating a coefficient map for each
    # indexing expression (normalize_e), and writing the map back to LoopIR
    # (get_loopir in index_start).
    # For example, when you have Assign statement:
    # y[n*4 - n*4 + 1] = 0.0
    # index_start will be called with e : n*4 - n*4 + 1.
    # Then, normalize_e will create a map of symbols and its coefficients.
    # The map for the expression `n*4 + 1` is:
    # { temporary_constant_symbol : 1, n : 4 }
    # and the map for the expression `n*4 - n*4 + 1` is:
    # { temporary_constant_symbol : 1, n : 0 }
    # This map concatnation is handled by concat_map function.
    def __init__(self, proc_cursor):
        self.C = Sym("temporary_constant_symbol")
        self.env = ChainMap()
        # TODO: dispatch to Z3 to reason about preds ranges
        for arg in proc_cursor._node().args:
            self.env[arg.name] = None
        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def concat_map(self, op, lhs, rhs):
        if op == "+":
            # if has same key: add value
            common = {key: (lhs[key] + rhs[key]) for key in lhs if key in rhs}
            return lhs | rhs | common
        elif op == "-":
            # has same key: sub value
            common = {key: (lhs[key] - rhs[key]) for key in lhs if key in rhs}
            # else, negate the rhs and cat map
            neg_rhs = {key: -rhs[key] for key in rhs}
            return lhs | neg_rhs | common
        elif op == "*":
            # rhs or lhs NEEDS to be constant
            assert len(rhs) == 1 or len(lhs) == 1
            # multiply the other one's value by that constant
            if len(rhs) == 1 and self.C in rhs:
                return {key: lhs[key] * rhs[self.C] for key in lhs}
            else:
                assert len(lhs) == 1 and self.C in lhs
                return {key: rhs[key] * lhs[self.C] for key in rhs}
        else:
            assert False, "bad case"

    def normalize_e(self, e):
        assert e.type.is_indexable(), f"{e} is not indexable!"

        if isinstance(e, LoopIR.Read):
            assert len(e.idx) == 0, "Indexing inside indexing does not make any sense"
            return {e.name: 1}
        elif isinstance(e, LoopIR.Const):
            return {self.C: e.val}
        elif isinstance(e, LoopIR.USub):
            e_map = self.normalize_e(e.arg)
            return {key: -e_map[key] for key in e_map}
        elif isinstance(e, LoopIR.BinOp):
            lhs_map = self.normalize_e(e.lhs)
            rhs_map = self.normalize_e(e.rhs)
            return self.concat_map(e.op, lhs_map, rhs_map)
        else:
            assert False, (
                "index_start should only be called by"
                + f" an indexing expression. e was {e}"
            )

    @staticmethod
    def has_div_mod_config(e):
        if isinstance(e, LoopIR.Read):
            return False
        elif isinstance(e, LoopIR.Const):
            return False
        elif isinstance(e, LoopIR.USub):
            return _DoNormalize.has_div_mod_config(e.arg)
        elif isinstance(e, LoopIR.BinOp):
            if e.op == "/" or e.op == "%":
                return True
            else:
                lhs = _DoNormalize.has_div_mod_config(e.lhs)
                rhs = _DoNormalize.has_div_mod_config(e.rhs)
                return lhs or rhs
        elif isinstance(e, LoopIR.ReadConfig):
            return True
        else:
            assert False, "bad case"

    # Call this when e is one indexing expression
    # e should be an indexing expression
    def index_start(self, e):
        def get_normalized_expr(e):
            # Make a map of symbols and coefficients
            n_map = self.normalize_e(e)

            new_e = LoopIR.Const(n_map.get(self.C, 0), T.int, e.srcinfo)

            delete_zero = [
                (n_map[v], v) for v in n_map if v != self.C and n_map[v] != 0
            ]

            return (new_e, delete_zero)

        def division_simplification(e):
            constant, normalization_list = get_normalized_expr(e.lhs)

            d = e.rhs.val

            non_divisible_terms = [
                (coeff, v) for coeff, v in normalization_list if coeff % d != 0
            ]

            if len(non_divisible_terms) == 0:
                normalization_list = [
                    (coeff // d, v) for coeff, v in normalization_list
                ]
                return generate_loopIR(
                    e.lhs, constant.update(val=constant.val // d), normalization_list
                )
            elif constant.val % d == 0:
                non_divisible_expr = generate_loopIR(
                    e.lhs, constant.update(val=0), non_divisible_terms
                )
                non_divisible_expr_range = AffineIndexRangeAnalysis(
                    non_divisible_expr, self.env
                ).result()

                if (
                    non_divisible_expr_range is not None
                    and 0 <= non_divisible_expr_range[0]
                    and non_divisible_expr_range[1] < d
                ):
                    divisible_terms = [
                        (coeff // d, v)
                        for coeff, v in normalization_list
                        if coeff % d == 0
                    ]
                    return generate_loopIR(
                        e.lhs, constant.update(val=constant.val // d), divisible_terms
                    )
            else:
                non_divisible_expr = generate_loopIR(
                    e.lhs, constant, non_divisible_terms
                )
                non_divisible_expr_range = AffineIndexRangeAnalysis(
                    non_divisible_expr, self.env
                ).result()

                if (
                    non_divisible_expr_range is not None
                    and 0 <= non_divisible_expr_range[0]
                    and non_divisible_expr_range[1] < d
                ):
                    divisible_terms = [
                        (coeff // d, v)
                        for coeff, v in normalization_list
                        if coeff % d == 0
                    ]
                    return generate_loopIR(
                        e.lhs, constant.update(val=0), divisible_terms
                    )

            new_lhs = generate_loopIR(e.lhs, constant, normalization_list)
            return LoopIR.BinOp("/", new_lhs, e.rhs, e.type, e.srcinfo)

        def modulo_simplification(e):
            constant, normalization_list = get_normalized_expr(e.lhs)

            m = e.rhs.val

            normalization_list = [
                (coeff, v) for coeff, v in normalization_list if coeff % m != 0
            ]

            if len(normalization_list) == 0:
                return constant.update(val=constant.val % m)

            if constant.val % m == 0:
                constant = constant.update(val=0)

            new_lhs = generate_loopIR(e.lhs, constant, normalization_list)
            new_lhs_range = AffineIndexRangeAnalysis(new_lhs, self.env).result()
            if new_lhs_range is not None and new_lhs_range[1] < m:
                assert new_lhs_range[0] >= 0
                return new_lhs

            return LoopIR.BinOp("%", new_lhs, e.rhs, e.type, e.srcinfo)

        def generate_loopIR(e_context, constant, normalization_list):
            def scale_read(coeff, key):
                return LoopIR.BinOp(
                    "*",
                    LoopIR.Const(coeff, T.int, e_context.srcinfo),
                    LoopIR.Read(key, [], e_context.type, e_context.srcinfo),
                    e_context.type,
                    e_context.srcinfo,
                )

            new_e = constant
            for coeff, v in sorted(normalization_list):
                if coeff > 0:
                    new_e = LoopIR.BinOp(
                        "+",
                        new_e,
                        scale_read(coeff, v),
                        e_context.type,
                        e_context.srcinfo,
                    )
                else:
                    new_e = LoopIR.BinOp(
                        "-",
                        new_e,
                        scale_read(-coeff, v),
                        e_context.type,
                        e_context.srcinfo,
                    )
            return new_e

        assert isinstance(e, LoopIR.expr)

        if isinstance(e, LoopIR.BinOp):
            new_lhs = self.index_start(e.lhs)
            new_rhs = self.index_start(e.rhs)
            e = e.update(lhs=new_lhs, rhs=new_rhs)

        if isinstance(e, LoopIR.BinOp) and e.op in ("/", "%"):
            assert isinstance(e.rhs, LoopIR.Const)
            if self.has_div_mod_config(e.lhs):
                return e

            if e.op == "/":
                return division_simplification(e)

            return modulo_simplification(e)

        # Div and mod special cases are handleded before, if that didn't succeed we cannot normalize
        # Skip ReadConfigs, they need careful handling because they're not Sym.
        if self.has_div_mod_config(e):
            return e

        constant, normalization_list = get_normalized_expr(e)
        return generate_loopIR(e, constant, normalization_list)

    def map_e(self, e):
        if e.type.is_indexable():
            return self.index_start(e)

        return super().map_e(e)

    def map_s(self, sc):
        s = sc._node()
        if isinstance(s, LoopIR.Seq):
            self.env = self.env.new_child()

            hi_range = AffineIndexRangeAnalysis(s.hi, self.env).result()
            if hi_range is not None:
                assert hi_range[0] >= 0
                if hi_range[1] == 0:
                    # We allow loop hi to be zero, however, that means that the loop
                    # variable doesn't have a defined range. We can set it to None
                    # since any simplification is not necessary since loop won't run
                    hi_range = None
                else:
                    hi_range = (0, hi_range[1] - 1)
                self.env[s.iter] = hi_range
            else:
                self.env[s.iter] = None

            new_s = super().map_s(sc)
            self.env = self.env.parents
            return new_s

        return super().map_s(sc)


class DoSimplify(Cursor_Rewrite):
    def __init__(self, proc_cursor):
        self.facts = ChainMap()
        new_procedure = _DoNormalize(proc_cursor).result()
        self.proc_cursor = ic.Cursor.root(new_procedure)

        super().__init__(self.proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def cfold(self, op, lhs, rhs):
        if op == "+":
            return lhs.val + rhs.val
        if op == "-":
            return lhs.val - rhs.val
        if op == "*":
            return lhs.val * rhs.val
        if op == "/":
            if lhs.type == T.f64 or lhs.type == T.f32:
                return lhs.val / rhs.val
            else:
                return lhs.val // rhs.val
        if op == "%":
            return lhs.val % rhs.val
        if op == "and":
            return lhs.val and rhs.val
        if op == "or":
            return lhs.val or rhs.val
        if op == "<":
            return lhs.val < rhs.val
        if op == ">":
            return lhs.val > rhs.val
        if op == "<=":
            return lhs.val <= rhs.val
        if op == ">=":
            return lhs.val >= rhs.val
        if op == "==":
            return lhs.val == rhs.val
        raise ValueError(f"Unknown operator ({op})")

    @staticmethod
    def is_quotient_remainder(e):
        """
        Checks if e is of the form (up to commutativity):
            N % K + K * (N / K)
        and returns N if so. Otherwise, returns None.
        """
        assert isinstance(e, LoopIR.BinOp)
        if e.op != "+":
            return None

        if isinstance(e.lhs, LoopIR.BinOp) and e.lhs.op == "%":
            assert isinstance(e.lhs.rhs, LoopIR.Const)
            num = e.lhs.lhs
            mod: LoopIR.Const = e.lhs.rhs
            rem = e.lhs
            quot = e.rhs
        elif isinstance(e.rhs, LoopIR.BinOp) and e.rhs.op == "%":
            assert isinstance(e.rhs.rhs, LoopIR.Const)
            num = e.rhs.lhs
            mod: LoopIR.Const = e.rhs.rhs
            rem = e.rhs
            quot = e.lhs
        else:
            return None

        # Validate form of remainder
        if not (
            isinstance(rem, LoopIR.BinOp)
            and rem.op == "%"
            and str(rem.lhs) == str(num)
            and str(rem.rhs) == str(mod)
        ):
            return None

        # Validate form of quotient
        if not (isinstance(quot, LoopIR.BinOp) and quot.op == "*"):
            return None

        def check_quot(const, div):
            if (
                isinstance(const, LoopIR.Const)
                and (isinstance(div, LoopIR.BinOp) and div.op == "/")
                and (str(const) == str(mod))
                and (str(div.lhs) == str(num))
                and (str(div.rhs) == str(mod))
            ):
                return num
            return None

        return check_quot(quot.lhs, quot.rhs) or check_quot(quot.rhs, quot.lhs)

    @staticmethod
    def simplify_div(e):
        """
        Simplifies expression of the of form:
            (c + Sigma a_i x_i) / d
        to:
            c / d + Sigma (a_i / d)x_i
            if c mod d = 0 and a_i mod d = 0 for all i
        """
        assert e.op == "/"

        def check_form(e):
            pass

    def map_binop(self, e: LoopIR.BinOp):
        lhs = self.map_e(e.lhs) or e.lhs
        rhs = self.map_e(e.rhs) or e.rhs

        if isinstance(lhs, LoopIR.Const) and isinstance(rhs, LoopIR.Const):
            return LoopIR.Const(self.cfold(e.op, lhs, rhs), lhs.type, lhs.srcinfo)

        if e.op == "+":
            if isinstance(lhs, LoopIR.Const) and lhs.val == 0:
                return rhs
            if isinstance(rhs, LoopIR.Const) and rhs.val == 0:
                return lhs
            if val := self.is_quotient_remainder(
                LoopIR.BinOp(e.op, lhs, rhs, lhs.type, lhs.srcinfo)
            ):
                return val
        elif e.op == "-":
            if isinstance(rhs, LoopIR.Const) and rhs.val == 0:
                return lhs
            if isinstance(lhs, LoopIR.BinOp) and lhs.op == "+":
                if lhs.lhs == rhs:
                    return lhs.rhs
                if lhs.rhs == rhs:
                    return lhs.lhs
        elif e.op == "*":
            if isinstance(lhs, LoopIR.Const) and lhs.val == 0:
                return LoopIR.Const(0, lhs.type, lhs.srcinfo)
            if isinstance(rhs, LoopIR.Const) and rhs.val == 0:
                return LoopIR.Const(0, lhs.type, lhs.srcinfo)
            if isinstance(lhs, LoopIR.Const) and lhs.val == 1:
                return rhs
            if isinstance(rhs, LoopIR.Const) and rhs.val == 1:
                return lhs
        elif e.op == "/":
            if isinstance(rhs, LoopIR.Const) and rhs.val == 1:
                return lhs
        elif e.op == "%":
            if isinstance(rhs, LoopIR.Const) and rhs.val == 1:
                return LoopIR.Const(0, lhs.type, lhs.srcinfo)

        return LoopIR.BinOp(e.op, lhs, rhs, e.type, e.srcinfo)

    def map_e(self, e):
        # If we get a match, then replace it with the known constant right away.
        # No need to run further simplify steps on this node.
        if const := self.is_known_constant(e):
            return const

        if isinstance(e, LoopIR.BinOp):
            e = self.map_binop(e)
        else:
            e = super().map_e(e) or e

        # After simplifying, we might match a known constant, so check again.
        if const := self.is_known_constant(e):
            return const

        return e

    def add_fact(self, cond):
        if (
            isinstance(cond, LoopIR.BinOp)
            and cond.op == "=="
            and isinstance(cond.rhs, LoopIR.Const)
        ):
            expr = cond.lhs
            const = cond.rhs
        elif (
            isinstance(cond, LoopIR.BinOp)
            and cond.op == "=="
            and isinstance(cond.lhs, LoopIR.Const)
        ):
            expr = cond.rhs
            const = cond.lhs
        else:
            return

        self.facts[str(expr)] = const

        # if we know that X / M == 0 then we also know that X % M == X.
        if isinstance(expr, LoopIR.BinOp) and expr.op == "/" and const.val == 0:
            mod_expr = LoopIR.BinOp("%", expr.lhs, expr.rhs, expr.type, expr.srcinfo)
            self.facts[str(mod_expr)] = expr.lhs

    def is_known_constant(self, e):
        if self.facts:
            return self.facts.get(str(e))
        return None

    def map_s(self, sc):
        s = sc._node()
        if isinstance(s, LoopIR.If):
            cond = self.map_e(s.cond)

            safe_cond = cond or s.cond

            # If constant true or false, then drop the branch
            if isinstance(safe_cond, LoopIR.Const):
                if safe_cond.val:
                    return super().map_stmts(sc.body())
                else:
                    return super().map_stmts(sc.orelse())

            # Try to use the condition while simplifying body
            self.facts = self.facts.new_child()
            self.add_fact(safe_cond)
            body = self.map_stmts(sc.body())
            self.facts = self.facts.parents

            # Try to use the negation while simplifying orelse
            self.facts = self.facts.new_child()
            # TODO: negate fact here
            orelse = self.map_stmts(sc.orelse())
            self.facts = self.facts.parents

            eff = self.map_eff(s.eff)
            if cond or body or orelse or eff:
                return [
                    s.update(
                        cond=safe_cond,
                        body=body or s.body,
                        orelse=orelse or s.orelse,
                        eff=eff or s.eff,
                    )
                ]
            return None
        elif isinstance(s, LoopIR.Seq):
            hi = self.map_e(s.hi)

            # Delete the loop if it would not run at all
            if isinstance(hi, LoopIR.Const) and hi.val == 0:
                return []

            # Delete the loop if it would have an empty body
            body = self.map_stmts(sc.body())
            if body == []:
                return []

            eff = self.map_eff(s.eff)
            if hi or body or eff:
                return [s.update(hi=hi or s.hi, body=body or s.body, eff=eff or s.eff)]

            return None
        else:
            return super().map_s(sc)


class DoAssertIf(Cursor_Rewrite):
    def __init__(self, proc_cursor, if_cursor, cond):
        self.if_stmt = if_cursor._node()

        assert isinstance(self.if_stmt, LoopIR.If)
        assert isinstance(cond, bool)

        self.cond = cond

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        if s is self.if_stmt:
            # check if the condition matches the asserted constant
            cond_node = LoopIR.Const(self.cond, T.bool, s.srcinfo)
            Check_ExprEqvInContext(self.orig_proc, s.cond, [s], cond_node)
            # if so, then we can simplify away the guard
            if self.cond:
                return self.map_stmts(sc.body())
            else:
                return self.map_stmts(sc.orelse())
        elif isinstance(s, LoopIR.Seq):
            body = self.map_stmts(sc.body())
            if not body:
                return []
            else:
                return [s.update(body=body)]

        return super().map_s(sc)


class DoDataReuse(Cursor_Rewrite):
    def __init__(self, proc_cursor, buf_cursor, rep_cursor):
        assert isinstance(buf_cursor._node(), LoopIR.Alloc)
        assert isinstance(rep_cursor._node(), LoopIR.Alloc)
        assert buf_cursor._node().type == rep_cursor._node().type

        self.buf_name = buf_cursor._node().name
        self.buf_dims = len(buf_cursor._node().type.shape())
        self.rep_name = rep_cursor._node().name
        self.rep_pat = rep_cursor._node()

        self.found_rep_alloc = False
        self.first_assn = True

        super().__init__(proc_cursor)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, sc):
        s = sc._node()
        # remove the allocation that we are eliminating through re-use
        if s is self.rep_pat:
            self.found_rep_alloc = True
            return []

        # make replacements after the first write to the buffer
        if self.found_rep_alloc:
            if (
                type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce
            ) and s.name == self.rep_name:
                name = self.buf_name
                rhs = self.map_e(s.rhs) or s.rhs

                # check whether the buffer we are trying to re-use
                # is live or not at this point in the execution
                if self.first_assn:
                    self.first_assn = False
                    Check_IsDeadAfter(
                        self.orig_proc._node(), [s], self.buf_name, self.buf_dims
                    )

                return [type(s)(name, s.type, None, s.idx, rhs, None, s.srcinfo)]

        return super().map_s(sc)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.rep_name:
            return e.update(name=self.buf_name)

        return super().map_e(e)


# TODO: This can probably be re-factored into a generic
# "Live Variables" analysis w.r.t. a context/stmt separation?
class _DoStageMem_FindBufData(LoopIR_Do):
    def __init__(self, proc, buf_name, stmt_start):
        self.buf_str = buf_name
        self.buf_sym = None
        self.buf_typ = None
        self.buf_mem = None
        self.stmt_start = stmt_start
        self.buf_map = ChainMap()
        self.orig_proc = proc

        for fa in self.orig_proc.args:
            if fa.type.is_numeric():
                self.buf_map[str(fa.name)] = (fa.name, fa.type, fa.mem)

        super().__init__(proc)

    def result(self):
        return self.buf_sym, self.buf_typ, self.buf_mem

    def push(self):
        self.buf_map = self.buf_map.new_child()

    def pop(self):
        self.buf_map = self.buf_map.parents

    def do_s(self, s):
        if s is self.stmt_start:
            if self.buf_str not in self.buf_map:
                raise SchedulingError(
                    f"no buffer or window "
                    f"named {self.buf_str} was live "
                    f"in the indicated statement block"
                )
            nm, typ, mem = self.buf_map[self.buf_str]
            self.buf_sym = nm
            self.buf_typ = typ
            self.buf_mem = mem

        if isinstance(s, LoopIR.Alloc):
            self.buf_map[str(s.name)] = (s.name, s.type, s.mem)
        if isinstance(s, LoopIR.WindowStmt):
            nm, typ, mem = self.buf_map[s.rhs.name]
            self.buf_map[str(s.name)] = (s.name, s.rhs.type, mem)
        elif isinstance(s, LoopIR.If):
            self.push()
            self.do_stmts(s.body)
            self.pop()
            self.push()
            self.do_stmts(s.orelse)
            self.pop()
        elif isinstance(s, LoopIR.Seq):
            self.push()
            self.do_stmts(s.body)
            self.pop()
        else:
            super().do_s(s)

    # short-circuit
    def do_e(self, e):
        pass


class DoStageMem(Cursor_Rewrite):
    def __init__(
        self,
        proc_cursor,
        buf_name,
        new_name,
        w_exprs,
        stmt_start,
        stmt_end,
        use_accum_zero=False,
    ):

        self.stmt_start = stmt_start._node()
        self.stmt_end = stmt_end._node()
        self.use_accum_zero = use_accum_zero

        nm, typ, mem = _DoStageMem_FindBufData(
            proc_cursor._node(), buf_name, self.stmt_start
        ).result()
        self.buf_name = nm  # this is a symbol
        self.buf_typ = typ if not isinstance(typ, T.Window) else typ.as_tensor
        self.buf_mem = mem

        self.w_exprs = w_exprs
        if len(w_exprs) != len(self.buf_typ.shape()):
            raise SchedulingError(
                f"expected windowing of '{buf_name}' "
                f"to have {len(self.buf_typ.shape())} indices, "
                f"but only got {len(w_exprs)}"
            )

        self.new_sizes = [
            LoopIR.BinOp("-", w[1], w[0], T.index, w[0].srcinfo)
            for w in w_exprs
            if isinstance(w, tuple)
        ]

        self.new_name = Sym(new_name)

        if all(isinstance(w, LoopIR.expr) for w in w_exprs):
            self.new_typ = typ.basetype()
        else:
            self.new_typ = T.Tensor(self.new_sizes, False, typ.basetype())

        self.found_stmt = False
        self.new_block = []
        self.in_block = False
        super().__init__(proc_cursor)
        assert self.found_stmt

        Check_Bounds(self.proc, self.new_block[0], self.new_block[1:])

        self.proc = InferEffects(self.proc).result()

    def rewrite_idx(self, idx):
        assert len(idx) == len(self.w_exprs)
        return [
            LoopIR.BinOp("-", i, w[0], T.index, i.srcinfo)
            for i, w in zip(idx, self.w_exprs)
            if isinstance(w, tuple)
        ]

    def rewrite_win(self, w_idx):
        assert len(w_idx) == len(self.w_exprs)

        def off_w(w, off):
            if isinstance(w, LoopIR.Interval):
                lo = LoopIR.BinOp("-", w.lo, off, T.index, w.srcinfo)
                hi = LoopIR.BinOp("-", w.hi, off, T.index, w.srcinfo)
                return LoopIR.Interval(lo, hi, w.srcinfo)
            else:
                assert isinstance(w, LoopIR.Point)
                pt = LoopIR.BinOp("-", w.pt, off, T.index, w.srcinfo)
                return LoopIR.Point(pt, w.srcinfo)

        return [off_w(w_i, w_e[0]) for w_i, w_e in zip(w_idx, self.w_exprs)]

    def map_stmts(self, stmts_c):
        """This method overload simply tries to find the indicated block"""
        if not self.in_block:
            for i, s1 in enumerate(stmts_c):
                if s1._node() is self.stmt_start:
                    for j, s2 in enumerate(stmts_c):
                        if s2._node() is self.stmt_end:
                            self.found_stmt = True
                            assert j >= i
                            pre = [s._node() for s in stmts_c[:i]]
                            post = [s._node() for s in stmts_c[j + 1 :]]
                            block = stmts_c[i : j + 1]

                            if self.use_accum_zero:
                                n_dims = len(self.buf_typ.shape())
                                Check_BufferReduceOnly(
                                    self.orig_proc._node(),
                                    [s._node() for s in block],
                                    self.buf_name,
                                    n_dims,
                                )

                            block = self.wrap_block(block)
                            self.new_block = block

                            return pre + block + post

        # fall through
        return super().map_stmts(stmts_c)

    def wrap_block(self, block_c):
        """This method rewrites the structure around the block.
        `map_s` and `map_e` below substitute the buffer
        name within the block."""
        block = [s._node() for s in block_c]
        orig_typ = self.buf_typ
        new_typ = self.new_typ
        mem = self.buf_mem
        shape = self.new_sizes

        n_dims = len(orig_typ.shape())
        basetyp = new_typ.basetype() if isinstance(new_typ, T.Tensor) else new_typ

        isR, isW = Check_BufferRW(self.orig_proc._node(), block, self.buf_name, n_dims)
        srcinfo = block[0].srcinfo

        new_alloc = [LoopIR.Alloc(self.new_name, new_typ, mem, None, srcinfo)]

        load_nest = []
        store_nest = []

        if isR:
            load_iter = [Sym(f"i{i}") for i, _ in enumerate(shape)]
            load_widx = [LoopIR.Read(s, [], T.index, srcinfo) for s in load_iter]

            cp_load_widx = load_widx.copy()
            load_ridx = []
            for w in self.w_exprs:
                if isinstance(w, tuple):
                    load_ridx.append(
                        LoopIR.BinOp("+", cp_load_widx.pop(0), w[0], T.index, srcinfo)
                    )
                else:
                    load_ridx.append(w)

            if self.use_accum_zero:
                load_rhs = LoopIR.Const(0.0, basetyp, srcinfo)
            else:
                load_rhs = LoopIR.Read(self.buf_name, load_ridx, basetyp, srcinfo)
            load_nest = [
                LoopIR.Assign(
                    self.new_name, basetyp, None, load_widx, load_rhs, None, srcinfo
                )
            ]

            for i, n in reversed(list(zip(load_iter, shape))):
                loop = LoopIR.Seq(i, n, load_nest, None, srcinfo)
                load_nest = [loop]

        if isW:
            store_iter = [Sym(f"i{i}") for i, _ in enumerate(shape)]
            store_ridx = [LoopIR.Read(s, [], T.index, srcinfo) for s in store_iter]
            cp_store_ridx = store_ridx.copy()
            store_widx = []
            for w in self.w_exprs:
                if isinstance(w, tuple):
                    store_widx.append(
                        LoopIR.BinOp("+", cp_store_ridx.pop(0), w[0], T.index, srcinfo)
                    )
                else:
                    store_widx.append(w)

            store_rhs = LoopIR.Read(self.new_name, store_ridx, basetyp, srcinfo)
            store_stmt = LoopIR.Reduce if self.use_accum_zero else LoopIR.Assign
            store_nest = [
                store_stmt(
                    self.buf_name, basetyp, None, store_widx, store_rhs, None, srcinfo
                )
            ]

            for i, n in reversed(list(zip(store_iter, shape))):
                loop = LoopIR.Seq(i, n, store_nest, None, srcinfo)
                store_nest = [loop]

        self.in_block = True
        block = self.map_stmts(block_c)
        self.in_block = False

        return new_alloc + load_nest + block + store_nest

    def map_s(self, sc):
        s = sc._node()
        new_s = super().map_s(sc)

        if self.in_block:
            if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
                if s.name is self.buf_name:
                    new_s = new_s[0] if new_s is not None else s
                    new_s = new_s.update(name=self.new_name)
                    idx = self.rewrite_idx(new_s.idx)
                    new_s = new_s.update(idx=idx)
                    return new_s

        return new_s

    def map_e(self, e):
        new_e = super().map_e(e)

        if self.in_block:
            if isinstance(e, LoopIR.Read):
                if e.name is self.buf_name:
                    new_e = new_e or e
                    new_e = new_e.update(name=self.new_name)

                    idx = self.rewrite_idx(new_e.idx)
                    return new_e.update(idx=idx)

            elif isinstance(e, LoopIR.WindowExpr):
                if e.name is self.buf_name:
                    new_e = new_e or e
                    w_idx = self.rewrite_win(new_e.idx)
                    return new_e.update(
                        name=self.new_name,
                        idx=w_idx,
                        type=T.Window(
                            self.new_typ, e.type.as_tensor, self.new_name, w_idx
                        ),
                    )

        return new_e


class DoStageWindow(Cursor_Rewrite):
    def __init__(self, proc_cursor, new_name, memory, expr):
        # Inputs
        self.new_name = Sym(new_name)
        self.memory = memory
        self.target_expr = expr._node()

        # Visitor state
        self._found_expr = False
        self._complete = False
        self._copy_code = None

        super().__init__(proc_cursor)
        Check_Aliasing(self.proc)

        self.proc = InferEffects(self.proc).result()

    def _stmt_writes_to_window(self, s):
        for eff in s.eff.reduces + s.eff.writes:
            if self.target_expr.name == eff.buffer:
                return True
        return False

    def _make_staged_alloc(self):
        """
        proc(Win[0:10, N, lo:hi])
        =>
        Staged : ty[10, hi - lo]
        for i0 in par(0, 10):
          for i1 in par(0, hi - lo):
            Staged[i0, i1] = Buf[0 + i0, N, lo + i1]
        proc(Staged[0:10, 0:(hi - lo)])
        """

        staged_extents = []  # e.g. 10, hi - lo
        staged_vars = []  # e.g. i0, i1
        staged_var_reads = []  # reads of staged_vars

        buf_points = []  # e.g. 0 + i0, N, lo + i1

        for idx in self.target_expr.idx:
            assert isinstance(idx, (LoopIR.Interval, LoopIR.Point))

            if isinstance(idx, LoopIR.Interval):
                assert isinstance(idx.hi.type, (T.Index, T.Size)), f"{idx.hi.type}"

                sym_i = Sym(f"i{len(staged_vars)}")
                staged_vars.append(sym_i)
                staged_extents.append(
                    LoopIR.BinOp("-", idx.hi, idx.lo, T.index, idx.srcinfo)
                )
                offset = LoopIR.Read(sym_i, [], T.index, idx.lo.srcinfo)
                buf_points.append(
                    LoopIR.BinOp("+", idx.lo, offset, T.index, idx.srcinfo)
                )
                staged_var_reads.append(LoopIR.Read(sym_i, [], T.index, idx.lo.srcinfo))
            elif isinstance(idx, LoopIR.Point):
                # TODO: test me!
                buf_points.append(idx.pt)

        assert staged_vars, "Window expression had no intervals"
        assert len(staged_vars) == len(staged_extents)

        # Staged : ty[10, hi - lo]
        srcinfo = self.target_expr.srcinfo
        data_type = self.target_expr.type.src_type.type
        alloc_type = T.Tensor(staged_extents, False, data_type)
        alloc = LoopIR.Alloc(self.new_name, alloc_type, self.memory, None, srcinfo)

        # Staged[i0, i1] = Buf[0 + i0, N, lo + i1]
        copy_stmt = LoopIR.Assign(
            self.new_name,
            data_type,
            None,
            staged_var_reads,
            LoopIR.Read(self.target_expr.name, buf_points, data_type, srcinfo),
            None,
            srcinfo,
        )

        # for i0 in par(0, 10):
        #     for i1 in par(0, hi - lo):
        for sym_i, extent_i in reversed(list(zip(staged_vars, staged_extents))):
            copy_stmt = LoopIR.Seq(sym_i, extent_i, [copy_stmt], None, srcinfo)

        # Staged[0:10, 0:(hi - lo)]
        w_extents = [
            LoopIR.Interval(LoopIR.Const(0, T.index, srcinfo), hi, srcinfo)
            for hi in staged_extents
        ]
        new_window = LoopIR.WindowExpr(
            self.new_name,
            w_extents,
            T.Window(data_type, alloc_type, self.new_name, w_extents),
            srcinfo,
        )

        return [alloc, copy_stmt], new_window

    def map_stmts(self, stmts):
        result = []

        for s in stmts:
            # TODO: be smarter about None here
            s = self.apply_s(s)

            if self._found_expr and not self._complete:
                assert len(s) == 1
                assert self._copy_code
                s = s[0]

                if self._stmt_writes_to_window(s):
                    raise NotImplementedError(
                        "StageWindow does not handle " "writes yet."
                    )
                s = self._copy_code + [s]
                self._complete = True

            result.extend(s)

        return result

    def map_e(self, e):
        if self._found_expr:
            return None

        if e is self.target_expr:
            self._found_expr = True
            self._copy_code, new_window = self._make_staged_alloc()
            return new_window

        return super().map_e(e)


class DoBoundAlloc(Cursor_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, bounds):
        self.alloc_site = alloc_cursor._node()
        self.bounds = bounds
        super().__init__(proc_cursor)

    def map_stmts(self, stmts_c):
        new_stmts = []
        for i, sc in enumerate(stmts_c):
            s = sc._node()
            if s is self.alloc_site:
                assert isinstance(s.type, T.Tensor)
                if len(self.bounds) != len(s.type.hi):
                    raise SchedulingError(
                        f"bound_alloc: dimensions do not match: "
                        f"{len(self.bounds)} != {len(s.type.hi)} (expected)"
                    )

                new_bounds = [
                    new if new else old for old, new in zip(s.type.hi, self.bounds)
                ]
                newtyp = T.Tensor(new_bounds, s.type.is_window, s.type.type)

                # TODO: CHECK THE BOUNDS OF ACCESSES IN stmts[i+1:] here

                s = LoopIR.Alloc(s.name, newtyp, s.mem, s.eff, s.srcinfo)
                return new_stmts + [s] + [s._node() for s in stmts_c[i + 1 :]]
            else:
                new_stmts += self.map_s(sc) or [s]

        return new_stmts


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Passes to export

__all__ = [
    "DoSplit",
    "DoUnroll",
    "DoInline",
    "DoPartialEval",
    "DoSetTypAndMem",
    "DoCallSwap",
    "DoBindExpr",
    "DoBindConfig",
    "DoLiftAlloc",
    "DoFissionLoops",
    "DoExtractMethod",
    "DoReorderStmt",
    "DoConfigWrite",
    "DoInlineWindow",
    "DoInsertPass",
    "DoDeletePass",
    "DoSimplify",
    "DoBoundAndGuard",
    "DoFuseLoop",
    "DoAddLoop",
    "DoDataReuse",
    "DoLiftScope",
    "DoPartitionLoop",
    "DoAssertIf",
    "DoSpecialize",
    "DoAddUnsafeGuard",
    "DoDeleteConfig",
    "DoFuseIf",
    "DoStageMem",
    "DoStageWindow",
    "DoBoundAlloc",
    "DoExpandDim",
    "DoRearrangeDim",
    "DoDivideDim",
    "DoMultiplyDim",
    "DoRemoveLoop",
    "DoLiftAllocSimple",
    "DoFissionAfterSimple",
    "DoProductLoop",
    "DoCommuteExpr",
    "DoMergeWrites",
]
