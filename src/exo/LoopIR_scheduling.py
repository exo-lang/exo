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
)
from .prelude import *
from .proc_eqv import get_strictest_eqv_proc


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

    pattern = f"for {name} in _:\n" f"  for {inner} in _: _{count}"
    return pattern


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reorder scheduling directive


# Take a conservative approach and allow stmt reordering only when they are
# writing to different buffers
# TODO: Do effectcheck's check_commutes-ish thing using SMT here
class _DoReorderStmt(LoopIR_Rewrite):
    def __init__(self, proc_cursor, f_cursor, s_cursor):
        proc = proc_cursor._node()
        self.f_stmt = f_cursor._node()
        self.s_stmt = s_cursor._node()
        # self.found_first = False

        # raise NotImplementedError("HIT REORDER STMTS")

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        for i, (s1, s2) in enumerate(zip(stmts, stmts[1:])):
            if s1 is self.f_stmt:
                if s2 is self.s_stmt:
                    Check_ReorderStmts(self.orig_proc, s1, s2)
                    return stmts[:i] + [s2, s1] + stmts[i + 2 :]

                raise SchedulingError(
                    "expected the second statement to be directly after the first"
                )

        return super().map_stmts(stmts)


class _PartitionLoop(LoopIR_Rewrite):
    def __init__(self, proc_cursor, loop_cursor, num):
        self.stmt = loop_cursor._node()
        self.partition_by = num
        self.second = False
        self.second_iter = None

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            assert isinstance(s, LoopIR.Seq)

            if not isinstance(s.hi, LoopIR.Const):
                raise SchedulingError("expected loop bound to be constant")

            if s.hi.val <= self.partition_by:
                raise SchedulingError(
                    "expected loop bound to be larger than partitioning value"
                )

            body = self.apply_stmts(s.body)
            first_loop = s.update(
                hi=LoopIR.Const(self.partition_by, T.int, s.srcinfo),
                body=body,
                eff=None,
            )

            # Should add partition_by to everything in body
            self.second = True

            new_iter = s.iter.copy()

            self.second_iter = new_iter

            second_body = SubstArgs(
                body, {s.iter: LoopIR.Read(new_iter, [], T.index, s.srcinfo)}
            ).result()

            second_loop = s.update(
                iter=new_iter,
                hi=LoopIR.Const(s.hi.val - self.partition_by, T.int, s.srcinfo),
                body=self.apply_stmts(second_body),
                eff=None,
            )

            return [first_loop] + [second_loop]

        return super().map_s(s)

    def map_e(self, e):
        if self.second:
            if isinstance(e, LoopIR.Read) and e.name == self.second_iter:
                assert e.type.is_indexable()
                return LoopIR.BinOp(
                    "+",
                    e,
                    LoopIR.Const(self.partition_by, T.int, e.srcinfo),
                    T.index,
                    e.srcinfo,
                )

        return super().map_e(e)


class _DoProductLoop(LoopIR_Rewrite):
    def __init__(self, proc_cursor, loop_cursor, new_name):
        self.stmt = loop_cursor._node()
        self.out_loop = self.stmt
        self.in_loop = self.out_loop.body[0]

        if len(self.out_loop.body) != 1 or not isinstance(self.in_loop, LoopIR.Seq):
            raise SchedulingError(
                f"expected loop directly inside of " f"{self.out_loop.iter} loop"
            )

        if not isinstance(self.in_loop.hi, LoopIR.Const):
            raise SchedulingError(
                f"expected the inner loop to have a constant bound, "
                f"got {self.in_loop.hi}."
            )
        self.inside = False
        self.new_var = Sym(new_name)

        super().__init__(proc_cursor._node())

    def map_s(self, s):
        styp = type(s)
        if s is self.stmt:
            self.inside = True
            body = self.map_stmts(s.body[0].body)
            self.inside = False
            new_hi = LoopIR.BinOp(
                "*", self.out_loop.hi, self.in_loop.hi, T.index, s.srcinfo
            )

            return [s.update(iter=self.new_var, hi=new_hi, body=body)]

        return super().map_s(s)

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


class _DoMergeWrites(LoopIR_Rewrite):
    def __init__(self, proc_cursor, f_cursor, s_cursor):
        self.proc = proc_cursor._node()
        self.s1 = f_cursor._node()
        self.s2 = s_cursor._node()

        try:
            assert len(self.s1.idx) == len(self.s2.idx)
            for i, j in zip(self.s1.idx, self.s2.idx):
                Check_ExprEqvInContext(self.proc, [self.s1, self.s2], i, j)
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

        super().__init__(self.proc)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        if isinstance(self.s2, LoopIR.Assign):
            for i, s in enumerate(stmts):
                if s is self.s2:
                    return stmts[: i - 1] + stmts[i:]
        else:
            for i, s in enumerate(stmts):
                if s is self.s1:
                    return stmts[:i] + [s.update(rhs=self.new_rhs)] + stmts[i + 2 :]

        return super().map_stmts(stmts)


class _Reorder(LoopIR_Rewrite):
    def __init__(self, proc_cursor, loop_cursor):
        self.stmt = loop_cursor._node()
        self.out_var = self.stmt.iter
        if len(self.stmt.body) != 1 or not isinstance(self.stmt.body[0], LoopIR.Seq):
            raise SchedulingError(
                f"expected loop directly inside of " f"{self.out_var} loop"
            )
        self.in_var = self.stmt.body[0].iter

        super().__init__(proc_cursor._node())

    def map_s(self, s):
        styp = type(s)
        if s is self.stmt:
            Check_ReorderLoops(self.orig_proc, self.stmt)

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
                return eff_bind(x, eff, pred=cond)  # TODO: , config_pred=cond_nz)

            # this is the actual body inside both for-loops
            body = s.body[0].body
            body_eff = get_effect_of_stmts(body)
            # blah
            inner_eff = do_bind(s.iter, s.hi, body_eff)
            outer_eff = do_bind(s.body[0].iter, s.body[0].hi, inner_eff)
            return [
                styp(
                    s.body[0].iter,
                    s.body[0].hi,
                    [styp(s.iter, s.hi, body, inner_eff, s.srcinfo)],
                    outer_eff,
                    s.body[0].srcinfo,
                )
            ]

        # fall-through
        return super().map_s(s)

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
# Split scheduling directive


class _Split(LoopIR_Rewrite):
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

        super().__init__(proc_cursor._node())

    def substitute(self, srcinfo):
        cnst = lambda x: LoopIR.Const(x, T.int, srcinfo)
        rd = lambda x: LoopIR.Read(x, [], T.index, srcinfo)
        op = lambda op, lhs, rhs: LoopIR.BinOp(op, lhs, rhs, T.index, srcinfo)

        return op("+", op("*", cnst(self.quot), rd(self.hi_i)), rd(self.lo_i))

    def cut_tail_sub(self, srcinfo):
        return self._cut_tail_sub

    def map_s(self, s):
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
                body = self.map_stmts(s.body)
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

                main_body = self.map_stmts(s.body)
                self._in_cut_tail = True
                tail_body = Alpha_Rename(self.map_stmts(s.body)).result()
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
                body = self.map_stmts(s.body)
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
        return super().map_s(s)

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


class _Unroll(LoopIR_Rewrite):
    def __init__(self, proc_cursor, loop_cursor):
        self.unroll_loop = loop_cursor._node()
        self.unroll_var = self.unroll_loop.iter
        self.unroll_itr = 0
        self.env = {}

        super().__init__(proc_cursor._node())

    def map_s(self, s):
        if s is self.unroll_loop:
            if not isinstance(s.hi, LoopIR.Const):
                raise SchedulingError(
                    f"expected loop '{s.iter}' to have constant bounds"
                )

            hi = s.hi.val
            if hi == 0:
                return []

            orig_body = s.body

            self.unroll_itr = 0

            body = Alpha_Rename(self.apply_stmts(orig_body)).result()
            for i in range(1, hi):
                self.unroll_itr = i
                body += Alpha_Rename(self.apply_stmts(orig_body)).result()

            return body

        # fall-through
        return super().map_s(s)

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


class _Inline(LoopIR_Rewrite):
    def __init__(self, proc_cursor, call_cursor):
        self.call_stmt = call_cursor._node()
        assert isinstance(self.call_stmt, LoopIR.Call)
        self.env = {}

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
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
        return super().map_s(s)

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


class _PartialEval(LoopIR_Rewrite):
    def __init__(self, proc, arg_vals):
        assert arg_vals, "Don't call _PartialEval without any substitutions"
        self.env = arg_vals
        self.proc = proc
        arg_types = {p.name: p.type for p in self.proc.args}

        # Validate env:
        for k, v in self.env.items():
            if not arg_types[k].is_indexable() and not arg_types[k].is_bool():
                raise SchedulingError(
                    "cannot partially evaluate numeric (non-index, non-bool) arguments"
                )
            if not isinstance(v, int):
                raise SchedulingError(
                    "cannot partially evaluate to a non-int, non-bool value"
                )

        super().__init__(self.proc)
        self.proc = self.proc.update(
            args=[a for a in self.proc.args if a.name not in arg_vals]
        )

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
class _SetTypAndMem(LoopIR_Rewrite):
    def __init__(self, proc_cursor, name, inst_no, basetyp=None, win=None, mem=None):
        ind = lambda x: 1 if x else 0
        assert ind(basetyp) + ind(win) + ind(mem) == 1
        self.name = name
        self.n_match = inst_no
        self.basetyp = basetyp
        self.win = win
        self.mem = mem

        super().__init__(proc_cursor._node())

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

    def map_s(self, s):
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
        return super().map_s(s)

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


class _CallSwap(LoopIR_Rewrite):
    def __init__(self, proc_cursor, call_cursor, new_subproc):
        self.call_stmt = call_cursor._node()
        assert isinstance(self.call_stmt, LoopIR.Call)
        self.new_subproc = new_subproc

        super().__init__(proc_cursor._node())

    def mod_eq(self):
        return self.eq_mod_config

    def map_s(self, s):
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
            mod_cfg = Check_ExtendEqv(self.orig_proc, [s], [s_new], configkeys)
            self.eq_mod_config = mod_cfg

            return [s_new]

        # fall-through
        return super().map_s(s)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self, e):
        return e

    def map_t(self, t):
        return t

    def map_eff(self, eff):
        return eff


class _InlineWindow(LoopIR_Rewrite):
    def __init__(self, proc_cursor, window_cursor):
        self.win_stmt = window_cursor._node()
        assert isinstance(self.win_stmt, LoopIR.WindowStmt)

        super().__init__(proc_cursor._node())

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def calc_idx(self, idxs):
        assert len(
            [w for w in self.win_stmt.rhs.idx if isinstance(w, LoopIR.Interval)]
        ) == len(idxs)

        new_idxs = []
        for w in self.win_stmt.rhs.idx:
            if isinstance(w, LoopIR.Interval):
                new_idxs.append(LoopIR.BinOp("+", w.lo, idxs[0], T.index, w.srcinfo))
                idxs.pop()
            else:
                new_idxs.append(w.pt)

        return new_idxs

    def map_s(self, s):
        if s is self.win_stmt:
            return []

        if isinstance(s, LoopIR.Assign) or isinstance(s, LoopIR.Reduce):
            if self.win_stmt.lhs == s.name:
                new_idxs = self.calc_idx(s.idx)

                return [
                    type(s)(
                        self.win_stmt.rhs.name,
                        s.type,
                        s.cast,
                        new_idxs,
                        s.rhs,
                        None,
                        s.srcinfo,
                    )
                ]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        assert isinstance(self.win_stmt.rhs, LoopIR.WindowExpr)

        # TODO: Add more safety check?
        if etyp is LoopIR.WindowExpr:
            if self.win_stmt.lhs == e.name:
                assert len(
                    [w for w in self.win_stmt.rhs.idx if isinstance(w, LoopIR.Interval)]
                ) == len(e.idx)
                idxs = e.idx
                new_idxs = []
                for w in self.win_stmt.rhs.idx:
                    if isinstance(w, LoopIR.Interval):
                        if isinstance(idxs[0], LoopIR.Interval):
                            # window again, so
                            # w.lo + idxs[0].lo : w.lo + idxs[0].hi
                            lo = LoopIR.BinOp("+", w.lo, idxs[0].lo, T.index, w.srcinfo)
                            hi = LoopIR.BinOp("+", w.lo, idxs[0].hi, T.index, w.srcinfo)
                            ivl = LoopIR.Interval(lo, hi, w.srcinfo)
                            new_idxs.append(ivl)
                        else:  # Point
                            p = LoopIR.Point(
                                LoopIR.BinOp("+", w.lo, idxs[0].pt, T.index, w.srcinfo),
                                w.srcinfo,
                            )
                            new_idxs.append(p)
                        idxs = idxs[1:]
                    else:
                        new_idxs.append(w)

                # repair window type..
                old_typ = self.win_stmt.rhs.type
                new_type = LoopIR.WindowType(
                    old_typ.src_type,
                    old_typ.as_tensor,
                    self.win_stmt.rhs.name,
                    new_idxs,
                )

                return LoopIR.WindowExpr(
                    self.win_stmt.rhs.name, new_idxs, new_type, e.srcinfo
                )

        elif etyp is LoopIR.Read:
            if self.win_stmt.lhs == e.name:
                new_idxs = self.calc_idx(e.idx)

                return LoopIR.Read(self.win_stmt.rhs.name, new_idxs, e.type, e.srcinfo)

        elif etyp is LoopIR.StrideExpr:
            if self.win_stmt.lhs == e.name:
                return LoopIR.StrideExpr(
                    self.win_stmt.rhs.name, e.dim, e.type, e.srcinfo
                )

        return super().map_e(e)


# TODO: Rewrite this to directly use stmt_cursor instead of after
class _ConfigWrite(LoopIR_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, config, field, expr, before=False):
        assert (
            isinstance(expr, LoopIR.Read)
            or isinstance(expr, LoopIR.StrideExpr)
            or isinstance(expr, LoopIR.Const)
        )

        self.stmt = stmt_cursor._node()
        self.config = config
        self.field = field
        self.expr = expr
        self.before = before

        self._new_cfgwrite_stmt = None

        super().__init__(proc_cursor._node())

        # check safety...
        mod_cfg = Check_DeleteConfigWrite(self.proc, [self._new_cfgwrite_stmt])
        self.eq_mod_config = mod_cfg

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def mod_eq(self):
        return self.eq_mod_config

    def map_stmts(self, stmts):
        body = []
        for i, s in enumerate(stmts):
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
                body += stmts[i + 1 :]
                return body

            else:
                # TODO: be smarter about None handling
                body += self.apply_s(s)

        return body


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Bind Expression scheduling directive


class _BindConfig_AnalysisSubst(LoopIR_Rewrite):
    def __init__(self, proc, keep_s, old_e, new_e):
        self.keep_s = keep_s
        self.old_e = old_e
        self.new_e = new_e
        super().__init__(proc)

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


class _BindConfig(LoopIR_Rewrite):
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

        super().__init__(proc_cursor._node())

        proc_analysis = _BindConfig_AnalysisSubst(
            self.proc, self.cfg_write_s, self.cfg_read_e, self.expr
        ).result()
        mod_cfg = Check_DeleteConfigWrite(proc_analysis, [self.cfg_write_s])
        self.eq_mod_config = mod_cfg

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def mod_eq(self):
        return self.eq_mod_config

    def process_block(self, block):
        if self.sub_done:
            return None

        new_block = []
        is_writeconfig_block = False

        modified = False

        for stmt in block:
            new_stmt = self.map_s(stmt)

            if self.found_expr and not self.placed_writeconfig:
                self.placed_writeconfig = True
                is_writeconfig_block = True
                wc = LoopIR.WriteConfig(
                    self.config, self.field, self.expr, None, self.expr.srcinfo
                )
                self.cfg_write_s = wc
                new_block.extend([wc])

            if new_stmt is None:
                new_block.append(stmt)
            else:
                new_block.extend(new_stmt)
                modified = True

        if is_writeconfig_block:
            self.sub_done = True

        if not modified:
            return None

        return new_block

    def map_s(self, s):
        if self.sub_done:
            return None  # TODO: is this right?

        # TODO: missing cases for multiple config writes. Subsequent writes are
        #   ignored.

        if isinstance(s, LoopIR.Seq):
            body = self.process_block(s.body)
            if body:
                return [s.update(body=body)]
            return None

        if isinstance(s, LoopIR.If):
            if_then = self.process_block(s.body)
            if_else = self.process_block(s.orelse)
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

        return super().map_s(s)

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


class _DoCommuteExpr(LoopIR_Rewrite):
    def __init__(self, proc_cursor, expr_cursors):
        self.exprs = [e._node() for e in expr_cursors]
        super().__init__(proc_cursor._node())
        self.proc = InferEffects(self.proc).result()

    def map_e(self, e):
        if e in self.exprs:
            assert isinstance(e, LoopIR.BinOp)
            return e.update(lhs=e.rhs, rhs=e.lhs)
        else:
            return super().map_e(e)


class _BindExpr(LoopIR_Rewrite):
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

        super().__init__(proc_cursor._node())

        # repair effects...
        self.proc = InferEffects(self.proc).result()

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
                stmt = [_stmt]

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

    def map_s(self, s):
        if self.found_write:
            return None

        if self.sub_done:
            return super().map_s(s)

        if isinstance(s, LoopIR.Seq):
            body = self.process_block(s.body)
            if body is None:
                return None
            else:
                return [s.update(body=body)]

        if isinstance(s, LoopIR.If):
            # TODO: our CSE here is very conservative. It won't look for
            #  matches between the then and else branches; in other words,
            #  it is restricted to a single basic block.
            if_then = self.process_block(s.body)
            if_else = self.process_block(s.orelse)
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

        return super().map_s(s)

    def map_e(self, e):
        if e in self.exprs and not self.sub_done:
            if not self.found_expr:
                # TODO: dirty hack. need real CSE-equality (i.e. modulo srcinfo)
                self.exprs = [x for x in self.exprs if str(e) == str(x)]
            self.found_expr = e
            return LoopIR.Read(self.new_name, [], e.type, e.srcinfo)
        else:
            return super().map_e(e)


class _DoStageAssn(LoopIR_Rewrite):
    def __init__(self, proc_cursor, new_name, assn_cursor):
        self.assn = assn_cursor._node()
        assert isinstance(self.assn, (LoopIR.Assign, LoopIR.Reduce))
        self.new_name = Sym(new_name)

        super().__init__(proc_cursor._node())

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        tmp = self.new_name
        if s is self.assn and isinstance(s, LoopIR.Assign):
            rdtmp = LoopIR.Read(tmp, [], s.type, s.srcinfo)
            return [
                # tmp : R
                LoopIR.Alloc(tmp, T.R, None, None, s.srcinfo),
                # tmp = rhs
                LoopIR.Assign(tmp, s.type, None, [], s.rhs, None, s.srcinfo),
                # lhs = tmp
                LoopIR.Assign(s.name, s.type, None, s.idx, rdtmp, None, s.srcinfo),
            ]
        elif s is self.assn and isinstance(s, LoopIR.Reduce):
            rdbuf = LoopIR.Read(s.name, s.idx, s.type, s.srcinfo)
            rdtmp = LoopIR.Read(tmp, [], s.type, s.srcinfo)
            return [
                # tmp : R
                LoopIR.Alloc(tmp, T.R, None, None, s.srcinfo),
                # tmp = lhs
                LoopIR.Assign(tmp, s.type, None, [], rdbuf, None, s.srcinfo),
                # tmp += rhs
                LoopIR.Reduce(tmp, s.type, None, [], s.rhs, None, s.srcinfo),
                # lhs = tmp
                LoopIR.Assign(s.name, s.type, None, s.idx, rdtmp, None, s.srcinfo),
            ]

        return super().map_s(s)


# Lift if no variable dependency
class _DoLiftIf(LoopIR_Rewrite):
    def __init__(self, proc_cursor, if_cursor, n_lifts):
        self.target = if_cursor._node()

        assert isinstance(self.target, LoopIR.If)
        assert is_pos_int(n_lifts)

        self.loop_deps = vars_in_expr(self.target.cond)

        self.n_lifts = n_lifts
        self.bubbling = False

        super().__init__(proc_cursor._node())

        if self.n_lifts:
            raise SchedulingError(
                f"Could not fully lift if statement! {self.n_lifts} lift(s) remain!",
                orig=self.orig_proc,
                proc=self.proc,
            )

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def resolve_lift(self, new_if):
        self.target = new_if
        self.n_lifts -= 1
        return [new_if]

    def map_s(self, s):
        if s is self.target:
            self.bubbling = True
            # Matching happens above this, no changes possible
            return None

        if not isinstance(s, (LoopIR.If, LoopIR.Seq)):
            # Only ifs and loops can be interchanged
            return None

        s2 = super().map_s(s)

        if self.n_lifts <= 0:
            # No lifts left, bubble up
            return s2

        outer = s2[0] if s2 else s

        if isinstance(outer, LoopIR.If):
            if len(outer.body) == 1 and outer.body[0] is self.target:
                #                    if INNER:
                # if OUTER:            if OUTER: A
                #   if INNER: A        else:     C
                #   else:     B  ~>  else:
                # else: C              if OUTER: B
                #                      else:     C
                stmt_a = self.target.body
                stmt_b = self.target.orelse
                stmt_c = outer.orelse

                if_ac = [s.update(body=stmt_a, orelse=stmt_c)]
                if stmt_b or stmt_c:
                    stmt_b = stmt_b or [LoopIR.Pass(None, self.target.srcinfo)]
                    if_bc = [s.update(body=stmt_b, orelse=stmt_c)]
                else:
                    if_bc = []

                new_if = self.target.update(body=if_ac, orelse=if_bc)
                return self.resolve_lift(new_if)

            if len(outer.orelse) == 1 and outer.orelse[0] is self.target:
                #                    if INNER:
                # if OUTER: A          if OUTER: A
                # else:                else:     B
                #   if INNER: B  ~>  else:
                #   else: C            if OUTER: A
                #                      else:     C
                stmt_a = outer.body
                stmt_b = self.target.body
                stmt_c = self.target.orelse

                if_ab = [s.update(body=stmt_a, orelse=stmt_b)]
                if_ac = [s.update(body=stmt_a, orelse=stmt_c)]

                new_if = self.target.update(body=if_ab, orelse=if_ac)
                return self.resolve_lift(new_if)

        if isinstance(s, LoopIR.Seq):
            if len(outer.body) == 1 and outer.body[0] is self.target:
                if s.iter in self.loop_deps:
                    raise SchedulingError("if statement depends on iteration variable")

                # for OUTER in _:      if INNER:
                #   if INNER: A    ~>    for OUTER in _: A
                #   else:     B        else:
                #                        for OUTER in _: B
                stmt_a = self.target.body
                stmt_b = self.target.orelse

                for_a = [s.update(body=stmt_a)]
                for_b = [s.update(body=stmt_b)] if stmt_b else []

                new_if = self.target.update(body=for_a, orelse=for_b)
                return self.resolve_lift(new_if)

        if self.bubbling:
            raise SchedulingError(
                "expected if statement to be directly nested in parents"
            )

        return s2

    def map_e(self, e):
        return None


class _DoExpandDim(LoopIR_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, alloc_dim, indexing):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert isinstance(alloc_dim, LoopIR.expr)
        assert isinstance(indexing, LoopIR.expr)

        self.alloc_sym = self.alloc_stmt.name
        self.alloc_dim = alloc_dim
        self.indexing = indexing
        self.alloc_type = None

        super().__init__(proc_cursor._node())

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.alloc_stmt:
            old_typ = s.type
            new_rngs = [self.alloc_dim]

            if isinstance(old_typ, T.Tensor):
                new_rngs += old_typ.shape()

            basetyp = old_typ.basetype()
            new_typ = T.Tensor(new_rngs, False, basetyp)
            self.alloc_type = new_typ

            return [LoopIR.Alloc(s.name, new_typ, s.mem, None, s.srcinfo)]

        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)) and s.name == self.alloc_sym:
            idx = [self.indexing] + self.apply_exprs(s.idx)
            rhs = self.apply_e(s.rhs)
            return [s.update(idx=idx, rhs=rhs, eff=None)]

        return super().map_s(s)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.alloc_sym:
            return e.update(idx=[self.indexing] + self.apply_exprs(e.idx))

        if isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            w_idx = self._map_list(self.map_w_access, e.idx) or e.idx
            idx = [LoopIR.Point(self.indexing, e.srcinfo)] + w_idx
            return e.update(
                idx=idx, type=T.Window(self.alloc_type, e.type.as_tensor, e.name, idx)
            )

        # fall-through
        return super().map_e(e)


class _DoRearrangeDim(LoopIR_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, dimensions):
        self.alloc_stmt = alloc_cursor._node()
        assert isinstance(self.alloc_stmt, LoopIR.Alloc)

        self.dimensions = dimensions

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        # simply change the dimension
        if s is self.alloc_stmt:
            # construct new_hi
            new_hi = [s.type.hi[i] for i in self.dimensions]
            # construct new_type
            new_type = LoopIR.Tensor(new_hi, s.type.is_window, s.type.type)

            return [LoopIR.Alloc(s.name, new_type, s.mem, None, s.srcinfo)]

        # Adjust the use-site
        if isinstance(s, LoopIR.Assign) or isinstance(s, LoopIR.Reduce):
            if s.name is self.alloc_stmt.name:
                # shuffle
                new_idx = [s.idx[i] for i in self.dimensions]
                return [
                    type(s)(s.name, s.type, s.cast, new_idx, s.rhs, None, s.srcinfo)
                ]

        return super().map_s(s)

    def map_e(self, e):
        # TODO: I am not sure what rearrange_dim should do in terms of StrideExpr
        if isinstance(e, LoopIR.Read) or isinstance(e, LoopIR.WindowExpr):
            if e.name is self.alloc_stmt.name:
                new_idx = [e.idx[i] for i in self.dimensions]
                return type(e)(e.name, new_idx, e.type, e.srcinfo)

        return super().map_e(e)


class _DoDivideDim(LoopIR_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, dim_idx, quotient):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert isinstance(dim_idx, int)
        assert isinstance(quotient, int)

        self.alloc_sym = self.alloc_stmt.name
        self.dim_idx = dim_idx
        self.quotient = quotient

        super().__init__(proc_cursor._node())

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def remap_idx(self, idx):
        orig_i = idx[self.dim_idx]
        srcinfo = orig_i.srcinfo
        quot = LoopIR.Const(self.quotient, T.int, srcinfo)
        hi = LoopIR.BinOp("/", orig_i, quot, orig_i.type, srcinfo)
        lo = LoopIR.BinOp("%", orig_i, quot, orig_i.type, srcinfo)
        return idx[: self.dim_idx] + [hi, lo] + idx[self.dim_idx + 1 :]

    def map_s(self, s):
        if s is self.alloc_stmt:
            old_typ = s.type
            old_shp = old_typ.shape()
            dim = old_shp[self.dim_idx]

            if not isinstance(dim, LoopIR.Const):
                raise SchedulingError(
                    f"Cannot divide non-literal dimension: " f"{str(dim)}"
                )
            if not dim.val % self.quotient == 0:
                raise SchedulingError(
                    f"Cannot divide {dim.val} evenly by " f"{self.quotient}"
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

        return super().map_s(s)

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


class _DoMultiplyDim(LoopIR_Rewrite):
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
                f"Cannot multiply with non-literal second " f"dimension: {str(lo_dim)}"
            )
        self.lo_val = lo_dim.val

        super().__init__(proc_cursor._node())

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

    def map_s(self, s):
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

        return super().map_s(s)

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


class _DoLiftAllocSimple(LoopIR_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, n_lifts):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert is_pos_int(n_lifts)

        self.n_lifts = n_lifts
        self.ctrl_ctxt = []
        self.lift_site = None

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.alloc_stmt:
            if self.n_lifts > len(self.ctrl_ctxt):
                raise SchedulingError(
                    "specified lift level {self.n_lifts} "
                    + "is higher than the number of loop "
                    + "{len(self.ctrl_ctxt)}"
                )
            self.lift_site = self.ctrl_ctxt[-self.n_lifts]

            return []

        elif isinstance(s, (LoopIR.If, LoopIR.Seq)):
            self.ctrl_ctxt.append(s)
            stmts = super().map_s(s)
            self.ctrl_ctxt.pop()

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

        return super().map_s(s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Lift Allocation scheduling directive

# TODO: Implement autolift_alloc's logic using high-level scheduling metaprogramming and
#       delete this code
class _LiftAlloc(LoopIR_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, n_lifts, mode, size, keep_dims):
        self.alloc_stmt = alloc_cursor._node()

        assert isinstance(self.alloc_stmt, LoopIR.Alloc)
        assert is_pos_int(n_lifts)

        if mode not in ("row", "col"):
            raise SchedulingError(f"Unknown lift mode {mode}, should be 'row' or 'col'")

        self.orig_proc = proc_cursor._node()
        self.alloc_sym = self.alloc_stmt.name
        self.alloc_deps = LoopIR_Dependencies(
            self.alloc_sym, self.orig_proc.body
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

        super().__init__(self.orig_proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def idx_mode(self, access, orig):
        if self.lift_mode == "row":
            return access + orig
        elif self.lift_mode == "col":
            return orig + access
        assert False

    def map_s(self, s):
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
            stmts = super().map_s(s)
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
            stmts = super().map_s(s)
            self._in_call_arg = False
            return stmts

        # fall-through
        return super().map_s(s)

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


class _VarsInExpr(LoopIR_Do):
    def __init__(self, expr):
        assert isinstance(expr, LoopIR.expr)

        self.vars = set()
        self.do_e(expr)

    def result(self):
        return self.vars

    def do_e(self, e):
        if isinstance(e, LoopIR.Read):
            self.vars.add(e.name)

        super().do_e(e)


def vars_in_expr(expr):
    return _VarsInExpr(expr).result()


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


class _DoDoubleFission:
    def __init__(self, proc_cursor, f_cursor, s_cursor, n_lifts):
        self.tgt_stmt1 = f_cursor._node()
        self.tgt_stmt2 = s_cursor._node()

        assert isinstance(self.tgt_stmt1, LoopIR.stmt)
        assert isinstance(self.tgt_stmt2, LoopIR.stmt)
        assert is_pos_int(n_lifts)
        self.orig_proc = proc_cursor._node()
        self.n_lifts = n_lifts

        self.hit_fission1 = False
        self.hit_fission2 = False

        pre_body, mid_body, post_body = self.map_stmts(self.orig_proc.body)
        self.proc = LoopIR.proc(
            name=self.orig_proc.name,
            args=self.orig_proc.args,
            preds=self.orig_proc.preds,
            body=pre_body + mid_body + post_body,
            instr=None,
            eff=self.orig_proc.eff,
            srcinfo=self.orig_proc.srcinfo,
        )

        self.proc = InferEffects(self.proc).result()

    def result(self):
        return self.proc

    def alloc_check(self, pre, post):
        if not _is_alloc_free(pre, post):
            raise SchedulingError(
                "Will not fission here, because "
                "an allocation might be buried "
                "in a different scope than some use-site"
            )

    def map_stmts(self, stmts):
        pre_stmts = []
        mid_stmts = []
        post_stmts = []
        for orig_s in stmts:
            pre, mid, post = self.map_s(orig_s)
            pre_stmts += pre
            mid_stmts += mid
            post_stmts += post

        return pre_stmts, mid_stmts, post_stmts

    def map_s(self, s):
        if s is self.tgt_stmt1:
            self.hit_fission1 = True
            return [s], [], []
        elif s is self.tgt_stmt2:
            self.hit_fission2 = True
            return [], [s], []

        elif isinstance(s, LoopIR.If):

            # first, check if we need to split the body
            pre, mid, post = self.map_stmts(s.body)
            fission_body = (
                len(pre) > 0 and len(mid) > 0 and len(post) > 0 and self.n_lifts > 0
            )
            if fission_body:
                self.n_lifts -= 1
                self.alloc_check(pre, mid)
                self.alloc_check(mid, post)
                pre = LoopIR.If(s.cond, pre, [], None, s.srcinfo)
                mid = LoopIR.If(s.cond, mid, s.orelse, None, s.srcinfo)
                post = LoopIR.If(s.cond, post, [], None, s.srcinfo)
                return [pre], [mid], [post]

            body = pre + mid + post

            # if we don't, then check if we need to split the or-else
            pre, mid, post = self.map_stmts(s.orelse)
            fission_orelse = (
                len(pre) > 0 and len(post) > 0 and len(mid) > 0 and self.n_lifts > 0
            )
            if fission_orelse:
                self.n_lifts -= 1
                self.alloc_check(pre, mid)
                self.alloc_check(mid, post)
                pre = LoopIR.If(s.cond, [], pre, None, s.srcinfo)
                mid = LoopIR.If(s.cond, body, mid, None, s.srcinfo)
                post = LoopIR.If(s.cond, [], post, None, s.srcinfo)
                return [pre], [mid], [post]

            orelse = pre + mid + post

            # if we neither split the body nor the or-else,
            # then we need to gather together the pre and post.
            single_stmt = LoopIR.If(s.cond, body, orelse, None, s.srcinfo)

        elif isinstance(s, LoopIR.Seq):

            # check if we need to split the loop
            pre, mid, post = self.map_stmts(s.body)
            do_fission = (
                len(pre) > 0 and len(post) > 0 and len(mid) > 0 and self.n_lifts > 0
            )
            if do_fission:
                self.n_lifts -= 1
                self.alloc_check(pre, mid)
                self.alloc_check(mid, post)

                # we can skip the loop iteration if the
                # body doesn't depend on the loop
                # and the body is idempotent
                if s.iter in _FV(pre) or not _is_idempotent(pre):
                    pre = [s.update(body=pre, eff=None)]
                    # since we are copying the binding of s.iter,
                    # we should perform an Alpha_Rename for safety
                    pre = Alpha_Rename(pre).result()
                if s.iter in _FV(mid) or not _is_idempotent(mid):
                    mid = [s.update(body=mid, eff=None)]
                if s.iter in _FV(post) or not _is_idempotent(post):
                    post = [s.update(body=post, eff=None)]
                    post = Alpha_Rename(post).result()

                return pre, mid, post

            single_stmt = s.update(body=pre + mid + post, eff=None)

        else:
            # all other statements cannot recursively
            # contain statements, so...
            single_stmt = s

        if self.hit_fission1 and not self.hit_fission2:
            return [], [single_stmt], []
        elif self.hit_fission2:
            return [], [], [single_stmt]
        else:
            return [single_stmt], [], []


class _DoRemoveLoop(LoopIR_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor):
        self.stmt = stmt_cursor._node()
        assert isinstance(self.stmt, LoopIR.stmt)
        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            # Check if we can remove the loop
            # Conditions are:
            # 1. Body does not depend on the loop iteration variable
            # 2. Body is idempotent
            # 3. The loop runs at least once
            # TODO: (3) could be checked statically using something similar to the legacy is_pos_int.

            if s.iter not in _FV(s.body):
                if _is_idempotent(s.body):
                    zero = LoopIR.Const(0, T.int, s.srcinfo)
                    cond = LoopIR.BinOp(">", s.hi, zero, T.bool, s.srcinfo)
                    body = self.apply_stmts(s.body)
                    guard = LoopIR.If(cond, body, [], None, s.srcinfo)
                    # remove loop and alpha rename
                    return Alpha_Rename([guard]).result()
                else:
                    raise SchedulingError(
                        "Cannot remove loop, loop body is " "not idempotent"
                    )
            else:
                raise SchedulingError(
                    f"Cannot remove loop, {s.iter} is not " "free in the loop body."
                )

        return super().map_s(s)


# This is same as original FissionAfter, except that
# this does not remove loop. We have separate remove_loop
# operator for that purpose.
class _DoFissionAfterSimple:
    def __init__(self, proc_cursor, stmt_cursor, n_lifts):
        self.tgt_stmt = stmt_cursor._node()
        assert isinstance(self.tgt_stmt, LoopIR.stmt)
        assert is_pos_int(n_lifts)
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
        return self.proc

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
class _FissionLoops:
    def __init__(self, proc_cursor, stmt_cursor, n_lifts):
        self.tgt_stmt = stmt_cursor._node()
        assert isinstance(self.tgt_stmt, LoopIR.stmt)
        assert is_pos_int(n_lifts)
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
        return self.proc

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


class _DoAddUnsafeGuard(LoopIR_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, cond):
        self.stmt = stmt_cursor._node()
        self.cond = cond
        self.in_loop = False

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            # Check_ExprEqvInContext(self.orig_proc, [s],
            #                       self.cond,
            #                       LoopIR.Const(True, T.bool, s.srcinfo))
            s1 = Alpha_Rename([s]).result()
            return [LoopIR.If(self.cond, s1, [], None, s.srcinfo)]

        return super().map_s(s)


class _DoSpecialize(LoopIR_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, conds):
        assert conds, "Must add at least one condition"
        self.stmt = stmt_cursor._node()
        self.conds = conds

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            else_br = Alpha_Rename([s]).result()
            for cond in reversed(self.conds):
                then_br = Alpha_Rename([s]).result()
                else_br = [LoopIR.If(cond, then_br, else_br, None, s.srcinfo)]
            return else_br

        return super().map_s(s)


def _get_constant_bound(e):
    if isinstance(e, LoopIR.BinOp) and e.op == "%":
        return e.rhs
    raise SchedulingError(f"Could not derive constant bound on {e}")


class _DoBoundAndGuard(LoopIR_Rewrite):
    def __init__(self, proc_cursor, loop_cursor):
        self.loop = loop_cursor._node()
        super().__init__(proc_cursor._node())

    def map_s(self, s):
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

        return super().map_s(s)


class _DoFuseLoop(LoopIR_Rewrite):
    def __init__(self, proc_cursor, f_cursor, s_cursor):
        self.loop1 = f_cursor._node()
        self.loop2 = s_cursor._node()
        self.modified_stmts = None

        super().__init__(proc_cursor._node())

        loop, body1, body2 = self.modified_stmts
        Check_FissionLoop(self.proc, loop, body1, body2)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        new_stmts = []

        for i, b in enumerate(stmts):
            if b is self.loop1:
                if i + 1 >= len(stmts) or stmts[i + 1] is not self.loop2:
                    raise SchedulingError(
                        "expected the two loops to be "
                        "fused to come one right after the other"
                    )

                loop1, loop2 = self.loop1, self.loop2

                # check if the loop bounds are equivalent
                Check_ExprEqvInContext(
                    self.orig_proc, [loop1, loop2], loop1.hi, loop2.hi
                )

                x = loop1.iter
                y = loop2.iter
                hi = loop1.hi
                body1 = loop1.body
                body2 = SubstArgs(
                    loop2.body, {y: LoopIR.Read(x, [], T.index, loop1.srcinfo)}
                ).result()
                loop = type(loop1)(x, hi, body1 + body2, None, loop1.srcinfo)
                self.modified_stmts = (loop, body1, body2)

                return stmts[:i] + [loop] + stmts[i + 2 :]

        # if we reached this point, we didn't find the loop
        return super().map_stmts(stmts)


class _DoFuseIf(LoopIR_Rewrite):
    def __init__(self, proc_cursor, f_cursor, s_cursor):
        self.if1 = f_cursor._node()
        self.if2 = s_cursor._node()

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        new_stmts = []

        found_first = False
        for stmt in stmts:
            if stmt is self.if1:
                found_first = True
                continue

            if found_first:
                found_first = False  # Must have been set on previous iteration

                if stmt is not self.if2:
                    raise SchedulingError(
                        "expected the second stmt to be "
                        "directly after the first stmt"
                    )

                # Check that conditions are identical
                if self.if1.cond != self.if2.cond:
                    raise SchedulingError("expected conditions to match")

                stmt = LoopIR.If(
                    self.if1.cond,
                    self.if1.body + self.if2.body,
                    self.if1.orelse + self.if2.orelse,
                    None,
                    self.if1.srcinfo,
                )

            new_stmts.extend(self.map_s(stmt))

        return new_stmts


class _DoAddLoop(LoopIR_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, var, hi, guard):
        self.stmt = stmt_cursor._node()
        self.var = var
        self.hi = hi
        self.guard = guard

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            if not _is_idempotent([s]):
                raise SchedulingError("expected stmt to be idempotent!")

            sym = Sym(self.var)

            new_s = s
            if self.guard:
                cond = LoopIR.BinOp(
                    "==",
                    LoopIR.Read(sym, [], T.index, s.srcinfo),
                    LoopIR.Const(0, T.int, s.srcinfo),
                    T.bool,
                    s.srcinfo,
                )
                new_s = LoopIR.If(cond, [s], [], None, s.srcinfo)

            ir = LoopIR.Seq(sym, self.hi, [new_s], None, new_s.srcinfo)
            return [ir]

        return super().map_s(s)


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


class _DoInsertPass(LoopIR_Rewrite):
    def __init__(self, proc_cursor, stmt_cursor, before=True):
        self.stmt = stmt_cursor._node()
        self.before = before
        super().__init__(proc_cursor._node())

    def map_s(self, s):
        if s is self.stmt:
            pass_s = LoopIR.Pass(eff_null(s.srcinfo), srcinfo=s.srcinfo)
            if self.before:
                return [pass_s, s]
            else:
                return [s, pass_s]
        return super().map_s(s)


class _DoDeleteConfig(LoopIR_Rewrite):
    def __init__(self, proc_cursor, config_cursor):
        self.stmt = config_cursor._node()
        self.eq_mod_config = set()
        super().__init__(proc_cursor._node())

    def mod_eq(self):
        return self.eq_mod_config

    def map_s(self, s):
        if s is self.stmt:
            mod_cfg = Check_DeleteConfigWrite(self.orig_proc, [self.stmt])
            self.eq_mod_config = mod_cfg
            return []
        else:
            return super().map_s(s)


class _DoDeletePass(LoopIR_Rewrite):
    def __init__(self, proc_cursor):
        super().__init__(proc_cursor._node())

    def map_s(self, s):
        if isinstance(s, LoopIR.Pass):
            return []

        elif isinstance(s, LoopIR.Seq):
            body = self.map_stmts(s.body)
            if body is None:
                return None
            elif body == []:
                return []
            else:
                return [s.update(body=body)]

        return super().map_s(s)


class _DoExtractMethod(LoopIR_Rewrite):
    def __init__(self, proc_cursor, name, stmt_cursor):
        self.match_stmt = stmt_cursor._node()
        assert isinstance(self.match_stmt, LoopIR.stmt)
        self.sub_proc_name = name
        self.new_subproc = None
        self.orig_proc = proc_cursor._node()

        self.var_types = ChainMap()

        for a in self.orig_proc.args:
            self.var_types[a.name] = a.type

        super().__init__(self.orig_proc)

    def subproc(self):
        return self.new_subproc

    def push(self):
        self.var_types = self.var_types.new_child()

    def pop(self):
        self.var_types = self.var_types.parents

    def map_s(self, s):
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
            body = self.map_stmts(s.body)
            self.pop()

            if body:
                return [s.update(body=body, eff=None)]

            return None
        elif isinstance(s, LoopIR.If):
            self.push()
            body = self.map_stmts(s.body)
            self.pop()
            self.push()
            orelse = self.map_stmts(s.orelse)
            self.pop()

            if body or orelse:
                return [
                    s.update(body=body or s.body, orelse=orelse or s.orlse, eff=None)
                ]

            return None

        return super().map_s(s)

    def map_e(self, e):
        return None


class _DoNormalize(LoopIR_Rewrite):
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
        super().__init__(proc_cursor._node())

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

    def has_div_mod_config(self, e):
        if isinstance(e, LoopIR.Read):
            return False
        elif isinstance(e, LoopIR.Const):
            return False
        elif isinstance(e, LoopIR.USub):
            return self.has_div_mod_config(e.arg)
        elif isinstance(e, LoopIR.BinOp):
            if e.op == "/" or e.op == "%":
                return True
            else:
                lhs = self.has_div_mod_config(e.lhs)
                rhs = self.has_div_mod_config(e.rhs)
                return lhs or rhs
        elif isinstance(e, LoopIR.ReadConfig):
            return True
        else:
            assert False, "bad case"

    # Call this when e is one indexing expression
    # e should be an indexing expression
    def index_start(self, e):
        assert isinstance(e, LoopIR.expr)
        # Div and mod need more subtle handling. Don't normalize for now.
        # Skip ReadConfigs, they need careful handling because they're not Sym.
        if self.has_div_mod_config(e):
            return e

        # Make a map of symbols and coefficients
        n_map = self.normalize_e(e)

        # Write back to LoopIR.expr
        def scale_read(coeff, key):
            return LoopIR.BinOp(
                "*",
                LoopIR.Const(coeff, T.int, e.srcinfo),
                LoopIR.Read(key, [], e.type, e.srcinfo),
                e.type,
                e.srcinfo,
            )

        new_e = LoopIR.Const(n_map.get(self.C, 0), T.int, e.srcinfo)

        delete_zero = [(n_map[v], v) for v in n_map if v != self.C and n_map[v] != 0]

        for coeff, v in sorted(delete_zero):
            if coeff > 0:
                new_e = LoopIR.BinOp(
                    "+", new_e, scale_read(coeff, v), e.type, e.srcinfo
                )
            else:
                new_e = LoopIR.BinOp(
                    "-", new_e, scale_read(-coeff, v), e.type, e.srcinfo
                )

        return new_e

    def map_e(self, e):
        if e.type.is_indexable():
            return self.index_start(e)

        return super().map_e(e)


class _DoSimplify(LoopIR_Rewrite):
    def __init__(self, proc_cursor):
        self.facts = ChainMap()
        proc = _DoNormalize(proc_cursor).result()

        super().__init__(proc)
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

    def map_s(self, s):
        if isinstance(s, LoopIR.If):
            cond = self.map_e(s.cond)

            safe_cond = cond or s.cond

            # If constant true or false, then drop the branch
            if isinstance(safe_cond, LoopIR.Const):
                if safe_cond.val:
                    return super().map_stmts(s.body)
                else:
                    return super().map_stmts(s.orelse)

            # Try to use the condition while simplifying body
            self.facts = self.facts.new_child()
            self.add_fact(safe_cond)
            body = self.map_stmts(s.body)
            self.facts = self.facts.parents

            # Try to use the negation while simplifying orelse
            self.facts = self.facts.new_child()
            # TODO: negate fact here
            orelse = self.map_stmts(s.orelse)
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
            body = self.map_stmts(s.body)
            if body == []:
                return []

            eff = self.map_eff(s.eff)
            if hi or body or eff:
                return [s.update(hi=hi or s.hi, body=body or s.body, eff=eff or s.eff)]

            return None
        else:
            return super().map_s(s)


class _AssertIf(LoopIR_Rewrite):
    def __init__(self, proc_cursor, if_cursor, cond):
        self.if_stmt = if_cursor._node()

        assert isinstance(self.if_stmt, LoopIR.If)
        assert isinstance(cond, bool)

        self.cond = cond

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.if_stmt:
            # TODO: Gilbert's SMT thing should do this safely
            if self.cond:
                return self.map_stmts(s.body)
            else:
                return self.map_stmts(s.orelse)
        elif isinstance(s, LoopIR.Seq):
            body = self.map_stmts(s.body)
            if not body:
                return []
            else:
                return [s.update(body=body)]

        return super().map_s(s)


# TODO: This analysis is overly conservative.
# However, it might be a bit involved to come up with
# a more precise analysis.
class _DoDataReuse(LoopIR_Rewrite):
    def __init__(self, proc_cursor, buf_cursor, rep_cursor):
        assert isinstance(buf_cursor._node(), LoopIR.Alloc)
        assert isinstance(rep_cursor._node(), LoopIR.Alloc)
        assert buf_cursor._node().type == rep_cursor._node().type

        self.buf_name = buf_cursor._node().name
        self.rep_name = rep_cursor._node().name
        self.rep_pat = rep_cursor._node()

        self.found_rep = False
        self.first_assn = False

        super().__init__(proc_cursor._node())

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        # Check that buf_name is only used
        # before the first assignment of rep_pat
        if self.first_assn:
            if self.buf_name in _FV([s]):
                raise SchedulingError(
                    "buf_name should not be used after the "
                    "first assignment of rep_pat"
                )

        if s is self.rep_pat:
            self.found_rep = True
            return []

        if self.found_rep:
            if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
                rhs = self.apply_e(s.rhs)
                name = s.name
                if s.name == self.rep_name:
                    name = self.buf_name
                    if not self.first_assn:
                        self.first_assn = True

                return s.update(name=name, cast=None, rhs=rhs, eff=None)

        return super().map_s(s)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.rep_name:
            return e.update(name=self.buf_name)

        return super().map_e(e)


# TODO: This can probably be re-factored into a generic
# "Live Variables" analysis w.r.t. a context/stmt separation?
class _DoStageMem_FindBufData(LoopIR_Do):
    def __init__(self, proc_cursor, buf_name, stmt_start):
        self.buf_str = buf_name
        self.buf_sym = None
        self.buf_typ = None
        self.buf_mem = None
        self.stmt_start = stmt_start
        self.buf_map = ChainMap()
        self.orig_proc = proc_cursor._node()

        for fa in self.orig_proc.args:
            if fa.type.is_numeric():
                self.buf_map[str(fa.name)] = (fa.name, fa.type, fa.mem)

        super().__init__(self.orig_proc)

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


class _DoStageMem(LoopIR_Rewrite):
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
            proc_cursor, buf_name, self.stmt_start
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
        super().__init__(proc_cursor._node())
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

    def map_stmts(self, stmts):
        """This method overload simply tries to find the indicated block"""
        if not self.in_block:
            for i, s1 in enumerate(stmts):
                if s1 is self.stmt_start:
                    for j, s2 in enumerate(stmts):
                        if s2 is self.stmt_end:
                            self.found_stmt = True
                            assert j >= i
                            pre = stmts[:i]
                            block = stmts[i : j + 1]
                            post = stmts[j + 1 :]

                            if self.use_accum_zero:
                                n_dims = len(self.buf_typ.shape())
                                Check_BufferReduceOnly(
                                    self.orig_proc, block, self.buf_name, n_dims
                                )

                            block = self.wrap_block(block)
                            self.new_block = block

                            return pre + block + post

        # fall through
        return super().map_stmts(stmts)

    def wrap_block(self, block):
        """This method rewrites the structure around the block.
        `map_s` and `map_e` below substitute the buffer
        name within the block."""
        orig_typ = self.buf_typ
        new_typ = self.new_typ
        mem = self.buf_mem
        shape = self.new_sizes

        n_dims = len(orig_typ.shape())
        basetyp = new_typ.basetype() if isinstance(new_typ, T.Tensor) else new_typ

        isR, isW = Check_BufferRW(self.orig_proc, block, self.buf_name, n_dims)
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
        block = self.map_stmts(block)
        self.in_block = False

        return new_alloc + load_nest + block + store_nest

    def map_s(self, s):
        new_s = super().map_s(s)

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


class _DoStageWindow(LoopIR_Rewrite):
    def __init__(self, proc_cursor, new_name, memory, expr):
        # Inputs
        self.new_name = Sym(new_name)
        self.memory = memory
        self.target_expr = expr._node()

        # Visitor state
        self._found_expr = False
        self._complete = False
        self._copy_code = None

        proc = InferEffects(proc_cursor._node()).result()

        super().__init__(proc)

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


class _DoBoundAlloc(LoopIR_Rewrite):
    def __init__(self, proc_cursor, alloc_cursor, bounds):
        self.alloc_site = alloc_cursor._node()
        self.bounds = bounds
        super().__init__(proc_cursor._node())

    def map_s(self, s):
        if s is self.alloc_site:
            assert isinstance(s.type, T.Tensor)
            if len(self.bounds) != len(s.type.hi):
                raise SchedulingError(
                    f"bound_alloc: dimensions do not match: {len(self.bounds)} "
                    f"!= {len(s.type.hi)} (expected)"
                )

            new_type = T.Tensor(
                [(new if new else old) for old, new in zip(s.type.hi, self.bounds)],
                s.type.is_window,
                s.type.type,
            )

            return [LoopIR.Alloc(s.name, new_type, s.mem, s.eff, s.srcinfo)]

        return super().map_s(s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Passes to export


class Schedules:
    DoReorder = _Reorder
    DoSplit = _Split
    DoUnroll = _Unroll
    DoInline = _Inline
    DoPartialEval = _PartialEval
    SetTypAndMem = _SetTypAndMem
    DoCallSwap = _CallSwap
    DoBindExpr = _BindExpr
    DoBindConfig = _BindConfig
    DoStageAssn = _DoStageAssn
    DoLiftAlloc = _LiftAlloc
    DoFissionLoops = _FissionLoops
    DoExtractMethod = _DoExtractMethod
    DoReorderStmt = _DoReorderStmt
    DoConfigWrite = _ConfigWrite
    DoInlineWindow = _InlineWindow
    DoInsertPass = _DoInsertPass
    DoDeletePass = _DoDeletePass
    DoSimplify = _DoSimplify
    DoBoundAndGuard = _DoBoundAndGuard
    DoFuseLoop = _DoFuseLoop
    DoAddLoop = _DoAddLoop
    DoDataReuse = _DoDataReuse
    DoLiftIf = _DoLiftIf
    DoDoubleFission = _DoDoubleFission
    DoPartitionLoop = _PartitionLoop
    DoAssertIf = _AssertIf
    DoSpecialize = _DoSpecialize
    DoAddUnsafeGuard = _DoAddUnsafeGuard
    DoDeleteConfig = _DoDeleteConfig
    DoFuseIf = _DoFuseIf
    DoStageMem = _DoStageMem
    DoStageWindow = _DoStageWindow
    DoBoundAlloc = _DoBoundAlloc
    DoExpandDim = _DoExpandDim
    DoRearrangeDim = _DoRearrangeDim
    DoDivideDim = _DoDivideDim
    DoMultiplyDim = _DoMultiplyDim
    DoRemoveLoop = _DoRemoveLoop
    DoLiftAllocSimple = _DoLiftAllocSimple
    DoFissionAfterSimple = _DoFissionAfterSimple
    DoProductLoop = _DoProductLoop
    DoCommuteExpr = _DoCommuteExpr
    DoMergeWrites = _DoMergeWrites
