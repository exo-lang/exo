import inspect
import re
import textwrap
from collections import ChainMap

from .API_types import ProcedureBase
from .LoopIR import (LoopIR, LoopIR_Rewrite, Alpha_Rename, LoopIR_Do,
                     SubstArgs, T, lift_to_eff_expr)
from .LoopIR_dataflow import LoopIR_Dependencies
from .LoopIR_effects import (Effects as E, eff_filter, eff_bind, eff_null,
                             get_effect_of_stmts)
from .effectcheck import InferEffects
from .prelude import *


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Scheduling Errors

from .new_eff import (
    SchedulingError,
    Check_ReorderStmts,
    Check_ReorderLoops,
)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Finding Names

def name_plus_count(namestr):
    results = re.search(r"^([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$", namestr)
    if not results:
        raise TypeError("expected name pattern of the form\n"
                        "  ident (# integer)?\n"
                        "where ident is the name of a variable "
                        "and (e.g.) '#2' may optionally be attached to mean "
                        "'the second occurence of that identifier")

    name        = results[1]
    count       = int(results[3]) if results[3] else None
    return name,count

def iter_name_to_pattern(namestr):
    name, count = name_plus_count(namestr)
    if count is not None:
        count   = f" #{count}"
    else:
        count   = ""

    pattern     = f"for {name} in _: _{count}"
    return pattern


def nested_iter_names_to_pattern(namestr, inner):
    name, count = name_plus_count(namestr)
    if count is not None:
        count   = f" #{count}"
    else:
        count   = ""
    assert is_valid_name(inner)

    pattern = (f"for {name} in _:\n"
               f"  for {inner} in _: _{count}")
    return pattern

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reorder scheduling directive


# Take a conservative approach and allow stmt reordering only when they are
# writing to different buffers
# TODO: Do effectcheck's check_commutes-ish thing using SMT here
class _DoReorderStmt(LoopIR_Rewrite):
    def __init__(self, proc, f_stmt, s_stmt):
        self.f_stmt = f_stmt
        self.s_stmt = s_stmt
        #self.found_first = False

        #raise NotImplementedError("HIT REORDER STMTS")

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        for i,s in enumerate(stmts):
            if s is self.f_stmt:
                if i+1 < len(stmts) and stmts[i+1] is self.s_stmt:

                    Check_ReorderStmts(self.orig_proc,
                                       self.f_stmt, self.s_stmt)

                    return (stmts[:i] +
                            [ self.s_stmt, self.f_stmt ] +
                            stmts[i+2:])
                else:
                    for k in stmts:
                        print(k)
                    raise SchedulingError("expected the second stmt to be "
                                          "directly after the first stmt")

        return super().map_stmts(stmts)


class _PartitionLoop(LoopIR_Rewrite):
    def __init__(self, proc, loop_stmt, num):
        self.stmt         = loop_stmt
        self.partition_by = num
        self.second       = False
        self.second_iter  = None

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            assert isinstance(s, LoopIR.ForAll)
            if not isinstance(s.hi, LoopIR.Const):
                raise SchedulingError("expected loop bound to be constant")
            if s.hi.val <= self.partition_by:
                raise SchedulingError("expected loop bound to be larger than "
                                      "partitioning value")

            body        = self.map_stmts(s.body)
            first_loop  = LoopIR.ForAll(s.iter,
                            LoopIR.Const(self.partition_by, T.int, s.srcinfo),
                            body, None, s.srcinfo)

            # Should add partition_by to everything in body
            self.second = True
            new_iter = s.iter.copy()
            self.second_iter = new_iter
            second_body = SubstArgs(body,
                    {s.iter: LoopIR.Read(new_iter, [], T.index, s.srcinfo)}).result()
            second_body = self.map_stmts(second_body)
            second_loop = LoopIR.ForAll(new_iter,
                            LoopIR.Const(s.hi.val - self.partition_by, T.int, s.srcinfo),
                            second_body, None, s.srcinfo)

            return [first_loop] + [second_loop]

        return super().map_s(s)

    def map_e(self, e):
        if self.second:
            if type(e) == LoopIR.Read and e.name == self.second_iter:
                assert e.type.is_indexable()
                return LoopIR.BinOp("+", e, LoopIR.Const(self.partition_by, T.int, e.srcinfo), T.index, e.srcinfo)

        return super().map_e(e)



class _Reorder(LoopIR_Rewrite):
    def __init__(self, proc, loop_stmt):
        self.stmt = loop_stmt
        self.out_var = loop_stmt.iter
        if (len(loop_stmt.body) != 1 or
                not isinstance(loop_stmt.body[0], (LoopIR.ForAll,LoopIR.Seq))):
            raise SchedulingError(f"expected loop directly inside of "
                                  f"{self.out_var} loop")
        self.in_var = loop_stmt.body[0].iter

        super().__init__(proc)

    def map_s(self, s):
        if s is self.stmt:
            Check_ReorderLoops(self.orig_proc, self.stmt)

            # short-hands for sanity
            def boolop(op,lhs,rhs):
                return LoopIR.BinOp(op,lhs,rhs,T.bool,s.srcinfo)
            def cnst(intval):
                return LoopIR.Const(intval, T.int, s.srcinfo)
            def rd(i):
                return LoopIR.Read(i, [], T.index, s.srcinfo)
            def rng(x, hi):
                lhs = boolop("<=", cnst(0), x)
                rhs = boolop("<", x, hi)
                return boolop("and", lhs, rhs)
            def do_bind(x, hi, eff):
                cond    = lift_to_eff_expr( rng(rd(x),hi) )
                cond_nz = boolop("<", cnst(0), hi)
                return eff_bind(x, eff, pred=cond)#TODO: , config_pred=cond_nz)

            # this is the actual body inside both for-loops
            body        = s.body[0].body
            body_eff    = get_effect_of_stmts(body)
            # blah
            inner_eff   = do_bind(s.iter, s.hi, body_eff)
            outer_eff   = do_bind(s.body[0].iter, s.body[0].hi, inner_eff)
            return [LoopIR.ForAll(s.body[0].iter, s.body[0].hi,
                        [LoopIR.ForAll(s.iter, s.hi,
                            body,
                            inner_eff, s.srcinfo)],
                        outer_eff, s.body[0].srcinfo)]

        # fall-through
        return super().map_s(s)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Split scheduling directive


class _Split(LoopIR_Rewrite):
    def __init__(self, proc, split_loop, quot, hi, lo,
                 tail='guard', perfect=False):
        self.split_loop     = split_loop
        self.split_var      = split_loop.iter
        self.quot = quot
        self.hi_i = Sym(hi)
        self.lo_i = Sym(lo)
        self.cut_i = Sym(lo)

        assert quot > 1

        # Tail strategies are 'cut', 'guard', and 'cut_and_guard'
        self._tail_strategy = tail
        if perfect:
            self._tail_strategy = 'perfect'
        self._in_cut_tail = False

        super().__init__(proc)

    def substitute(self, srcinfo):
        cnst = lambda x: LoopIR.Const(x, T.int, srcinfo)
        rd   = lambda x: LoopIR.Read(x, [], T.index, srcinfo)
        op   = lambda op,lhs,rhs: LoopIR.BinOp(op,lhs,rhs,T.index, srcinfo)

        return op('+', op('*', cnst(self.quot),
                               rd(self.hi_i)),
                       rd(self.lo_i))

    def cut_tail_sub(self, srcinfo):
        return self._cut_tail_sub

    def map_s(self, s):
        if s is self.split_loop:
            # short-hands for sanity
            def boolop(op,lhs,rhs):
                return LoopIR.BinOp(op,lhs,rhs,T.bool,s.srcinfo)
            def szop(op,lhs,rhs):
                return LoopIR.BinOp(op,lhs,rhs,lhs.type,s.srcinfo)
            def cnst(intval):
                return LoopIR.Const(intval, T.int, s.srcinfo)
            def rd(i):
                return LoopIR.Read(i, [], T.index, s.srcinfo)
            def ceildiv(lhs,rhs):
                assert isinstance(rhs, LoopIR.Const) and rhs.val > 1
                rhs_1 = cnst(rhs.val-1)
                return szop("/", szop("+", lhs, rhs_1), rhs)

            def rng(x, hi):
                lhs = boolop("<=", cnst(0), x)
                rhs = boolop("<", x, hi)
                return boolop("and", lhs, rhs)
            def do_bind(x, hi, eff):
                cond    = lift_to_eff_expr( rng(rd(x),hi) )
                cond_nz = lift_to_eff_expr( boolop("<", cnst(0), hi) )
                return eff_bind(x, eff, pred=cond, config_pred=cond_nz)

            # in the simple case, wrap body in a guard
            if self._tail_strategy == 'guard':
                body    = self.map_stmts(s.body)
                body_eff= get_effect_of_stmts(body)
                idx_sub = self.substitute(s.srcinfo)
                cond    = boolop("<", idx_sub, s.hi)
                # condition for guarded loop is applied to effects
                body_eff= eff_filter(lift_to_eff_expr(cond), body_eff)
                body    = [LoopIR.If(cond, body, [], body_eff, s.srcinfo)]

                lo_rng  = cnst(self.quot)
                hi_rng  = ceildiv(s.hi, lo_rng)

                # pred for inner loop is: 0 <= lo <= lo_rng
                inner_eff   = do_bind(self.lo_i, lo_rng, body_eff)

                return [LoopIR.ForAll(self.hi_i, hi_rng,
                            [LoopIR.ForAll(self.lo_i, lo_rng,
                                body,
                                inner_eff, s.srcinfo)],
                            s.eff, s.srcinfo)]

            # an alternate scheme is to split the loop in two
            # by cutting off the tail into a second loop
            elif (self._tail_strategy == 'cut' or
                    self._tail_strategy == 'cut_and_guard'):
                # if N == s.hi and Q == self.quot, then
                #   we want Ncut == (N-Q+1)/Q
                Q       = cnst(self.quot)
                N       = s.hi
                Ncut    = szop("/", N, Q) # floor div

                # and then for the tail loop, we want to
                # iterate from 0 to Ntail
                # where Ntail == N % Q
                Ntail   = szop("%", N, Q)
                # in that loop we want the iteration variable to
                # be mapped instead to (Ncut*Q + cut_i)
                self._cut_tail_sub = szop("+", rd(self.cut_i),
                                               szop("*", Ncut, Q))

                main_body = self.map_stmts(s.body)
                self._in_cut_tail = True
                tail_body = Alpha_Rename(self.map_stmts(s.body)).result()
                self._in_cut_tail = False

                main_eff    = get_effect_of_stmts(main_body)
                tail_eff    = get_effect_of_stmts(tail_body)
                lo_eff      = do_bind(self.lo_i, Q, main_eff)
                hi_eff      = do_bind(self.hi_i, Ncut, lo_eff)
                tail_eff    = do_bind(self.cut_i, Ntail, tail_eff)

                if self._tail_strategy == 'cut_and_guard':
                    body = [LoopIR.ForAll(self.cut_i, Ntail,
                            tail_body,
                            tail_eff, s.srcinfo)]
                    body_eff= get_effect_of_stmts(body)
                    cond = boolop(">", Ntail, LoopIR.Const(0, T.int, s.srcinfo))
                    body_eff= eff_filter(lift_to_eff_expr(cond), body_eff)

                    loops = [LoopIR.ForAll(self.hi_i, Ncut,
                                [LoopIR.ForAll(self.lo_i, Q,
                                    main_body,
                                    lo_eff, s.srcinfo)],
                                hi_eff, s.srcinfo),
                             LoopIR.If(cond, body, [], body_eff, s.srcinfo)]

                else:
                    loops = [LoopIR.ForAll(self.hi_i, Ncut,
                                [LoopIR.ForAll(self.lo_i, Q,
                                    main_body,
                                    lo_eff, s.srcinfo)],
                                hi_eff, s.srcinfo),
                             LoopIR.ForAll(self.cut_i, Ntail,
                                tail_body,
                                tail_eff, s.srcinfo)]

                return loops

            elif self._tail_strategy == 'perfect':
                if not isinstance(s.hi, LoopIR.Const):
                    raise SchedulingError(
                        f"cannot perfectly split the '{s.iter}' loop "
                        f"unless it has a constant bound")
                elif s.hi.val % self.quot != 0:
                    raise SchedulingError(
                        f"cannot perfectly split the '{s.iter}' loop "
                        f"because {self.quot} does not evenly divide "
                        f"{s.hi.val}")

                # otherwise, we're good to go
                body    = self.map_stmts(s.body)
                body_eff= get_effect_of_stmts(body)

                lo_rng  = cnst(self.quot)
                hi_rng  = cnst(s.hi.val // self.quot)

                # pred for inner loop is: 0 <= lo <= lo_rng
                inner_eff   = do_bind(self.lo_i, lo_rng, body_eff)

                return [LoopIR.ForAll(self.hi_i, hi_rng,
                            [LoopIR.ForAll(self.lo_i, lo_rng,
                                body,
                                inner_eff, s.srcinfo)],
                            s.eff, s.srcinfo)]

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
    def __init__(self, proc, unroll_loop):
        self.orig_proc      = proc
        self.unroll_loop    = unroll_loop
        self.unroll_var     = unroll_loop.iter
        self.unroll_itr     = 0
        self.env            = {}

        super().__init__(proc)

    def map_s(self, s):
        if s is self.unroll_loop:
            # if isinstance(s, LoopIR.ForAll):
            #    # unroll this to loops!!
            #    if s.iter is self.unroll_var:
            if not isinstance(s.hi, LoopIR.Const):
                raise SchedulingError(f"expected loop '{s.iter}' "
                                      f"to have constant bounds")
            # if len(s.body) != 1:
            #    raise SchedulingError(f"expected loop '{s.iter}' "
            #                          f"to have only one body")
            hi      = s.hi.val
            if hi == 0:
                return []

            #assert hi > 0
            orig_body = s.body

            self.unroll_itr = 0

            body    = Alpha_Rename(self.map_stmts(orig_body)).result()
            for i in range(1,hi):
                self.unroll_itr = i
                nxtbody = Alpha_Rename(self.map_stmts(orig_body)).result()
                body   += nxtbody

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
    def __init__(self, proc, call_stmt):
        assert isinstance(call_stmt, LoopIR.Call)
        self.orig_proc  = proc
        self.call_stmt  = call_stmt
        self.env        = {}

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.call_stmt:
            # handle potential window expressions in call positions
            win_binds = []

            def map_bind(nm, a):
                if isinstance(a, LoopIR.WindowExpr):
                    stmt = LoopIR.WindowStmt(nm, a, eff_null(a.srcinfo),
                                             a.srcinfo)
                    win_binds.append(stmt)
                    return LoopIR.Read(nm, [], a.type, a.srcinfo)
                else:
                    return a

            # first, set-up a binding from sub-proc arguments
            # to supplied expressions at the call-site
            call_bind = {xd.name: map_bind(xd.name, a)
                         for xd, a in zip(s.f.args, s.args)}

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
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Partial Evaluation scheduling directive


class _PartialEval(LoopIR_Rewrite):
    def __init__(self, proc, arg_vals):
        assert arg_vals, "Don't call _PartialEval without any substitutions"
        self.env = arg_vals

        arg_types = {p.name: p.type for p in proc.args}

        # Validate env:
        for k, v in self.env.items():
            if not arg_types[k].is_indexable() and not arg_types[k].is_bool():
                raise SchedulingError("cannot partially evaluate "
                                      "numeric (non-index, non-bool) arguments")
            if not isinstance(v, int):
                raise SchedulingError("cannot partially evaluate "
                                      "to a non-int, non-bool value")

        self.orig_proc = proc

        args = [self.map_fnarg(a) for a in proc.args
                if a.name not in self.env]
        preds = [self.map_e(p) for p in self.orig_proc.preds]
        body = self.map_stmts(self.orig_proc.body)
        eff = self.map_eff(self.orig_proc.eff)

        self.proc = LoopIR.proc(name=self.orig_proc.name,
                                args=args,
                                preds=preds,
                                body=body,
                                instr=None,
                                eff=eff,
                                srcinfo=self.orig_proc.srcinfo)

    def map_e(self,e):
        if isinstance(e, LoopIR.Read):
            if e.type.is_indexable():
                assert len(e.idx) == 0
                if e.name in self.env:
                    return LoopIR.Const(self.env[e.name], T.int, e.srcinfo)
            elif e.type.is_bool():
                if e.name in self.env:
                    return LoopIR.Const(self.env[e.name], T.bool, e.srcinfo)

        return super().map_e(e)

    def map_eff_e(self,e):
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
class _SetTypAndMem(LoopIR_Rewrite):
    def __init__(self, proc, name, inst_no, basetyp=None, win=None, mem=None):
        ind = lambda x: 1 if x else 0
        assert ind(basetyp) + ind(win) + ind(mem) == 1
        self.orig_proc  = proc
        self.name       = name
        self.n_match    = inst_no
        self.basetyp    = basetyp
        self.win        = win
        self.mem        = mem

        super().__init__(proc)

    def check_inst(self):
        # otherwise, handle instance counting...
        if self.n_match is None:
            return True
        else:
            self.n_match = self.n_match-1
            return (self.n_match == 0)

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
        typ     = a.type
        mem     = a.mem
        if self.basetyp is not None:
            if not a.type.is_numeric():
                raise SchedulingError("cannot change the precision of a "
                                      "non-numeric argument")
            typ = self.change_precision(typ)
        elif self.win is not None:
            if not a.type.is_tensor_or_window():
                raise SchedulingError("cannot change windowing of a "
                                      "non-tensor/window argument")
            typ = self.change_window(typ)
        else:
            assert self.mem is not None
            if not a.type.is_numeric():
                raise SchedulingError("cannot change the memory of a "
                                      "non-numeric argument")
            mem = self.mem

        return LoopIR.fnarg( a.name, typ, mem, a.srcinfo )

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
                    raise SchedulingError("cannot change an allocation to "
                                          "be or not be a window")
                else:
                    assert self.mem is not None
                    mem = self.mem

                return [LoopIR.Alloc( s.name, typ, mem, s.eff, s.srcinfo )]

        # fall-through
        return super().map_s(s)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Call Swap scheduling directive


class _CallSwap(LoopIR_Rewrite):
    def __init__(self, proc, call_stmt, new_subproc):
        assert isinstance(call_stmt, LoopIR.Call)
        self.orig_proc  = proc
        self.call_stmt  = call_stmt
        self.new_subproc = new_subproc

        super().__init__(proc)

    def map_s(self, s):
        if s is self.call_stmt:
            return [ LoopIR.Call(self.new_subproc, s.args, None, s.srcinfo) ]

        # fall-through
        return super().map_s(s)

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff


class _InlineWindow(LoopIR_Rewrite):
    def __init__(self, proc, stmt):
        assert (isinstance(stmt, LoopIR.WindowStmt))

        self.orig_proc = proc
        self.win_stmt  = stmt

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.win_stmt:
            return []

        return super().map_s(s)

    def map_e(self, e):
        etyp    = type(e)
        assert isinstance(self.win_stmt.rhs, LoopIR.WindowExpr)

        # TODO: Add more safety check?
        if etyp is LoopIR.WindowExpr:
            if self.win_stmt.lhs == e.name:
                assert (len([w for w in self.win_stmt.rhs.idx
                             if isinstance(w, LoopIR.Interval)]) == len(e.idx))
                idxs     = e.idx
                new_idxs = []
                for w in self.win_stmt.rhs.idx:
                    if isinstance(w, LoopIR.Interval):
                        if isinstance(idxs[0], LoopIR.Interval):
                            # window again, so
                            # w.lo + idxs[0].lo : w.lo + idxs[0].hi
                            lo = LoopIR.BinOp("+", w.lo, idxs[0].lo,
                                              T.index, w.srcinfo)
                            hi = LoopIR.BinOp("+", w.lo, idxs[0].hi,
                                              T.index, w.srcinfo)
                            ivl = LoopIR.Interval(lo, hi, w.srcinfo)
                            new_idxs.append(ivl)
                        else:  # Point
                            p = LoopIR.Point(LoopIR.BinOp("+", w.lo, idxs[0].pt,
                                                           T.index, w.srcinfo),
                                             w.srcinfo)
                            new_idxs.append(p)
                        idxs = idxs[1:]
                    else:
                        new_idxs.append(w)

                # repair window type..
                old_typ = self.win_stmt.rhs.type
                new_type = LoopIR.WindowType(old_typ.src_type,
                                             old_typ.as_tensor,
                                             self.win_stmt.rhs.name,
                                             new_idxs)


                return LoopIR.WindowExpr( self.win_stmt.rhs.name,
                                          new_idxs, new_type, e.srcinfo )

        elif etyp is LoopIR.Read:
            if self.win_stmt.lhs == e.name:
                assert (len([w for w in self.win_stmt.rhs.idx
                             if isinstance(w, LoopIR.Interval)]) == len(e.idx))
                idxs     = e.idx
                new_idxs = []
                for w in self.win_stmt.rhs.idx:
                    if isinstance(w, LoopIR.Interval):
                        new_idxs.append(
                            LoopIR.BinOp("+", w.lo, idxs[0], T.index,
                                         w.srcinfo))
                        idxs.pop()
                    else:
                        new_idxs.append(w.pt)

                return LoopIR.Read( self.win_stmt.rhs.name,
                                    new_idxs, e.type, e.srcinfo )

        elif etyp is LoopIR.StrideExpr:
            if self.win_stmt.lhs == e.name:
                return LoopIR.StrideExpr( self.win_stmt.rhs.name, e.dim, e.type, e.srcinfo )

        return super().map_e(e)



class _ConfigWriteRoot(LoopIR_Rewrite):
    def __init__(self, proc, config, field, expr):
        assert (isinstance(expr, LoopIR.Read)
                or isinstance(expr, LoopIR.StrideExpr)
                or isinstance(expr, LoopIR.Const))

        self.orig_proc = proc

        c_str = [LoopIR.WriteConfig(config, field, expr, None, self.orig_proc.srcinfo)]
        proc.body = c_str + proc.body

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()



class _ConfigWriteAfter(LoopIR_Rewrite):
    def __init__(self, proc, stmt, config, field, expr):
        assert (isinstance(expr, LoopIR.Read)
                or isinstance(expr, LoopIR.StrideExpr)
                or isinstance(expr, LoopIR.Const))

        self.orig_proc = proc
        self.stmt = stmt
        self.config = config
        self.field = field
        self.expr = expr

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        body = []
        for s in stmts:
            body += self.map_s(s)
            if s is self.stmt:
                c_str = LoopIR.WriteConfig(self.config, self.field, self.expr,
                                           None, s.srcinfo)
                body.append(c_str)

        return body

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Bind Expression scheduling directive

class _BindConfig(LoopIR_Rewrite):
    def __init__(self, proc, config, field, expr):
        assert isinstance(expr, LoopIR.Read)

        self.orig_proc = proc
        self.config    = config
        self.field     = field
        self.expr      = expr
        self.found_expr= False
        self.placed_writeconfig = False
        self.sub_over  = False

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def process_block(self, block):
        if self.sub_over:
            return block

        new_block = []
        is_writeconfig_block = False

        for stmt in block:
            stmt = self.map_s(stmt)

            if self.found_expr and not self.placed_writeconfig:
                self.placed_writeconfig = True
                is_writeconfig_block    = True
                wc = LoopIR.WriteConfig( self.config, self.field,
                                         self.expr, None,
                                         self.expr.srcinfo )
                new_block.extend([wc])

            new_block.extend(stmt)

        if is_writeconfig_block:
            self.sub_over = True

        return new_block

    def map_s(self, s):
        if self.sub_over:
            return super().map_s(s)

        if isinstance(s, LoopIR.ForAll):
            body = self.process_block(s.body)
            return [LoopIR.ForAll(s.iter, s.hi, body, s.eff, s.srcinfo)]

        if isinstance(s, LoopIR.If):
            if_then = self.process_block(s.body)
            if_else = self.process_block(s.orelse)
            cond    = self.map_e(s.cond)
            return [LoopIR.If(cond, if_then, if_else, s.eff, s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        if e is self.expr and not self.sub_over:
            assert not self.found_expr
            self.found_expr = True

            return LoopIR.ReadConfig( self.config, self.field, e.type, e.srcinfo)
        else:
            return super().map_e(e)



class _BindExpr(LoopIR_Rewrite):
    def __init__(self, proc, new_name, exprs, cse=False):
        assert all(isinstance(expr, LoopIR.expr) for expr in exprs)
        assert all(expr.type.is_numeric() for expr in exprs)
        assert exprs

        self.orig_proc = proc
        self.new_name = Sym(new_name)
        self.exprs = exprs if cse else [exprs[0]]
        self.cse = cse
        self.found_expr = None
        self.placed_alloc = False
        self.sub_over = False

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def process_block(self, block):
        if self.sub_over:
            return block

        new_block = []
        is_alloc_block = False

        for stmt in block:
            stmt = self.map_s(stmt)

            if self.found_expr and not self.placed_alloc:
                self.placed_alloc = True
                is_alloc_block = True
                alloc = LoopIR.Alloc(self.new_name, T.R, None, None,
                                     self.found_expr.srcinfo)
                # TODO Fix Assign, probably wrong
                assign = LoopIR.Assign(self.new_name, T.R, None, [],
                                       self.found_expr, None,
                                       self.found_expr.srcinfo)
                new_block.extend([alloc, assign])

            new_block.extend(stmt)

        # If this is the block containing the new alloc, stop substituting
        if is_alloc_block:
            self.sub_over = True

        return new_block

    def map_s(self, s):
        if self.sub_over:
            return super().map_s(s)

        if isinstance(s, LoopIR.ForAll):
            body = self.process_block(s.body)
            return [LoopIR.ForAll(s.iter, s.hi, body, s.eff, s.srcinfo)]

        if isinstance(s, LoopIR.If):
            # TODO: our CSE here is very conservative. It won't look for
            #  matches between the then and else branches; in other words,
            #  it is restricted to a single basic block.
            if_then = self.process_block(s.body)
            if_else = self.process_block(s.orelse)
            return [LoopIR.If(s.cond, if_then, if_else, s.eff, s.srcinfo)]

        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            e = self.exprs[0]
            # bind LHS when self.cse == True
            if (self.cse and
                    isinstance(e, LoopIR.Read) and
                    e.name == s.name and
                    e.type == s.type and
                    all([True for i, j in zip(e.idx, s.idx) if i == j])):
                rhs = self.map_e(s.rhs)

                return [type(s)(self.new_name, s.type, None, [], rhs, None,
                                s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        if e in self.exprs and not self.sub_over:
            if not self.found_expr:
                # TODO: dirty hack. need real CSE-equality (i.e. modulo srcinfo)
                self.exprs = [x for x in self.exprs if str(e) == str(x)]
            self.found_expr = e
            return LoopIR.Read(self.new_name, [], e.type, e.srcinfo)
        else:
            return super().map_e(e)


class _DoStageAssn(LoopIR_Rewrite):
    def __init__(self, proc, new_name, assn):
        assert isinstance(assn, (LoopIR.Assign, LoopIR.Reduce))
        self.assn = assn
        self.new_name = Sym(new_name)

        super().__init__(proc)

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
                LoopIR.Assign(s.name, s.type, None, s.idx, rdtmp, None,
                              s.srcinfo)
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
                LoopIR.Assign(s.name, s.type, None, s.idx, rdtmp, None,
                              s.srcinfo)
            ]

        return super().map_s(s)


class _DoParToSeq(LoopIR_Rewrite):
    def __init__(self, proc, par_stmt):
        assert isinstance(par_stmt, LoopIR.ForAll)

        self.par_stmt = par_stmt
        super().__init__(proc)

    def map_s(self, s):
        if s is self.par_stmt:
            body = self.map_stmts(s.body)
            return [LoopIR.Seq(s.iter, s.hi, body, s.eff, s.srcinfo)]
        else:
            return super().map_s(s)



#Lift if no variable dependency
class _DoLiftIf(LoopIR_Rewrite):
    def __init__(self, proc, if_stmt, n_lifts):
        assert isinstance(if_stmt, LoopIR.If)
        assert is_pos_int(n_lifts)

        self.orig_proc    = proc
        self.if_stmt      = if_stmt
        self.if_deps      = vars_in_expr(if_stmt.cond)

        self.n_lifts      = n_lifts

        self.ctrl_ctxt    = []
        self.lift_sites   = []

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.if_stmt:
            n_up = min(self.n_lifts, len(self.ctrl_ctxt))
            self.lift_sites = self.ctrl_ctxt[-n_up:]

            # erase the statement from this location
            return []

        elif isinstance(s, (LoopIR.ForAll, LoopIR.Seq)): #TODO: Need to lift if??
            # handle recursive part of pass at this statement
            self.ctrl_ctxt.append(s)
            orig_body = super().map_stmts(s.body)
            self.ctrl_ctxt.pop()

            # splice in lifted statement at the point to lift-to
            if s in self.lift_sites:
                if s.iter in self.if_deps:
                    raise SchedulingError("If statement condition should not depend on "+
                                          "loop variable")
                if len(s.body) != 1:
                    raise SchedulingError("expected if statement to be directly inside "+
                                          "the loop")

                body   = [type(s)(s.iter, s.hi, self.if_stmt.body, None, s.srcinfo)]
                orelse = []
                if len(self.if_stmt.orelse) > 0:
                    orelse = [type(s)(s.iter, s.hi, self.if_stmt.orelse, None, s.srcinfo)]
                self.if_stmt = LoopIR.If(self.if_stmt.cond, body, orelse, None, s.srcinfo)

                return [self.if_stmt]

            return [type(s)(s.iter, s.hi, orig_body, None, s.srcinfo)]

        # fall-through
        return super().map_s(s)



class _DoExpandDim(LoopIR_Rewrite):
    def __init__(self, proc, alloc_stmt, alloc_dim, indexing):
        assert isinstance(alloc_stmt, LoopIR.Alloc)
        assert isinstance(alloc_dim, LoopIR.expr)
        assert isinstance(indexing, LoopIR.expr)

        self.orig_proc    = proc
        self.alloc_stmt   = alloc_stmt
        self.alloc_sym    = alloc_stmt.name
        self.alloc_dim    = alloc_dim
        self.indexing     = indexing
        self.alloc_type   = None

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.alloc_stmt:
            new_typ = s.type
            new_rngs = [self.alloc_dim]

            if isinstance(new_typ, T.Tensor):
                new_rngs += new_typ.shape()

            new_typ = new_typ.basetype()
            new_typ = T.Tensor(new_rngs, False, new_typ)
            self.alloc_type = new_typ

            return [LoopIR.Alloc(s.name, new_typ, s.mem, None, s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        if isinstance(e, LoopIR.Read) and e.name == self.alloc_sym:
            idx = [self.indexing] + e.idx
            return LoopIR.Read(e.name, idx, e.type, e.srcinfo)

        if isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            idx = [LoopIR.Point(self.indexing, e.srcinfo)] + e.idx
            win_typ = T.Window(self.alloc_type, e.type.as_tensor, e.name, idx)
            return LoopIR.WindowExpr(e.name, idx, win_typ, e.srcinfo)

        # fall-through
        return super().map_e(e)


class _DoRearrangeDim(LoopIR_Rewrite):
    def __init__(self, proc, alloc_stmt, dimensions):
        assert isinstance(alloc_stmt, LoopIR.Alloc)

        self.alloc_stmt = alloc_stmt
        self.dimensions = dimensions

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        # simply change the dimension
        if s is self.alloc_stmt:
            # construct new_hi
            new_hi   = [s.type.hi[i] for i in self.dimensions]
            # construct new_type
            new_type = LoopIR.Tensor(new_hi, s.type.is_window, s.type.type)

            return [LoopIR.Alloc(s.name, new_type, s.mem, None, s.srcinfo)]

        # Adjust the use-site
        if isinstance(s, LoopIR.Assign) or isinstance(s, LoopIR.Reduce):
            if s.name is self.alloc_stmt.name:
                # shuffle
                new_idx = [s.idx[i] for i in self.dimensions]
                return [type(s)(s.name, s.type, s.cast, new_idx, s.rhs, None, s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        # TODO: I am not sure what rearrange_dim should do in terms of StrideExpr
        if isinstance(e, LoopIR.Read) or isinstance(e, LoopIR.WindowExpr):
            if e.name is self.alloc_stmt.name:
                new_idx = [e.idx[i] for i in self.dimensions]
                return type(e)(e.name, new_idx, e.type, e.srcinfo)

        return super().map_e(e)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Lift Allocation scheduling directive

class _LiftAlloc(LoopIR_Rewrite):
    def __init__(self, proc, alloc_stmt, n_lifts, mode, size, keep_dims):
        assert isinstance(alloc_stmt, LoopIR.Alloc)
        assert is_pos_int(n_lifts)

        if mode not in ('row', 'col'):
            raise SchedulingError(
                f"Unknown lift mode {mode}, should be 'row' or 'col'")

        self.orig_proc    = proc
        self.alloc_stmt   = alloc_stmt
        self.alloc_sym    = alloc_stmt.name
        self.alloc_deps   = LoopIR_Dependencies(self.alloc_sym,
                                                proc.body).result()
        self.lift_mode    = mode
        self.lift_size    = size
        self.keep_dims    = keep_dims

        self.n_lifts      = n_lifts

        self.ctrl_ctxt    = []
        self.lift_site    = None

        self.lifted_stmt  = None
        self.access_idxs  = None
        self.alloc_type   = None
        self._in_call_arg = False

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def idx_mode(self, access, orig):
        if self.lift_mode == 'row':
            return access + orig
        elif self.lift_mode == 'col':
            return orig + access
        assert False

    def map_s(self, s):
        if s is self.alloc_stmt:
            # mark the point we want to lift this alloc-stmt to
            n_up = min(self.n_lifts, len(self.ctrl_ctxt))
            self.lift_site = self.ctrl_ctxt[-n_up]

            # extract the ranges and variables of enclosing loops
            idxs, rngs = self.get_ctxt_itrs_and_rngs(n_up)

            # compute the lifted allocation buffer type, and
            # the new allocation statement
            new_typ = s.type
            new_rngs = []
            for r in rngs:
                if isinstance(r, LoopIR.Const):
                    if r.val > 0:
                        new_rngs.append(r)
                    else:
                        assert False, "why loop bound is negative?"
                else:
                    new_rngs.append(
                        LoopIR.BinOp("+", r,
                                     LoopIR.Const(1, T.int, r.srcinfo),
                                     T.index, r.srcinfo))

            if isinstance(new_typ, T.Tensor):
                if self.lift_mode == 'row':
                    new_rngs += new_typ.shape()
                elif self.lift_mode == 'col':
                    new_rngs = new_typ.shape() + new_rngs
                else:
                    assert False

                new_typ = new_typ.basetype()
            if len(new_rngs) > 0:
                new_typ = T.Tensor(new_rngs, False, new_typ)

            # effect remains null
            self.lifted_stmt = LoopIR.Alloc(s.name, new_typ, s.mem,
                                            None, s.srcinfo)
            self.access_idxs = idxs
            self.alloc_type = new_typ

            # erase the statement from this location
            return []

        elif isinstance(s, (LoopIR.If, LoopIR.ForAll, LoopIR.Seq)):
            # handle recursive part of pass at this statement
            self.ctrl_ctxt.append(s)
            stmts = super().map_s(s)
            self.ctrl_ctxt.pop()

            # splice in lifted statement at the point to lift-to
            if s is self.lift_site:
                stmts = [self.lifted_stmt] + stmts

            return stmts

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            # in this case, we may need to substitute the
            # buffer name on the lhs of the assignment/reduction
            if s.name is self.alloc_sym:
                assert self.access_idxs is not None
                idx = self.idx_mode(
                    [LoopIR.Read(i, [], T.index, s.srcinfo)
                     for i in self.access_idxs], s.idx)
                rhs = self.map_e(s.rhs)
                # return allocation or reduction...
                return [type(s)(s.name, s.type, s.cast, idx, rhs, None,
                                s.srcinfo)]

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
            if len(self.access_idxs) == 0:
                return e

            # if self._in_call_arg:
            if e.type.is_real_scalar():
                idx = self.idx_mode(
                    [LoopIR.Read(i, [], T.index, e.srcinfo)
                     for i in self.access_idxs],
                    e.idx)
                return LoopIR.Read(e.name, idx, e.type, e.srcinfo)
            else:
                assert self._in_call_arg
                assert len(e.idx) == 0
                # then we need to replace this read with a
                # windowing expression
                access = [LoopIR.Point(LoopIR.Read(i, [], T.index, e.srcinfo),
                                       e.srcinfo)
                          for i in self.access_idxs]
                orig = [LoopIR.Interval(LoopIR.Const(0, T.int, e.srcinfo),
                                        hi,
                                        e.srcinfo)
                        for hi in e.type.shape()]
                idx = self.idx_mode(access, orig)
                tensor_type = (e.type.as_tensor if isinstance(e.type, T.Window)
                               else e.type)
                win_typ = T.Window(self.alloc_type, tensor_type, e.name, idx)
                return LoopIR.WindowExpr(e.name, idx, win_typ, e.srcinfo)

        if isinstance(e, LoopIR.WindowExpr) and e.name == self.alloc_sym:
            assert self.access_idxs is not None
            if len(self.access_idxs) == 0:
                return e
            # otherwise, extend windowing with accesses...

            idx = self.idx_mode(
                [LoopIR.Point(LoopIR.Read(i, [], T.index, e.srcinfo),
                              e.srcinfo)
                 for i in self.access_idxs], e.idx)
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
            elif isinstance(s, LoopIR.ForAll):
                # note, do not accrue false dependencies
                if s.iter in self.alloc_deps or self.keep_dims:
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
                                    f"be less than for-loop bound {s.hi.val}")
                        elif isinstance(s.hi, LoopIR.BinOp) and s.hi.op == '%':
                            assert isinstance(s.hi.rhs, LoopIR.Const)
                            if s.hi.rhs.val > self.lift_size:
                                raise SchedulingError(
                                    f"Lift size cannot "
                                    f"be less than for-loop bound {s.hi}")
                        else:
                            raise NotImplementedError

                        rngs.append(LoopIR.Const(self.lift_size, T.int, s.srcinfo))
                    else:
                        rngs.append(s.hi)
            elif isinstance(s, LoopIR.Seq):
                pass
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
            if type(s) is LoopIR.Reduce: # Allow reduce
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
        self._fvs     = set()
        self._bound   = set()

        self.do_stmts(stmts)

    def result(self):
        return self._fvs

    def do_s(self, s):
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if s.name not in self._bound:
                self._fvs.add(s.name)
        elif isinstance(s, LoopIR.ForAll):
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
        elif styp is LoopIR.ForAll:
            return _is_idempotent(s.body)
        else:
            return True

    return all( _stmt(s) for s in stmts )


class _DoDoubleFission:
    def __init__(self, proc, stmt1, stmt2, n_lifts):
        assert isinstance(stmt1, LoopIR.stmt)
        assert isinstance(stmt2, LoopIR.stmt)
        assert is_pos_int(n_lifts)
        self.orig_proc      = proc
        self.tgt_stmt1      = stmt1
        self.tgt_stmt2      = stmt2
        self.n_lifts        = n_lifts

        self.hit_fission1   = False
        self.hit_fission2   = False

        pre_body, mid_body, post_body = self.map_stmts(proc.body)
        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = self.orig_proc.args,
                                preds   = self.orig_proc.preds,
                                body    = pre_body + mid_body + post_body,
                                instr   = None,
                                eff     = self.orig_proc.eff,
                                srcinfo = self.orig_proc.srcinfo)

        self.proc = InferEffects(self.proc).result()

    def result(self):
        return self.proc

    def alloc_check(self, pre, post):
        if not _is_alloc_free(pre, post):
            raise SchedulingError("Will not fission here, because "
                                  "an allocation might be buried "
                                  "in a different scope than some use-site")

    def map_stmts(self, stmts):
        pre_stmts           = []
        mid_stmts           = []
        post_stmts          = []
        for orig_s in stmts:
            pre, mid, post  = self.map_s(orig_s)
            pre_stmts      += pre
            mid_stmts      += mid
            post_stmts     += post

        return (pre_stmts, mid_stmts, post_stmts)


    def map_s(self, s):
        if s is self.tgt_stmt1:
            self.hit_fission1 = True
            return ([s],[],[])
        elif s is self.tgt_stmt2:
            self.hit_fission2 = True
            return ([],[s],[])

        elif isinstance(s, LoopIR.If):

            # first, check if we need to split the body
            pre, mid, post = self.map_stmts(s.body)
            fission_body = (len(pre) > 0 and len(mid) > 0 and len(post) > 0 and self.n_lifts > 0)
            if fission_body:
                self.n_lifts -= 1
                self.alloc_check(pre, mid)
                self.alloc_check(mid, post)
                pre = LoopIR.If(s.cond, pre, [], None, s.srcinfo)
                mid = LoopIR.If(s.cond, mid, s.orelse, None, s.srcinfo)
                post = LoopIR.If(s.cond, post, [], None, s.srcinfo)
                return ([pre],[mid],[post])

            body = pre+mid+post

            # if we don't, then check if we need to split the or-else
            pre, mid, post       = self.map_stmts(s.orelse)
            fission_orelse  = (len(pre) > 0 and len(post) > 0 and len(mid) > 0 and
                               self.n_lifts > 0)
            if fission_orelse:
                self.n_lifts -= 1
                self.alloc_check(pre, mid)
                self.alloc_check(mid, post)
                pre        = LoopIR.If(s.cond, [], pre, None, s.srcinfo)
                mid        = LoopIR.If(s.cond, body, mid, None, s.srcinfo)
                post       = LoopIR.If(s.cond, [], post, None, s.srcinfo)
                return ([pre],[mid],[post])

            orelse = pre+mid+post

            # if we neither split the body nor the or-else,
            # then we need to gather together the pre and post.
            single_stmt = LoopIR.If(s.cond, body, orelse, None, s.srcinfo)

        elif isinstance(s, LoopIR.ForAll) or isinstance(s, LoopIR.Seq):

            # check if we need to split the loop
            pre, mid, post = self.map_stmts(s.body)
            do_fission = (len(pre) > 0 and len(post) > 0 and len(mid) > 0 and
                          self.n_lifts > 0)
            if do_fission:
                self.n_lifts -= 1
                self.alloc_check(pre, mid)
                self.alloc_check(mid, post)

                # we can skip the loop iteration if the
                # body doesn't depend on the loop
                # and the body is idempotent
                if s.iter in _FV(pre) or not _is_idempotent(pre):
                    pre    = [LoopIR.ForAll(s.iter, s.hi, pre, None, s.srcinfo)]
                    # since we are copying the binding of s.iter,
                    # we should perform an Alpha_Rename for safety
                    pre    = Alpha_Rename(pre).result()
                if s.iter in _FV(mid) or not _is_idempotent(mid):
                    mid    = [LoopIR.ForAll(s.iter, s.hi, mid, None, s.srcinfo)]
                if s.iter in _FV(post) or not _is_idempotent(post):
                    post   = [LoopIR.ForAll(s.iter, s.hi, post, None, s.srcinfo)]
                    post   = Alpha_Rename(post).result()

                return (pre,mid,post)

            single_stmt = LoopIR.ForAll(s.iter, s.hi, pre+mid+post, None, s.srcinfo)

        else:
            # all other statements cannot recursively
            # contain statements, so...
            single_stmt = s

        if self.hit_fission1 and not self.hit_fission2:
            return ([],[single_stmt],[])
        elif self.hit_fission2:
            return ([],[],[single_stmt])
        else:
            return ([single_stmt],[],[])



# structure is weird enough to skip using the Rewrite-pass super-class
class _FissionLoops:
    def __init__(self, proc, stmt, n_lifts):
        assert isinstance(stmt, LoopIR.stmt)
        assert is_pos_int(n_lifts)
        self.orig_proc      = proc
        self.tgt_stmt       = stmt
        self.n_lifts        = n_lifts

        self.hit_fission    = False     # signal to map_stmts

        pre_body, post_body = self.map_stmts(proc.body)
        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = self.orig_proc.args,
                                preds   = self.orig_proc.preds,
                                body    = pre_body + post_body,
                                instr   = None,
                                eff     = self.orig_proc.eff,
                                srcinfo = self.orig_proc.srcinfo)
        self.proc = InferEffects(self.proc).result()

    def result(self):
        return self.proc

    def alloc_check(self, pre, post):
        if not _is_alloc_free(pre, post):
            raise SchedulingError("Will not fission here, because "
                                  "an allocation might be buried "
                                  "in a different scope than some use-site")

    # returns a pair of stmt-lists
    # for those statements occurring before and
    # after the fission point
    def map_stmts(self, stmts):
        pre_stmts           = []
        post_stmts          = []
        for orig_s in stmts:
            pre, post       = self.map_s(orig_s)
            pre_stmts      += pre
            post_stmts     += post

        return (pre_stmts, post_stmts)

    # see map_stmts comment
    def map_s(self, s):
        if s is self.tgt_stmt:
            #assert self.hit_fission == False
            self.hit_fission = True
            # none-the-less make sure we return this statement in
            # the pre-fission position
            return ([s],[])

        elif isinstance(s, LoopIR.If):

            # first, check if we need to split the body
            pre, post = self.map_stmts(s.body)
            fission_body = (len(pre) > 0 and len(post) > 0 and self.n_lifts > 0)
            if fission_body:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre = LoopIR.If(s.cond, pre, [], None, s.srcinfo)
                post = LoopIR.If(s.cond, post, s.orelse, None, s.srcinfo)
                return ([pre],[post])

            body = pre+post

            # if we don't, then check if we need to split the or-else
            pre, post       = self.map_stmts(s.orelse)
            fission_orelse  = (len(pre) > 0 and len(post) > 0 and
                               self.n_lifts > 0)
            if fission_orelse:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre         = LoopIR.If(s.cond, body, pre, None, s.srcinfo)
                post        = LoopIR.If(s.cond, [LoopIR.Pass(None, s.srcinfo)],
                                                post, None, s.srcinfo)
                return ([pre],[post])

            orelse = pre+post

            # if we neither split the body nor the or-else,
            # then we need to gather together the pre and post.
            single_stmt = LoopIR.If(s.cond, body, orelse, None, s.srcinfo)

        elif isinstance(s, LoopIR.ForAll) or isinstance(s, LoopIR.Seq):

            # check if we need to split the loop
            pre, post = self.map_stmts(s.body)
            do_fission = (len(pre) > 0 and len(post) > 0 and
                          self.n_lifts > 0)
            if do_fission:
                self.n_lifts -= 1
                self.alloc_check(pre, post)

                # we can skip the loop iteration if the
                # body doesn't depend on the loop
                # and the body is idempotent
                if s.iter in _FV(pre) or not _is_idempotent(pre):
                    pre     = [LoopIR.ForAll(s.iter, s.hi, pre, None, s.srcinfo)]
                    # since we are copying the binding of s.iter,
                    # we should perform an Alpha_Rename for safety
                    pre         = Alpha_Rename(pre).result()
                if s.iter in _FV(post) or not _is_idempotent(post):
                    post    = [LoopIR.ForAll(s.iter, s.hi, post, None, s.srcinfo)]

                return (pre,post)

            # if we didn't split, then compose pre and post of the body
            single_stmt = LoopIR.ForAll(s.iter, s.hi, pre+post, None, s.srcinfo)

        else:
            # all other statements cannot recursively
            # contain statements, so...
            single_stmt = s

        if self.hit_fission:
            return ([],[single_stmt])
        else:
            return ([single_stmt],[])


class _DoAddUnsafeGuard(LoopIR_Rewrite):
    def __init__(self, proc, stmt, cond):
        self.stmt = stmt
        self.cond = cond
        self.in_loop = False

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            s1 = Alpha_Rename([s]).result()
            return [LoopIR.If(self.cond, s1, [], None, s.srcinfo)]

        return super().map_s(s)


class _DoSpecialize(LoopIR_Rewrite):
    def __init__(self, proc, stmt, conds):
        assert conds, "Must add at least one condition"
        self.stmt = stmt
        self.conds = conds
        self.in_loop = False

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            else_br = Alpha_Rename([s]).result()
            for cond in reversed(self.conds):
                then_br = Alpha_Rename([s]).result()
                else_br = [LoopIR.If(cond, then_br, else_br, None, s.srcinfo)]
            return else_br

        return super().map_s(s)


class _DoAddGuard(LoopIR_Rewrite):
    def __init__(self, proc, stmt, itr_stmt, val):
        assert val == 0

        self.stmt = stmt
        self.loop = itr_stmt
        self.itr = itr_stmt.iter
        self.val = val
        self.in_loop = False

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.loop:
            self.in_loop = True
            hi = self.map_e(s.hi)
            body = self.map_stmts(s.body)
            eff = self.map_eff(s.eff)
            self.in_loop = False
            return [type(s)( s.iter, hi, body, eff, s.srcinfo )]
        if s is self.stmt:
            if not self.in_loop:
                raise SchedulingError(f"statement is not inside the loop {self.itr}")
            if self.itr in _FV([s]):
                raise SchedulingError(f"expected {self.itr} not to be used"+
                                      " in statement {s}")
            if not _is_idempotent([s]):
                raise SchedulingError(f"statement {s} is not idempotent")

            cond = LoopIR.BinOp('==', LoopIR.Read(self.itr, [], T.index, s.srcinfo),
                                      LoopIR.Const(self.val, T.int, s.srcinfo), T.bool, s.srcinfo)
            return [LoopIR.If(cond, [s], [], None, s.srcinfo)]

        return super().map_s(s)


def _get_constant_bound(e):
    if isinstance(e, LoopIR.BinOp) and e.op == '%':
        return e.rhs
    raise SchedulingError(f'Could not derive constant bound on {e}')


class _DoBoundAndGuard(LoopIR_Rewrite):
    def __init__(self, proc, loop):
        self.loop = loop
        super().__init__(proc)

    def map_s(self, s):
        if s == self.loop:
            assert isinstance(s, LoopIR.ForAll)
            bound = _get_constant_bound(s.hi)
            guard = LoopIR.If(
                LoopIR.BinOp('<',
                             LoopIR.Read(s.iter, [], T.index, s.srcinfo),
                             s.hi,
                             T.bool,
                             s.srcinfo),
                s.body,
                [],
                None,
                s.srcinfo
            )
            return [LoopIR.ForAll(s.iter, bound, [guard], None, s.srcinfo)]

        return super().map_s(s)


class _DoMergeGuard(LoopIR_Rewrite):
    def __init__(self, proc, stmt1, stmt2):
        assert isinstance(stmt1, LoopIR.If)
        assert isinstance(stmt2, LoopIR.If)

        self.stmt1 = stmt1
        self.stmt2 = stmt2
        self.found_first = False

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        new_stmts = []

        for b in stmts:
            if self.found_first:
                if b != self.stmt2:
                    raise SchedulingError("expected the second stmt to be "
                                          "directly after the first stmt")
                self.found_first = False

                body = self.stmt1.body + self.stmt2.body
                orelse = self.stmt1.orelse + self.stmt2.orelse
                b = LoopIR.If(b.cond, body, orelse, None, b.srcinfo)

            if b is self.stmt1:
                self.found_first = True
                continue

            for s in self.map_s(b):
                new_stmts.append(s)

        return new_stmts


class _DoFuseLoop(LoopIR_Rewrite):
    def __init__(self, proc, loop1, loop2):
        self.loop1 = loop1
        self.loop2 = loop2
        self.found_first = False

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_stmts(self, stmts):
        new_stmts = []

        for b in stmts:
            if self.found_first:
                if b != self.loop2:
                    raise SchedulingError("expected the second stmt to be "
                                          "directly after the first stmt")
                self.found_first = False

                # TODO: Is this enough??
                # Check that loop is equivalent
                if self.loop1.iter.name() != self.loop2.iter.name():
                    raise SchedulingError("expected loop iteration variable "
                                          "to match")

                # Structural match
                if self.loop1.hi != self.loop2.hi:
                    raise SchedulingError("Loop bounds do not match!")

                # TODO: Check sth about stmts? Safe for Seq loops? etc. etc.

                body1 = SubstArgs(
                    self.loop1.body,
                    {self.loop1.iter: LoopIR.Read(self.loop2.iter, [], T.index,
                                                  self.loop2.srcinfo)}
                ).result()
                body = body1 + self.loop2.body

                b = type(self.loop1)(self.loop2.iter, self.loop2.hi, body, None,
                                     b.srcinfo)

            if b is self.loop1:
                self.found_first = True
                continue

            for s in self.map_s(b):
                new_stmts.append(s)

        return new_stmts

class _DoFuseIf(LoopIR_Rewrite):
    def __init__(self, proc, if1, if2):
        self.if1 = if1
        self.if2 = if2

        super().__init__(proc)

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
                    raise SchedulingError("expected the second stmt to be "
                                          "directly after the first stmt")

                # Check that conditions are identical
                if self.if1.cond != self.if2.cond:
                    raise SchedulingError("expected conditions to match")

                stmt = LoopIR.If(
                    self.if1.cond,
                    self.if1.body + self.if2.body,
                    self.if1.orelse + self.if2.orelse,
                    None,
                    self.if1.srcinfo
                )

            new_stmts.extend(self.map_s(stmt))

        return new_stmts


class _DoAddLoop(LoopIR_Rewrite):
    def __init__(self, proc, stmt, var, hi):
        self.stmt = stmt
        self.var  = var
        self.hi   = hi

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.stmt:
            if not _is_idempotent([s]):
                raise SchedulingError("expected stmt to be idempotent!")

            sym = Sym(self.var)
            hi  = LoopIR.Const(self.hi, T.int, s.srcinfo)
            ir  = LoopIR.ForAll(sym, hi, [s], None, s.srcinfo)
            return [ir]

        return super().map_s(s)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Factor out a sub-statement as a Procedure scheduling directive

def _make_closure(name, stmts, var_types):
    FVs     = _FV(stmts)
    info    = stmts[0].srcinfo

    # work out the calling arguments (args) and sub-proc args (fnargs)
    args    = []
    fnargs  = []

    # first, scan over all the arguments and convert them.
    # accumulate all size symbols separately
    sizes   = set()
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
    sizes   = list(sizes)
    args    = [ LoopIR.Read(sz, [], T.size, info) for sz in sizes ] + args
    fnargs  = [ LoopIR.fnarg(sz, T.size, None, info)
                for sz in sizes ] + fnargs

    eff     = None
    # TODO: raise NotImplementedError("need to figure out effect of new closure")
    closure = LoopIR.proc(name, fnargs, [], stmts, None, eff, info)

    return closure, args


class _DoInsertPass(LoopIR_Rewrite):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        super().__init__(proc)

    def map_s(self, s):
        if s is self.stmt:
            return [LoopIR.Pass(eff_null(s.srcinfo), srcinfo=s.srcinfo), s]
        return super().map_s(s)


class _DoDeleteConfig(LoopIR_Rewrite):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        super().__init__(proc)

    def map_s(self, s):
        if s is self.stmt:
            return []
        else:
            return super().map_s(s)


class _DoDeletePass(LoopIR_Rewrite):
    def map_s(self, s):
        if isinstance(s, LoopIR.Pass):
            return []
        else:
            return super().map_s(s)


class _DoExtractMethod(LoopIR_Rewrite):
    def __init__(self, proc, name, stmt):
        assert isinstance(stmt, LoopIR.stmt)
        self.orig_proc      = proc
        self.sub_proc_name  = name
        self.match_stmt     = stmt
        self.new_subproc    = None

        self.var_types      = ChainMap()

        for a in proc.args:
            self.var_types[a.name] = a.type

        super().__init__(proc)

    def subproc(self):
        return self.new_subproc

    def push(self):
        self.var_types = self.var_types.new_child()

    def pop(self):
        self.var_types = self.var_types.parents

    def map_s(self, s):
        if s is self.match_stmt:
            subproc, args = _make_closure(self.sub_proc_name,
                                          [s], self.var_types)
            self.new_subproc = subproc
            return [LoopIR.Call(subproc, args, None, s.srcinfo)]
        elif isinstance(s, LoopIR.Alloc):
            self.var_types[s.name] = s.type
            return [s]
        elif isinstance(s, LoopIR.ForAll):
            self.push()
            self.var_types[s.iter] = T.index
            body = self.map_stmts(s.body)
            self.pop()
            return [LoopIR.ForAll(s.iter, s.hi, body, None, s.srcinfo)]
        elif isinstance(s, LoopIR.If):
            self.push()
            body = self.map_stmts(s.body)
            self.pop()
            self.push()
            orelse = self.map_stmts(s.orelse)
            self.pop()
            return [LoopIR.If(s.cond, body, orelse, None, s.srcinfo)]
        else:
            return super().map_s(s)

    def map_e(self, e):
        return e


class _DoSimplify(LoopIR_Rewrite):
    def __init__(self, proc):
        self.facts = ChainMap()
        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def cfold(self, op, lhs, rhs):
        if op == '+':
            return lhs.val + rhs.val
        if op == '-':
            return lhs.val - rhs.val
        if op == '*':
            return lhs.val * rhs.val
        if op == '/':
            if lhs.type == T.f64 or lhs.type == T.f32:
                return lhs.val / rhs.val
            else:
                return lhs.val // rhs.val
        if op == '%':
            return lhs.val % rhs.val
        if op == 'and':
            return lhs.val and rhs.val
        if op == 'or':
            return lhs.val or rhs.val
        if op == '<':
            return lhs.val < rhs.val
        if op == '>':
            return lhs.val > rhs.val
        if op == '<=':
            return lhs.val <= rhs.val
        if op == '>=':
            return lhs.val >= rhs.val
        if op == '==':
            return lhs.val == rhs.val
        raise ValueError(f'Unknown operator ({op})')

    @staticmethod
    def is_quotient_remainder(e):
        """
        Checks if e is of the form (up to commutativity):
            N % K + K * (N / K)
        and returns N if so. Otherwise, returns None.
        """
        assert isinstance(e, LoopIR.BinOp)
        if e.op != '+':
            return None

        if isinstance(e.lhs, LoopIR.BinOp) and e.lhs.op == '%':
            assert isinstance(e.lhs.rhs, LoopIR.Const)
            num = e.lhs.lhs
            mod: LoopIR.Const = e.lhs.rhs
            rem = e.lhs
            quot = e.rhs
        elif isinstance(e.rhs, LoopIR.BinOp) and e.rhs.op == '%':
            assert isinstance(e.rhs.rhs, LoopIR.Const)
            num = e.rhs.lhs
            mod: LoopIR.Const = e.rhs.rhs
            rem = e.rhs
            quot = e.lhs
        else:
            return None

        # Validate form of remainder
        if not (isinstance(rem, LoopIR.BinOp) and rem.op == '%'
                and str(rem.lhs) == str(num) and str(rem.rhs) == str(mod)):
            return None

        # Validate form of quotient
        if not (isinstance(quot, LoopIR.BinOp) and quot.op == '*'):
            return None

        def check_quot(const, div):
            if (isinstance(const, LoopIR.Const)
                    and (isinstance(div, LoopIR.BinOp) and div.op == '/')
                    and (str(const) == str(mod))
                    and (str(div.lhs) == str(num))
                    and (str(div.rhs) == str(mod))):
                return num
            return None

        return check_quot(quot.lhs, quot.rhs) or check_quot(quot.rhs, quot.lhs)

    def map_binop(self, e: LoopIR.BinOp):
        lhs = self.map_e(e.lhs)
        rhs = self.map_e(e.rhs)

        if isinstance(lhs, LoopIR.Const) and isinstance(rhs, LoopIR.Const):
            return LoopIR.Const(self.cfold(e.op, lhs, rhs), lhs.type,
                                lhs.srcinfo)

        if e.op == '+':
            if isinstance(lhs, LoopIR.Const) and lhs.val == 0:
                return rhs
            if isinstance(rhs, LoopIR.Const) and rhs.val == 0:
                return lhs
            if val := self.is_quotient_remainder(
                    LoopIR.BinOp(e.op, lhs, rhs, lhs.type, lhs.srcinfo)):
                return val
        elif e.op == '-':
            if isinstance(rhs, LoopIR.Const) and rhs.val == 0:
                return lhs
            if isinstance(lhs, LoopIR.BinOp) and lhs.op == '+':
                if lhs.lhs == rhs:
                    return lhs.rhs
                if lhs.rhs == rhs:
                    return lhs.lhs
        elif e.op == '*':
            if isinstance(lhs, LoopIR.Const) and lhs.val == 0:
                return LoopIR.Const(0, lhs.type, lhs.srcinfo)
            if isinstance(rhs, LoopIR.Const) and rhs.val == 0:
                return LoopIR.Const(0, lhs.type, lhs.srcinfo)
            if isinstance(lhs, LoopIR.Const) and lhs.val == 1:
                return rhs
            if isinstance(rhs, LoopIR.Const) and rhs.val == 1:
                return lhs
        elif e.op == '/':
            if isinstance(rhs, LoopIR.Const) and rhs.val == 1:
                return lhs
        elif e.op == '%':
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
            e = super().map_e(e)

        # After simplifying, we might match a known constant, so check again.
        if const := self.is_known_constant(e):
            return const

        return e

    def add_fact(self, cond):
        if (isinstance(cond, LoopIR.BinOp) and cond.op == '=='
                and isinstance(cond.rhs, LoopIR.Const)):
            expr = cond.lhs
            const = cond.rhs
        elif (isinstance(cond, LoopIR.BinOp) and cond.op == '=='
              and isinstance(cond.lhs, LoopIR.Const)):
            expr = cond.rhs
            const = cond.lhs
        else:
            return

        self.facts[str(expr)] = const

        # if we know that X / M == 0 then we also know that X % M == X.
        if (isinstance(expr, LoopIR.BinOp) and expr.op == '/'
                and const.val == 0):
            mod_expr = LoopIR.BinOp('%', expr.lhs, expr.rhs, expr.type,
                                    expr.srcinfo)
            self.facts[str(mod_expr)] = expr.lhs

    def is_known_constant(self, e):
        if self.facts:
            return self.facts.get(str(e))
        return None

    def map_s(self, s):
        if isinstance(s, LoopIR.If):
            cond = self.map_e(s.cond)

            # If constant true or false, then drop the branch
            if isinstance(cond, LoopIR.Const):
                if cond.val:
                    return super().map_stmts(s.body)
                else:
                    return super().map_stmts(s.orelse)

            # Try to use the condition while simplifying body
            self.facts = self.facts.new_child()
            self.add_fact(cond)
            body = self.map_stmts(s.body)
            self.facts = self.facts.parents

            # Try to use the negation while simplifying orelse
            self.facts = self.facts.new_child()
            # TODO: negate fact here
            orelse = self.map_stmts(s.orelse)
            self.facts = self.facts.parents

            return [LoopIR.If(cond, body, orelse, self.map_eff(s.eff),
                              s.srcinfo)]
        elif isinstance(s, LoopIR.ForAll):
            hi = self.map_e(s.hi)
            # Delete the loop if it would not run at all
            if isinstance(hi, LoopIR.Const) and hi.val == 0:
                return []

            # Delete the loop if it would have an empty body
            body = self.map_stmts(s.body)
            if not body:
                return []
            return [LoopIR.ForAll(s.iter, hi, body, self.map_eff(s.eff),
                                  s.srcinfo)]
        else:
            return super().map_s(s)


class _AssertIf(LoopIR_Rewrite):
    def __init__(self, proc, if_stmt, cond):
        assert type(if_stmt) is LoopIR.If
        assert type(cond) is bool

        self.if_stmt = if_stmt
        self.cond    = cond

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        if s is self.if_stmt:
            # TODO: Gilbert's SMT thing should do this safely
            if self.cond:
                return self.map_stmts(s.body)
            else:
                return self.map_stmts(s.orelse)

        return super().map_s(s)


class _DoDataReuse(LoopIR_Rewrite):
    def __init__(self, proc, buf_pat, rep_pat):
        assert type(buf_pat) is LoopIR.Alloc
        assert type(rep_pat) is LoopIR.Alloc
        assert buf_pat.type == rep_pat.type

        self.buf_name = buf_pat.name
        self.rep_name = rep_pat.name
        self.rep_pat = rep_pat

        self.found_rep = False
        self.first_assn = False

        super().__init__(proc)

        self.proc = InferEffects(self.proc).result()

    def map_s(self, s):
        # Check that buf_name is only used before the first assignment of rep_pat
        if self.first_assn:
            if self.buf_name in _FV([s]):
                raise SchedulingError("buf_name should not be used after the "
                                      "first  assignment of rep_pat")

        if s is self.rep_pat:
            self.found_rep = True
            return []

        if self.found_rep:
            if type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce:
                rhs = self.map_e(s.rhs)
                name = s.name
                if s.name == self.rep_name:
                    name  = self.buf_name
                    if not self.first_assn:
                        self.first_assn = True

                return [type(s)(name, s.type, None, s.idx, rhs, None, s.srcinfo)]


        return super().map_s(s)

    def map_e(self, e):
        if type(e) is LoopIR.Read and e.name == self.rep_name:
            return LoopIR.Read(self.buf_name, e.idx, e.type, e.srcinfo)

        return super().map_e(e)

class _DoStageWindow(LoopIR_Rewrite):
    def __init__(self, proc, new_name, memory, expr):
        # Inputs
        self.new_name = Sym(new_name)
        self.memory = memory
        self.target_expr = expr

        # Visitor state
        self._found_expr = False
        self._complete = False
        self._copy_code = None

        proc = InferEffects(proc).result()

        super().__init__(proc)

    def _stmt_writes_to_window(self, s):
        for eff in s.eff.reduces + s.eff.writes:
            if self.target_expr.name == eff.buffer:
                return True
        return False

    def _make_staged_alloc(self):
        '''
        proc(Win[0:10, N, lo:hi])
        =>
        Staged : ty[10, hi - lo]
        for i0 in par(0, 10):
          for i1 in par(0, hi - lo):
            Staged[i0, i1] = Buf[0 + i0, N, lo + i1]
        proc(Staged[0:10, 0:(hi - lo)])
        '''

        staged_extents = []  # e.g. 10, hi - lo
        staged_vars = []  # e.g. i0, i1
        staged_var_reads = []  # reads of staged_vars

        buf_points = []  # e.g. 0 + i0, N, lo + i1

        for idx in self.target_expr.idx:
            assert isinstance(idx, (LoopIR.Interval, LoopIR.Point))

            if isinstance(idx, LoopIR.Interval):
                assert isinstance(idx.hi.type, (T.Index, T.Size)), \
                    f'{idx.hi.type}'

                sym_i = Sym(f'i{len(staged_vars)}')
                staged_vars.append(sym_i)
                staged_extents.append(
                    LoopIR.BinOp('-', idx.hi, idx.lo, T.index, idx.srcinfo)
                )
                offset = LoopIR.Read(sym_i, [], T.index, idx.lo.srcinfo)
                buf_points.append(
                    LoopIR.BinOp('+', idx.lo, offset, T.index, idx.srcinfo)
                )
                staged_var_reads.append(
                    LoopIR.Read(sym_i, [], T.index, idx.lo.srcinfo)
                )
            elif isinstance(idx, LoopIR.Point):
                # TODO: test me!
                buf_points.append(idx.pt)

        assert staged_vars, "Window expression had no intervals"
        assert len(staged_vars) == len(staged_extents)

        # Staged : ty[10, hi - lo]
        srcinfo = self.target_expr.srcinfo
        data_type = self.target_expr.type.src_type.type
        alloc_type = T.Tensor(staged_extents, False, data_type)
        alloc = LoopIR.Alloc(self.new_name, alloc_type, self.memory, None,
                             srcinfo)

        # Staged[i0, i1] = Buf[0 + i0, N, lo + i1]
        copy_stmt = LoopIR.Assign(
            self.new_name,
            data_type,
            None,
            staged_var_reads,
            LoopIR.Read(self.target_expr.name, buf_points,
                        data_type, srcinfo),
            None,
            srcinfo
        )

        # for i0 in par(0, 10):
        #     for i1 in par(0, hi - lo):
        for sym_i, extent_i in reversed(list(zip(staged_vars, staged_extents))):
            copy_stmt = LoopIR.ForAll(sym_i, extent_i, [copy_stmt], None,
                                      srcinfo)

        # Staged[0:10, 0:(hi - lo)]
        w_extents = [
            LoopIR.Interval(LoopIR.Const(0, T.index, srcinfo), hi, srcinfo)
            for hi in staged_extents
        ]
        new_window = LoopIR.WindowExpr(
            self.new_name,
            w_extents,
            T.Window(
                data_type,
                alloc_type,
                self.new_name,
                w_extents
            ),
            srcinfo
        )

        return [alloc, copy_stmt], new_window

    def map_stmts(self, stmts):
        result = []
        for s in stmts:
            s = self.map_s(s)
            if self._found_expr and not self._complete:
                assert len(s) == 1
                assert self._copy_code
                s = s[0]

                if self._stmt_writes_to_window(s):
                    raise NotImplementedError('StageWindow does not handle '
                                              'writes yet.')
                s = self._copy_code + [s]
                self._complete = True
            result.extend(s)
        return result

    def map_e(self, e):
        if self._found_expr:
            return e
        if e is self.target_expr:
            self._found_expr = True
            self._copy_code, new_window = self._make_staged_alloc()
            return new_window
        return super().map_e(e)


class _DoBoundAlloc(LoopIR_Rewrite):
    def __init__(self, proc, alloc_site, bounds):
        self.alloc_site = alloc_site
        self.bounds = bounds
        super().__init__(proc)

    def map_s(self, s):
        if s is self.alloc_site:
            assert isinstance(s.type, T.Tensor)
            if len(self.bounds) != len(s.type.hi):
                raise SchedulingError(
                    f'bound_alloc: dimensions do not match: {len(self.bounds)} '
                    f'!= {len(s.type.hi)} (expected)')

            new_type = T.Tensor(
                [(new if new else old)
                 for old, new in zip(s.type.hi, self.bounds)],
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
    DoParToSeq = _DoParToSeq
    DoReorderStmt = _DoReorderStmt
    DoConfigWriteAfter = _ConfigWriteAfter
    DoConfigWriteRoot = _ConfigWriteRoot
    DoInlineWindow = _InlineWindow
    DoInsertPass = _DoInsertPass
    DoDeletePass = _DoDeletePass
    DoSimplify = _DoSimplify
    DoAddGuard = _DoAddGuard
    DoBoundAndGuard = _DoBoundAndGuard
    DoMergeGuard = _DoMergeGuard
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
    DoStageWindow = _DoStageWindow
    DoBoundAlloc = _DoBoundAlloc
    DoExpandDim    = _DoExpandDim
    DoRearrangeDim  = _DoRearrangeDim
