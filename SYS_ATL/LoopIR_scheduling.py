from .prelude import *
from .LoopIR import LoopIR, LoopIR_Rewrite, Alpha_Rename, LoopIR_Do, SubstArgs
from .LoopIR import T
from .LoopIR import lift_to_eff_expr
from .LoopIR_effects import Effects as E
from .LoopIR_effects import get_effect_of_stmts
from .LoopIR_effects import (eff_union, eff_filter, eff_bind,
                             eff_null, eff_remove_buf)
from .effectcheck import InferEffects
import re

from collections import defaultdict, ChainMap

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Scheduling Errors

class SchedulingError(Exception):
    pass


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Finding Names

def name_plus_count(namestr):
    results = re.search(r"^([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$", namestr)
    if not results:
        raise TypeError("expected name pattern of the form\n"+
                        "  ident (# integer)?\n"+
                        "where ident is the name of a variable "+
                        "and (e.g.) '#2' may optionally be attached to mean "+
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

    pattern     = (f"for {name} in _:\n"+
                   f"  for {inner} in _: _{count}")
    return pattern

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reorder scheduling directive


class _Reorder(LoopIR_Rewrite):
    def __init__(self, proc, loop_stmt):
        self.stmt    = loop_stmt
        self.out_var = loop_stmt.iter
        if ( len(loop_stmt.body) != 1 or
             type(loop_stmt.body[0]) is not LoopIR.ForAll ):
            raise SchedulingError(f"expected loop directly inside of "+
                                  f"{self.out_var} loop")
        self.in_var  = loop_stmt.body[0].iter

        super().__init__(proc)

    def map_s(self, s):
        if s == self.stmt:
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
                return eff_bind(x, eff, pred=cond, config_pred=cond_nz)

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
        if s == self.split_loop:
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
                assert type(rhs) is LoopIR.Const and rhs.val > 1
                rhs_1 = cnst(rhs.val-1)
                return szop("/", szop("+", lhs, rhs_1), rhs)

            def rng(x, hi):
                lhs = boolop("<=", cnst(0), x)
                rhs = boolop("<", x, hi)
                return boolop("and", lhs, rhs)
            def do_bind(x, hi, eff):
                cond    = lift_to_eff_expr( rng(rd(x),hi) )
                cond_nz = boolop("<", cnst(0), hi)
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
                if type(s.hi) is not LoopIR.Const:
                    raise SchedulingError(
                        f"cannot perfectly split the '{s.iter}' loop "+
                        f"unless it has a constant bound")
                elif s.hi.val % self.quot != 0:
                    raise SchedulingError(
                        f"cannot perfectly split the '{s.iter}' loop "+
                        f"because {self.quot} does not evenly divide "+
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
        if type(e) is LoopIR.Read:
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
        if type(e) is E.Var:
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
        if s == self.unroll_loop:
        #if type(s) is LoopIR.ForAll:
        #    # unroll this to loops!!
        #    if s.iter is self.unroll_var:
            if type(s.hi) is not LoopIR.Const:
                raise SchedulingError(f"expected loop '{s.iter}' "+
                                      f"to have constant bounds")
            #if len(s.body) != 1:
            #    raise SchedulingError(f"expected loop '{s.iter}' "+
            #                          f"to have only one body")
            hi      = s.hi.val
            assert hi > 0
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
        if type(e) is LoopIR.Read:
            if e.type is T.index:
            # This is an unrolled variable, substitute it!
                if e.name is self.unroll_var:
                    return LoopIR.Const(self.unroll_itr, T.index, e.srcinfo)

        # fall-through
        return super().map_e(e)

    def map_eff_e(self, e):
        if type(e) is E.Var:
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
        assert type(call_stmt) is LoopIR.Call
        self.orig_proc  = proc
        self.call_stmt  = call_stmt
        self.env        = {}

        super().__init__(proc)

    def map_s(self, s):
        if s == self.call_stmt:
            # handle potential window expressions in call positions
            win_binds   = []
            def map_bind(nm, a):
                if type(a) is LoopIR.WindowExpr:
                    stmt = LoopIR.WindowStmt( nm, a, eff_null(a.srcinfo),
                                                     a.srcinfo )
                    win_binds.append(stmt)
                    return LoopIR.Read( nm, [], a.type, a.srcinfo )
                else:
                    return a

            # first, set-up a binding from sub-proc arguments
            # to supplied expressions at the call-site
            call_bind   = { xd.name : map_bind(xd.name, a)
                            for xd,a in zip(s.f.args,s.args) }


            # whenever we copy code we need to alpha-rename for safety
            body        = Alpha_Rename(s.f.body).result()

            # then we will substitute the bindings for the call
            body        = SubstArgs(body, call_bind).result()

            # note that all sub-procedure assertions must be true
            # even if not asserted, or else this call being inlined
            # wouldn't have been valid to make in the first place

            # the code to splice in at this point
            return win_binds + body

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
        self.env        = {}
        arg_gap         = len(proc.args) - len(arg_vals)
        assert arg_gap >= 0
        arg_vals = list(arg_vals) + [ None for _ in range(arg_gap) ]

        self.orig_proc = proc

        # bind values for partial evaluation
        for v, a in zip(arg_vals, proc.args):
            if v is None:
                pass
            elif a.type.is_indexable():
                if type(v) is int:
                    self.env[a.name] = v
                else:
                    raise SchedulingError("cannot partially evaluate "+
                                          "to a non-int value")
            else:
                raise SchedulingError("cannot partially evaluate "+
                                      "numeric (non-index) arguments")

        args    = [ self.map_fnarg(a) for v,a in zip(arg_vals, proc.args)
                                      if v is None ]
        preds   = [ self.map_e(p) for p in self.orig_proc.preds ]
        body    = self.map_stmts(self.orig_proc.body)
        eff     = self.map_eff(self.orig_proc.eff)

        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = args,
                                preds   = preds,
                                body    = body,
                                instr   = None,
                                eff     = eff,
                                srcinfo = self.orig_proc.srcinfo)

    def map_e(self,e):
        if type(e) is LoopIR.Read:
            if e.type.is_indexable():
                assert len(e.idx) == 0
                if e.name in self.env:
                    return LoopIR.Const(self.env[e.name], T.int, e.srcinfo)

        return super().map_e(e)

    def map_eff_e(self,e):
        if type(e) is E.Var:
            assert e.type.is_indexable()
            if e.name in self.env:
                return E.Const(self.env[e.name], T.int, e.srcinfo)

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
        elif type(t) is T.Tensor:
            assert t.type.is_real_scalar()
            return T.Tensor(t.hi, t.is_window, self.basetyp)
        else:
            assert False, "bad case"

    def change_window(self, t):
        assert type(t) is T.Tensor
        assert type(self.win) is bool
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
                raise SchedulingError("cannot change the precision of a "+
                                      "non-numeric argument")
            typ = self.change_precision(typ)
        elif self.win is not None:
            if not a.type.is_tensor_or_window():
                raise SchedulingError("cannot change windowing of a "+
                                      "non-tensor/window argument")
            typ = self.change_window(typ)
        else:
            assert self.mem is not None
            if not a.type.is_numeric():
                raise SchedulingError("cannot change the memory of a "+
                                      "non-numeric argument")
            mem = self.mem

        return LoopIR.fnarg( a.name, typ, mem, a.srcinfo )

    def map_s(self, s):
        if self.early_exit():
            return [s]

        if type(s) is LoopIR.Alloc and str(s.name) == self.name:
            if self.check_inst():

                # if that passed, we definitely found the symbol being pointed at
                # So attempt the substitution
                typ     = s.type
                assert typ.is_numeric()
                mem     = s.mem
                if self.basetyp is not None:
                    typ = self.change_precision(typ)
                elif self.win is not None:
                    raise SchedulingError("cannot change an allocation to "+
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
        assert type(call_stmt) is LoopIR.Call
        self.orig_proc  = proc
        self.call_stmt  = call_stmt
        self.new_subproc = new_subproc

        super().__init__(proc)

    def map_s(self, s):
        if s == self.call_stmt:
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


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Bind Expression scheduling directive


class _BindExpr(LoopIR_Rewrite):
    def __init__(self, proc, new_name, expr):
        assert isinstance(expr, LoopIR.expr)
        assert expr.type == T.R
        self.orig_proc  = proc
        self.new_name   = Sym(new_name)
        self.expr       = expr
        self.found_expr = False

        super().__init__(proc)

    def map_s(self, s):
        # handle recursive part of pass at this statement
        stmts = super().map_s(s)
        if self.found_expr:
            # TODO Fix Assign, probably wrong
            stmts = [ LoopIR.Alloc(self.new_name, T.R, None, None, s.srcinfo),
                      LoopIR.Assign(self.new_name, s.type, s.cast, [],
                                    self.expr, None, self.expr.srcinfo )
                    ] + stmts
            self.found_expr = False

        return stmts

    def map_e(self, e):
        if e is self.expr:
            self.found_expr = True
            return LoopIR.Read(self.new_name, [], e.type, e.srcinfo)
        else:
            return super().map_e(e)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Lift Allocation scheduling directive

# data-flow dependencies between variable names
class _Alloc_Dependencies(LoopIR_Do):
    def __init__(self, buf_sym, stmts):
        self._buf_sym = buf_sym
        self._lhs     = None
        self._depends = defaultdict(lambda: set())
        self._alias   = defaultdict(lambda: None)

        self.do_stmts(stmts)

    def result(self):
        depends = self._depends[self._buf_sym]
        new     = list(depends)
        done    = []
        while True:
            if len(new) == 0:
                break
            sym = new.pop()
            done.append(sym)
            d = self._depends[sym]
            depends.update(d)
            [ new.append(s) for s in d if s not in done ]

        return depends

    def do_s(self, s):
        if type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce:
            lhs         = self._alias[s.name] or s.name
            self._lhs   = lhs
            self._depends[lhs].add(lhs)
        elif type(s) is LoopIR.WindowStmt:
            rhs_buf     = self._alias[s.rhs.name] or s.rhs.name
            self._alias[s.lhs] = rhs_buf
            self._lhs   = rhs_buf
            self._depends[rhs_buf].add(rhs_buf)
        elif type(s) is LoopIR.Call:
            # internal dependencies of each argument
            for a in s.args:
                self.do_e(a)
            # giant cross-bar of dependencies on the arguments
            for fa, a in zip(s.f.args, s.args):
                if fa.type.is_numeric():
                    name = self._alias[a.name] or a.name
                    # handle any potential indexing of this variable
                    for aa in s.args:
                        if aa.type.is_indexable():
                            self.do_e(aa)
                    # if this buffer is being written,
                    # then handle dependencies on other buffers
                    maybe_write = self.analyze_eff(s.f.eff, fa.name,
                                                   write=True)
                    if maybe_write:
                        self._lhs = name
                        for faa, aa in zip(s.f.args, s.args):
                            if faa.type.is_numeric():
                                maybe_read  = self.analyze_eff(s.f.eff, faa.name,
                                                               read=True)
                                if maybe_read:
                                    self.do_e(aa)
                        self._lhs = None

            # already handled all sub-terms above
            # don't do the usual statement processing
            return

        super().do_s(s)
        self._lhs = None

    def analyze_eff(self, eff, buf, write=False, read=False):
        if read:
            if any(es.buffer == buf for es in eff.reads):
                return True
        if write:
            if any(es.buffer == buf for es in eff.writes):
                return True
        if read or write:
            if any(es.buffer == buf for es in eff.reduces):
                return True

        return False

    def do_e(self, e):
        if type(e) is LoopIR.Read or type(e) is LoopIR.WindowExpr:
            def visit_idx(e):
                if type(e) is LoopIR.Read:
                    for i in e.idx:
                        self.do_e(i)
                else:
                    for w in e.idx:
                        if type(w) is LoopIR.Interval:
                            self.do_e(w.lo)
                            self.do_e(w.hi)
                        else:
                            self.do_e(w.pt)

            lhs     = self._lhs
            name    = self._alias[e.name] or e.name
            self._lhs = name
            if lhs:
                self._depends[lhs].add(name)
            visit_idx(e)
            self._lhs = lhs
            visit_idx(e)

        else:
            super().do_e(e)

    def do_t(self, t):
        pass
    def do_eff(self, eff):
        pass

class _LiftAlloc(LoopIR_Rewrite):
    def __init__(self, proc, alloc_stmt, n_lifts, mode, size):
        assert type(alloc_stmt) is LoopIR.Alloc
        assert is_pos_int(n_lifts)
        self.orig_proc      = proc
        self.alloc_stmt     = alloc_stmt
        self.alloc_sym      = alloc_stmt.name
        self.alloc_deps     = _Alloc_Dependencies(self.alloc_sym,
                                                  proc.body).result()
        self.lift_mode      = mode
        self.lift_size      = size

        self.n_lifts        = n_lifts

        self.ctrl_ctxt      = []
        self.lift_site      = None

        self.lifted_stmt    = None
        self.access_idxs    = None
        self.alloc_type     = None
        self._in_call_arg   = False

        super().__init__(proc)

        # repair effects...
        self.proc = InferEffects(self.proc).result()

    def idx_mode(self, access, orig):
        if self.lift_mode == 'row':
            return access + orig
        elif self.lift_mode == 'col':
            return orig + access
        else:
            raise SchedulingError(f"Unknown lift mode {self.lift_mode},"+
                                   "should be 'row' or 'col'")

    def map_s(self, s):
        if s == self.alloc_stmt:
            # mark the point we want to lift this alloc-stmt to
            n_up = min( self.n_lifts, len(self.ctrl_ctxt) )
            self.lift_site = self.ctrl_ctxt[-n_up]

            # extract the ranges and variables of enclosing loops
            idxs, rngs = self.get_ctxt_itrs_and_rngs(n_up)

            # compute the lifted allocation buffer type, and
            # the new allocation statement
            new_typ = s.type
            new_rngs = []
            for r in rngs:
                if type(r) is LoopIR.Const:
                    if r.val > 0:
                        new_rngs.append(r)
                    else:
                        assert False, "why loop bound is negative?"
                else:
                    new_rngs.append(
                            LoopIR.BinOp("+", r,
                                LoopIR.Const(1, T.int, r.srcinfo),
                                T.index, r.srcinfo) )

            if type(new_typ) is T.Tensor:
                if self.lift_mode == 'row':
                    new_rngs += new_typ.shape()
                elif self.lift_mode == 'col':
                    new_rngs = new_typ.shape() + new_rngs
                else:
                    raise SchedulingError(f"Unknown lift mode {self.lift_mode},"+
                                           "should be 'row' or 'col'")

                new_typ = new_typ.basetype()
            if len(new_rngs) > 0:
                new_typ = T.Tensor(new_rngs, False, new_typ)

            # effect remains null
            self.lifted_stmt = LoopIR.Alloc( s.name, new_typ, s.mem,
                                             None, s.srcinfo )
            self.access_idxs = idxs
            self.alloc_type  = new_typ

            # erase the statement from this location
            return []

        elif type(s) is LoopIR.If or type(s) is LoopIR.ForAll:
            # handle recursive part of pass at this statement
            self.ctrl_ctxt.append(s)
            stmts = super().map_s(s)
            self.ctrl_ctxt.pop()

            # splice in lifted statement at the point to lift-to
            if s == self.lift_site:
                stmts = [ self.lifted_stmt ] + stmts

            return stmts

        elif type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce:
            # in this case, we may need to substitute the
            # buffer name on the lhs of the assignment/reduction
            if s.name == self.alloc_sym:
                assert self.access_idxs is not None
                idx = self.idx_mode(
                        [ LoopIR.Read(i, [], T.index, s.srcinfo)
                        for i in self.access_idxs ] , s.idx)
                rhs = self.map_e(s.rhs)
                # return allocation or reduction...
                return [ type(s)( s.name, s.type, s.cast,
                                  idx, rhs, None, s.srcinfo ) ]

        elif type(s) is LoopIR.Call:
            # substitution in call arguments currently unsupported;
            # so setting flag here
            self._in_call_arg = True
            stmts = super().map_s(s)
            self._in_call_arg = False
            return stmts

        # fall-through
        return super().map_s(s)

    def map_e(self, e):
        if type(e) is LoopIR.Read and e.name == self.alloc_sym:
            assert self.access_idxs is not None
            if len(self.access_idxs) == 0:
                return e

            #if self._in_call_arg:
            if e.type.is_real_scalar():
                idx = self.idx_mode(
                        [ LoopIR.Read(i, [], T.index, e.srcinfo)
                        for i in self.access_idxs ] , e.idx)
                return LoopIR.Read( e.name, idx, e.type, e.srcinfo )
            else:
                assert self._in_call_arg
                assert len(e.idx) == 0
                # then we need to replace this read with a
                # windowing expression
                idx = self.idx_mode(
                        [ LoopIR.Point(LoopIR.Read(i, [], T.index, e.srcinfo),
                                     e.srcinfo)
                             for i in self.access_idxs ] ,
                        [ LoopIR.Interval(LoopIR.Const(0,T.int,e.srcinfo),
                                         hi, e.srcinfo)
                             for hi in e.type.shape() ])
                tensor_type = (e.type.as_tensor if type(e.type) is T.Window
                               else e.type)
                win_typ     = T.Window( self.alloc_type, tensor_type,
                                        e.name, idx )
                return LoopIR.WindowExpr( e.name, idx, win_typ, e.srcinfo )

        if type(e) is LoopIR.WindowExpr and e.name == self.alloc_sym:
            assert self.access_idxs is not None
            if len(self.access_idxs) == 0:
                return e
            # otherwise, extend windowing with accesses...

            idx = self.idx_mode(
                    [ LoopIR.Point(LoopIR.Read(i, [], T.index, e.srcinfo),
                                 e.srcinfo)
                    for i in self.access_idxs ] , e.idx)
            win_typ     = T.Window( self.alloc_type, e.type.as_tensor,
                                    e.name, idx )
            return LoopIR.WindowExpr( e.name, idx, win_typ, e.srcinfo )

        # fall-through
        return super().map_e(e)

    def get_ctxt_itrs_and_rngs(self, n_up):
        rngs    = []
        idxs    = []
        for s in self.ctrl_ctxt[-n_up:]:
            if type(s) is LoopIR.If:
                # if-statements do not affect allocations
                # note that this may miss opportunities to
                # shrink the allocation by being aware of
                # guards; oh well.
                continue
            elif type(s) is LoopIR.ForAll:
                # note, do not accrue false dependencies
                if s.iter in self.alloc_deps:
                    idxs.append(s.iter)
                    if type(s.hi) == LoopIR.Read:
                        assert s.hi.type.is_indexable()
                        assert len(s.hi.idx) == 0
                    elif type(s.hi) == LoopIR.Const:
                        assert s.hi.type == T.int
                    elif type(s.hi) == LoopIR.BinOp:
                        assert s.hi.type.is_indexable()
                    else:
                        assert False, "bad case"

                    if self.lift_size != None:
                        assert type(self.lift_size) is int
                        # TODO: More robust checking of
                        # self.lift_size >= s.hi
                        if type(s.hi) == LoopIR.Const:
                            if s.hi.val > self.lift_size:
                                raise SchedulingError(f"Lift size cannot "+
                                      f"be less than for-loop bound {s.hi.val}")
                        elif (type(s.hi) == LoopIR.BinOp and
                                s.hi.op == '%'):
                            assert type(s.hi.rhs) is LoopIR.Const
                            if s.hi.rhs.val > self.lift_size:
                                raise SchedulingError(f"Lift size cannot "+
                                      f"be less than for-loop bound {s.hi}")
                        else:
                            raise NotImplementedError

                        rngs.append(LoopIR.Const(self.lift_size, T.int, s.srcinfo))
                    else:
                        rngs.append(s.hi)
            else: assert False, "bad case"

        return (idxs, rngs)


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
        if type(s) is LoopIR.Alloc:
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
        if type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce:
            if s.name not in self._bound:
                self._fvs.add(s.name)
        elif type(s) is LoopIR.ForAll:
            self._bound.add(s.iter)
        elif type(s) is LoopIR.Alloc:
            self._bound.add(s.name)

        super().do_s(s)

    def do_e(self, e):
        if type(e) is LoopIR.Read:
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
        elif styp is LoopIR.ForAll:
            return _is_idempotent(s.body)
        else:
            return True

    return all( _stmt(s) for s in stmts )


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

    def result(self):
        return self.proc

    def alloc_check(self, pre, post):
        if not _is_alloc_free(pre, post):
            raise SchedulingError("Will not fission here, because "+
                                  "an allocation might be buried "+
                                  "in a different scope than some use-site")

    # returns a pair of stmt-lists
    # for those statements occuring before and
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
        if s == self.tgt_stmt:
            assert self.hit_fission == False
            self.hit_fission = True
            # none-the-less make sure we return this statement in
            # the pre-fission position
            return ([s],[])

        elif type(s) is LoopIR.If:

            # first, check if we need to split the body
            pre, post       = self.map_stmts(s.body)
            fission_body    = (len(pre) > 0 and len(post) > 0 and
                               self.n_lifts > 0)
            if fission_body:
                self.n_lifts -= 1
                self.alloc_check(pre, post)
                pre         = LoopIR.If(s.cond, pre, [], None, s.srcinfo)
                post        = LoopIR.If(s.cond, post, s.orelse, None, s.srcinfo)
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
                post        = LoopIR.If(s.cond, [], post, None, s.srcinfo)
                return ([pre],[post])

            orelse = pre+post

            # if we neither split the body nor the or-else,
            # then we need to gather together the pre and post.
            single_stmt = LoopIR.If(s.cond, body, orelse, None, s.srcinfo)

        elif type(s) is LoopIR.ForAll:

            # check if we need to split the loop
            pre, post       = self.map_stmts(s.body)
            do_fission      = (len(pre) > 0 and len(post) > 0 and
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
                if s.iter in _FV(post) or not _is_idempotent(pre):
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
                if type(sz) is Sym:
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

class _DoFactorOut(LoopIR_Rewrite):
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
        if s == self.match_stmt:
            subproc, args = _make_closure(self.sub_proc_name,
                                          [s], self.var_types)
            self.new_subproc = subproc
            return [LoopIR.Call(subproc, args, None, s.srcinfo)]
        elif type(s) is LoopIR.Alloc:
            self.var_types[s.name] = s.type
            return [s]
        elif type(s) is LoopIR.ForAll:
            self.push()
            self.var_types[s.iter] = T.index
            body    = self.map_stmts(s.body)
            self.pop()
            return [LoopIR.ForAll(s.iter, s.hi, body, None, s.srcinfo)]
        elif type(s) is LoopIR.If:
            self.push()
            body    = self.map_stmts(s.body)
            self.pop()
            self.push()
            orelse  = self.map_stmts(s.orelse)
            self.pop()
            return [LoopIR.If(s.cond, body, orelse, None, s.srcinfo)]
        else:
            return super().map_s(s)

    def map_e(self, e):
        return e







# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Passes to export

class Schedules:
    DoReorder           = _Reorder
    DoSplit             = _Split
    DoUnroll            = _Unroll
    DoInline            = _Inline
    DoPartialEval       = _PartialEval
    SetTypAndMem        = _SetTypAndMem
    DoCallSwap          = _CallSwap
    DoBindExpr          = _BindExpr
    DoLiftAlloc         = _LiftAlloc
    DoFissionLoops      = _FissionLoops
    DoFactorOut         = _DoFactorOut
