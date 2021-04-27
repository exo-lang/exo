from .prelude import *
from .LoopIR import LoopIR, LoopIR_Rewrite, Alpha_Rename, LoopIR_Do, SubstArgs
from .LoopIR import T
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

#
#   current name descriptor language
#
#       d    ::= e
#              | e > e
#
#       e    ::= prim[int]
#              | prim
#
#       prim ::= name-string
#

def name_str_2_symbols(proc, desc):
    assert type(proc) is LoopIR.proc
    # parse regular expression
    #   either name[int]
    #       or name
    name = re.search(r"^(\w+)", desc).group(0)
    idx = re.search(r"\[([0-9_]+)\]", desc)
    if idx is not None:
        idx = int(idx.group(1))
        # idx is a non-negative integer if present
        assert idx > 0
    else:
        idx = None

    # find all occurrences of name
    sym_list = []

    # search proc signature for symbol
    for a in proc.args:
        if str(a.name) == name:
            sym_list.append(a.name)

    def find_sym_stmt(node, nm):
        if type(node) is LoopIR.If:
            for b in node.body:
                find_sym_stmt(b, nm)
            for b in node.orelse:
                find_sym_stmt(b, nm)
        elif type(node) is LoopIR.Alloc:
            if str(node.name) == nm:
                sym_list.append(node.name)
        elif type(node) is LoopIR.ForAll:
            if str(node.iter) == nm:
                sym_list.append(node.iter)
            for b in node.body:
                find_sym_stmt(b, nm)

    # search proc body
    for b in proc.body:
        find_sym_stmt(b, name)

    if idx is not None:
        assert len(sym_list) >= idx
        res = [sym_list[idx-1]]
    else:
        res = sym_list

    return res

def name_str_2_pairs(proc, out_desc, in_desc):
    assert type(proc) is LoopIR.proc
    # parse regular expression
    #   either name[int]
    #       or name
    out_name = re.search(r"^(\w+)", out_desc).group(0)
    in_name = re.search(r"^(\w+)", in_desc).group(0)
    out_idx = re.search(r"\[([0-9_]+)\]", out_desc)
    in_idx = re.search(r"\[([0-9_]+)\]", in_desc)

    out_idx = int(out_idx.group(1)) if out_idx is not None else None
    in_idx  = int(in_idx.group(1)) if in_idx is not None else None

    # idx is a non-negative integer if present
    for idx in [out_idx, in_idx]:
        if idx is not None:
            assert idx > 0

    # find all occurrences of name
    pair_list = []
    out_cnt = 0
    in_cnt  = 0
    def find_sym_stmt(node, out_sym):
        if type(node) is LoopIR.If:
            for b in node.body:
                find_sym_stmt(b, out_sym)
            for b in node.orelse:
                find_sym_stmt(b, out_sym)
        elif type(node) is LoopIR.ForAll:
            # first, search for the outer name
            if out_sym is None and str(node.iter) == out_name:
                nonlocal out_cnt
                out_cnt += 1
                if out_idx is None or out_idx is out_cnt:
                    out_sym = node.iter
                    for b in node.body:
                        find_sym_stmt(b, out_sym)
                    out_sym = None
            # if we are inside of an outer name match...
            elif out_sym is not None and str(node.iter) == in_name:
                nonlocal in_cnt
                in_cnt += 1
                if in_idx is None or in_idx is in_cnt:
                    pair_list.append( (out_sym, node.iter) )
            for b in node.body:
                find_sym_stmt(b, out_sym)

    # search proc body
    for b in proc.body:
        find_sym_stmt(b, None)

    return pair_list


"""
Here is an example of nested naming insanity
The numbers give us unique identifiers for the names

for j =  0
    for i =  1
        for j =  2
            for i =  3
                for i =  4

searching for j,i
pair_list = [
    0,1
    0,3
    0,4
    2,3
    2,4
]
"""


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reorder scheduling directive


class _Reorder(LoopIR_Rewrite):
    def __init__(self, proc, out_var, in_var):
        self.out_var = out_var
        self.in_var  = in_var
        super().__init__(proc)

    def map_s(self, s):
        if type(s) is LoopIR.ForAll:
            if s.iter == self.out_var:
                if len(s.body) != 1 or type(s.body[0]) is not LoopIR.ForAll:
                    raise SchedulingError(f"expected loop directly inside of "+
                                          f"{self.out_var} loop")
                elif s.body[0].iter != self.in_var:
                    raise SchedulingError(f"expected loop directly inside of "+
                                          f"{self.out_var} loop to have "+
                                          f"iteration variable {self.in_var}")
                else:
                    # this is the actual body inside both for-loops
                    body = s.body[0].body
                    return [LoopIR.ForAll(s.body[0].iter, s.body[0].hi,
                                [LoopIR.ForAll(s.iter, s.hi,
                                    body,
                                    s.eff, s.srcinfo)],
                                s.body[0].eff, s.body[0].srcinfo)]

        # fall-through
        return super().map_s(s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Split scheduling directive


class _Split(LoopIR_Rewrite):
    def __init__(self, proc, split_var, quot, hi, lo, cut_tail=False):
        self.split_var = split_var
        self.quot = quot
        self.hi_i = Sym(hi)
        self.lo_i = Sym(lo)
        self.cut_i = Sym(lo)

        self._tail_strategy = 'cut' if cut_tail else 'guard'
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
        if type(s) is LoopIR.ForAll:
            # Split this to two loops!!
            if s.iter is self.split_var:
                # short-hands for sanity
                def boolop(op,lhs,rhs):
                    return LoopIR.BinOp(op,lhs,rhs,T.bool,s.srcinfo)
                def szop(op,lhs,rhs):
                    return LoopIR.BinOp(op,lhs,rhs,lhs.type,s.srcinfo)
                def cnst(intval):
                    return LoopIR.Const(intval, T.int, s.srcinfo)
                def rd(i):
                    return LoopIR.Read(i, [], T.index, s.srcinfo)

                # in the simple case, wrap body in a guard
                if self._tail_strategy == 'guard':
                    body    = self.map_stmts(s.body)
                    idx_sub = self.substitute(s.srcinfo)
                    cond    = boolop("<", idx_sub, s.hi)
                    body    = [LoopIR.If(cond, body, [], s.eff, s.srcinfo)]
                    lo_rng  = cnst(self.quot)
                    hi_rng  = szop("/", s.hi, lo_rng)

                    return [LoopIR.ForAll(self.hi_i, hi_rng,
                                [LoopIR.ForAll(self.lo_i, lo_rng,
                                    body,
                                    s.eff, s.srcinfo)],
                                s.eff, s.srcinfo)]

                # an alternate scheme is to split the loop in two
                # by cutting off the tail into a second loop
                elif self._tail_strategy == 'cut':
                    # if N == s.hi and Q == self.quot, then
                    #   we want Ncut == (N-Q+1)/Q
                    Q       = cnst(self.quot)
                    N       = s.hi
                    NQ1     = szop("+", szop("-", N, Q), cnst(1))
                    Ncut    = szop("/", NQ1, Q)

                    # and then for the tail loop, we want to
                    # iterate from 0 to Ntail
                    # where Ntail == (N - Ncut*Q)
                    Ntail   = szop("-", N, szop("*", Ncut, Q))
                    # in that loop we want the iteration variable to
                    # be mapped instead to (Ncut*Q + cut_i)
                    self._cut_tail_sub = szop("+", rd(self.cut_i),
                                                   szop("*", Ncut, Q))

                    main_body = self.map_stmts(s.body)
                    self._in_cut_tail = True
                    tail_body = Alpha_Rename(self.map_stmts(s.body)).result()
                    self._in_cut_tail = False

                    loops = [LoopIR.ForAll(self.hi_i, Ncut,
                                [LoopIR.ForAll(self.lo_i, Q,
                                    main_body,
                                    s.eff, s.srcinfo)],
                                s.eff, s.srcinfo),
                             LoopIR.ForAll(self.cut_i, Ntail,
                                tail_body,
                                s.eff, s.srcinfo)]

                    return loops

                else: assert False, f"bad tail strategy"

        # fall-through
        return super().map_s(s)

    def map_e(self, e):
        if type(e) is LoopIR.Read:
            if e.type is T.index:
                # This is a splitted variable, substitute it!
                if e.name is self.split_var:
                    if self._in_cut_tail:
                        return self.cut_tail_sub(e.srcinfo)
                    else:
                        return self.substitute(e.srcinfo)

        # fall-through
        return super().map_e(e)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unroll scheduling directive


class _Unroll(LoopIR_Rewrite):
    def __init__(self, proc, unroll_var):
        self.orig_proc  = proc
        self.unroll_var = unroll_var
        self.unroll_itr = 0
        self.env        = {}

        super().__init__(proc)

    def map_s(self, s):
        if type(s) is LoopIR.ForAll:
            # unroll this to loops!!
            if s.iter is self.unroll_var:
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
                # This is a unrolled variable, substitute it!
                if e.name is self.unroll_var:
                    return LoopIR.Const(self.unroll_itr, T.index,
                                        e.srcinfo)

        # fall-through
        return super().map_e(e)


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
            # first, set-up a binding from sub-proc arguments
            # to supplied expressions at the call-site
            call_bind   = { xd.name : a for xd,a in zip(s.f.args,s.args) }

            # whenever we copy code we need to alpha-rename for safety
            body        = Alpha_Rename(s.f.body).result()

            # then we will substitute the bindings for the call
            body        = SubstArgs(body, call_bind).result()

            # the code to splice in at this point
            return body

        # fall-through
        return super().map_s(s)

    def map_e(self, e):
        return e


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

    def map_e(self, e):
        return e


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

        self.do_stmts(stmts)

    def result(self):
        return self._depends[self._buf_sym]

    def do_s(self, s):
        if type(s) is LoopIR.Assign or type(s) is LoopIR.Reduce:
            self._lhs = s.name
            self._depends[s.name].add(s.name)
        elif type(s) is LoopIR.Call:
            # giant cross-bar of dependencies on the arguments
            for fa, a in zip(s.f.args, s.args):
                maybe_out = (not fa.effect or fa.effect != T.In)
                buf_arg   = (a.type.is_numeric() and type(a) is LoopIR.Read)
                if buf_arg and maybe_out:
                    self._lhs = a.name
                    for aa in s.args:
                        maybe_in = (not fa.effect or fa.effect != T.Out)
                        if maybe_in:
                            self.do_e(aa)

            # already handled all sub-terms above
            # don't do the usual statement processing
            return
        else:
            self._lhs = None

        super().do_s(s)

    def do_e(self, e):
        if type(e) is LoopIR.Read:
            if self._lhs:
                self._depends[self._lhs].add(e.name)

        super().do_e(e)


class _LiftAlloc(LoopIR_Rewrite):
    def __init__(self, proc, alloc_stmt, n_lifts):
        assert type(alloc_stmt) is LoopIR.Alloc
        assert is_pos_int(n_lifts)
        self.orig_proc      = proc
        self.alloc_stmt     = alloc_stmt
        self.alloc_sym      = alloc_stmt.name
        self.alloc_deps     = _Alloc_Dependencies(self.alloc_sym,
                                                  proc.body).result()
        self.n_lifts        = n_lifts

        self.ctrl_ctxt      = []
        self.lift_site      = None

        self.lifted_stmt    = None
        self.access_idxs    = None
        self._in_call_arg   = False

        super().__init__(proc)

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
            if len(rngs) > 0:
                #TODO: Fix stride!
                new_typ = T.Tensor(rngs, [], new_typ)

            # TODO: What is the effect here?
            self.lifted_stmt = LoopIR.Alloc( s.name, new_typ, s.mem,
                                             None, s.srcinfo )
            self.access_idxs = idxs

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
                idx = [ LoopIR.Read(i, [], T.index, s.srcinfo)
                        for i in self.access_idxs ] + s.idx
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
            if self._in_call_arg:
                raise SchedulingError("cannot lift allocation of "+
                                      f"'{self.alloc_sym}' because it "+
                                      "was passed into a sub-procedure "+
                                      "call; TODO: fix this!")
            idx = [ LoopIR.Read(i, [], T.index, e.srcinfo)
                    for i in self.access_idxs ] + e.idx
            return LoopIR.Read( e.name, idx, e.type, e.srcinfo )

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
                        assert s.hi.type == T.size
                        assert len(s.hi.idx) == 0
                        rngs.append(s.hi)
                    elif type(s.hi) == LoopIR.Const:
                        assert s.hi.type == T.int
                        rngs.append(s.hi)
                    else:
                        raise SchedulingError("Can only lift through loops "+
                                              "with simple range bounds, "+
                                              "i.e. a variable or constant")
            else: assert False, "bad case"

        return (idxs, rngs)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Fissioning at a Statement scheduling directive

class _Is_Alloc_Free(LoopIR_Do):
    def __init__(self, stmts):
        self._is_alloc_free = True

        self.do_stmts(stmts)

    def result(self):
        return self._is_alloc_free

    def do_s(self, s):
        if type(s) is LoopIR.Alloc:
            self._is_alloc_free = False

        super().do_s(s)

    def do_e(self, e):
        pass

def _is_alloc_free(stmts):
    return _Is_Alloc_Free(stmts).result()


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

    def alloc_check(self, stmts):
        if not _is_alloc_free(stmts):
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
                self.alloc_check(pre)
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
                self.alloc_check(pre)
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
                self.alloc_check(pre)

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
    DoCallSwap          = _CallSwap
    DoBindExpr          = _BindExpr
    DoLiftAlloc         = _LiftAlloc
    DoFissionLoops      = _FissionLoops
    DoFactorOut         = _DoFactorOut
