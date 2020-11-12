from .prelude import *
from .LoopIR import LoopIR
from . import shared_types as T
import re

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
    for sz in proc.sizes:
        if str(sz) == name:
            sym_list.append(sz)
    for a in proc.args:
        if str(a.name) == name:
            sym_list.append(a.name)

    def find_sym_stmt(node, nm):
        if type(node) is LoopIR.Seq:
            find_sym_stmt(node.s0, nm)
            find_sym_stmt(node.s1, nm)
        elif type(node) is LoopIR.If:
            find_sym_stmt(node.body, nm)
        elif type(node) is LoopIR.Alloc:
            if str(node.name) == nm:
                sym_list.append(node.name)
        elif type(node) is LoopIR.ForAll:
            if str(node.iter) == nm:
                sym_list.append(node.iter)
            find_sym_stmt(node.body, nm)

    # search proc body
    find_sym_stmt(proc.body, name)

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
        if type(node) is LoopIR.Seq:
            find_sym_stmt(node.s0, out_sym)
            find_sym_stmt(node.s1, out_sym)
        elif type(node) is LoopIR.If:
            find_sym_stmt(node.body, out_sym)
        elif type(node) is LoopIR.ForAll:
            # first, search for the outer name
            if out_sym is None and str(node.iter) == out_name:
                nonlocal out_cnt
                out_cnt += 1
                if out_idx is None or out_idx is out_cnt:
                    out_sym = node.iter
                    find_sym_stmt(node.body, out_sym)
                    out_sym = None
            # if we are inside of an outer name match...
            elif out_sym is not None and str(node.iter) == in_name:
                nonlocal in_cnt
                in_cnt += 1
                if in_idx is None or in_idx is in_cnt:
                    pair_list.append( (out_sym, node.iter) )
            find_sym_stmt(node.body, out_sym)

    # search proc body
    find_sym_stmt(proc.body, None)

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
# Generic Tree Transformation Pass

class _LoopIR_Rewrite:
    def __init__(self, proc, *args, **kwargs):
        self.orig_proc  = proc

        body = self.map_s(self.orig_proc.body)

        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                sizes   = self.orig_proc.sizes,
                                args    = self.orig_proc.args,
                                body    = body,
                                srcinfo = self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return styp( s.name, [ self.map_a(a) for a in s.idx ],
                         self.map_e(s.rhs), s.srcinfo )
        elif styp is LoopIR.Seq:
            return LoopIR.Seq( self.map_s(s.s0), self.map_s(s.s1),
                               s.srcinfo )
        elif styp is LoopIR.If:
            orelse = self.map_s(s.orelse) if s.orelse else None
            return LoopIR.If( self.map_p(s.cond), self.map_s(s.body),
                              orelse, s.srcinfo )
        elif styp is LoopIR.ForAll:
            return LoopIR.ForAll( s.iter, self.map_a(s.hi), self.map_s(s.body),
                              s.srcinfo )
        elif styp is LoopIR.Instr:
            return LoopIR.Instr( s.op, self.map_s(s.body), s.srcinfo )
        else:
            return s

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            return LoopIR.Read( e.name, [ self.map_a(a) for a in e.idx ],
                                e.srcinfo )
        elif etyp is LoopIR.BinOp:
            return LoopIR.BinOp( e.op, self.map_e(e.lhs), self.map_e(e.rhs),
                                 e.srcinfo )
        elif etyp is LoopIR.Select:
            return LoopIR.Select( self.map_p(e.cond), self.map_e(e.body),
                                  e.srcinfo )
        else:
            return e

    def map_p(self, p):
        ptyp = type(p)
        if ptyp is LoopIR.Cmp:
            return LoopIR.Cmp( p.op, self.map_a(p.lhs), self.map_a(p.rhs),
                               p.srcinfo )
        elif ptyp is LoopIR.And or ptyp is LoopIR.Or:
            return ptyp( self.map_p(p.lhs), self.map_p(p.rhs),
                         p.srcinfo )
        else:
            return p

    def map_a(self, a):
        atyp = type(a)
        if atyp is LoopIR.AScale:
            return LoopIR.AScale( a.coeff, self.map_a(a.rhs), a.srcinfo )
        elif atyp is LoopIR.AScaleDiv:
            return LoopIR.AScaleDiv( self.map_a(a.lhs), a.quotient, a.srcinfo )
        elif atyp is LoopIR.AAdd or atyp is LoopIR.ASub:
            return atyp( self.map_a(a.lhs), self.map_a(a.rhs), a.srcinfo )
        else:
            return a


class _Alpha_Rename(_LoopIR_Rewrite):
    def __init__(self, node):
        self.env    = {}
        if isinstance(node, LoopIR.stmt):
            self.node = self.map_s(node)
        elif isinstance(node, LoopIR.expr):
            self.node = self.map_e(node)
        elif isinstance(node, LoopIR.pred):
            self.node = self.map_p(node)
        elif isinstance(node, LoopIR.aexpr):
            self.node = self.map_a(node)

    def result(self):
        return self.node

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            nm = self.env[s.name] if s.name in self.env else s.name
            return styp( nm, [ self.map_a(a) for a in s.idx ],
                         self.map_e(s.rhs), s.srcinfo )
        elif styp is LoopIR.ForAll:
            itr = s.iter.copy()
            self.env[s.iter] = itr
            return LoopIR.ForAll( itr, self.map_a(s.hi), self.map_s(s.body),
                                  s.srcinfo )
        elif styp is LoopIR.Alloc:
            nm = s.name.copy()
            self.env[s.name] = nm
            return LoopIR.Alloc( nm, s.type, s.mem, s.srcinfo )

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.Read( nm, [ self.map_a(a) for a in e.idx ],
                                e.srcinfo )

        return super().map_e(e)

    def map_a(self, a):
        atyp = type(a)
        if atyp is LoopIR.AVar:
            nm = self.env[a.name] if a.name in self.env else a.name
            return LoopIR.AVar( nm, a.srcinfo )

        return super().map_a(a)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Reorder scheduling directive


class _Reorder(_LoopIR_Rewrite):
    def __init__(self, proc, out_var, in_var):
        self.out_var = out_var
        self.in_var  = in_var
        super().__init__(proc)

    def map_s(self, s):
        if type(s) is LoopIR.ForAll:
            if s.iter == self.out_var:
                if type(s.body) is not LoopIR.ForAll:
                    raise SchedulingError(f"expected loop directly inside of "+
                                          f"{self.out_var} loop")
                elif s.body.iter != self.in_var:
                    raise SchedulingError(f"expected loop directly inside of "+
                                          f"{self.out_var} loop to have "+
                                          f"iteration variable {self.in_var}")
                else:
                    body = s.body.body
                    # wrap outer loop; now inner loop
                    body = LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
                    # wrap inner loop; now outer loop
                    body = LoopIR.ForAll(s.body.iter, s.body.hi,
                                         body, s.body.srcinfo)
                    return body

        # fall-through
        return super().map_s(s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Split scheduling directive


class _Split(_LoopIR_Rewrite):
    def __init__(self, proc, split_var, quot, hi, lo):
        self.split_var = split_var
        self.quot = quot
        self.hi_i = Sym(hi)
        self.lo_i = Sym(lo)

        super().__init__(proc)

    def substitute(self, srcinfo):
        return LoopIR.AAdd(
                LoopIR.AScale(self.quot, LoopIR.AVar(self.hi_i, srcinfo),
                    srcinfo), LoopIR.AVar(self.lo_i, srcinfo), srcinfo)

    def map_s(self, s):
        if type(s) is LoopIR.ForAll:
            # Split this to two loops!!
            if s.iter is self.split_var:
                body = self.map_s(s.body)
                cond    = LoopIR.Cmp("<", self.substitute(s.srcinfo), s.hi,
                                     s.srcinfo)
                body    = LoopIR.If(cond, body, None, s.srcinfo)
                hi_hi   = LoopIR.AScaleDiv(s.hi, self.quot, s.srcinfo)
                lo_hi   = LoopIR.AConst(self.quot, s.srcinfo)

                return LoopIR.ForAll(self.hi_i, hi_hi,
                            LoopIR.ForAll(self.lo_i, lo_hi,
                                body,
                                s.srcinfo),
                            s.srcinfo)

        # fall-through
        return super().map_s(s)

    def map_a(self, a):
        if type(a) is LoopIR.AVar:
            # This is a splitted variable, substitute it!
            if a.name is self.split_var:
                return self.substitute(a.srcinfo)

        # fall-through
        return super().map_a(a)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unroll scheduling directive


class _Unroll(_LoopIR_Rewrite):
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
                if type(s.hi) is not LoopIR.AConst:
                    raise SchedulingError(f"expected loop '{s.iter}' "+
                                          f"to have constant bounds")
                hi      = s.hi.val
                assert hi > 0
                orig_body = s.body

                self.unroll_itr = 0

                body    = _Alpha_Rename(self.map_s(orig_body)).result()
                for i in range(1,hi):
                    self.unroll_itr = i
                    nxtbody = _Alpha_Rename(self.map_s(orig_body)).result()
                    body = LoopIR.Seq(body, nxtbody, s.srcinfo)

                return body
            else:
                body = self.map_s(s.body)
                return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)

        # fall-through
        return super().map_s(s)

    def map_a(self, a):
        if type(a) is LoopIR.AVar:
            # This is a unrolled variable, substitute it!
            if a.name is self.unroll_var:
                return LoopIR.AConst(self.unroll_itr, a.srcinfo)
            else:
                return a

        # fall-through
        return super().map_a(a)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Passes to export

class Schedules:
    DoReorder   = _Reorder
    DoSplit     = _Split
    DoUnroll    = _Unroll
