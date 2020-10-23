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
    # TODO! Handle idx
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

    print(pair_list)

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

class _Reorder:
    def __init__(self, proc, out_var, in_var):
        self.orig_proc = proc
        self.out_var = out_var
        self.in_var  = in_var

        body = self.reorder_s(self.orig_proc.body)

        self.proc = LoopIR.proc(name  = self.orig_proc.name,
                               sizes  = self.orig_proc.sizes,
                               args   = self.orig_proc.args,
                               body   = body,
                               srcinfo= self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def reorder_s(self, s):
        styp = type(s)

        if styp is LoopIR.Seq:
            s0 = self.reorder_s(s.s0)
            s1 = self.reorder_s(s.s1)
            return LoopIR.Seq(s0, s1, s.srcinfo)
        elif styp is LoopIR.If:
            body = self.reorder_s(s.body)
            return LoopIR.If(s.cond, body, s.srcinfo)
        elif styp is LoopIR.ForAll:
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
            else:
                body = self.reorder_s(s.body)
                return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
        else:
            return s



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Split scheduling directive


class _Split:
    def __init__(self, proc, split_var, quot, hi, lo):
        self.orig_proc = proc
        self.split_var = split_var
        self.quot = quot
        self.hi_i = Sym(hi)
        self.lo_i = Sym(lo)

        body = self.split_s(self.orig_proc.body)

        self.proc = LoopIR.proc(name  = self.orig_proc.name,
                               sizes  = self.orig_proc.sizes,
                               args   = self.orig_proc.args,
                               body   = body,
                               srcinfo= self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def substitute(self, srcinfo):
        return LoopIR.AAdd(
                LoopIR.AScale(self.quot, LoopIR.AVar(self.hi_i, srcinfo),
                    srcinfo), LoopIR.AVar(self.lo_i, srcinfo), srcinfo)

    def split_s(self, s):
        styp = type(s)

        if styp is LoopIR.Seq:
            s0 = self.split_s(s.s0)
            s1 = self.split_s(s.s1)
            return LoopIR.Seq(s0, s1, s.srcinfo)
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            idx = [self.split_a(i) for i in s.idx]
            rhs = self.split_e(s.rhs)
            IRnode = (LoopIR.Assign if styp is LoopIR.Assign else
                      LoopIR.Reduce)
            return IRnode(s.name, idx, rhs, s.srcinfo)
        elif styp is LoopIR.If:
            cond = self.split_p(s.cond)
            body = self.split_s(s.body)
            return LoopIR.If(cond, body, s.srcinfo)
        elif styp is LoopIR.ForAll:
            body = self.split_s(s.body)
            # Split this to two loops!!
            if s.iter is self.split_var:
                cond    = LoopIR.Cmp("<", self.substitute(s.srcinfo), s.hi,
                                     s.srcinfo)
                body    = LoopIR.If(cond, body, s.srcinfo)
                hi_hi   = LoopIR.AScaleDiv(s.hi, self.quot, s.srcinfo)
                lo_hi   = LoopIR.AConst(self.quot, s.srcinfo)

                return LoopIR.ForAll(self.hi_i, hi_hi,
                            LoopIR.ForAll(self.lo_i, lo_hi,
                                body,
                                s.srcinfo),
                            s.srcinfo)
            else:
                return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
        else:
            return s


    def split_e(self, e):
        if type(e) is LoopIR.Read:
            idx = [self.split_a(i) for i in e.idx]
            return LoopIR.Read(e.name, idx, e.srcinfo)
        elif type(e) is LoopIR.BinOp:
            lhs = self.split_e(e.lhs)
            rhs = self.split_e(e.rhs)
            return LoopIR.BinOp(e.op, lhs, rhs, e.srcinfo)
        elif type(e) is LoopIR.Select:
            pred = self.split_p(e.cond)
            body = self.split_e(e.body)
            return LoopIR.Select(pred, body, e.srcinfo)
        else:
            return e

    def split_a(self, a):
        atyp = type(a)

        if atyp is LoopIR.AVar:
            # This is a splitted variable, substitute it!
            if a.name is self.split_var:
                return self.substitute(a.srcinfo)
            else:
                return a
        elif atyp is LoopIR.AScale:
            rhs = self.split_a(a.rhs)
            return LoopIR.AScale(a.coeff, rhs, a.srcinfo)
        elif atyp is LoopIR.AScaleDiv:
            lhs = self.split_a(a.lhs)
            return LoopIR.AScaleDiv(lhs, a.quotient, a.srcinfo)
        elif atyp is LoopIR.AAdd:
            lhs = self.split_a(a.lhs)
            rhs = self.split_a(a.rhs)
            return LoopIR.AAdd(lhs, rhs, a.srcinfo)
        elif atyp is LoopIR.ASub:
            lhs = self.split_a(a.lhs)
            rhs = self.split_a(a.rhs)
            return LoopIR.ASub(lhs, rhs, a.srcinfo)
        else:
            return a

    def split_p(self, p):
        if type(p) is LoopIR.Cmp:
            lhs = self.split_a(p.lhs)
            rhs = self.split_a(p.rhs)
            return LoopIR.Cmp(p.op, lhs, rhs, p.srcinfo)
        elif type(p) is LoopIR.And:
            lhs = self.split_p(p.lhs)
            rhs = self.split_p(p.rhs)
            return LoopIR.And(lhs, rhs, p.srcinfo)
        elif type(p) is LoopIR.Or:
            lhs = self.split_p(p.lhs)
            rhs = self.split_p(p.rhs)
            return LoopIR.Or(lhs, rhs, p.srcinfo)
        else:
            return p



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unroll scheduling directive


class _Unroll:
    def __init__(self, proc, unroll_var):
        self.orig_proc  = proc
        self.unroll_var = unroll_var

        self.env        = {}

        body = self.unroll_s(self.orig_proc.body)

        self.proc = LoopIR.proc(name  = self.orig_proc.name,
                               sizes  = self.orig_proc.sizes,
                               args   = self.orig_proc.args,
                               body   = body,
                               srcinfo= self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def alpha_buf_rename(self):
        styp = type(s)

        if styp is LoopIR.Seq:
            s0 = self.alpha_buf_rename(s.s0)
            s1 = self.alpha_buf_rename(s.s1)
            return LoopIR.Seq(s0, s1, s.srcinfo)
        elif styp is LoopIR.Alloc:
            nm      = s.name.copy()
            self.env[s.name] = nm
            return LoopIR.Alloc(nm, s.type, s.srcinfo)
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            rhs     = self.alpha_sub(s.rhs)
            IRnode  = (LoopIR.Assign if styp is LoopIR.Assign else
                       LoopIR.Reduce)
            # replace the name?
            nm      = self.env[s.name] if s.name in self.env else s.name
            return IRnode(nm, s.idx, rhs, s.srcinfo)
        elif styp is LoopIR.If:
            body    = self.alpha_sub(s.body)
            return LoopIR.If(s.cond, body, s.srcinfo)
        elif styp is LoopIR.ForAll:
            body    = self.alpha_sub(s.body)
            return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
        else:
            return s

    def alpha_sub(self, e):
        etyp = type(e)

        if etyp is LoopIR.Seq:
            s0 = self.alpha_sub(e.s0)
            s1 = self.alpha_sub(e.s1)
            return LoopIR.Seq(s0, s1, e.srcinfo)
        elif etyp is LoopIR.Assign:
            rhs     = self.alpha_sub(e.rhs)
            return LoopIR.Assign(e.name, e.idx, rhs, e.srcinfo)
        elif etyp is LoopIR.Reduce:
            rhs     = self.alpha_sub(e.rhs)
            return LoopIR.Reduce(e.name, e.idx, rhs, e.srcinfo)
        elif etyp is LoopIR.If:
            body    = self.alpha_sub(e.body)
            return LoopIR.If(e.cond, body, e.srcinfo)
        elif etyp is LoopIR.ForAll:
            body    = self.alpha_sub(e.body)
            return LoopIR.ForAll(e.iter, e.hi, body, e.srcinfo)

        elif etyp is LoopIR.Read:
            nm  = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.Read(nm, e.idx, e.srcinfo)
        elif etyp is LoopIR.BinOp:
            lhs = self.alpha_sub(e.lhs)
            rhs = self.alpha_sub(e.rhs)
            return LoopIR.BinOp(e.op, lhs, rhs, e.srcinfo)
        elif etyp is LoopIR.Select:
            body = self.alpha_sub(e.body)
            return LoopIR.Select(e.pred, body, e.srcinfo)

        else:
            return e

    def unroll_s(self, s, unroll_itr=0):
        styp = type(s)

        if styp is LoopIR.Seq:
            s0 = self.unroll_s(s.s0, unroll_itr)
            s1 = self.unroll_s(s.s1, unroll_itr)
            return LoopIR.Seq(s0, s1, s.srcinfo)
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            idx = [self.unroll_a(i, unroll_itr) for i in s.idx]
            rhs = self.unroll_e(s.rhs, unroll_itr)
            IRnode = (LoopIR.Assign if styp is LoopIR.Assign else
                      LoopIR.Reduce)
            return IRnode(s.name, idx, rhs, s.srcinfo)
        elif styp is LoopIR.If:
            cond = self.unroll_p(s.cond, unroll_itr)
            body = self.unroll_s(s.body, unroll_itr)
            return LoopIR.If(cond, body, s.srcinfo)
        elif styp is LoopIR.ForAll:
            # unroll this to loops!!
            if s.iter is self.unroll_var:
                if type(s.hi) is not LoopIR.AConst:
                    raise SchedulingError(f"expected loop '{s.iter}' "+
                                          f"to have constant bounds")
                hi      = s.hi.val
                assert hi > 0
                orig_body = s.body

                body    = self.alpha_buf_rename(self.unroll_s(orig_body, 0))
                for i in range(1,hi):
                    nxtbody = self.alpha_buf_rename(self.unroll_s(orig_body, i))
                    body = LoopIR.Seq(body, nxtbody, s.srcinfo)

                return body
            else:
                body = self.unroll_s(s.body, unroll_itr)
                return LoopIR.ForAll(s.iter, s.hi, body, s.srcinfo)
        else:
            return s


    def unroll_e(self, e, unroll_itr=0):
        if type(e) is LoopIR.Read:
            idx = [self.unroll_a(i, unroll_itr) for i in e.idx]
            return LoopIR.Read(e.name, idx, e.srcinfo)
        elif type(e) is LoopIR.BinOp:
            lhs = self.unroll_e(e.lhs, unroll_itr)
            rhs = self.unroll_e(e.rhs, unroll_itr)
            return LoopIR.BinOp(e.op, lhs, rhs, e.srcinfo)
        elif type(e) is LoopIR.Select:
            pred = self.unroll_p(e.cond, unroll_itr)
            body = self.unroll_e(e.body, unroll_itr)
            return LoopIR.Select(pred, body, e.srcinfo)
        else:
            return e

    def unroll_a(self, a, unroll_itr=0):
        atyp = type(a)

        if atyp is LoopIR.AVar:
            # This is a unrolled variable, substitute it!
            if a.name is self.unroll_var:
                return LoopIR.AConst(unroll_itr, a.srcinfo)
            else:
                return a
        elif atyp is LoopIR.AScale:
            rhs = self.unroll_a(a.rhs, unroll_itr)
            return LoopIR.AScale(a.coeff, rhs, a.srcinfo)
        elif atyp is LoopIR.AScaleDiv:
            lhs = self.unroll_a(a.lhs, unroll_itr)
            return LoopIR.AScaleDiv(lhs, a.quotient, a.srcinfo)
        elif atyp is LoopIR.AAdd:
            lhs = self.unroll_a(a.lhs, unroll_itr)
            rhs = self.unroll_a(a.rhs, unroll_itr)
            return LoopIR.AAdd(lhs, rhs, a.srcinfo)
        elif atyp is LoopIR.ASub:
            lhs = self.unroll_a(a.lhs, unroll_itr)
            rhs = self.unroll_a(a.rhs, unroll_itr)
            return LoopIR.ASub(lhs, rhs, a.srcinfo)
        else:
            return a

    def unroll_p(self, p, unroll_itr=0):
        if type(p) is LoopIR.Cmp:
            lhs = self.unroll_a(p.lhs, unroll_itr)
            rhs = self.unroll_a(p.rhs, unroll_itr)
            return LoopIR.Cmp(p.op, lhs, rhs, p.srcinfo)
        elif type(p) is LoopIR.And:
            lhs = self.unroll_p(p.lhs, unroll_itr)
            rhs = self.unroll_p(p.rhs, unroll_itr)
            return LoopIR.And(lhs, rhs, p.srcinfo)
        elif type(p) is LoopIR.Or:
            lhs = self.unroll_p(p.lhs, unroll_itr)
            rhs = self.unroll_p(p.rhs, unroll_itr)
            return LoopIR.Or(lhs, rhs, p.srcinfo)
        else:
            return p


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Passes to export

class Schedules:
    DoReorder   = _Reorder
    DoSplit     = _Split
    DoUnroll    = _Unroll
