from .prelude import *
from .LoopIR import LoopIR, LoopIR_Rewrite, Alpha_Rename, LoopIR_Do
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
                                    s.srcinfo)],
                                s.body[0].srcinfo)]

        # fall-through
        return super().map_s(s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Split scheduling directive


class _Split(LoopIR_Rewrite):
    def __init__(self, proc, split_var, quot, hi, lo):
        self.split_var = split_var
        self.quot = quot
        self.hi_i = Sym(hi)
        self.lo_i = Sym(lo)

        super().__init__(proc)

    def substitute(self, srcinfo):
        return LoopIR.BinOp("+",
                    LoopIR.BinOp("*",
                                 LoopIR.Const(self.quot, T.int, srcinfo),
                                 LoopIR.Read(self.hi_i, [], T.index, srcinfo),
                                 T.index, srcinfo),
                    LoopIR.Read(self.lo_i, [], T.index, srcinfo),
                    T.index, srcinfo)

    def map_s(self, s):
        if type(s) is LoopIR.ForAll:
            # Split this to two loops!!
            if s.iter is self.split_var:
                body = self.map_stmts(s.body)
                cond    = LoopIR.BinOp("<", self.substitute(s.srcinfo), s.hi,
                                        T.bool, s.srcinfo)
                body    = LoopIR.If(cond, body, [], s.srcinfo)
                lo_hi   = LoopIR.Const(self.quot, T.int, s.srcinfo)
                hi_hi   = LoopIR.BinOp("/", s.hi, lo_hi, T.index, s.srcinfo)

                return [LoopIR.ForAll(self.hi_i, hi_hi,
                            [LoopIR.ForAll(self.lo_i, lo_hi,
                                [body],
                                s.srcinfo)],
                            s.srcinfo)]

        # fall-through
        return super().map_s(s)

    def map_e(self, e):
        if type(e) is LoopIR.Read:
            if e.type is T.index:
                # This is a splitted variable, substitute it!
                if e.name is self.split_var:
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
# The Passes to export

class Schedules:
    DoReorder   = _Reorder
    DoSplit     = _Split
    DoUnroll    = _Unroll
