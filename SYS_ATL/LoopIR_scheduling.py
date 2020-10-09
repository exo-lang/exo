
from .prelude import *
from .LoopIR import LoopIR
from . import shared_types as T

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
    name = desc # extract name
    idx  = None
    # idx is a non-negative integer if present

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
            find_sym_stmt(node.s0)
            find_sym_stmt(node.s1)
        elif type(node) is LoopIR.If:
            find_sym_stmt(node.body)
        elif type(node) is LoopIR.Alloc:
            if str(node.name) == nm:
                sym_list.append(a.name)
        elif type(node) is LoopIR.ForAll:
            if str(node.iter) == nm:
                sym_list.append(a.name)
            find_sym_stmt(node.body)
    # search proc body
    find_sym_stmt(body, name))

    return sym_list

def name_str_2_pairs(proc, out_desc, in_desc):
    assert type(proc) is LoopIR.proc
    # parse regular expression
    #   either name[int]
    #       or name
    out_name = out_desc # extract name
    idx  = None
    in_name = in_desc
    # idx is a non-negative integer if present

    # find all occurrences of name
    pair_list = []
    out_sym   = None
    def find_sym_stmt(node):
        if type(node) is LoopIR.Seq:
            find_sym_stmt(node.s0)
            find_sym_stmt(node.s1)
        elif type(node) is LoopIR.If:
            find_sym_stmt(node.body)
        elif type(node) is LoopIR.ForAll:
            # first, search for the outer name
            if out_sym is None and str(node.iter) == out_name:
                out_sym = node.iter
                find_sym_stmt(node.body)
                out_sym = None
            # if we are inside of an outer name match...
            elif out_sym is not None and str(node.iter) == in_name:
                pair_list.append( (out_sym,node.iter) )
            find_sym_stmt(node.body)
    # search proc body
    find_sym_stmt(body)

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
# Scheduling Passes

class _Reorder:
    pass

class _Split:
    pass

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The Passes to export

class Schedules:
    DoReorder   = _Reorder
    DoSplit     = _Split
