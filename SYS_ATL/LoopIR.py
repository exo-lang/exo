from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR_effects import Effects as E

from .memory import Memory

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Untyped AST

front_ops = {
    "+":    True,
    "-":    True,
    "*":    True,
    "/":    True,
    "%":    True,
    #
    "<":    True,
    ">":    True,
    "<=":   True,
    ">=":   True,
    "==":   True,
    #
    "and":  True,
    "or":   True,
}

UAST = ADT("""
module UAST {
    proc    = ( name?           name,
                fnarg*          args,
                expr*           preds,
                stmt*           body,
                string?         instr,
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                mem?            mem,
                srcinfo         srcinfo )

    stmt    = Assign  ( sym name, expr* idx, expr rhs )
            | Reduce  ( sym name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | ForAll  ( sym iter,  expr cond,   stmt* body )
            | Alloc   ( sym name, type type, mem? mem )
            | Call    ( loopir_proc f, expr* args )
            attributes( srcinfo srcinfo )

    expr    = Read    ( sym name, expr* idx )
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            | ParRange( expr lo, expr hi ) -- only use for loop cond
            attributes( srcinfo srcinfo )

} """, {
    'name':         is_valid_name,
    'sym':          lambda x: type(x) is Sym,
    'type':         T.is_type,
    'mem':          lambda x: isinstance(x, Memory),
    'loopir_proc':  lambda x: type(x) is LoopIR.proc,
    'op':           lambda x: x in front_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo,
})


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern AST
#   - used to specify pattern-matches

PAST = ADT("""
module PAST {

    stmt    = Assign  ( name name, expr* idx, expr rhs )
            | Reduce  ( name name, expr* idx, expr rhs )
            | Pass    ()
            | If      ( expr cond, stmt* body,  stmt* orelse )
            | ForAll  ( name iter, expr hi,     stmt* body )
            | Alloc   ( name name ) -- may want to add type & mem back in?
            | Call    ( name f, expr* args )
            | S_Hole  ()
            attributes( srcinfo srcinfo )

    expr    = Read    ( name name, expr* idx )
            | E_Hole  ()
            | Const   ( object val )
            | USub    ( expr arg ) -- i.e.  -(...)
            | BinOp   ( op op, expr lhs, expr rhs )
            attributes( srcinfo srcinfo )

} """, {
    'name':         lambda x: x == '_' or is_valid_name(x),
    'op':           lambda x: x in front_ops,
    'srcinfo':      lambda x: type(x) is SrcInfo,
})

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR

bin_ops = {
    "+":    True,
    "-":    True,
    "*":    True,
    "/":    True,
    "%":    True,

    "and":  True,
    "or":   True,

    "<":    True,
    ">":    True,
    "<=":   True,
    ">=":   True,
    "==":   True,
}

LoopIR = ADT("""
module LoopIR {
    proc    = ( name            name,
                fnarg*          args,
                expr*           preds,
                stmt*           body,
                string?         instr,
                effect?         eff,
                srcinfo         srcinfo )

    fnarg   = ( sym             name,
                type            type,
                mem?            mem,
                srcinfo         srcinfo )

    stmt    = Assign ( sym name, expr* idx, expr rhs )
            | Reduce ( sym name, expr* idx, expr rhs )
        --  | Alias  ( sym name, expr rhs ) -- rhs has to be a slice expr?
            | Pass   ()
            | If     ( expr cond, stmt* body, stmt* orelse )
            | ForAll ( sym iter, expr hi, stmt* body )
            | Alloc  ( sym name, type type, mem? mem )
            | Free   ( sym name, type type, mem? mem )
            | Call   ( proc f, expr* args )
            attributes( effect? eff, srcinfo srcinfo )

    expr    = Read( sym name, expr* idx )
        --  | Slice( sym name, slice_idx* idx )
            | Const( object val )
            | BinOp( binop op, expr lhs, expr rhs )
            attributes( type type, srcinfo srcinfo )

    -- slice_idx = SlicePoint( expr val )
    --           | SliceRange( expr lo, expr hi )
    --           attributes(srcinfo srcinfo)

} """, {
    'name':     is_valid_name,
    'sym':      lambda x: type(x) is Sym,
    'type':     T.is_type,
    'effect':   lambda x: type(x) is E.effect,
    'mem':      lambda x: isinstance(x, Memory),
    'binop':    lambda x: x in bin_ops,
    'srcinfo':  lambda x: type(x) is SrcInfo,
})

# make proc be a hashable object
@extclass(LoopIR.proc)
def __hash__(self):
    return id(self)
del __hash__

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def __init__(self, proc, instr=None, *args, **kwargs):
        self.orig_proc  = proc

        body = self.map_stmts(self.orig_proc.body)

        self.proc = LoopIR.proc(name    = self.orig_proc.name,
                                args    = self.orig_proc.args,
                                preds   = self.orig_proc.preds,
                                body    = body,
                                instr   = instr,
                                eff     = self.orig_proc.eff,
                                srcinfo = self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def map_stmts(self, stmts):
        return [ s for b in stmts
                   for s in self.map_s(b) ]

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return [styp( s.name, [ self.map_e(a) for a in s.idx ],
                          self.map_e(s.rhs), s.eff, s.srcinfo )]
        elif styp is LoopIR.If:
            return [LoopIR.If( self.map_e(s.cond), self.map_stmts(s.body),
                               self.map_stmts(s.orelse), s.eff, s.srcinfo )]
        elif styp is LoopIR.ForAll:
            return [LoopIR.ForAll( s.iter, self.map_e(s.hi),
                                   self.map_stmts(s.body), s.eff, s.srcinfo )]
        else:
            return [s]

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            return LoopIR.Read( e.name, [ self.map_e(a) for a in e.idx ],
                                e.type, e.srcinfo )
        elif etyp is LoopIR.BinOp:
            return LoopIR.BinOp( e.op, self.map_e(e.lhs), self.map_e(e.rhs),
                                 e.type, e.srcinfo )
        else:
            return e

class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc  = proc

        self.do_stmts(self.proc.body)

    def do_stmts(self, stmts):
        for s in stmts:
            self.do_s(s)

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
        elif styp is LoopIR.ForAll:
            self.do_e(s.hi)
            self.do_stmts(s.body)
        elif styp is LoopIR.Call:
            for e in s.args:
                self.do_e(e)
        else:
            pass

    def do_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            for e in e.idx:
                self.do_e(e)
        elif etyp is LoopIR.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        else:
            return e


class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env    = {}
        self.node   = []
        for n in node:
            assert isinstance(n, LoopIR.stmt)  # only case handled for now
            self.node += self.map_s(n)

    def result(self):
        return self.node

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            nm = self.env[s.name] if s.name in self.env else s.name
            return [styp( nm, [ self.map_e(a) for a in s.idx ],
                         self.map_e(s.rhs), s.eff, s.srcinfo )]
        elif styp is LoopIR.ForAll:
            itr = s.iter.copy()
            self.env[s.iter] = itr
            return [LoopIR.ForAll( itr, self.map_e(s.hi),
                                   self.map_stmts(s.body),
                                   s.eff, s.srcinfo )]
        elif styp is LoopIR.Alloc:
            nm = s.name.copy()
            self.env[s.name] = nm
            return [LoopIR.Alloc( nm, s.type, s.mem, s.eff, s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.Read( nm, [ self.map_e(a) for a in e.idx ],
                                e.type, e.srcinfo )

        return super().map_e(e)


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, stmts, binding):
        assert isinstance(stmts, list)
        assert isinstance(binding, dict)
        assert all( isinstance(v, LoopIR.expr) for v in binding.values() )
        self.env    = binding
        self.stmts  = []
        for s in stmts:
            assert isinstance(s, LoopIR.stmt)  # only case handled for now
            self.stmts += self.map_s(s)

    def result(self):
        return self.stmts

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name in self.env:
                e = self.env[s.name]
                assert type(e) is LoopIR.Read and len(e.idx) == 0
                return [styp( e.name, [ self.map_e(a) for a in s.idx ],
                             self.map_e(s.rhs), s.eff, s.srcinfo )]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            if e.name in self.env:
                if len(e.idx) == 0:
                    return self.env[e.name]
                else:
                    sub_e = self.env[e.name]
                    assert type(sub_e) is LoopIR.Read and len(sub_e.idx) == 0
                    return LoopIR.Read( sub_e.name,
                                        [ self.map_e(a) for a in e.idx ],
                                        e.type, e.srcinfo )

        return super().map_e(e)
