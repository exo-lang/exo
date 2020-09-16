from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

from .mem_analysis import MemoryAnalysis

import numpy as np

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler

# top level compiler function called by tests!
def run_compile(proc_list,c_file,h_file):
    # take proc_list
    # for each p in proc_list:
    #   run Compiler() pass to get (decl, def)
    #
    # check for name conflicts between procs
    #
    # write out c_file and h_file

    #fwd_decls = "#include <cstdio>\n" + "#include <cstring>\n\n"
    fwd_decls = ""

    body = f"#include \"{h_file}\"\n\n"
    for p in proc_list:
        p       = MemoryAnalysis(p).result()
        d, b    = Compiler(p).comp_top()
        fwd_decls += d
        body += b
        body += '\n'

    f_header = open(h_file, "w")
    f_header.write(fwd_decls)
    f_header.close()

    f_cpp = open(c_file, "w")
    f_cpp.write(body)
    f_cpp.close()

def _eshape(typ,env):
    return tuple( r if is_pos_int(r) else env[r]
                  for r in typ.shape() )

def _simple_typecheck_buffer(typ, buf, env):
    if type(buf) is not np.ndarray:
        return False
    elif buf.dtype != float and buf.dtype != np.float64:
        return False

    if typ is T.R:
        if tuple(buf.shape) != (1,):
            return False
    else:
        shape = _eshape(typ,env)
        if shape != tuple(buf.shape):
            return False

    return True

class Compiler:
    def __init__(self, proc, **kwargs):
        assert type(proc) is LoopIR.proc

        self.proc   = proc
        self.env    = Environment()
        self.names  = set()

        # setup, size argument binding
        for sz in proc.sizes:
            self.env[sz] = self.new_varname(sz, True)

        # setup, buffer argument binding
        for a in proc.args:
            self.env[a.name] = self.new_varname(a.name, True)

    def comp_top(self):
        stmt_str = self.comp_s(self.proc.body)

        assert self.proc.name != None, "expected names for compilation"
        name = self.proc.name
        sizes = self.proc.sizes
        args = self.proc.args
        size_str = ""
        arg_str = ""
        typ_comment_str = ""
        for size in sizes:
            size_str    += f"int {size},"
        for arg in args:
            arg_str     += f" float* {arg.name},"
            typ_comment_str += f" {arg.name} : {arg.type} {arg.effect},"

        # Generate headers here?
        proc_decl = ( f"// {name}({typ_comment_str[:-1]} )\n"
                    + f"void {name}({size_str}{arg_str[:-1]});\n"
                    )
        proc_def =  ( f"// {name}({typ_comment_str[:-1]} )\n"
                    + f"void {name}({size_str}{arg_str[:-1]}) {{\n"
                    + stmt_str + "\n"
                    + "}\n"
                    )

        #return proc_decl, proc_def
        return proc_decl, proc_def

    def new_varname(self, symbol, force_literal=False):
        s = str(symbol) if force_literal else repr(symbol)
        assert s not in self.names, "name conflict!"
        self.env[symbol] = s
        self.names.add(s)
        return self.env[symbol]

    def idx_str(self, idx_list):
        idx = ""
        for a in idx_list:
            idx += (f"[{self.comp_a(a)}]")
        return idx

    def comp_s(self, s):
        styp    = type(s)

        if styp is LoopIR.Seq:
            first = self.comp_s(s.s0)
            second = self.comp_s(s.s1)

            return (f"{first}\n\n{second}")
        elif styp is LoopIR.Pass:
            return (f"; // # NO-OzP :")
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            lbuf = s.name
            idx = self.idx_str(s.idx)
            rhs  = self.comp_e(s.rhs)
            if styp is LoopIR.Assign:
                return (f"{lbuf}{idx} = {rhs};")
            else:
                return (f"{lbuf}{idx} += {rhs};")
        elif styp is LoopIR.If:
            cond = self.comp_p(s.cond)
            body = self.comp_s(s.body)
            return (f"if ({cond}) {{\n"+
                    f"{body}\n"+
                    f"}}\n")

            # TODO: Do we have to push env here??
            #self.env.push()
            #self.env.pop()
        elif styp is LoopIR.ForAll:
            hi      = self.env[s.hi] # this should be a string
            itr     = self.new_varname(s.iter) # allocate a new string
            body    = self.comp_s(s.body)
            return (f"for (int {itr}=0; {itr} < {hi}; {itr}++) {{\n"+
                    f"{body}\n"+
                    f"}}")
        elif styp is LoopIR.Alloc:
            name = self.new_varname(s.name)
            if s.type is T.R:
                return (f"int {name};")
            else:
                size = _eshape(s.type, self.env)
                #TODO: Maybe randomize?
                return (f"float *{name} = (float*) malloc ({size[0]} * sizeof(float));")
        elif styp is LoopIR.Free:
            name = self.env[s.name]
            return f"free({name});"
        else: assert False, "bad case"

    def comp_e(self, e):
        etyp    = type(e)

        if etyp is LoopIR.Read:
            buf = self.env[e.name]
            #idx = ( (0,) if len(e.idx) == 0
            #             else tuple( self.comp_a(a) for a in e.idx ))
            #return buf[idx]
            idx = self.idx_str(e.idx)
            return (f"{e.name}" + idx)
        elif etyp is LoopIR.Const:
            return str(e.val)
        elif etyp is LoopIR.BinOp:
            lhs, rhs = self.comp_e(e.lhs), self.comp_e(e.rhs)
            if e.op == "+":
                return (f"{lhs} + {rhs}")
            elif e.op == "-":
                return (f"{lhs} - {rhs}")
            elif e.op == "*":
                return (f"{lhs} * {rhs}")
            elif e.op == "/":
                return (f"{lhs} / {rhs}")
        elif etyp is LoopIR.Select:
            cond    = self.comp_p(e.cond)
            if cond:
                body = self.comp_e(e.body)
                return (f"{body}")
            else:
                return ("0.0")
        else: assert False, "bad case"

    def comp_a(self, a):
        atyp    = type(a)

        if atyp is LoopIR.AVar or atyp is LoopIR.ASize:
            return self.env[a.name]
        elif atyp is LoopIR.AConst:
            return str(a.val)
        elif atyp is LoopIR.AScale:
            return (f"{a.coeff} * {self.comp_a(a.rhs)}")
        elif atyp is LoopIR.AAdd:
            return (f"{self.comp_a(a.lhs)} + {self.comp_a(a.rhs)}")
        elif atyp is LoopIR.ASub:
            return (f"{self.comp_a(a.lhs)} - {self.comp_a(a.rhs)}")
        else: assert False, "bad case"

    def comp_p(self, p):
        ptyp = type(p)

        if ptyp is LoopIR.BConst:
            return (f"{p.val}")
        elif ptyp is LoopIR.Cmp:
            lhs, rhs = self.comp_a(p.lhs), self.comp_a(p.rhs)
            if p.op == "==":
                return (f"{lhs} == {rhs}")
            elif p.op == "<":
                return (f"{lhs} < {rhs}")
            elif p.op == ">":
                return (f"{lhs} > {rhs}")
            elif p.op == "<=":
                return (f"{lhs} <= {rhs}")
            elif p.op == ">=":
                return (f"{lhs} >= {rhs}")
            else: assert False, "bad case"
        elif ptyp is LoopIR.And or ptyp is LoopIR.Or:
            lhs, rhs = self.comp_p(p.lhs), self.comp_p(p.rhs)
            if ptyp is LoopIR.And:
                return (f"{lhs} && {rhs}")
            elif ptyp is LoopIR.Or:
                return (f"{lhs} || {rhs}")
            else: assert False, "bad case"
        else: assert False, "bad case"
