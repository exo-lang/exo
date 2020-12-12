from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo
import re

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

from .mem_analysis import MemoryAnalysis

import numpy as np
import os


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

op_prec = {
    "or":     10,
    #
    "and":    20,
    #
    "==":     30,
    #
    "<":      40,
    ">":      40,
    "<=":     40,
    ">=":     40,
    #
    "+":      50,
    "-":      50,
    #
    "*":      60,
    "/":      60,
    "%":      60,
}

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler

# top level compiler function called by tests!


def run_compile(proc_list, path, c_file, h_file):
    # take proc_list
    # for each p in proc_list:
    #   run Compiler() pass to get (decl, def)
    #
    # check for name conflicts between procs
    #
    # write out c_file and h_file

    fwd_decls = "#include <stdio.h>\n" + "#include <stdlib.h>\n\n"

    body = (f"#include \"{h_file}\"\n\n"+
             "int _floor_div(int num, int quot) {\n"+
             "  int off = (num<0)? quot-1 : 0;\n"
             "  return (num-off)/quot;\n"
             "}\n\n")
    for p in proc_list:
        p = MemoryAnalysis(p).result()
        d, b = Compiler(p).comp_top()
        fwd_decls += d
        body += b
        body += '\n'

    with open(os.path.join(path, h_file), "w") as f_header:
        f_header.write(fwd_decls)

    with open(os.path.join(path, c_file), "w") as f_cpp:
        f_cpp.write(body)


def _type_shape(typ, env):
    return tuple(str(r) if is_pos_int(r) else env[r]
                 for r in typ.shape())


def _type_size(typ, env):
    szs = _type_shape(typ, env)
    return ("*").join(szs)


def _type_idx(typ, idx, env):
    assert type(typ) is T.Tensor
    szs = _type_shape(typ, env)
    assert len(szs) == len(idx)
    s = idx[0]
    for i, n in zip(idx[1:], szs[1:]):
        s = f"({s}) * {n} + ({i})"
    return s


class Compiler:
    def __init__(self, proc, **kwargs):
        assert type(proc) is LoopIR.proc

        self.proc   = proc
        self.env    = Environment()
        self.names  = Environment()
        self.envtyp = Environment()
        self._tab   = ""
        self._lines = []
        self._scalar_refs = set()

        assert self.proc.name != None, "expected names for compilation"
        name = self.proc.name
        arg_str = ""
        typ_comment_str = ""

        for a in proc.args:
            name_arg = self.new_varname(a.name, typ=a.type)
            # setup, size argument binding
            if a.type == T.size:
                arg_str += f" int {name_arg},"
            # setup, arguments
            else:
                if a.type == T.R:
                    self._scalar_refs.add(a.name)
                arg_str += f" float* {name_arg},"
                mem = f" @{a.mem}" if a.mem else ""
                typ_comment_str += f" {name_arg} : {a.type} @{a.effect}{mem},"

        self.comp_stmts(self.proc.body)

        # Generate headers here?
        proc_decl = (f"// {name}({typ_comment_str[:-1]} )\n"
                     + f"void {name}({arg_str[:-1]});\n"
                     )
        proc_def = (f"// {name}({typ_comment_str[:-1]} )\n"
                    + f"void {name}({arg_str[:-1]}) {{\n"
                    + "\n".join(self._lines) + "\n"
                      + "}\n"
                    )

        self.proc_decl = proc_decl
        self.proc_def = proc_def

    def add_line(self, line):
        self._lines.append(self._tab+line+"\n")

    def comp_stmts(self, stmts):
        self.push()
        for b in stmts:
            self.comp_s(b)
        self.pop()

    def comp_top(self):
        return self.proc_decl, self.proc_def

    def new_varname(self, symbol, typ=None):
        strnm   = str(symbol)
        if strnm not in self.names:
            pass
        else:
            s = self.names[strnm]
            while s in self.names:
                m = re.match('^(.*)_([0-9]*)$', s)
                if not m:
                    s = s + "_1"
                else:
                    s = f"{m[1]}_{int(m[2]) + 1}"
            self.names[strnm]   = s
            strnm               = s

        self.names[strnm]   = strnm
        self.env[symbol]    = strnm
        if typ is not None:
            self.envtyp[symbol] = typ
        return strnm

    def push(self,only=None):
        if only is None:
            self.env.push()
            self.names.push()
            self._tab = self._tab + "  "
        elif only == 'env':
            self.env.push()
            self.names.push()
        elif only == 'tab':
            self._tab = self._tab + "  "
        else:
            assert False, f"BAD only parameter {only}"

    def pop(self):
        self.env.pop()
        self.names.pop()
        self._tab = self._tab[:-2]

    def access_str(self, nm, idx_list):
        buf = self.env[nm]
        type = self.envtyp[nm]
        idxs = [self.comp_e(i) for i in idx_list]
        idx = _type_idx(type, idxs, self.env)
        return f"{buf}[{idx}]"

    def comp_s(self, s):
        styp = type(s)

        if styp is LoopIR.Pass:
            self.add_line("; // NO-OP")
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name in self._scalar_refs:
                lhs = f"*{self.env[s.name]}"
            elif self.envtyp[s.name] is T.R:
                lhs = self.env[s.name]
            else:
                lhs = self.access_str(s.name, s.idx)
            rhs = self.comp_e(s.rhs)
            if styp is LoopIR.Assign:
                self.add_line(f"{lhs} = {rhs};")
            else:
                self.add_line(f"{lhs} += {rhs};")
        elif styp is LoopIR.If:
            cond = self.comp_e(s.cond)
            self.add_line(f"if ({cond}) {{")
            self.push()
            self.comp_stmts(s.body)
            self.pop()
            if len(s.orelse) > 0:
                self.add_line("} else {")
                self.push()
                self.comp_stmts(s.orelse)
                self.pop()
            self.add_line("}")

        elif styp is LoopIR.ForAll:
            hi = self.comp_e(s.hi)
            self.push(only='env')
            itr = self.new_varname(s.iter, typ=T.index)  # allocate a new string
            self.add_line(f"for (int {itr}=0; {itr} < {hi}; {itr}++) {{")
            self.push(only='tab')
            self.comp_stmts(s.body)
            self.pop()
            self.add_line("}")

        elif styp is LoopIR.Alloc:
            name = self.new_varname(s.name, typ=s.type)
            if s.type is T.R:
                #TODO: broken
                #If we have float* in the function, should we dereference
                #upon reading??
                self.add_line(f"float {name};")
            else:
                size = _type_size(s.type, self.env)
                self.add_line(f"float *{name} = " +
                        f"(float*) malloc ({size} * sizeof(float));")
        elif styp is LoopIR.Free:
            if s.type is not T.R:
                name = self.env[s.name]
                self.add_line(f"free({name});")
        elif styp is LoopIR.Instr:
            return s.op.compile(s.body, self)
        else:
            assert False, "bad case"

    def comp_e(self, e, prec=0):
        etyp = type(e)

        if etyp is LoopIR.Read:
            rtyp = self.envtyp[e.name]
            if e.name in self._scalar_refs:
                return f"*{self.env[e.name]}"
            elif type(rtyp) is not T.Tensor:
                return self.env[e.name]
            else:
                return self.access_str(e.name, e.idx)
        elif etyp is LoopIR.Const:
            return str(e.val)
        elif etyp is LoopIR.BinOp:
            local_prec  = op_prec[e.op]
            int_div     = (e.op == "/" and not e.type.is_numeric())
            if int_div:
                local_prec = 0
            op  = e.op
            if op == "and":
                op = "&&"
            elif op == "or":
                op = "||"

            lhs = self.comp_e(e.lhs, local_prec)
            rhs = self.comp_e(e.rhs, local_prec+1)

            if int_div:
                return f"_floor_div({lhs}, {rhs})"

            s = f"{lhs} {op} {rhs}"
            if local_prec < prec:
                s = f"({s})"

            return s
        else:
            assert False, "bad case"
