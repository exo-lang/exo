from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR

from .mem_analysis import MemoryAnalysis

import numpy as np
import os

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

    f_header = open(os.path.join(path, h_file), "w")
    f_header.write(fwd_decls)
    f_header.close()

    f_cpp = open(os.path.join(path, c_file), "w")
    f_cpp.write(body)
    f_cpp.close()


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

        self.proc = proc
        self.env = Environment()
        self.envtyp = Environment()

        assert self.proc.name != None, "expected names for compilation"
        name = self.proc.name
        size_str = ""
        arg_str = ""
        typ_comment_str = ""

        # setup, size argument binding
        for sz in proc.sizes:
            size = self.new_varname(sz, force_literal=True)
            size_str += f" int {size},"

        # setup, buffer argument binding
        for a in proc.args:
            name_arg = self.new_varname(a.name, typ=a.type, force_literal=True)
            arg_str += f" float* {name_arg},"
            mem = f" @{a.mem}" if a.mem else ""
            typ_comment_str += f" {name_arg} : {a.type} @{a.effect}{mem},"

        self.env.push()
        stmt_str = self.comp_s(self.proc.body)
        self.env.pop()

        # Generate headers here?
        proc_decl = (f"// {name}({typ_comment_str[:-1]} )\n"
                     + f"void {name}({size_str}{arg_str[:-1]});\n"
                     )
        proc_def = (f"// {name}({typ_comment_str[:-1]} )\n"
                    + f"void {name}({size_str}{arg_str[:-1]}) {{\n"
                    + stmt_str + "\n"
                      + "}\n"
                    )

        self.proc_decl = proc_decl
        self.proc_def = proc_def

    def comp_top(self):
        return self.proc_decl, self.proc_def

    def new_varname(self, symbol, typ=None, force_literal=False):
        s = str(symbol) if force_literal else repr(symbol)

        assert symbol not in self.env, "name conflict!"
        self.env[symbol] = s

        if typ is not None:
            self.envtyp[symbol] = typ
        return self.env[symbol]

    def access_str(self, nm, idx_list):
        buf = self.env[nm]
        type = self.envtyp[nm]
        idxs = [self.comp_a(i) for i in idx_list]
        idx = _type_idx(type, idxs, self.env)
        return f"{buf}[{idx}]"

    def comp_s(self, s):
        styp = type(s)

        if styp is LoopIR.Seq:
            first = self.comp_s(s.s0)
            second = self.comp_s(s.s1)

            return (f"{first}\n\n{second}")
        elif styp is LoopIR.Pass:
            return (f"; // # NO-OzP :")
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if self.envtyp[s.name] is T.R:
                lhs = self.env[s.name]
            else:
                lhs = self.access_str(s.name, s.idx)
            rhs = self.comp_e(s.rhs)
            if styp is LoopIR.Assign:
                return (f"{lhs} = {rhs};")
            else:
                return (f"{lhs} += {rhs};")
        elif styp is LoopIR.If:
            cond = self.comp_p(s.cond)
            self.env.push()
            body = self.comp_s(s.body)
            self.env.pop()
            ret = (f"if ({cond}) {{\n" +
                  f"{body}\n" +
                  f"}}\n")

            if s.orelse:
                ebody = self.comp_s(s.orelse)
                ret += (f"else {{\n" +
                       f"{ebody}\n" +
                       f"}}\n")

            return ret

        elif styp is LoopIR.ForAll:
            hi = self.comp_a(s.hi)
            itr = self.new_varname(s.iter)  # allocate a new string
            self.env.push()
            body = self.comp_s(s.body)
            self.env.pop()
            return (f"for (int {itr}=0; {itr} < {hi}; {itr}++) {{\n" +
                    f"{body}\n" +
                    f"}}")
        elif styp is LoopIR.Alloc:
            name = self.new_varname(s.name, typ=s.type)
            if s.type is T.R:
                return (f"float {name};")
            else:
                size = _type_size(s.type, self.env)
                return (f"float *{name} = " +
                        f"(float*) malloc ({size} * sizeof(float));")
        elif styp is LoopIR.Free:
            if s.type is not T.R:
                name = self.env[s.name]
                return f"free({name});"
        elif styp is LoopIR.Instr:
            return s.op.compile(s.body, self)
        else:
            assert False, "bad case"

    def comp_e(self, e):
        etyp = type(e)

        if etyp is LoopIR.Read:
            if self.envtyp[e.name] is T.R:
                return self.env[e.name]
            else:
                return self.access_str(e.name, e.idx)
        elif etyp is LoopIR.Const:
            return str(e.val)
        elif etyp is LoopIR.BinOp:
            lhs, rhs = self.comp_e(e.lhs), self.comp_e(e.rhs)
            if e.op == "+":
                return f"({lhs} + {rhs})"
            elif e.op == "-":
                return f"({lhs} - {rhs})"
            elif e.op == "*":
                return f"({lhs} * {rhs})"
            elif e.op == "/":
                return f"({lhs} / {rhs})"
        elif etyp is LoopIR.Select:
            cond = self.comp_p(e.cond)
            body = self.comp_e(e.body)
            return f"(({cond})? {body} : 0.0)"
        else:
            assert False, "bad case"

    def comp_a(self, a):
        atyp = type(a)

        if atyp is LoopIR.AVar or atyp is LoopIR.ASize:
            return self.env[a.name]
        elif atyp is LoopIR.AConst:
            return str(a.val)
        elif atyp is LoopIR.AScale:
            return f"({a.coeff} * {self.comp_a(a.rhs)})"
        elif atyp is LoopIR.AAdd:
            return f"({self.comp_a(a.lhs)} + {self.comp_a(a.rhs)})"
        elif atyp is LoopIR.ASub:
            return f"({self.comp_a(a.lhs)} - {self.comp_a(a.rhs)})"
        elif atyp is LoopIR.AScaleDiv:
            assert a.quotient > 0
            return f"_floor_div({self.comp_a(a.lhs)}, {a.quotient})"
        elif atyp is LoopIR.AMod:
            assert a.divisor > 0
            return f"{self.comp_a(a.lhs)} % {a.divisor})"
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
            else:
                assert False, "bad case"
        elif ptyp is LoopIR.And or ptyp is LoopIR.Or:
            lhs, rhs = self.comp_p(p.lhs), self.comp_p(p.rhs)
            if ptyp is LoopIR.And:
                return (f"{lhs} && {rhs}")
            elif ptyp is LoopIR.Or:
                return (f"{lhs} || {rhs}")
            else:
                assert False, "bad case"
        else:
            assert False, "bad case"
