from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo
import re

from .prelude import *

from . import shared_types as T
from .LoopIR import LoopIR, LoopIR_Do

from .mem_analysis import MemoryAnalysis
from .prec_analysis import PrecisionAnalysis
from .memory import MemGenError

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

class LoopIR_SubProcs(LoopIR_Do):
    def __init__(self, proc):
        self._subprocs = set()
        super().__init__(proc)

    def result(self):
        return self._subprocs

    # to improve efficiency
    def do_e(self,e):
        pass

    def do_s(self, s):
        if type(s) is LoopIR.Call:
            self._subprocs.add(s.f)
        else:
            super().do_s(s)

def find_all_subprocs(proc_list):
    to_visit    = [ p for p in reversed(proc_list) ] # ** see below
    queued      = set(to_visit)
    proc_list   = []
    visited     = set(proc_list)

    # ** to_visit is reversed so that in the simple case of requesting e.g.
    # run_compile([p1, p2], ...) the generated C-code will list the def.
    # of p1 before p2

    # flood-fill algorithm to produce a topological-sort/order
    while len(to_visit) > 0:
        p = to_visit.pop(0) # de-queue
        visited.add(p)
        proc_list.append(p)

        subp = LoopIR_SubProcs(p).result()
        for sp in subp:
            assert sp not in visited, "found cycle in the call graph"
            if sp not in queued:
                queued.add(sp)
                to_visit.append(sp) #en-queue

    return [ p for p in reversed(proc_list) ]

class LoopIR_FindMems(LoopIR_Do):
    def __init__(self, proc):
        self._mems = set()
        for a in proc.args:
            if a.mem:
                self._mems.add(a.mem)
        super().__init__(proc)

    def result(self):
        return self._mems

    # to improve efficiency
    def do_e(self,e):
        pass

    def do_s(self, s):
        if type(s) is LoopIR.Alloc:
            if s.mem:
                self._mems.add(s.mem)
        else:
            super().do_s(s)

def find_all_mems(proc_list):
    mems = set()
    for p in proc_list:
        mems.update( LoopIR_FindMems(p).result() )

    return [ m for m in mems ]

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler Entry-points

# top level compiler function called by tests!


def run_compile(proc_list, path, c_file, h_file, malloc=False):

    fwd_decls, body = compile_to_strings(proc_list)

    includes = "#include <stdio.h>\n" + "#include <stdlib.h>\n"

    if malloc:
        includes += ("#include <stdint.h>\n"+
                     "#include <assert.h>\n"+
                     "#include <string.h>\n")

        with open(os.path.dirname(os.path.realpath(__file__)) + "/malloc.c", "r") as f_malloc:
            m_lines = f_malloc.readlines()
            m_lines[0] = m_lines[0].format(heap_size = 100000)
            body = "".join(m_lines) + body

    fwd_decls = includes + "\n"+ fwd_decls

    with open(os.path.join(path, h_file), "w") as f_header:
        f_header.write(fwd_decls)

    with open(os.path.join(path, c_file), "w") as f_cpp:
        f_cpp.write("#include \"" + h_file + "\"\n\n");
        f_cpp.write(body)


def compile_to_strings(proc_list):

    # get transitive closure of call-graph
    orig_procs  = set(proc_list)
    proc_list   = find_all_subprocs(proc_list)
    mem_list    = find_all_mems(proc_list)

    # check for name conflicts between procs
    used_names  = set()
    for p in proc_list:
        if p.name in used_names:
            raise Exception(f"Cannot compile multiple "+
                            f"procedures named '{p.name}'")
        used_names.add(p.name)

    body = ["int _floor_div(int num, int quot) {",
            "  int off = (num>=0)? 0 : quot-1;",
            "  return (num-off)/quot;",
            "}",
            "\n"]

    for m in mem_list:
        if m._global:
            body.append(m._global)
            body.append("\n")

    fwd_decls = []

    for p in proc_list:
        p = MemoryAnalysis(p).result()
        p = PrecisionAnalysis(p).result()
        d, b = Compiler(p).comp_top()
        # only dump .h-file forward declarations for requested procedures
        if p in orig_procs:
            fwd_decls.append(d)
        body.append(b)

    return ("\n".join(fwd_decls), "\n".join(body))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler


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
        self.mems   = dict()
        self._tab   = ""
        self._lines = []
        self._scalar_refs = set()

        assert self.proc.name != None, "expected names for compilation"
        name            = self.proc.name
        arg_strs        = []
        typ_comments    = []

        for a in proc.args:
            mem = a.mem if a.type.is_numeric() else None
            name_arg = self.new_varname(a.name, typ=a.type, mem=mem)
            # setup, size argument binding
            if a.type == T.size:
                arg_strs.append(f"int {name_arg}")
                typ_comments.append(f"{name_arg} : size")
            # setup, index argument binding
            elif a.type == T.index:
                arg_strs.append(f"int {name_arg}")
                typ_comments.append(f"{name_arg} : index")
            # setup, arguments
            else:
                if a.type.is_real_scalar():
                    self._scalar_refs.add(a.name)
                ctyp = a.type.basetype().ctype()
                arg_strs.append(f"{ctyp}* {name_arg}")
                mem             = f" @{a.mem.name()}" if a.mem else ""
                comment_str     = f"{name_arg} : {a.type} {mem}"
                typ_comments.append(comment_str)

        self.comp_stmts(self.proc.body)

        # Generate headers here?
        comment     = (f"// {name}(\n"+
                        ',\n'.join(['//     '+s for s in typ_comments])+
                        '\n'+
                        "// )\n")
        proc_decl   = (comment+
                       f"void {name}( {', '.join(arg_strs)} );\n")
        proc_def    = (comment+
                       f"void {name}( {', '.join(arg_strs)} ) {{\n"+
                        "\n".join(self._lines) + "\n"+
                        "}\n")

        self.proc_decl = proc_decl
        self.proc_def = proc_def

    def add_line(self, line):
        self._lines.append(self._tab+line)

    def comp_stmts(self, stmts):
        for b in stmts:
            self.comp_s(b)

    def comp_top(self):
        return self.proc_decl, self.proc_def

    def new_varname(self, symbol, typ, mem=None):
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
        self.envtyp[symbol] = typ
        if mem is not None:
            self.mems[symbol] = mem
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

    def shape_strs(self, shape):
        s = [self.env[nm] for nm in shape]
        return s

    def comp_s(self, s):
        styp = type(s)

        if styp is LoopIR.Pass:
            self.add_line("; // NO-OP")
        elif styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name in self._scalar_refs:
                lhs = f"*{self.env[s.name]}"
            elif self.envtyp[s.name].is_real_scalar():
                lhs = self.env[s.name]
            else:
                lhs = self.access_str(s.name, s.idx)
            rhs = self.comp_e(s.rhs)

            cast = ""
            if s.cast:
                cast = f"({s.cast})"
            if s.cast and type(s.rhs) is LoopIR.BinOp:
                rhs = "(" + rhs + ")"

            mem = self.mems[s.name]
            if styp is LoopIR.Assign:
                if not mem._write:
                    raise MemGenError(f"{s.srcinfo}: cannot write to buffer "+
                                      f"'{s.name}' in memory '{mem.name()}'")
                self.add_line(f"{lhs} = {cast}{rhs};")
            else:
                if not mem._reduce:
                    raise MemGenError(f"{s.srcinfo}: cannot reduce to buffer "+
                                      f"'{s.name}' in memory '{mem.name()}'")
                self.add_line(f"{lhs} += {cast}{rhs};")
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
            name = self.new_varname(s.name, typ=s.type, mem=s.mem)
            assert s.type.basetype().is_real_scalar()
            ctype = s.type.basetype().ctype()
            line = s.mem._alloc( name,
                                 ctype,
                                 self.shape_strs( s.type.shape() ),
                                 None )

            self.add_line(line)
        elif styp is LoopIR.Free:
            name = self.env[s.name]
            assert s.type.basetype().is_real_scalar()
            ctype = s.type.basetype().ctype()
            line = s.mem._free( name,
                                ctype,
                                self.shape_strs( s.type.shape() ),
                                None )
            self.add_line(line)
        elif styp is LoopIR.Call:
            args    = [ self.comp_e(e, call_arg=True) for e in s.args ]
            if s.f.instr is not None:
                d = dict()
                assert len(s.f.args) == len(args)
                for i in range(len(args)):
                    d[str(s.f.args[i].name)] = args[i]

                self.add_line(f"{s.f.instr.format(**d)}")
            else:
                fname   = s.f.name
                self.add_line(f"{fname}({','.join(args)})")
        else:
            assert False, "bad case"

    def comp_e(self, e, prec=0, call_arg=False):
        etyp = type(e)

        if etyp is LoopIR.Read:
            rtyp = self.envtyp[e.name]
            if call_arg:
                if rtyp is T.size or rtyp is T.index:
                    return self.env[e.name]
                elif e.name in self._scalar_refs:
                    return self.env[e.name]
                elif type(rtyp) is T.Tensor:
                    return self.env[e.name]
                else:
                    assert rtyp is T.R
                    return f"&{self.env[e.name]}"
            else:
                if rtyp.is_indexable():
                    return self.env[e.name]

                mem = self.mems[e.name]
                if not mem._read:
                    raise MemGenError(f"{s.srcinfo}: cannot read from buffer "+
                                      f"'{s.name}' in memory '{mem.name()}'")

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
