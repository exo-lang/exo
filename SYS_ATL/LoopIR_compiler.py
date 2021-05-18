from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo
import re

from collections import defaultdict

from .prelude import *

from .LoopIR import T
from .LoopIR import LoopIR, LoopIR_Do

from .mem_analysis import MemoryAnalysis
from .prec_analysis import PrecisionAnalysis
from .win_analysis import WindowAnalysis
from .memory import MemGenError

import numpy as np
import os


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

CacheDict = lambda: defaultdict(CacheDict)

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

_window_struct_shorthand = {
    T.f32       : 'f32',
    T.f64       : 'f64',
    T.i8        : 'i8',
}

def window_struct(basetyp, n_dims):
    assert n_dims >= 1
    sname = f"systl_win_{n_dims}{_window_struct_shorthand[basetyp]}"

    sdef = (f"struct {sname}{{\n"+
            f"    {basetyp.ctype()} *data;\n"+
            f"    int strides[{n_dims}];\n"+
            f"}};")

    return sname, sdef


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

        with open(os.path.dirname(os.path.realpath(__file__)) +
                                  "/malloc.c", "r") as f_malloc:
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
    struct_defns = set()

    for p in proc_list:
        p       = MemoryAnalysis(p).result()
        p       = PrecisionAnalysis(p).result()
        p       = WindowAnalysis(p).result()
        comp    = Compiler(p)
        d, b    = comp.comp_top()
        struct_defns = struct_defns.union(comp.struct_defns())
        # only dump .h-file forward declarations for requested procedures
        if p in orig_procs:
            fwd_decls.append(d)
        body.append(b)

    # add struct definitions before the other forward declarations
    fwd_decls = list(struct_defns) + fwd_decls

    return ("\n".join(fwd_decls), "\n".join(body))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler

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

        self.window_defns = set()
        self.window_cache = CacheDict()

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
                if a.type.is_win():
                    wintyp = self.get_window_type(a.type)
                    arg_strs.append(f"struct {wintyp} {name_arg}")
                else:
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

    def struct_defns(self):
        return self.window_defns

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
        idx_expr = self.get_idx_offset(buf,type,idxs)
        if not type.is_win():
            return f"{buf}[{idx_expr}]"
        else:
            return f"{buf}.data[{idx_expr}]"

    def shape_strs(self, shape, prec=100):
        return [ self.comp_e(s,prec=prec) for s in shape ]

    def tensor_strides(self, shape, prec=100):
        szs = self.shape_strs(shape, max(prec,61))
        assert len(szs) >= 1
        strides = ["1"]
        s = szs[-1]
        for sz in reversed(szs[:-1]):
            strides.append(s)
            s = f"{sz} * {s}"
        strides = list(reversed(strides))
        return strides

    # works for any tensor or window type
    def get_strides(self, name, typ, prec=100):
        if typ.is_win():
            return [ f"{name}.strides[{i}]" for i in range(len(typ.shape())) ]
        else:
            return self.tensor_strides(typ.shape(), prec)

    def get_idx_offset(self, name, typ, idx):
        strides = self.get_strides(name, typ, prec=61)
        assert len(strides) == len(idx)
        acc = " + ".join([ f"{i} * {s}" for i,s in zip(idx,strides) ])
        return acc

    #def _type_idx(self, typ, idx):
        #szs = self.shape_strs(typ.shape(), prec=61)
        #assert len(szs) == len(idx)
        #s = idx[0]
        #for i, n in zip(idx[1:], szs[1:]):
        #    s = f"({s}) * {n} + ({i})"
        #return s

    def get_window_type(self, typ):
        if type(typ) is T.Window:
            base    = typ.as_tensor.basetype()
            n_dims  = len(typ.as_tensor.shape())
        elif type(typ) is T.Tensor and typ.is_window:
            base    = typ.basetype()
            n_dims  = len(typ.shape())
        else: assert False, f"not a window type: {typ}"

        lookup = self.window_cache[base][n_dims]
        if type(lookup) is str:
            return lookup
        else:
            name, defn = window_struct(base, n_dims)
            self.window_defns.add(defn)
            self.window_cache[base][n_dims] = name
            return name

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
            if s.type.basetype() != s.rhs.type.basetype():
                cast = f"({s.type.ctype()})"
                if type(s.rhs) is LoopIR.BinOp:
                    rhs = f"({rhs})"

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
        elif styp is LoopIR.WindowStmt:
            win_struct  = self.get_window_type(s.rhs.type)
            rhs         = self.comp_e(s.rhs)
            assert type(s.rhs) is LoopIR.WindowExpr
            mem         = self.mems[s.rhs.name]
            lhs         = self.new_varname(s.lhs, typ=s.rhs.type, mem=mem)
            self.add_line(f"struct {win_struct} {lhs} = {rhs};")
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
            assert all(a.type.is_win() == fna.type.is_win()
                       for a, fna in zip(s.args, s.f.args))
            args    = [ self.comp_e(e, call_arg=True) for e in s.args ]
            if s.f.instr is not None:
                d = dict()
                assert len(s.f.args) == len(args)
                for i in range(len(args)):
                    d[str(s.f.args[i].name)] = args[i]

                self.add_line(f"{s.f.instr.format(**d)}")
            else:
                fname   = s.f.name
                self.add_line(f"{fname}({','.join(args)});")
        else:
            assert False, "bad case"

    def comp_e(self, e, prec=0, call_arg=False):
        etyp = type(e)

        if etyp is LoopIR.Read:
            rtyp = self.envtyp[e.name]
            if call_arg:
                assert len(e.idx) == 0
                if rtyp.is_indexable():
                    return self.env[e.name]
                elif e.name in self._scalar_refs:
                    return self.env[e.name]
                elif rtyp.is_tensor_or_window():
                    return self.env[e.name]
                else:
                    assert rtyp.is_real_scalar()
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
                elif not rtyp.is_tensor_or_window():
                    return self.env[e.name]
                else:
                    return self.access_str(e.name, e.idx)
        elif etyp is LoopIR.WindowExpr:
            win_struct  = self.get_window_type(e.type)
            base        = self.env[e.name]
            basetyp     = self.envtyp[e.name]
            mem         = self.mems[e.name]
            if not mem._read:
                assert False, "TODO: handle GEMMINI Windowing"

            # compute offset to new data pointer
            def w_lo(w):
                return (w.lo if type(w) is LoopIR.Interval else w.pt)

            idxs        = [ self.comp_e(w_lo(w)) for w in e.idx ]
            idx_expr    = self.get_idx_offset(base, basetyp, idxs)
            if not basetyp.is_win():
                dataptr = f"{base} + {idx_expr}"
            else:
                dataptr = f"{base}.data + {idx_expr}"

            # compute new window strides
            strides     = self.get_strides(base, basetyp, prec=0)
            assert len(strides) == len(e.idx)
            assert len(strides) > 0
            strides     = [ s for s,w in zip(strides,e.idx)
                              if type(w) is LoopIR.Interval ]

            struct_str = f"(struct {win_struct}){{ {dataptr}, {','.join(strides)} }}"

            return struct_str
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
