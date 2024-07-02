import functools
import re
import textwrap
from collections import ChainMap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .LoopIR import LoopIR, LoopIR_Do, get_writes_of_stmts, T, CIR
from .configs import ConfigError
from .mem_analysis import MemoryAnalysis
from .memory import MemGenError, Memory, DRAM, StaticMemory
from .parallel_analysis import ParallelAnalysis
from .prec_analysis import PrecisionAnalysis
from .prelude import *
from .win_analysis import WindowAnalysis
from .range_analysis import IndexRangeEnvironment


def sanitize_str(s):
    return re.sub(r"\W", "_", s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

CacheDict = lambda: defaultdict(CacheDict)

op_prec = {
    "or": 10,
    #
    "and": 20,
    #
    "==": 30,
    #
    "<": 40,
    ">": 40,
    "<=": 40,
    ">=": 40,
    #
    "+": 50,
    "-": 50,
    #
    "*": 60,
    "/": 60,
    "%": 60,
    # unary minus
    "~": 70,
}


def lift_to_cir(e, range_env):
    assert e.type.is_indexable(), "why are you here?"

    is_non_neg = lambda e: range_env.check_expr_bound(0, IndexRangeEnvironment.leq, e)

    if isinstance(e, LoopIR.Read):
        return CIR.Read(e.name, is_non_neg(e))
    elif isinstance(e, LoopIR.Const):
        return CIR.Const(e.val)
    elif isinstance(e, LoopIR.BinOp):
        lhs = lift_to_cir(e.lhs, range_env)
        rhs = lift_to_cir(e.rhs, range_env)
        return CIR.BinOp(e.op, lhs, rhs, is_non_neg(e))
    elif isinstance(e, LoopIR.USub):
        arg = lift_to_cir(e.arg, range_env)
        return CIR.USub(arg, is_non_neg(e))
    else:
        assert False, "bad case!"


operations = {
    "+": lambda x, y: x + y,
    "-": lambda x, y: x - y,
    "*": lambda x, y: x * y,
    "/": lambda x, y: x / y,
    "%": lambda x, y: x % y,
}


def simplify_cir(e):
    if isinstance(e, (CIR.Read, CIR.Const, CIR.Stride)):
        return e

    elif isinstance(e, CIR.BinOp):
        lhs = simplify_cir(e.lhs)
        rhs = simplify_cir(e.rhs)

        if isinstance(lhs, CIR.Const) and isinstance(rhs, CIR.Const):
            return CIR.Const(operations[e.op](lhs.val, rhs.val))

        if isinstance(lhs, CIR.Const) and lhs.val == 0:
            if e.op == "+":
                return rhs
            elif e.op == "*" or e.op == "/":
                return CIR.Const(0)
            elif e.op == "-":
                pass  # cannot simplify
            else:
                assert False

        if isinstance(rhs, CIR.Const) and rhs.val == 0:
            if e.op == "+" or e.op == "-":
                return lhs
            elif e.op == "*":
                return CIR.Const(0)
            elif e.op == "/":
                assert False, "division by zero??"
            else:
                assert False, "bad case"

        if isinstance(lhs, CIR.Const) and lhs.val == 1 and e.op == "*":
            return rhs

        if isinstance(rhs, CIR.Const) and rhs.val == 1 and (e.op == "*" or e.op == "/"):
            return lhs

        return CIR.BinOp(e.op, lhs, rhs, e.is_non_neg)
    elif isinstance(e, CIR.USub):
        arg = simplify_cir(e.arg)
        if isinstance(arg, CIR.USub):
            return arg.arg
        if isinstance(arg, CIR.Const):
            return arg.update(val=-(arg.val))
        return e.update(arg=arg)
    else:
        assert False, "bad case!"


class LoopIR_SubProcs(LoopIR_Do):
    def __init__(self, proc):
        self._subprocs = set()
        if proc.instr is None:
            super().__init__(proc)

    def result(self):
        return self._subprocs

    # to improve efficiency
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Call):
            self._subprocs.add(s.f)
        else:
            super().do_s(s)


def find_all_subprocs(proc_list):
    all_procs = []
    seen = set()

    def walk(proc, visited):
        if proc in seen:
            return

        all_procs.append(proc)
        seen.add(proc)

        for sp in LoopIR_SubProcs(proc).result():
            if sp in visited:
                raise ValueError(f"found call cycle involving {sp.name}")
            walk(sp, visited | {proc})

    for proc in proc_list:
        walk(proc, set())

    # Reverse for C declaration order.
    return list(reversed(all_procs))


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
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.mem:
                self._mems.add(s.mem)
        else:
            super().do_s(s)

    def do_t(self, t):
        pass


class LoopIR_FindBuiltIns(LoopIR_Do):
    def __init__(self, proc):
        self._builtins = set()
        super().__init__(proc)

    def result(self):
        return self._builtins

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.BuiltIn):
            self._builtins.add(e.f)
        else:
            super().do_e(e)

    def do_t(self, t):
        pass


class LoopIR_FindConfigs(LoopIR_Do):
    def __init__(self, proc):
        self._configs = set()
        super().__init__(proc)

    def result(self):
        return self._configs

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.ReadConfig):
            self._configs.add(e.config)
        else:
            super().do_e(e)

    def do_s(self, s):
        if isinstance(s, LoopIR.WriteConfig):
            self._configs.add(s.config)
        super().do_s(s)

    def do_t(self, t):
        pass


def find_all_mems(proc_list):
    mems = set()
    for p in proc_list:
        mems.update(LoopIR_FindMems(p).result())

    return [m for m in mems]


def find_all_builtins(proc_list):
    builtins = set()
    for p in proc_list:
        builtins.update(LoopIR_FindBuiltIns(p).result())

    return [b for b in builtins]


def find_all_configs(proc_list):
    configs = set()
    for p in proc_list:
        configs.update(LoopIR_FindConfigs(p).result())

    return list(configs)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class WindowStruct:
    name: str
    definition: str


@functools.cache
def _window_struct(typename, ctype, n_dims, is_const) -> WindowStruct:
    const_kwd = "const " if is_const else ""
    const_suffix = "c" if is_const else ""

    sname = f"exo_win_{n_dims}{typename}{const_suffix}"
    sdef = (
        f"struct {sname}{{\n"
        f"    {const_kwd}{ctype} * const data;\n"
        f"    const int_fast32_t strides[{n_dims}];\n"
        f"}};"
    )

    sdef_guard = sname.upper()
    sdef = f"""#ifndef {sdef_guard}
#define {sdef_guard}
{sdef}
#endif"""

    return WindowStruct(sname, sdef)


def window_struct(base_type, n_dims, is_const) -> WindowStruct:
    assert n_dims >= 1

    _window_struct_shorthand = {
        T.f16: "f16",
        T.f32: "f32",
        T.f64: "f64",
        T.i8: "i8",
        T.ui8: "ui8",
        T.ui16: "ui16",
        T.i32: "i32",
    }

    return _window_struct(
        _window_struct_shorthand[base_type], base_type.ctype(), n_dims, is_const
    )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler Entry-points

# top level compiler function called by tests!


def run_compile(proc_list, h_file_name: str):
    file_stem = str(Path(h_file_name).stem)
    lib_name = sanitize_str(file_stem)
    fwd_decls, body = compile_to_strings(lib_name, proc_list)

    source = f'#include "{h_file_name}"\n\n{body}'

    header_guard = f"{lib_name}_H".upper()
    header = f"""
#pragma once
#ifndef {header_guard}
#define {header_guard}

#ifdef __cplusplus
extern "C" {{
#endif

{fwd_decls}

#ifdef __cplusplus
}}
#endif
#endif  // {header_guard}
"""

    return source, header


_static_helpers = {
    "exo_floor_div": textwrap.dedent(
        """
        static int exo_floor_div(int num, int quot) {
          int off = (num>=0)? 0 : quot-1;
          return (num-off)/quot;
        }
        """
    ),
}


def compile_to_strings(lib_name, proc_list):
    # Get transitive closure of call-graph
    orig_procs = [id(p) for p in proc_list]

    def from_lines(x):
        return "\n".join(x)

    proc_list = list(sorted(find_all_subprocs(proc_list), key=lambda x: x.name))

    # Header contents
    ctxt_name, ctxt_def = _compile_context_struct(find_all_configs(proc_list), lib_name)
    struct_defns = set()
    public_fwd_decls = []

    # Body contents
    memory_code = _compile_memories(find_all_mems(proc_list))
    builtin_code = _compile_builtins(find_all_builtins(proc_list))
    private_fwd_decls = []
    proc_bodies = []
    instrs_global = []

    needed_helpers = set()

    # Compile proc bodies
    seen_procs = set()
    for p in proc_list:
        if p.name in seen_procs:
            raise TypeError(f"multiple procs named {p.name}")
        seen_procs.add(p.name)

        # don't compile instruction procedures, but add a comment.
        if p.instr is not None:
            argstr = ",".join([str(a.name) for a in p.args])
            proc_bodies.extend(
                [
                    "",
                    '/* relying on the following instruction..."',
                    f"{p.name}({argstr})",
                    p.instr.c_instr,
                    "*/",
                ]
            )
            if p.instr.c_global:
                instrs_global.append(p.instr.c_global)
        else:
            is_public_decl = id(p) in orig_procs

            p = ParallelAnalysis().run(p)
            p = PrecisionAnalysis().run(p)
            p = WindowAnalysis().apply_proc(p)
            p = MemoryAnalysis().run(p)

            comp = Compiler(p, ctxt_name, is_public_decl=is_public_decl)
            d, b = comp.comp_top()
            struct_defns |= comp.struct_defns()
            needed_helpers |= comp.needed_helpers()

            if is_public_decl:
                public_fwd_decls.append(d)
            else:
                private_fwd_decls.append(d)

            proc_bodies.append(b)

    # Structs are just blobs of code... still sort them for output stability
    struct_defns = [x.definition for x in sorted(struct_defns, key=lambda x: x.name)]

    header_contents = f"""
#include <stdint.h>
#include <stdbool.h>

// Compiler feature macros adapted from Hedley (public domain)
// https://github.com/nemequ/hedley

#if defined(__has_builtin)
#  define EXO_HAS_BUILTIN(builtin) __has_builtin(builtin)
#else
#  define EXO_HAS_BUILTIN(builtin) (0)
#endif

#if EXO_HAS_BUILTIN(__builtin_assume)
#  define EXO_ASSUME(expr) __builtin_assume(expr)
#elif EXO_HAS_BUILTIN(__builtin_unreachable)
#  define EXO_ASSUME(expr) \\
      ((void)((expr) ? 1 : (__builtin_unreachable(), 1)))
#else
#  define EXO_ASSUME(expr) ((void)(expr))
#endif

{from_lines(ctxt_def)}
{from_lines(struct_defns)}
{from_lines(public_fwd_decls)}
"""

    helper_code = [_static_helpers[v] for v in needed_helpers]
    body_contents = [
        helper_code,
        instrs_global,
        memory_code,
        builtin_code,
        private_fwd_decls,
        proc_bodies,
    ]
    body_contents = list(filter(lambda x: x, body_contents))  # filter empty lines
    body_contents = map(from_lines, body_contents)
    body_contents = from_lines(body_contents)
    body_contents += "\n"  # New line at end of file
    return header_contents, body_contents


def _compile_builtins(builtins):
    builtin_code = []
    for b in sorted(builtins, key=lambda x: x.name()):
        if glb := b.globl():
            builtin_code.append(glb)
    return builtin_code


def _compile_memories(mems):
    memory_code = []
    for m in sorted(mems, key=lambda x: x.name()):
        memory_code.append(m.global_())
    return memory_code


def _compile_context_struct(configs, lib_name):
    if not configs:
        return "void", []

    ctxt_name = f"{lib_name}_Context"
    ctxt_def = [f"typedef struct {ctxt_name} {{ ", f""]

    seen = set()
    for c in sorted(configs, key=lambda x: x.name()):
        name = c.name()

        if name in seen:
            raise TypeError(f"multiple configs named {name}")
        seen.add(name)

        if c.is_allow_rw():
            sdef_lines = c.c_struct_def()
            sdef_lines = [f"    {line}" for line in sdef_lines]
            ctxt_def += sdef_lines
            ctxt_def += [""]
        else:
            ctxt_def += [f"// config '{name}' not materialized", ""]

    ctxt_def += [f"}} {ctxt_name};"]
    return ctxt_name, ctxt_def


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler


class Compiler:
    def __init__(self, proc, ctxt_name, *, is_public_decl):
        assert isinstance(proc, LoopIR.proc)

        self.proc = proc
        self.ctxt_name = ctxt_name
        self.env = ChainMap()
        self.range_env = IndexRangeEnvironment(proc, fast=False)
        self.names = ChainMap()
        self.envtyp = dict()
        self.mems = dict()
        self._tab = ""
        self._lines = []
        self._scalar_refs = set()
        self._needed_helpers = set()
        self.window_defns = set()
        self._known_strides = {}

        assert self.proc.name is not None, "expected names for compilation"
        name = self.proc.name
        arg_strs = []
        typ_comments = []

        # reserve the first "ctxt" argument
        self.new_varname(Sym("ctxt"), None)
        arg_strs.append(f"{ctxt_name} *ctxt")

        self.non_const = set(e for e, _ in get_writes_of_stmts(self.proc.body))

        for a in proc.args:
            mem = a.mem if a.type.is_numeric() else None
            name_arg = self.new_varname(a.name, typ=a.type, mem=mem)
            if a.type in (T.size, T.index, T.bool, T.stride):
                arg_strs.append(f"{a.type.ctype()} {name_arg}")
                typ_comments.append(f"{name_arg} : {a.type}")
            # setup, arguments
            else:
                assert a.type.is_numeric()
                assert a.type.basetype() != T.R
                if a.type.is_real_scalar():
                    self._scalar_refs.add(a.name)
                if a.type.is_win():
                    wintyp = self.get_window_type(a)
                    arg_strs.append(f"struct {wintyp} {name_arg}")
                else:
                    const_kwd = "const " if a.name not in self.non_const else ""
                    ctyp = a.type.basetype().ctype()
                    arg_strs.append(f"{const_kwd}{ctyp}* {name_arg}")
                mem = f" @{a.mem.name()}" if a.mem else ""
                comment_str = f"{name_arg} : {a.type}{mem}"
                typ_comments.append(comment_str)

        for pred in proc.preds:
            if isinstance(pred, LoopIR.Const):
                # TODO: filter these out earlier?
                continue

            if (
                isinstance(pred, LoopIR.BinOp)
                and pred.op == "=="
                and isinstance(pred.lhs, LoopIR.StrideExpr)
                and isinstance(pred.rhs, LoopIR.Const)
            ):
                self._known_strides[(pred.lhs.name, pred.lhs.dim)] = CIR.Const(
                    pred.rhs.val
                )
                self.add_line(f"// assert {pred}")
            else:
                # Default to just informing the compiler about the constraint
                # on a best-effort basis
                self.add_line(f"EXO_ASSUME({self.comp_e(pred)});")

        if not self.static_memory_check(self.proc):
            raise MemGenError("Cannot generate static memory in non-leaf procs")

        self.comp_stmts(self.proc.body)

        static_kwd = "" if is_public_decl else "static "

        # Generate headers here?
        comment = (
            f"// {name}(\n" + ",\n".join(["//     " + s for s in typ_comments]) + "\n"
            "// )\n"
        )
        proc_decl = comment + f"{static_kwd}void {name}( {', '.join(arg_strs)} );\n"
        proc_def = (
            comment
            + f"{static_kwd}void {name}( {', '.join(arg_strs)} ) {{\n"
            + "\n".join(self._lines)
            + "\n"
            "}\n"
        )

        self.proc_decl = proc_decl
        self.proc_def = proc_def

    def static_memory_check(self, proc):
        def allocates_static_memory(stmts):
            check = False
            for s in stmts:
                if isinstance(s, LoopIR.Alloc):
                    mem = s.mem
                    assert issubclass(mem, Memory)
                    check |= issubclass(mem, StaticMemory)
                elif isinstance(s, LoopIR.For):
                    check |= allocates_static_memory(s.body)
                elif isinstance(s, LoopIR.If):
                    check |= allocates_static_memory(s.body)
                    check |= allocates_static_memory(s.orelse)
            return check

        def is_leaf_proc(stmts):
            check = True
            for s in stmts:
                if isinstance(s, LoopIR.Call):
                    # Since intrinsics don't allocate memory, we can ignore
                    # them for leaf-node classification purposes. We want
                    # to avoid nested procs that both allocate static memory.
                    check &= s.f.instr is not None
                elif isinstance(s, LoopIR.For):
                    check &= is_leaf_proc(s.body)
                elif isinstance(s, LoopIR.If):
                    check &= is_leaf_proc(s.body)
                    check &= is_leaf_proc(s.orelse)
            return check

        return not allocates_static_memory(proc.body) or is_leaf_proc(proc.body)

    def add_line(self, line):
        if line:
            self._lines.append(self._tab + line)

    def comp_stmts(self, stmts):
        for b in stmts:
            self.comp_s(b)

    def comp_top(self):
        return self.proc_decl, self.proc_def

    def struct_defns(self):
        return self.window_defns

    def needed_helpers(self):
        return self._needed_helpers

    def new_varname(self, symbol, typ, mem=None):
        strnm = str(symbol)
        if strnm not in self.names:
            pass
        else:
            s = self.names[strnm]
            while s in self.names:
                m = re.match(r"^(.*)_([0-9]*)$", s)
                if not m:
                    s = s + "_1"
                else:
                    s = f"{m[1]}_{int(m[2]) + 1}"
            self.names[strnm] = s
            strnm = s

        self.names[strnm] = strnm
        self.env[symbol] = strnm
        self.envtyp[symbol] = typ
        if mem is not None:
            self.mems[symbol] = mem
        else:
            self.mems[symbol] = DRAM
        return strnm

    def push(self, only=None):
        if only is None:
            self.env = self.env.new_child()
            self.range_env.enter_scope()
            self.names = self.names.new_child()
            self._tab = self._tab + "  "
        elif only == "env":
            self.env = self.env.new_child()
            self.range_env.enter_scope()
            self.names = self.names.new_child()
        elif only == "tab":
            self._tab = self._tab + "  "
        else:
            assert False, f"BAD only parameter {only}"

    def pop(self):
        self.env = self.env.parents
        self.range_env.exit_scope()
        self.names = self.names.parents
        self._tab = self._tab[:-2]

    def comp_cir(self, e, env, prec) -> str:
        if isinstance(e, CIR.Read):
            return env[e.name]

        elif isinstance(e, CIR.Const):
            return str(e.val)

        elif isinstance(e, CIR.BinOp):
            local_prec = op_prec[e.op]

            lhs = self.comp_cir(e.lhs, env, local_prec)
            rhs = self.comp_cir(e.rhs, env, local_prec)

            if isinstance(e.rhs, CIR.BinOp) and (e.op == "-" or e.op == "/"):
                rhs = f"({rhs})"

            if e.op == "/":
                if (isinstance(e.lhs, (CIR.Read, CIR.BinOp)) and e.lhs.is_non_neg) or (
                    isinstance(e.lhs, CIR.Const) and e.lhs.val > 0
                ):
                    return f"({lhs} / {rhs})"
                else:
                    return self._call_static_helper("exo_floor_div", lhs, rhs)

            s = f"{lhs} {e.op} {rhs}"
            if local_prec < prec:
                s = f"({s})"

            return s

        elif isinstance(e, CIR.Stride):
            return f"{e.name}.strides[{e.dim}]"
        elif isinstance(e, CIR.USub):
            return f'-{self.comp_cir(e.arg, env, op_prec["~"])}'
        else:
            assert False, "bad case!"

    def access_str(self, nm, idx_list) -> str:
        type = self.envtyp[nm]
        cirs = [lift_to_cir(i, self.range_env) for i in idx_list]
        idx_expr = self.get_idx_offset(nm, type, cirs)
        idx_expr_s = self.comp_cir(simplify_cir(idx_expr), self.env, prec=0)
        buf = self.env[nm]
        if not type.is_win():
            return f"{buf}[{idx_expr_s}]"
        else:
            return f"{buf}.data[{idx_expr_s}]"

    def shape_strs(self, shape, prec=100) -> str:
        comp_res = [
            self.comp_cir(simplify_cir(lift_to_cir(i, self.range_env)), self.env, prec)
            for i in shape
        ]
        return comp_res

    def tensor_strides(self, shape) -> CIR:
        szs = [lift_to_cir(i, self.range_env) for i in shape]
        assert len(szs) >= 1
        strides = [CIR.Const(1)]
        s = szs[-1]
        for sz in reversed(szs[:-1]):
            strides.append(s)
            s = CIR.BinOp("*", sz, s, True)
        strides = list(reversed(strides))

        return strides

    # works for any tensor or window type
    def get_strides(self, name: Sym, typ) -> CIR:
        if typ.is_win():
            res = []
            for i in range(len(typ.shape())):
                if stride := self._known_strides.get((name, i)):
                    res.append(stride)
                else:
                    res.append(CIR.Stride(name, i))

            return res
        else:
            return self.tensor_strides(typ.shape())

    def get_idx_offset(self, name: Sym, typ, idx) -> CIR:
        strides = self.get_strides(name, typ)
        assert len(strides) == len(idx)
        acc = CIR.BinOp("*", idx[0], strides[0], True)
        for i, s in zip(idx[1:], strides[1:]):
            new = CIR.BinOp("*", i, s, True)
            acc = CIR.BinOp("+", acc, new, True)

        return acc

    def get_window_type(self, typ, is_const=None):
        assert isinstance(typ, T.Window) or (
            isinstance(typ, LoopIR.fnarg) and typ.type.is_win()
        )

        if isinstance(typ, T.Window):
            base = typ.as_tensor.basetype()
            n_dims = len(typ.as_tensor.shape())
            if is_const is None:
                is_const = typ.src_buf not in self.non_const
        else:
            base = typ.type.basetype()
            n_dims = len(typ.type.shape())
            if is_const is None:
                is_const = typ.name not in self.non_const

        win = window_struct(base, n_dims, is_const)
        self.window_defns.add(win)
        return win.name

    def comp_s(self, s):
        if isinstance(s, LoopIR.Pass):
            self.add_line("; // NO-OP")
        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            if s.name in self._scalar_refs:
                lhs = f"*{self.env[s.name]}"
            elif self.envtyp[s.name].is_real_scalar():
                lhs = self.env[s.name]
            else:
                lhs = self.access_str(s.name, s.idx)
            rhs = self.comp_e(s.rhs)

            # possibly cast!
            lbtyp = s.type.basetype()
            rbtyp = s.rhs.type.basetype()
            if lbtyp != rbtyp:
                assert s.type.is_real_scalar()
                assert s.rhs.type.is_real_scalar()

                rhs = f"({lbtyp.ctype()})({rhs})"

            mem: Memory = self.mems[s.name]
            if isinstance(s, LoopIR.Assign):
                self.add_line(mem.write(s, lhs, rhs))
            else:
                self.add_line(mem.reduce(s, lhs, rhs))

        elif isinstance(s, LoopIR.WriteConfig):
            if not s.config.is_allow_rw():
                raise ConfigError(
                    f"{s.srcinfo}: cannot write to config '{s.config.name()}'"
                )

            nm = s.config.name()
            rhs = self.comp_e(s.rhs)

            # possibly cast!
            ltyp = s.config.lookup_type(s.field)
            rtyp = s.rhs.type
            if ltyp != rtyp and not ltyp.is_indexable():
                assert ltyp.is_real_scalar()
                assert rtyp.is_real_scalar()

                rhs = f"({ltyp.ctype()})({rhs})"

            self.add_line(f"ctxt->{nm}.{s.field} = {rhs};")

        elif isinstance(s, LoopIR.WindowStmt):
            win_struct = self.get_window_type(s.rhs.type)
            rhs = self.comp_e(s.rhs)
            assert isinstance(s.rhs, LoopIR.WindowExpr)
            mem = self.mems[s.rhs.name]
            name = self.new_varname(s.name, typ=s.rhs.type, mem=mem)
            self.add_line(f"struct {win_struct} {name} = {rhs};")
        elif isinstance(s, LoopIR.If):
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

        elif isinstance(s, LoopIR.For):
            lo = self.comp_e(s.lo)
            hi = self.comp_e(s.hi)
            self.push(only="env")
            itr = self.new_varname(s.iter, typ=T.index)  # allocate a new string
            self.range_env.add_loop_iter(
                s.iter,
                s.lo,
                s.hi,
            )
            if isinstance(s.loop_mode, LoopIR.Par):
                self.add_line(f"#pragma omp parallel for")
            self.add_line(f"for (int_fast32_t {itr} = {lo}; {itr} < {hi}; {itr}++) {{")
            self.push(only="tab")
            self.comp_stmts(s.body)
            self.pop()
            self.add_line("}")

        elif isinstance(s, LoopIR.Alloc):
            name = self.new_varname(s.name, typ=s.type, mem=s.mem)
            assert s.type.basetype().is_real_scalar()
            assert s.type.basetype() != T.R
            ctype = s.type.basetype().ctype()
            mem = s.mem or DRAM
            line = mem.alloc(name, ctype, self.shape_strs(s.type.shape()), s.srcinfo)

            self.add_line(line)
        elif isinstance(s, LoopIR.Free):
            name = self.env[s.name]
            assert s.type.basetype().is_real_scalar()
            ctype = s.type.basetype().ctype()
            mem = s.mem or DRAM
            line = mem.free(name, ctype, self.shape_strs(s.type.shape()), s.srcinfo)
            self.add_line(line)
        elif isinstance(s, LoopIR.Call):
            assert all(
                a.type.is_win() == fna.type.is_win() for a, fna in zip(s.args, s.f.args)
            )
            args = [self.comp_fnarg(e, s.f, i) for i, e in enumerate(s.args)]
            if s.f.instr is not None:
                d = dict()
                assert len(s.f.args) == len(args)
                for i in range(len(args)):
                    arg_name = str(s.f.args[i].name)
                    d[arg_name] = f"({args[i]})"
                    arg_type = s.args[i].type
                    if arg_type.is_win():
                        assert isinstance(s.args[i], LoopIR.WindowExpr)
                        data, _ = self.window_struct_fields(s.args[i])
                        d[f"{arg_name}_data"] = data
                        d[f"{arg_name}_int"] = self.env[s.args[i].name]
                    else:
                        d[f"{arg_name}_data"] = f"({args[i]})"

                self.add_line(f"{s.f.instr.c_instr.format(**d)}")
            else:
                fname = s.f.name
                args = ["ctxt"] + args
                self.add_line(f"{fname}({','.join(args)});")
        else:
            assert False, "bad case"

    def comp_fnarg(self, e, fn, i, *, prec=0):
        if isinstance(e, LoopIR.Read):
            assert not e.idx
            rtyp = self.envtyp[e.name]
            if rtyp.is_indexable():
                return self.env[e.name]
            elif rtyp is T.bool:
                return self.env[e.name]
            elif rtyp is T.stride:
                return self.env[e.name]
            elif e.name in self._scalar_refs:
                return self.env[e.name]
            elif rtyp.is_tensor_or_window():
                return self.env[e.name]
            else:
                assert rtyp.is_real_scalar()
                return f"&{self.env[e.name]}"
        elif isinstance(e, LoopIR.WindowExpr):
            if isinstance(fn, LoopIR.proc):
                callee_buf = fn.args[i].name
                is_const = callee_buf not in set(
                    x for x, _ in get_writes_of_stmts(fn.body)
                )
            else:
                raise NotImplementedError("Passing windows to built-ins")
            win_struct = self.get_window_type(e.type, is_const)
            data, strides = self.window_struct_fields(e)
            return f"(struct {win_struct}){{ &{data}, {{ {strides} }} }}"
        else:
            return self.comp_e(e, prec)

    def comp_e(self, e, prec=0):
        if isinstance(e, LoopIR.Read):
            rtyp = self.envtyp[e.name]
            if rtyp.is_indexable() or rtyp is T.bool or rtyp == T.stride:
                return self.env[e.name]

            mem: Memory = self.mems[e.name]

            if not mem.can_read():
                raise MemGenError(
                    f"{e.srcinfo}: cannot read from buffer "
                    f"'{e.name}' in memory '{mem.name()}'"
                )

            if e.name in self._scalar_refs:
                return f"*{self.env[e.name]}"
            elif not rtyp.is_tensor_or_window():
                return self.env[e.name]
            else:
                return self.access_str(e.name, e.idx)

        elif isinstance(e, LoopIR.WindowExpr):
            win_struct = self.get_window_type(e.type)
            data, strides = self.window_struct_fields(e)
            return f"(struct {win_struct}){{ &{data}, {{ {strides} }} }}"

        elif isinstance(e, LoopIR.Const):
            if isinstance(e.val, bool):
                return "true" if e.val else "false"
            elif e.type.is_indexable():
                return f"{int(e.val)}"
            elif e.type == T.f64:
                return f"{float(e.val)}"
            elif e.type == T.f32:
                return f"{float(e.val)}f"
            else:
                return f"(({e.type.ctype()}) {str(e.val)})"

        elif isinstance(e, LoopIR.BinOp):
            local_prec = op_prec[e.op]
            int_div = e.op == "/" and not e.type.is_numeric()
            if int_div:
                local_prec = 0
            op = e.op
            if op == "and":
                op = "&&"
            elif op == "or":
                op = "||"

            lhs = self.comp_e(e.lhs, local_prec)
            rhs = self.comp_e(e.rhs, local_prec + 1)

            if int_div:
                if self.range_env.check_expr_bound(0, IndexRangeEnvironment.leq, e):
                    # TODO: too many parens?
                    return f"(({lhs}) / ({rhs}))"
                return self._call_static_helper("exo_floor_div", lhs, rhs)

            s = f"{lhs} {op} {rhs}"
            if local_prec < prec:
                s = f"({s})"

            return s
        elif isinstance(e, LoopIR.USub):
            return f'-{self.comp_e(e.arg, op_prec["~"])}'

        elif isinstance(e, LoopIR.BuiltIn):
            args = [self.comp_fnarg(a, e, i) for i, a in enumerate(e.args)]
            return e.f.compile(args)

        elif isinstance(e, LoopIR.StrideExpr):
            basetyp = self.envtyp[e.name]
            stride = self.get_strides(e.name, basetyp)[e.dim]
            return self.comp_cir(simplify_cir(stride), self.env, prec=0)

        elif isinstance(e, LoopIR.ReadConfig):
            if not e.config.is_allow_rw():
                raise ConfigError(
                    f"{e.srcinfo}: cannot read from config '{e.config.name()}'"
                )
            return f"ctxt->{e.config.name()}.{e.field}"

        else:
            assert False, "bad case"

    def _call_static_helper(self, helper, *args):
        self._needed_helpers.add(helper)
        return f'{helper}({", ".join(map(str, args))})'

    def window_struct_fields(self, e):
        base = self.env[e.name]
        basetyp = self.envtyp[e.name]
        mem: Memory = self.mems[e.name]

        # compute offset to new data pointer
        def w_lo(w):
            return w.lo if isinstance(w, LoopIR.Interval) else w.pt

        cirs = [lift_to_cir(w_lo(w), self.range_env) for w in e.idx]
        idxs = [self.comp_cir(simplify_cir(i), self.env, prec=0) for i in cirs]

        # compute new window strides
        all_strides = self.get_strides(e.name, basetyp)
        all_strides_s = [
            self.comp_cir(simplify_cir(i), self.env, prec=0) for i in all_strides
        ]
        assert 0 < len(all_strides_s) == len(e.idx)
        dataptr = mem.window(basetyp, base, idxs, all_strides_s, e.srcinfo)
        strides = ", ".join(
            s for s, w in zip(all_strides_s, e.idx) if isinstance(w, LoopIR.Interval)
        )
        return dataptr, strides
