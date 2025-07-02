import functools
import re
import textwrap
import warnings
from collections import ChainMap
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict

from ..core.LoopIR import LoopIR, LoopIR_Do, get_writes_of_stmts, T, CIR
from ..core.configs import ConfigError
from .mem_analysis import MemoryAnalysis
from ..core.memory import (
    MemGenError,
    MemWin,
    AllocableMemWin,
    Memory,
    SpecialWindow,
    BarrierType,
    DRAM,
    StaticMemory,
    WindowStructCtx,
    SpecialWindowFromMemoryCtx,
)
from .parallel_analysis import ParallelAnalysis
from .prec_analysis import PrecisionAnalysis
from ..core.prelude import *
from .win_analysis import WindowAnalysis
from ..rewrite.range_analysis import IndexRangeEnvironment

from ..spork.async_config import (
    BaseAsyncConfig,
    CudaDeviceFunction,
    InstrTimelineAnalysis,
)
from ..spork.base_with_context import (
    BaseWithContext,
    is_if_holding_with,
    ExtWithContext,
)
from ..spork.loop_modes import LoopMode, Seq, Par, _CodegenPar
from ..spork.barrier_usage import BarrierUsage, BarrierUsageAnalysis, SyncInfo
from ..spork import timelines
from ..spork.cuda_backend import loopir_lower_cuda, h_snippet_for_cuda
from ..spork import excut


def sanitize_str(s):
    return re.sub(r"\W", "_", s)


T_shorthand = {
    T.f16: "f16",
    T.f32: "f32",
    T.f64: "f64",
    T.i8: "i8",
    T.ui8: "ui8",
    T.ui16: "ui16",
    T.i32: "i32",
}

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


class LoopIR_FindMemWins(LoopIR_Do):
    def __init__(self, proc):
        self._memwins = set()
        for a in proc.args:
            if a.mem:
                self._memwins.add(a.mem)
        super().__init__(proc)

    def result(self):
        return self._memwins

    # to improve efficiency
    def do_e(self, e):
        pass

    def do_s(self, s):
        if isinstance(s, LoopIR.Alloc):
            if s.mem:
                self._memwins.add(s.mem)
        elif isinstance(s, LoopIR.WindowStmt):
            if s.special_window:
                self._memwins.add(s.special_window)
        else:
            super().do_s(s)

    def do_t(self, t):
        pass


class LoopIR_FindExterns(LoopIR_Do):
    def __init__(self, proc):
        self._externs = set()
        super().__init__(proc)

    def result(self):
        return self._externs

    # to improve efficiency
    def do_e(self, e):
        if isinstance(e, LoopIR.Extern):
            self._externs.add((e.f, e.type.basetype().ctype()))
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


def find_all_memwins(proc_list):
    memwins = set()
    for p in proc_list:
        memwins.update(LoopIR_FindMemWins(p).result())
    return memwins


def find_all_externs(proc_list):
    externs = set()
    for p in proc_list:
        externs.update(LoopIR_FindExterns(p).result())

    return externs


def find_all_configs(proc_list):
    configs = set()
    for p in proc_list:
        configs.update(LoopIR_FindConfigs(p).result())

    return list(configs)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


@dataclass
class WindowStruct:
    name: str
    definition: str
    dataptr: str
    separate_dataptr: bool
    is_const: bool
    emit_definition: bool


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR Compiler Entry-points

# top level compiler function called by tests!


def run_compile(proc_list, file_stem: str):
    lib_name = sanitize_str(file_stem)
    fwd_decls, body, ext_lines = ext_compile_to_strings(lib_name, proc_list)
    used_cuda = "cu" in ext_lines

    source = f'#include "{file_stem}.h"\n\n{body}'

    header_guard = f"{lib_name}_H".upper()
    header = f"""
#pragma once
#ifndef {header_guard}
#define {header_guard}
{h_snippet_for_cuda if used_cuda else ""}\

#ifdef __cplusplus
extern "C" {{
#endif

{fwd_decls}

#ifdef __cplusplus
}}
#endif
#endif  // {header_guard}
"""

    ext_snippets = {"c": source, "h": header}

    # Gather any non .c, .h files
    for ext, lines in ext_lines.items():
        if ext == "c" or ext == "h":
            continue
        elif ext == "cuh":
            lines2 = ["#pragma once", f'#include "{file_stem}.h"']
            lines2.extend(lines)  # Most of the code
            text = "\n".join(lines2)
        elif ext == "cu":
            text = "\n".join([f'#include "{file_stem}.cuh"'] + lines)
        else:
            # A bit crappy we have per-file-extension logic here.
            assert "Add case for file extension"
        ext_snippets[ext] = text

    return ext_snippets


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


def join_ext_lines(lines):
    if lines:
        return "\n".join(["\n"] + lines + ["\n"])
    else:
        return ""


def compile_to_strings(lib_name, proc_list):
    """Legacy wrapper, for procs that don't generate extension files"""
    header, body, ext = ext_compile_to_strings(lib_name, proc_list)
    assert not ext
    return header, body


def ext_compile_to_strings(lib_name, proc_list):
    # Get transitive closure of call-graph
    orig_procs = [id(p) for p in proc_list]

    def from_lines(x):
        return "\n".join(x)

    proc_list = list(sorted(find_all_subprocs(proc_list), key=lambda x: x.name))

    # Header contents
    ctxt_name, ctxt_def = _compile_context_struct(find_all_configs(proc_list), lib_name)
    window_struct_cache = WindowStructCache()
    public_fwd_decls = []
    used_cuda = False

    # Body contents
    private_fwd_decls = []
    proc_bodies = []
    instrs_c_global = []
    tagged_cu_utils: List[Tuple[str, str]] = []  # (name, cu_util)
    tagged_cu_includes: List[Tuple[str, str]] = []  # (name, header name)
    analyzed_public_procs = []
    analyzed_private_procs = []
    ext_lines = {}

    needed_helpers = set()

    # Compile proc bodies
    seen_procs = set()
    for p in proc_list:
        # don't compile instruction procedures, but add a comment.
        if instr := p.instr:
            arg_list = [str(a.name) for a in p.args]
            if kwargs := instr._formatted_tparam_kwargs:
                arg_list.append(kwargs)
            argstr = ",".join(arg_list)
            instr_name = f"{p.name}({argstr})"
            proc_bodies.extend(
                [
                    "",
                    '/* relying on the following instruction..."',
                    instr_name,
                    "\n".join(p.instr.instr_format),
                    "*/",
                ]
            )
            if instr.c_global:
                instrs_c_global.append(instr.c_global)
            if instr.cu_utils:
                for util in instr.cu_utils:
                    tagged_cu_utils.append((instr_name, util))
            if instr.cu_includes:
                for header_name in instr.cu_includes:
                    tagged_cu_includes.append((instr_name, header_name))

        else:
            if p.name in seen_procs:
                raise TypeError(f"multiple non-instr procs named {p.name}")
            seen_procs.add(p.name)

            is_public_decl = id(p) in orig_procs

            p = ParallelAnalysis().run(p)
            p = PrecisionAnalysis().run(p)
            p = WindowAnalysis().apply_proc(p)
            p = MemoryAnalysis().run(p)
            instr_tl_analysis = InstrTimelineAnalysis()
            p = instr_tl_analysis.run(p)
            barrier_uses: Optional[Dict[Sym, BarrierUsage]]
            barrier_uses = None
            proc_uses_cuda = (
                timelines.cuda_in_order_instr in instr_tl_analysis.instr_tl_seen
            )
            if instr_tl_analysis.contains_sync:
                # Don't force non-CUDA Exo users to waste time here
                barrier_usage_analysis = BarrierUsageAnalysis(p)
                barrier_uses = barrier_usage_analysis.uses

            comp = Compiler(
                p,
                lib_name,
                ctxt_name,
                window_struct_cache,
                barrier_uses,
                proc_uses_cuda,
                is_public_decl=is_public_decl,
            )
            d, b = comp.comp_top()
            needed_helpers |= comp.needed_helpers()
            used_cuda |= proc_uses_cuda

            if is_public_decl:
                public_fwd_decls.append(d)
            else:
                private_fwd_decls.append(d)

            proc_bodies.append(b)

            if is_public_decl:
                analyzed_public_procs.append(p)
            else:
                analyzed_private_procs.append(p)
            for ext, snippets in comp.ext_lines().items():
                ext_lines.setdefault(ext, []).extend(snippets)

    # Memories and structs are just blobs of code...
    # still sort them for output stability
    header_memwins, header_memwin_code, body_memwin_code = _compile_memwins(
        analyzed_public_procs, analyzed_private_procs
    )

    (
        header_struct_defns,
        body_struct_defns,
    ) = window_struct_cache.sorted_header_body_definitions(header_memwins)

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
{from_lines(header_memwin_code)}
{from_lines(header_struct_defns)}
{from_lines(public_fwd_decls)}
{join_ext_lines(ext_lines.get("h"))}"""

    extern_code = _compile_externs(
        find_all_externs(analyzed_public_procs + analyzed_private_procs)
    )

    helper_code = [_static_helpers[v] for v in needed_helpers]
    body_contents = [helper_code, instrs_c_global, body_memwin_code, body_struct_defns]
    if lines := ext_lines.get("c"):
        body_contents.append(lines)
    body_contents.extend(
        [
            extern_code,
            private_fwd_decls,
            proc_bodies,
        ]
    )
    body_contents = list(filter(lambda x: x, body_contents))  # filter empty lines
    body_contents = map(from_lines, body_contents)
    body_contents = from_lines(body_contents)
    body_contents += "\n"  # New line at end of file

    # Add cu_includes, cu_util, window definitions to .cuh file, if it exists.
    if (cuh_lines := ext_lines.get("cuh")) is not None:
        prepend_tagged_cuh_ext_lines(
            False,
            lib_name,
            tagged_cu_utils,
            ext_lines,
        )
        ext_lines["cuh"] = body_memwin_code + body_struct_defns + ext_lines["cuh"]
        # Moved CUDA includes to the top.
        # clangd seems to get really confused if includes are in the wrong place.
        prepend_tagged_cuh_ext_lines(
            True,
            lib_name,
            tagged_cu_includes,
            ext_lines,
        )

    return header_contents, body_contents, ext_lines


def _compile_externs(externs):
    extern_code = []
    for f, t in sorted(externs, key=lambda x: x[0].name() + x[1]):
        if glb := f.globl(t):
            extern_code.append(glb)
    return extern_code


def _compile_memwins(public_procs, private_procs):
    assert public_procs or not private_procs, "Only have private procs?"

    """Return (header memwin set, header memwin code, C body memwin code)"""
    all_memwins = find_all_memwins(public_procs + private_procs)

    # Memories used as part of public proc args must be defined in public header
    header_memwins = set()
    for p in public_procs:
        if p.instr is None:
            for arg in p.args:
                memwin = arg.mem or DRAM
                assert memwin in all_memwins
                header_memwins.add(arg.mem or DRAM)

    header_memwin_code = []
    body_memwin_code = []
    for m in sorted(all_memwins, key=lambda x: x.name()):
        code_list = header_memwin_code if m in header_memwins else body_memwin_code
        code_list.append(m.global_())
    return header_memwins, header_memwin_code, body_memwin_code


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
    def __init__(
        self,
        proc,
        lib_name,
        ctxt_name,
        window_struct_cache,
        barrier_uses,
        used_cuda,
        *,
        is_public_decl,
    ):
        assert isinstance(proc, LoopIR.proc)
        assert isinstance(window_struct_cache, WindowStructCache)

        self.lib_name = lib_name
        self.proc = proc
        self.ctxt_name = ctxt_name
        self.env = ChainMap()
        self.force_names = dict()  # For ExtWithContext.force_names
        self.range_env = IndexRangeEnvironment(proc, fast=False)
        self.names = ChainMap()
        self.envtyp = dict()
        self.mems = dict()
        self._tab = ""
        self._lines = []
        self._scalar_refs = set()
        self._needed_helpers = set()
        self.window_struct_cache = window_struct_cache
        self.barrier_uses = barrier_uses
        self._used_cuda = used_cuda
        self._known_strides = {}
        self._in_cuda_function = False
        self._cuda_kernel_count = 0

        # Additional lines for each file extension
        # Since Exo was originally written for only .c and .h files,
        # we have a lot of special treatment for these files,
        # handled separately from this (see comp_top).
        self._ext_lines = {}

        assert self.proc.name is not None, "expected names for compilation"
        name = self.proc.name
        arg_strs = []
        typ_comments = []

        # reserve the first "ctxt" argument
        self.new_varname(Sym("ctxt"), None)
        arg_strs.append(f"{ctxt_name} *ctxt")

        # See self.is_const
        self.global_non_const = set(e for e, _ in get_writes_of_stmts(self.proc.body))
        self.force_const = set()  # For ExtWithContext.force_const

        for a in proc.args:
            mem = a.mem if a.type.is_numeric() else None
            name_arg = self.new_varname(a.name, typ=a.type, mem=mem)
            if a.type.is_real_scalar():
                self._scalar_refs.add(a.name)
            self.append_fnarg_decl(a, name_arg, arg_strs, typ_comments)

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

    def is_const(self, sym: Sym):
        assert isinstance(sym, Sym)
        return sym not in self.global_non_const or sym in self.force_const

    def append_fnarg_decl(
        self,
        a: LoopIR.fnarg,
        name_arg: str,
        arg_strs: List[str],
        typ_comments: List[str],
        *,
        force_pass_by_value=False,
    ):
        """Compile a LoopIR.fnarg to C function argument declaration(s).

        Appends function arguments (e.g. `int* foo`) and type comments
        to the given lists, respectively.
        Side effect: triggers compilation of memory definitions
        and window struct declarations as needed.
        """
        assert isinstance(a, LoopIR.fnarg)
        mem = a.mem if a.type.is_numeric() else None
        if a.type in (T.size, T.index, T.bool, T.stride):
            arg_strs.append(f"{a.type.ctype()} {name_arg}")
            typ_comments.append(f"{name_arg} : {a.type}")
        # setup, arguments
        else:
            assert a.type.is_numeric()
            assert a.type.basetype() != T.R
            is_const = self.is_const(a.name)
            if a.type.is_real_scalar():
                if force_pass_by_value:
                    arg_strs.append(f"{a.type.ctype()} {name_arg}")
                else:
                    arg_strs.append(
                        f"{'const ' if is_const else ''}{a.type.ctype()}* {name_arg}"
                    )
            else:
                assert a.type.is_tensor_or_window()
                window_struct = self.get_window_struct(
                    a, mem or DRAM, is_const, a.type.is_win()
                )
                if a.type.is_win():
                    if window_struct.separate_dataptr:
                        arg_strs.append(
                            f"{window_struct.dataptr} {dataptr_name(name_arg)}"
                        )
                        typ_comments.append("    (Separate window data pointer)")
                    arg_strs.append(f"struct {window_struct.name} {name_arg}")
                else:
                    arg_strs.append(f"{window_struct.dataptr} {name_arg}")
            memstr = f" @{a.mem.name()}" if a.mem else ""
            comment_str = f"{name_arg} : {a.type}{memstr}"
            typ_comments.append(comment_str)

    def static_memory_check(self, proc):
        def allocates_static_memory(stmts):
            check = False
            for s in stmts:
                if isinstance(s, LoopIR.Alloc):
                    mem = s.mem
                    assert issubclass(mem, AllocableMemWin)
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

    def ext_lines(self):
        return self._ext_lines

    def needed_helpers(self):
        return self._needed_helpers

    def used_cuda(self):
        return self._used_cuda

    def new_varname(self, symbol, typ, mem=None):
        strnm = str(symbol)

        # Reserve "exo_" prefix for internal use.
        if strnm.lower().startswith("exo_"):
            strnm = "exo_user_" + strnm

        if forced_name := self.force_names.get(symbol):
            strnm = forced_name
        elif strnm not in self.names:
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
            assert issubclass(mem, MemWin)
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
        buf = self.env[nm]
        if nm in self._scalar_refs:
            return f"*{buf}"
        elif not idx_list:
            return buf
        type = self.envtyp[nm]
        cirs = [lift_to_cir(i, self.range_env) for i in idx_list]
        idx_expr = self.get_idx_offset(nm, type, cirs)
        idx_expr_s = self.comp_cir(simplify_cir(idx_expr), self.env, prec=0)
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

    def get_strides_s(self, name: Sym, typ) -> List[str]:
        all_strides = self.get_strides(name, typ)
        return [self.comp_cir(simplify_cir(i), self.env, prec=0) for i in all_strides]

    def get_idx_offset(self, name: Sym, typ, idx) -> CIR:
        strides = self.get_strides(name, typ)
        assert len(strides) == len(idx)
        acc = CIR.BinOp("*", idx[0], strides[0], True)
        for i, s in zip(idx[1:], strides[1:]):
            new = CIR.BinOp("*", i, s, True)
            acc = CIR.BinOp("+", acc, new, True)

        return acc

    def get_window_struct(self, node, mem, is_const=None, emit_definition=True):
        typ = node.type
        assert isinstance(typ, T.Window) or (
            isinstance(node, LoopIR.fnarg) and typ.is_tensor_or_window()
        )

        if isinstance(typ, T.Window):
            base = typ.as_tensor.basetype()
            n_dims = len(typ.as_tensor.shape())
            if is_const is None:
                is_const = self.is_const(typ.src_buf)
        else:
            base = typ.type.basetype()
            n_dims = len(typ.shape())
            if is_const is None:
                is_const = self.is_const(node.name)

        return self.window_struct_cache.get(
            mem, base, n_dims, is_const, node.srcinfo, emit_definition
        )

    def comp_s(self, s):
        if isinstance(s, LoopIR.Pass):
            self.add_line("; // NO-OP")
        elif isinstance(s, LoopIR.SyncStmt):
            if s.lowered is None:
                raise TypeError(
                    f"{s.srcinfo}: SyncStmt not allowed here "
                    "(or internal compiler error -- missing lowered barrier)"
                )
            sync_type = s.sync_type
            barrier_lines = s.lowered
            self.add_line(f"// {s.sync_type.format_stmt(s.barriers)}")
            assert not isinstance(barrier_lines, str), "expect List[str]"
            for line in barrier_lines:
                self.add_line(line)
        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            typ = self.envtyp[s.name]
            idx = []
            if not typ.is_real_scalar():
                idx = s.idx
            lhs = self.access_str(s.name, idx)
            rhs = self.comp_e(s.rhs)

            # possibly cast!
            lbtyp = s.type.basetype()
            rbtyp = s.rhs.type.basetype()
            if lbtyp != rbtyp:
                assert s.type.is_real_scalar()
                assert s.rhs.type.is_real_scalar()

                rhs = f"({lbtyp.ctype()})({rhs})"

            mem: MemWin = self.mems[s.name]
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
            rhs = s.rhs
            assert isinstance(rhs, LoopIR.WindowExpr)
            input_winmem = self.mems[rhs.name]
            input_win_struct = self.get_window_struct(rhs, input_winmem)
            (
                w_type,
                w_def,
                d_type,
                d_def,
                layout,
                separate_dataptr,
            ) = self.unpack_window_expr(rhs, input_winmem, input_win_struct.is_const)

            output_winmem = s.special_window or input_winmem
            name = self.new_varname(s.name, typ=rhs.type, mem=output_winmem)

            if not s.special_window:
                output_win_struct = input_win_struct
            else:
                # Special case, creating a special window
                # We pass the temporary expressions from unpack_window_expr to
                # the SpecialWindow creation callback.
                assert issubclass(output_winmem, SpecialWindow)
                assert issubclass(input_winmem, output_winmem.source_memory_type())
                tensor_type = rhs.type.as_tensor_type()
                scalar_type = tensor_type.basetype()
                output_win_struct = self.get_window_struct(rhs, output_winmem)
                ctx = SpecialWindowFromMemoryCtx(
                    d_def,
                    layout,
                    output_win_struct.dataptr,
                    output_win_struct.name,
                    tensor_type,
                    self.shape_strs(tensor_type.shape()),
                    output_win_struct.is_const,
                    scalar_type.ctype(),
                    T_shorthand[scalar_type],
                    s.srcinfo,
                )
                tmp = output_winmem.from_memory(ctx)

                # Substitute window definition for codegen, replacing temporary window.
                separate_dataptr = output_winmem.separate_dataptr()
                if separate_dataptr:
                    assert len(tmp) == 2
                    d_def, w_def = tmp
                else:
                    assert isinstance(tmp, str)
                    d_def, w_def = None, tmp

            if separate_dataptr:
                cref = ""
                if self._in_cuda_function:
                    # HACK needed for CUtensorMap; if we copy the CUtensorMap
                    # in CUDA code, then it won't be in grid constant memory,
                    # and cp.async.bulk won't work anymore.
                    cref = " const&"
                self.add_line(
                    f"{output_win_struct.dataptr}{cref} {dataptr_name(name)} = {d_def};"
                )
            self.add_line(f"struct {output_win_struct.name} {name} = {w_def};")

        elif is_if_holding_with(s, LoopIR):  # must be before .If case
            ctx = s.cond.val
            if isinstance(ctx, ExtWithContext):
                # Modify Sym state as specified by ExtWithContext.
                # Please read the comment in ExtWithContext and ensure it's
                # correct ... in particular handling nested ExtWithContexts.
                self.push(only="env")
                old_force_names = self.force_names
                old_force_const = self.force_const
                old_scalar_refs = self._scalar_refs
                for nm in ctx.reserved_names:
                    # TODO We can prevent using the reserved name inside, but
                    # we don't retroactively undo its usage outside.
                    self.names[nm] = nm
                self.force_names = dict(old_force_names)
                for sym, nm in ctx.force_names.items():
                    self.names[nm] = nm
                    self.force_names[sym] = nm
                    if sym in self.env:
                        self.env[sym] = nm
                self.force_const = old_force_const | ctx.force_const
                self._scalar_refs = ctx.scalar_refs  # ignore old scalar_refs

                # Reset indentation and redirect text lines for compiled subtree
                # to new location (per-file-extension lines dict). We defer
                # extending the list so that nested ExtWithContext works.
                old_lines = self._lines
                old_tab = self._tab
                self._lines = []
                self._tab = ""

                # Add code snippets
                for ext, snippet in ctx.ext_snippets.items():
                    self._ext_lines.setdefault(ext, []).append(snippet)

                # Compile body, with prefix and suffix.
                # Note ordering after snippets are added, as promised in ExtWithContext.
                self.add_line(ctx.body_prefix)  # Might not really be just 1 line...
                self._tab += "  "
                self.comp_stmts(s.body)
                self._tab = ""
                self.add_line(ctx.body_suffix)

                # Deferred extension of lines dict
                self._ext_lines.setdefault(ctx.body_ext, []).extend(self._lines)

                # Restore Sym state
                self._scalar_refs = old_scalar_refs
                self.force_const = old_force_const
                self.force_names = old_force_names
                self.pop()  # Rolls back reserved names

                # Restore old lines list and indentation
                self._tab = old_tab
                self._lines = old_lines

                # Add kernel launch syntax
                self.add_line(ctx.launch)

            elif isinstance(ctx, CudaDeviceFunction):
                spork_ctx = SporkLoweringCtx(
                    self.lib_name, self.proc.name, self._cuda_kernel_count, self
                )
                lowered = loopir_lower_cuda(s, spork_ctx)
                # print(lowered)
                assert self._used_cuda
                assert not self._in_cuda_function
                self._in_cuda_function = True
                self.comp_s(lowered)
                self._cuda_kernel_count += 1
                self._in_cuda_function = False

            # Must appear last (fallback case)
            elif isinstance(ctx, BaseAsyncConfig):
                self.add_line("{")
                self.push()
                self.add_line(f"// {ctx}")
                self.comp_stmts(s.body)
                self.pop()
                self.add_line("}")

            else:
                raise TypeError(f"Unknown with stmt context type {type(ctx)}")

        # If statement that is not disguising a with statement
        # (remove note when this hack is fixed)
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
            sym_range = self.range_env.add_loop_iter(
                s.iter,
                s.lo,
                s.hi,
            )

            loop_mode = s.loop_mode
            emit_loop = True

            if isinstance(s.loop_mode, Par):
                self.add_line(f"#pragma omp parallel for")
            elif isinstance(s.loop_mode, Seq):
                unroll = s.loop_mode.pragma_unroll
                if unroll is not None:
                    unroll_str = f" {unroll}" if unroll > 0 else ""
                    self.add_line(f"#pragma unroll{unroll_str}")
            elif isinstance(loop_mode, _CodegenPar):
                # This is not valid C; if we add non-cuda backends we may have
                # to add config options to _CodegenPar to tweak lowering syntax.
                conds = []
                if (bdd := loop_mode.static_bounds[0]) is not None:
                    conds.append(f"{itr} >= {bdd}")
                if (bdd := loop_mode.static_bounds[1]) is not None:
                    conds.append(f"{itr} < {bdd}")
                if conds:
                    cond = " && ".join(conds)
                    maybe_unused = ""
                else:
                    cond = "1"
                    maybe_unused = "[[maybe_unused]] "
                if comment := loop_mode.comment:
                    assert "\n" not in comment
                    self.add_line(f"// {comment}")
                self.add_line(
                    f"if ({maybe_unused}int {itr} = {loop_mode.c_index}; {cond}) {{"
                )
                emit_loop = False
            else:
                raise TypeError(
                    f"{s.srcinfo}: unexpected loop mode {loop_mode.loop_mode_name()} in {s.iter} loop"
                )

            if emit_loop:
                ctype = "int" if self._in_cuda_function else "int_fast32_t"
                self.add_line(f"for ({ctype} {itr} = {lo}; {itr} < {hi}; {itr}++) {{")

            self.push(only="tab")
            self.comp_stmts(s.body)
            self.pop()
            self.add_line("}")

        elif isinstance(s, LoopIR.Alloc):
            name = self.new_varname(s.name, typ=s.type, mem=s.mem)
            if not s.type.is_barrier():
                assert s.type.basetype().is_real_scalar()
                assert s.type.basetype() != T.R
                ctype = s.type.basetype().ctype()
                shape_strs = self.shape_strs(s.type.shape())
            else:
                assert issubclass(s.mem, BarrierType)
                ctype = None  # Use in the future if we externalize BarrierType?
                shape_strs = ()
            mem = s.mem or DRAM
            line = mem.alloc(name, ctype, shape_strs, s.srcinfo)
            self.add_line(line)
        elif isinstance(s, LoopIR.Free):
            name = self.env[s.name]
            if s.type.is_barrier():
                pass
            else:
                assert s.type.basetype().is_real_scalar()
                ctype = s.type.basetype().ctype()
                mem = s.mem or DRAM
                line = mem.free(name, ctype, self.shape_strs(s.type.shape()), s.srcinfo)
                self.add_line(line)

        elif isinstance(s, LoopIR.Call):
            assert all(
                a.type.is_win() == fna.type.is_win() for a, fna in zip(s.args, s.f.args)
            )
            arg_tups = [self.comp_fnarg(e, s.f, i) for i, e in enumerate(s.args)]
            if s.f.instr is not None:
                d = dict()
                assert len(s.f.args) == len(arg_tups)
                for i in range(len(arg_tups)):
                    arg_name = str(s.f.args[i].name)
                    c_args, instr_data, instr_layout = arg_tups[i]
                    arg_type = s.args[i].type
                    if arg_type.is_win():
                        if not isinstance(s.args[i], LoopIR.WindowExpr):
                            # comp_fnarg requires this for {arg_name}_data
                            raise TypeError(
                                f"{s.srcinfo}: Argument {arg_name} must be a "
                                f"window expression created at the call site "
                                f"of {s.f.name}"
                            )
                        # c_args = (window,) or (dataptr, layout) (depending on
                        # separate_dataptr); [-1] gets the window/layout
                        d[arg_name] = f"({c_args[-1]})"
                        d[f"{arg_name}_data"] = instr_data
                        d[f"{arg_name}_layout"] = instr_layout
                        # Special case for AMX instrs
                        d[f"{arg_name}_int"] = self.env[s.args[i].name]
                        assert instr_data
                    else:
                        assert (
                            len(c_args) == 1
                        ), "didn't expect multiple c_args for non-window"
                        arg = f"({c_args[0]})"
                        d[arg_name] = arg
                        # Exo 1 does this; unclear why for non-windows
                        d[f"{arg_name}_data"] = arg

                for line in s.f.instr.instr_format:
                    self.add_line(f"{line.format(**d)}")
            else:
                fname = s.f.name
                args = ["ctxt"]
                for tups in arg_tups:
                    c_args = tups[0]
                    args.extend(c_args)
                self.add_line(f"{fname}({','.join(args)});")
        else:
            assert False, "bad case"

    def comp_fnarg(self, e, fn, i, *, prec=0, force_pass_by_value=False):
        """Returns (c_args : tuple,
                    instr_data : Optional[str],
                    instr_layout : Optional[str])

        c_args is a tuple (length 1 or 2) of formatted arguments.
        Length 2 only occurs for separate_dataptr windows: (dataptr, layout).

        instr_data is for formatting instr_format windows; passed as {arg_name}_data.
        This is needed both for compatibility with Exo 1 and for allowing
        access to the dataptr when separate_dataptr is True.

        instr_layout is similar, passed as {arg_name}_layout.
        This is an untyped initializer for the window layout (e.g. strides).

        We assume that all scalar values are passed by pointer,
        unless force_pass_by_value is True.
        """
        assert isinstance(fn, LoopIR.proc)
        mem = fn.args[i].mem
        is_const = None
        if isinstance(e, LoopIR.WindowExpr):
            callee_buf = fn.args[i].name
            is_const = callee_buf not in set(x for x, _ in get_writes_of_stmts(fn.body))
        return self.comp_fnarg_impl(e, mem, is_const, prec, force_pass_by_value)

    def comp_fnarg_impl(self, e, mem, is_const, prec, force_pass_by_value):
        if isinstance(e, LoopIR.Read):
            assert not e.idx
            rtyp = self.envtyp[e.name]
            if rtyp.is_indexable():
                return (self.env[e.name],), None, None
            elif rtyp is T.bool:
                return (self.env[e.name],), None, None
            elif rtyp is T.stride:
                return (self.env[e.name],), None, None
            elif e.name in self._scalar_refs:
                star = "*" if force_pass_by_value else ""
                return (f"{star}{self.env[e.name]}",), None, None
            elif rtyp.is_tensor_or_window():
                c_window = self.env[e.name]
                if mem and mem.separate_dataptr():
                    # This data path is exercised for calling normal
                    # functions, but the omitted instr_data is only
                    # used for instr, which can't use this code path.
                    c_data = dataptr_name(c_window)
                    return (c_data, c_window), None, None
                else:
                    return (c_window,), None, None
            else:
                assert rtyp.is_real_scalar()
                amp = "" if force_pass_by_value else "&"
                return (f"{amp}{self.env[e.name]}",), None, None
        elif isinstance(e, LoopIR.WindowExpr):
            assert is_const is not None
            _, w_def, _, d_def, layout, separate_dataptr = self.unpack_window_expr(
                e, mem, is_const
            )
            if separate_dataptr:
                return (d_def, w_def), d_def, layout
            else:
                return (w_def,), d_def, layout
        else:
            return (self.comp_e(e, prec),), None, None

    def comp_e(self, e, prec=0):
        if isinstance(e, LoopIR.Read):
            rtyp = self.envtyp[e.name]
            if rtyp.is_indexable() or rtyp is T.bool or rtyp == T.stride:
                return self.env[e.name]

            mem: MemWin = self.mems[e.name]

            if not mem.can_read():
                raise MemGenError(
                    f"{e.srcinfo}: cannot read from buffer "
                    f"'{e.name}' in memory '{mem.name()}'"
                )

            return self.access_str(e.name, e.idx)

        elif isinstance(e, LoopIR.WindowExpr):
            # WindowExpr needs to be handled differently depending on usage
            #   * WindowStmt
            #   * Passing to function
            #   * Passing to instr
            # see unpack_window_expr and get strings from there
            assert 0, "Unexpected standalone WindowExpr"

        elif isinstance(e, LoopIR.Const):
            if isinstance(e.val, bool):
                return "true" if e.val else "false"
            elif e.type.is_indexable():
                return f"{int(e.val)}"
            elif e.type == T.f64:
                return f"{float(e.val)}"
            elif e.type == T.f32:
                return f"{float(e.val)}f"
            elif e.type == T.with_context:
                assert False, "should be handled when compiling LoopIR.If"
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

        elif isinstance(e, LoopIR.Extern):
            args = [self.comp_e(a) for a in e.args]
            return e.f.compile(args, e.type.basetype().ctype())

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

    def unpack_window_expr(self, e: LoopIR.WindowExpr, src_memwin: type, is_const=None):
        """(w_type, w_def, d_type, d_def, layout, separate_dataptr)

        w_type, w_def: C typename and initialization for window struct

        d_type: C typename for data pointer

        d_def: "data" passed through from src_memwin.window(...)

        layout: untyped C braced initializer for layout portion of window

        separate_dataptr: If True, the window is defined with a
          separate data pointer {d_type} {name} = {d_def}
        """
        win_struct = self.get_window_struct(e, src_memwin, is_const)
        w_type = win_struct.name
        d_type = win_struct.dataptr
        separate_dataptr = win_struct.separate_dataptr

        base = self.env[e.name]
        basetyp = self.envtyp[e.name].as_tensor_type()

        # compute offset to new data pointer
        def w_lo(w):
            return w.lo if isinstance(w, LoopIR.Interval) else w.pt

        cirs = [lift_to_cir(w_lo(w), self.range_env) for w in e.idx]
        idxs = [self.comp_cir(simplify_cir(i), self.env, prec=0) for i in cirs]

        # compute new window strides
        all_strides_s = self.get_strides_s(e.name, basetyp)
        assert 0 < len(all_strides_s) == len(e.idx)
        if separate_dataptr and basetyp.is_win():
            window_in_expr = dataptr_name(base), base
        else:
            window_in_expr = base
        callback_result = src_memwin.window(
            basetyp, window_in_expr, idxs, all_strides_s, e.srcinfo
        )
        if isinstance(callback_result, str):
            # Base case, no custom layout
            assert (
                not separate_dataptr
            ), "MemWin must define custom layout for separate_dataptr"
            strides = ", ".join(
                s
                for s, w in zip(all_strides_s, e.idx)
                if isinstance(w, LoopIR.Interval)
            )
            layout = f"{{ {strides} }}"
            d_def = callback_result
            w_def = f"(struct {w_type}){{ &{d_def}, {layout} }}"
        else:
            # Custom layout case
            assert len(callback_result) == 2
            d_def, layout = callback_result
            if separate_dataptr:
                w_def = f"(struct {w_type}) {layout}"
            else:
                w_def = f"(struct {w_type}){{ {d_def}, {layout} }}"  # not &data
            # This could be an optional MemWin.window_remove_dims(...) callback
            if any(isinstance(w, LoopIR.Point) for w in e.idx):
                raise MemGenError(
                    f"{e.srcinfo}: {src_memwin.name()} window from {e.name} doesn't support removing dimensions (single Point coordinate in window indices)"
                )

        return w_type, w_def, d_type, d_def, layout, separate_dataptr

    def _call_static_helper(self, helper, *args):
        self._needed_helpers.add(helper)
        return f'{helper}({", ".join(map(str, args))})'


def dataptr_name(wname):
    """C variable name used to store the separate dataptr of a window

    We prepend the (reserved) exo_data_ prefix, but this is
    complicated by the fact that sometimes C variables are stored in
    structs (e.g. exo_deviceArgs) and we need to avoid modifying the
    struct name.

    """
    fragments = wname.split(".")
    fragments[-1] = "exo_data_" + fragments[-1]
    return ".".join(fragments)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Assemble includes and exo_CudaUtil namespace in .cuh file
# from list of pairs of (required_by: str, content: str)
# where content is a header name or a cu_util blob.
# We remove exact duplicate strings.
# This is also where we add the excut string table, needed for test tracing
def prepend_tagged_cuh_ext_lines(
    is_includes,
    lib_name,
    tagged_content,
    ext_lines,
):
    combined: [List[str], str] = []  # ([required_by], content)
    index_dict = {}

    for tag, content in tagged_content:
        idx = index_dict.get(content)
        if idx is None:
            index_dict[content] = len(combined)
            combined.append(([tag], content))
        else:
            combined[idx][0].append(tag)

    lines = []
    if is_includes:
        # Alphabetize include files
        # Note however we do NOT sort util source code, since later utils
        # may require earlier ones to compile correctly!
        combined.sort(key=lambda tup: tup[1])
    else:
        # Begin namespace
        lines.append("")
        lines.append(f"namespace exo_CudaUtil_{lib_name} {{")
        # Paste in weird excut code
        lines.extend(excut.generate_c_str_table_lines())

    for tags, content in combined:
        for tag in tags:
            lines.append(f"/* Required by {tag} */")
        if is_includes:
            lines.append(f"#include <{content}>")
        else:
            for line in content.split("\n"):
                if not line or line.isspace():
                    lines.append("")
                else:
                    lines.append(line)

    if not is_includes:
        lines.append("}  // end namespace")

    ext_lines["cuh"] = lines + ext_lines.setdefault("cuh", [])


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cached collection of window struct definitions


class WindowStructCache(object):
    __slots__ = ["_key_to_name", "_name_to_struct"]

    def __init__(self):
        self._key_to_name = {}
        self._name_to_struct = {}

    def _add_to_cache(self, key_tuple, srcinfo) -> WindowStruct:
        memwin, base_type, n_dims, is_const = key_tuple
        type_shorthand = T_shorthand[base_type]
        separate_dataptr = memwin.separate_dataptr()

        ctx = WindowStructCtx(
            base_type.ctype(),
            type_shorthand,
            n_dims,
            is_const,
            separate_dataptr,
            srcinfo,
        )
        c_dataptr, c_window = memwin.window_definition(ctx)

        assert isinstance(c_dataptr, str)
        assert isinstance(c_window, str)
        assert isinstance(separate_dataptr, bool)

        assert ctx._struct_name is not None, "MemWin didn't name the struct"
        sname = ctx._struct_name

        self._key_to_name[key_tuple] = sname

        sdef = f"""#ifndef {ctx._guard_macro}
#define {ctx._guard_macro}
{c_window}
#endif"""

        v = self._name_to_struct.get(sname)

        if v is None:
            v = WindowStruct(
                ctx._struct_name,
                sdef,
                c_dataptr,
                separate_dataptr,
                is_const,
                False,  # emit_definition flag; modified outside this function
            )
            self._name_to_struct[sname] = v
        elif v.definition != sdef:
            # Since windows are keyed based on MemWin type, and derived MemWin
            # types inherit an identical window struct if not overriden,
            # it's valid to have a struct name collision here.
            # But we validate that the collision is due to a duplicate
            # identical struct, and not a true name incompatibility.
            for key_tuple2, sname2 in self._key_to_name.values():
                if sname2 == sname:
                    memwin2, base_type2, n_dims2, is_const2 = key_tuple2
                    type_shorthand2 = T_shorthand[base_type2]
                    raise ValueError(
                        f"""Window name collision for {sname}:
{memwin.name()}, {type_shorthand}, n_dims={n_dims}, is_const={is_const};
{memwin2.name()}, {type_shorthand2}, n_dims={n_dims2}, is_const={is_const2}"""
                    )

        return v

    def get(
        self, memwin, base_type, n_dims, is_const, srcinfo, emit_definition
    ) -> WindowStruct:
        key_tuple = (memwin, base_type, n_dims, is_const)
        sname = self._key_to_name.get(key_tuple)
        if sname is None:
            v = self._add_to_cache(key_tuple, srcinfo)
        else:
            v = self._name_to_struct[sname]
        v.emit_definition |= emit_definition
        return v

    def sorted_header_body_definitions(self, header_memwins):
        header_snames = set()
        for key_tuple, sname in self._key_to_name.items():
            memwin, _, _, _ = key_tuple
            if memwin in header_memwins:
                header_snames.add(sname)

        sorted_pairs = sorted(self._name_to_struct.items())
        h_definitions = []
        c_definitions = []
        for sname, struct in sorted_pairs:
            if struct.emit_definition:
                lst = h_definitions if sname in header_snames else c_definitions
                lst.append(struct.definition)
        return h_definitions, c_definitions


class SporkLoweringCtx(object):
    """Communication object between main LoopIR compiler and Spork backend.

    The task of the spork backend is to transform a subtree of LoopIR
    to a new subtree of LoopIR that the main compiler is able to
    understand.  Usually, the backend will return a tree rooted with
    an ExtWithContext to redirect the generated subtree C-like code to
    separate files for accelerator code (e.g. .cuh or .cu code for
    cuda).

    """

    __slots__ = [
        "_lib_name",
        "_proc_name",
        "_kernel_index",
        "_compiler",
    ]

    _lib_name: str
    _proc_name: str
    _kernel_index: int
    _compiler: Compiler

    def __init__(self, lib_name, proc_name, kernel_index, compiler):
        self._lib_name = lib_name
        self._proc_name = proc_name
        self._kernel_index = kernel_index
        self._compiler = compiler

    def lib_name(self):
        return self._lib_name

    def proc_name(self):
        return self._proc_name

    def kernel_index(self):
        return self._kernel_index

    def sym_c_name(self, sym: Sym):
        assert isinstance(sym, Sym)
        return self._compiler.env[sym]

    def sym_type(self, sym: Sym, overrides: Dict[Sym, LoopIR.type]):
        assert isinstance(sym, Sym)
        return overrides.get(sym) or self._compiler.envtyp[sym]

    def sym_mem(self, sym: Sym):
        assert isinstance(sym, Sym)
        return self._compiler.mems[sym]

    def sym_is_scalar_ref(self, sym: Sym):
        assert isinstance(sym, Sym)
        return sym in self._compiler._scalar_refs

    def is_const(self, sym: Sym):
        assert isinstance(sym, Sym)
        return self._compiler.is_const(sym)

    def append_fnarg_decl(
        self,
        a: LoopIR.fnarg,
        name_arg: str,
        arg_strs: List[str],
        typ_comments: List[str],
        *,
        force_pass_by_value=False,
    ):
        return self._compiler.append_fnarg_decl(
            a, name_arg, arg_strs, typ_comments, force_pass_by_value=force_pass_by_value
        )

    def fnarg_values(self, e, is_const, force_pass_by_value):
        mem = self._compiler.mems[e.name]
        return self._compiler.comp_fnarg_impl(e, mem, is_const, 0, force_pass_by_value)[
            0
        ]

    def get_barrier_usage(self, name: Sym) -> BarrierUsage:
        return self._compiler.barrier_uses[name]
