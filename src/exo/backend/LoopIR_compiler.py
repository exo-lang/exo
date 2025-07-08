import functools
import re
import textwrap
import warnings
from collections import ChainMap
from collections import defaultdict
from dataclasses import dataclass, field
from math import prod
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Set, Type

from ..core.cir import CIR, CIR_Wrapper, simplify_cir
from ..core.c_window import WindowFeatures
from ..core.instr_class import InstrWindowArg, InstrNonWindowArg, InstrArgs
from ..core.LoopIR import LoopIR, LoopIR_Do, get_writes_of_stmts, T
from ..core.configs import ConfigError
from .mem_analysis import MemoryAnalysis
from ..core.c_window import WindowIndexer, WindowEncoder
from ..core.memory import (
    MemIncludeC,
    MemGlobalC,
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
from ..core.c_window import (
    UtilInjector,
    WindowFeatures,
    WindowEncoderArgs,
    WindowIndexerArgs,
    WindowIndexerResult,
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
    # getattr
    ".": 80,
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


@dataclass(slots=True, frozen=True)
class UtilInjectorImpl(UtilInjector):
    tag: str
    tagged_c_utils: List[Tuple[str, str]]  # (name, c_util)
    tagged_c_includes: List[Tuple[str, str]]  # (name, header name)
    tagged_cu_utils: List[Tuple[str, str]]  # (name, cu_util)
    tagged_cu_includes: List[Tuple[str, str]]  # (name, header name)

    def with_tag(self, new_tag):
        return UtilInjectorImpl(
            new_tag,
            self.tagged_c_utils,
            self.tagged_c_includes,
            self.tagged_cu_utils,
            self.tagged_cu_includes,
        )

    def add_c_util(self, code):
        """Add snippet of C code at global scope to appear before your code"""
        self.tagged_c_utils.append(self.tag, code)

    def add_c_include(self, header_name):
        """Add header file to generated C code"""
        self.tagged_c_includes.append(self.tag, header_name)

    def add_cu_util(self, code):
        """Add CUDA utility to appear before your code"""
        self.tagged_cu_utils.append(self.tag, code)

    def add_cu_include(self, header_name):
        """Add header file to generated CUDA code"""
        self.tagged_cu_includes.append(self.tag, header_name)


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
            cuh_lines = ["#pragma once"]
            cuh_lines.append(f'#include "{file_stem}.h"')
            cuh_lines.append("#if EXO_EXCUT_bENABLE_LOG")
            cuh_lines.append(f'#include "{file_stem}.excut_str_table"')
            cuh_lines.append("#endif")
            cuh_lines.extend(lines)  # Most of the code
            text = "\n".join(cuh_lines)
        elif ext == "cu":
            text = "\n".join([f'#include "{file_stem}.cuh"'] + lines)
        else:
            # A bit crappy we have per-file-extension logic here.
            assert "Add case for file extension"
        ext_snippets[ext] = text

    # excut stuff for CUDA tests
    if used_cuda:
        ext_snippets["excut_str_table"] = excut.generate_excut_str_table_header(
            f"exo_CudaUtil_{lib_name}"
        )

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
    public_fwd_decls = []
    used_cuda = False

    # Body contents
    private_fwd_decls = []
    proc_bodies = []
    tagged_c_utils: List[Tuple[str, str]] = []  # (name, cu_util)
    tagged_c_includes: List[Tuple[str, str]] = []  # (name, cu_util)
    tagged_cu_utils: List[Tuple[str, str]] = []  # (name, cu_util)
    tagged_cu_includes: List[Tuple[str, str]] = []  # (name, header name)
    util_injector = UtilInjectorImpl(
        "", tagged_c_utils, tagged_c_includes, tagged_cu_utils, tagged_cu_includes
    )
    analyzed_public_procs = []
    analyzed_private_procs = []
    ext_lines = {}

    needed_helpers = set()
    mem_code_builder = MemCodeBuilder()
    header_memwins = set()

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
                    "\n".join(p.instr.instr_format or ""),
                    "*/",
                ]
            )
            if instr.c_utils:
                for util in instr.c_utils:
                    tagged_c_utils.append((instr_name, util))
            if instr.c_includes:
                for header_name in instr.c_includes:
                    tagged_c_includes.append((instr_name, header_name))
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
                barrier_uses,
                proc_uses_cuda,
                util_injector,
                mem_code_builder,
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
                for a in p.args:
                    header_memwins.add(a.mem or DRAM)
            else:
                analyzed_private_procs.append(p)
            for ext, snippets in comp.ext_lines().items():
                ext_lines.setdefault(ext, []).extend(snippets)

    memgen = mem_code_builder.generate_code(header_memwins)

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

{from_lines(memgen.h_includes)}
{from_lines(ctxt_def)}
{from_lines(memgen.h_code)}
{from_lines(public_fwd_decls)}
{join_ext_lines(ext_lines.get("h"))}"""

    extern_code = _compile_externs(
        find_all_externs(analyzed_public_procs + analyzed_private_procs)
    )

    body_contents = [memgen.c_includes]
    body_contents.append(make_utility_lines(True, None, tagged_c_includes))
    helper_code = [_static_helpers[v] for v in needed_helpers]
    body_contents.append(helper_code)
    body_contents.append(memgen.c_code)
    body_contents.append(make_utility_lines(False, None, tagged_c_utils))
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
        # Moved CUDA includes to the top.
        # clangd seems to get really confused if includes are in the wrong place.
        cu_include_lines = make_utility_lines(True, None, tagged_cu_includes)
        cu_util_lines = make_utility_lines(
            False, f"exo_CudaUtil_{lib_name}", tagged_cu_utils
        )
        ext_lines["cuh"] = (
            memgen.c_includes
            + cu_include_lines
            + memgen.c_code
            + cu_util_lines
            + cuh_lines
        )

    return header_contents, body_contents, ext_lines


def _compile_externs(externs):
    extern_code = []
    for f, t in sorted(externs, key=lambda x: x[0].name() + x[1]):
        if glb := f.globl(t):
            extern_code.append(glb)
    return extern_code


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
        barrier_uses,
        used_cuda,
        util_injector,
        mem_code_builder,
        *,
        is_public_decl,
    ):
        assert isinstance(proc, LoopIR.proc)

        self.lib_name = lib_name
        self.proc = proc
        self.ctxt_name = ctxt_name
        self.env = ChainMap()
        self.force_names = dict()  # For ExtWithContext.force_names
        self.range_env = IndexRangeEnvironment(proc, fast=False)
        self.names = ChainMap()
        self.envtyp = dict()
        self.env_window_features = dict()  # Sym -> WindowFeatures
        self.mems = dict()
        self._tab = ""
        self._lines = []
        self._scalar_refs = set()
        self._needed_helpers = set()
        self.barrier_uses = barrier_uses
        self._used_cuda = used_cuda
        self._known_strides = {}
        self._in_cuda_function = False
        self._cuda_kernel_count = 0
        self._util_injector = util_injector
        self._mem_code_builder = mem_code_builder

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
            self.new_varname(a.name, typ=a.type, mem=mem)
            if a.type.is_real_scalar():
                self._scalar_refs.add(a.name)

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
                    int(pred.rhs.val)
                )
                self.add_line(f"// assert {pred}")
            else:
                # Default to just informing the compiler about the constraint
                # on a best-effort basis
                self.add_line(f"EXO_ASSUME({self.comp_e(pred)});")

        for a in proc.args:
            # NOTE: Moved below preds, so that known_strides gets filled
            # before initializing new variables.
            self.init_window_features(a, a.name)
            self.append_fnarg_decl(a, self.env[a.name], arg_strs, typ_comments)

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

                # Need to have init_window_features(a, a.name) before this
                encoder = self.env_window_features[a.name].get_encoder()

                if a.type.is_win():
                    if encoder.separate_dataptr():
                        arg_strs.append(
                            f"{encoder.dataptr_ctype()} {dataptr_name(name_arg)}"
                        )
                        typ_comments.append("    (Separate window data pointer)")
                    arg_strs.append(f"struct {encoder.exo_struct_name()} {name_arg}")
                else:
                    arg_strs.append(f"{encoder.dataptr_ctype()} {name_arg}")
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

    def new_varname(self, symbol, typ, mem=None) -> str:
        """Init envs & MemWin for new variable, except env_window_features.

        Give back C name for the variable (env[symbol]).
        Note, env_window_features must be initialized separately due to
        an ordering issue with known_strides.

        """
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

        # Record LoopIR type
        self.envtyp[symbol] = typ

        # Record MemWin type
        if mem is not None:
            assert issubclass(mem, MemWin)
        else:
            mem = DRAM
        self.mems[symbol] = mem
        self._mem_code_builder.register_memwin(mem)

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

    def wrap_cir(self, e, origin_story, origin_index=None) -> CIR_Wrapper:
        if isinstance(e, LoopIR.expr):
            e = lift_to_cir(e, self.range_env)
        elif isinstance(e, int):
            e = CIR.Const(e)
        assert isinstance(e, CIR.expr)
        if origin_index is not None:
            origin_story = f"{origin_story}[{origin_index}]"
        return CIR_Wrapper(e, self, origin_story)

    def window_expr_to_cir(
        self, e: LoopIR.WindowExpr
    ) -> Tuple[List[CIR_Wrapper], List[CIR_Wrapper], SrcInfo]:
        """Convert WindowExpr to WindowFeatures.new_window args"""
        w_idxs = []
        w_intervals = []
        for i, w in enumerate(e.idx):
            if isinstance(w, LoopIR.Point):
                lo = w.pt
                w_intervals.append(None)
            else:
                lo = w.lo
                w_intervals.append(
                    self.wrap_cir(w.hi, f"{e.name} interval_sizes", i)
                    - lift_to_cir(w.lo, self.range_env)
                )
            w_idxs.append(self.wrap_cir(lo, f"{e.name} offsets", i))
        return w_idxs, w_intervals, e.srcinfo

    def comp_cir(self, e, prec) -> str:
        env = self.env
        if isinstance(e, CIR.Read):
            return env[e.name]

        elif isinstance(e, CIR.ReadSeparateDataptr):
            return dataptr_name(env[e.name])

        elif isinstance(e, CIR.Const):
            return str(e.val)

        elif isinstance(e, CIR.BinOp):
            local_prec = op_prec[e.op]

            lhs = self.comp_cir(e.lhs, local_prec)
            rhs = self.comp_cir(e.rhs, local_prec)

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

        elif isinstance(e, CIR.USub):
            return f'-{self.comp_cir(e.arg, op_prec["~"])}'
        elif isinstance(e, CIR.AddressOf):
            return "&" + self.comp_cir(e.arg, op_prec["~"])
        elif isinstance(e, CIR.Indexed):
            ptr = self.comp_cir(e.ptr, op_prec["."])
            idx = self.comp_cir(e.idx, 0)
            return f"{ptr}[{idx}]"
        elif isinstance(e, CIR.GetAttr):
            arg = self.comp_cir(e.arg, op_prec["."])
            return f"{arg}.{e.attr}"
        elif isinstance(e, CIR.Custom):
            str_args = [self.comp_cir(a) for a in e.args]
            text = e.callback(*str_args)
            if prec > 0:
                text = f"({text})"
            return text
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
        idx_expr_s = self.comp_cir(simplify_cir(idx_expr), prec=0)
        if not type.is_win():
            return f"{buf}[{idx_expr_s}]"
        else:
            return f"{buf}.data[{idx_expr_s}]"

    def shape_strs(self, shape, prec=100) -> str:
        comp_res = [
            self.comp_cir(simplify_cir(lift_to_cir(i, self.range_env)), prec)
            for i in shape
        ]
        return comp_res

    # TODO remove
    def tensor_strides(self, shape) -> List[CIR.expr]:
        szs = [lift_to_cir(i, self.range_env) for i in shape]
        assert len(szs) >= 1
        strides = [CIR.Const(1)]
        wrapped_stride = self.wrap_cir(szs[-1], "tensor_strides")
        for sz in reversed(szs[:-1]):
            s = wrapped_stride.exo_get_cir()
            if hasattr(s, "is_non_neg"):
                s = s.update(is_non_neg=True)  # TODO rethink is_non_neg
            strides.append(s)
            wrapped_stride = sz * wrapped_stride
        strides.reverse()
        return strides

    # works for any tensor or window type
    # TODO remove
    def get_strides(self, name: Sym, typ) -> List[CIR.expr]:
        if typ.is_win():
            res = []
            for i in range(len(typ.shape())):
                if stride := self._known_strides.get((name, i)):
                    res.append(stride)
                else:
                    # TODO externalize soon
                    expr = CIR_Wrapper(
                        CIR.Read(name, False), self, "get_strides"
                    ).strides[i]
                    res.append(expr.exo_get_cir())
            return res
        else:
            return self.tensor_strides(typ.shape())

    # TODO remove
    def get_strides_s(self, name: Sym, typ) -> List[str]:
        all_strides = self.get_strides(name, typ)
        return [self.comp_cir(simplify_cir(i), prec=0) for i in all_strides]

    def get_idx_offset(self, name: Sym, typ, idx) -> CIR:
        strides = self.get_strides(name, typ)
        assert len(strides) == len(idx)
        acc = CIR.BinOp("*", idx[0], strides[0], True)
        for i, s in zip(idx[1:], strides[1:]):
            new = CIR.BinOp("*", i, s, True)
            acc = CIR.BinOp("+", acc, new, True)

        return acc

    def init_window_features(self, node, symbol):
        """Init env_window_features for variable and add global memory code"""
        typ = self.envtyp[symbol]
        if not typ.is_tensor_or_window():
            return
        strnm = self.env[symbol]
        mem = self.mems[symbol]

        srcinfo = node.srcinfo
        basetype_name = str(typ.basetype())
        const = self.is_const(symbol)
        utils = self._util_injector.with_tag(mem.name())

        def kvetch(message):
            raise MemGenError(f"{srcinfo}: {typ} @ {mem.name()} is invalid: {message}")

        def wrap_cir(obj, attr, idx):
            if isinstance(obj, int):
                obj = CIR.Const(obj)
            else:
                obj = obj.exo_get_cir()
            return CIR_Wrapper(obj, self, f"{symbol} {attr}[{idx}]")

        # Analyze packed tensor shape
        shape = typ.shape()
        n_dims = len(shape)
        packed_tensor_shape = mem.packed_tensor_shape(basetype_name)
        n_packed_dims = len(packed_tensor_shape)
        n_array_dims = n_dims - n_packed_dims
        if n_array_dims < 0:
            kvetch(
                f"must be at least {n_packed_dims}-dimensional (for packed tensor shape {packed_tensor_shape})"
            )

        cir_array_interval_sizes = [
            simplify_cir(lift_to_cir(e, self.range_env)) for e in shape[:n_array_dims]
        ]
        cir_packed_interval_sizes = [
            simplify_cir(lift_to_cir(e, self.range_env)) for e in shape[n_array_dims:]
        ]

        packed_const_shape: List[int] = []
        for c in cir_packed_interval_sizes:
            if isinstance(c, CIR.Const):
                packed_const_shape.append(c.val)
            else:
                actual = [str(c) for c in cir_packed_interval_sizes]
                kvetch(f"Required constant packed tensor shape, not {actual}")
        if tuple(packed_const_shape) != tuple(packed_tensor_shape):
            kvetch(
                f"{packed_const_shape} packed tensor shape not supported; expect {packed_tensor_shape}"
            )
        scalars_per_packed_tensor = prod(packed_const_shape)

        # Get encoder and indexer if possible. Analyze stride support
        cw_sym = CIR_Wrapper(CIR.Read(symbol, False), self, strnm)
        encoder, indexer = None, None
        if mem.has_window_encoder():
            encoder = mem.make_window_encoder(basetype_name, n_dims, const)
            self._mem_code_builder.register_window_encoder(encoder)
        if mem.has_window_indexer():
            indexer = mem.make_window_indexer(basetype_name, n_dims, const)
        supports_strides = True
        if n_array_dims > 0 and encoder:
            try:
                encoder.decode_array_stride_as_packed(utils, cw_sym, 0)
            except NotImplementedError:
                supports_strides = False

        # Handle differences between unpacking window structs (MemWin-customized)
        # vs allocated tensors (built in logic).
        cw_array_strides = []
        cw_array_offsets = []
        if typ.is_win():
            # Unpack dataptr, array offsets, maybe strides, from window
            if encoder is None:
                kvetch(
                    "cannot create a window when the MemWin type defines no WindowEncoder"
                )
            if encoder.separate_dataptr():
                cw_dataptr = CIR_Wrapper(CIR.ReadSeparateDataptr(symbol), self, strnm)
            else:
                cw_dataptr = cw_sym.data
            for n in range(n_array_dims):
                offset = encoder.decode_array_offset(utils, cw_sym, n)
                cw_array_offsets.append(wrap_cir(offset, "array_offsets", n))
                if supports_strides:
                    if stride := self._known_strides.get((symbol, n)):
                        # Translate stride units from scalars to packed tensors
                        stride = wrap_cir(stride, "array_strides_as_packed", n)
                        cw_array_strides.append(stride / scalars_per_packed_tensor)
                    else:
                        stride = encoder.decode_array_stride_as_packed(utils, cw_sym, n)
                        cw_array_strides.append(
                            wrap_cir(stride, "array_strides_as_packed", n)
                        )
        else:
            # dataptr = allocated tensor name
            # array_offsets = all 0
            # strides = defaults, if not disabled
            assert isinstance(typ, LoopIR.Tensor)
            cw_dataptr = cw_sym
            for n in range(n_array_dims):
                cw_array_offsets.append(wrap_cir(0, "array_offsets", n))
                if supports_strides:
                    stride = wrap_cir(1, "array_strides_as_packed", n)
                    for i in range(n + 1, n_array_dims):
                        stride *= cir_array_interval_sizes[i]
                    cw_array_strides.append(stride)

        # Make features object.
        features = WindowFeatures()
        features._mem = mem
        features._scalars_per_packed_tensor = scalars_per_packed_tensor
        features._varname = cw_sym
        features._dataptr = cw_dataptr
        features._array_strides_as_packed = cw_array_strides
        features._array_offsets = cw_array_offsets
        features._array_interval_sizes = [
            wrap_cir(c, "array_interval_sizes", i)
            for i, c in enumerate(cir_array_interval_sizes)
        ]
        features._packed_offsets = [
            wrap_cir(0, "packed_offsets", i) for i in range(n_packed_dims)
        ]
        features._packed_interval_sizes = [
            wrap_cir(c, "packed_interval_sizes", i)
            for i, c in enumerate(cir_packed_interval_sizes)
        ]
        features._encoder = encoder
        features._indexer = indexer
        features._legacy_basetyp = typ
        features._srcinfo = srcinfo
        self.env_window_features[symbol] = features

    def debug_comment_window_features(self, features: WindowFeatures):
        self.add_line("/*")
        self.add_line(f"mem = {features.get_mem().name()}")
        self.add_line(
            f"scalars_per_packed_tensor = {features._scalars_per_packed_tensor}"
        )
        self.add_line(f"raw_name = {features.get_raw_name()!r}")
        self.add_line(f"dataptr = {features.get_dataptr()}")
        for n in range(features.n_array_dims()):
            offset = features.get_array_offset(n)
            size = features.get_array_interval_size(n)
            stride = None
            if features._array_strides_as_packed:
                stride = features.get_array_stride_as_packed(n)
            self.add_line(f"array[{n}]: {offset}, {size}, {stride}")
        for n in range(features.n_packed_dims()):
            offset = features.get_packed_offset(n)
            size = features.get_packed_interval_size(n)
            self.add_line(f"packed[{n}]: {offset}, {size}")
        self.add_line(f"encoder: {type(features._encoder).__name__}")
        self.add_line(f"indexer: {type(features._indexer).__name__}")
        self.add_line("*/")

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
            self.add_line(f"// {sync_type.format_stmt(s.barriers)}")
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
            is_const = self.is_const(rhs.name)
            if not is_const:
                self.global_non_const.add(s.name)

            output_winmem = s.special_window or input_winmem
            name = self.new_varname(s.name, typ=rhs.type, mem=output_winmem)
            self.init_window_features(s, s.name)

            # Unpack features of input window, and modify based on WindowExpr
            in_features = self.env_window_features[rhs.name].new_window(
                *self.window_expr_to_cir(rhs)
            )

            # Unpack features of output window.
            out_features = self.env_window_features[s.name]
            out_encoder: WindowEncoder = out_features.get_encoder()
            if s.special_window and not issubclass(input_winmem, SpecialWindow):
                encode_window = out_encoder.encode_special_window
                encode_dataptr = out_encoder.encode_special_separate_dataptr
            else:
                encode_window = out_encoder.encode_window
                encode_dataptr = out_encoder.encode_separate_dataptr
            utils = self._util_injector.with_tag(output_winmem.name())

            # Initialize separate dataptr
            if out_encoder.separate_dataptr():
                d_def = encode_dataptr(utils, in_features)
                cref = ""
                if self._in_cuda_function:
                    # HACK needed for CUtensorMap; if we copy the CUtensorMap
                    # in CUDA code, then it won't be in grid constant memory,
                    # and cp.async.bulk won't work anymore.
                    cref = " const&"
                self.add_line(
                    f"{out_encoder.dataptr_ctype()}{cref} {dataptr_name(name)} = {d_def};"
                )
            # Initialize window struct.
            w_def = encode_window(utils, in_features)
            self.add_line(f"struct {out_encoder.exo_struct_name()} {name} = {w_def};")

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
            self.init_window_features(s, s.name)
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
            fn = s.f
            assert all(
                a.type.is_win() == fna.type.is_win() for a, fna in zip(s.args, fn.args)
            )
            if fn.instr is not None:
                try:
                    args_dict = dict()
                    for i, e in enumerate(s.args):
                        fnarg = fn.args[i]
                        args_dict[str(fnarg.name)] = self.comp_fnarg(e, fn, i)
                    lines = fn.instr.codegen(InstrArgs(args_dict))
                    assert not isinstance(lines, str), "codegen() must give List[str]"
                    for line in lines:
                        self.add_line(line)
                except Exception as e:
                    raise ValueError(
                        f"Failed to compile {fn.name}; this could be invalid usage, or a bug in the @instr implementation: {e}"
                    ) from e
            else:
                args = ["ctxt"]
                for i, e in enumerate(s.args):
                    args.extend(self.comp_fnarg(e, fn, i).to_arg_strs())
                self.add_line(f"{fn.name}({', '.join(args)});")

        else:
            assert False, "bad case"

    def comp_fnarg(self, e, fn, i, *, force_pass_by_value=False):
        """Returns InstrWindowArg or InstrNonWindowArg"""
        assert isinstance(fn, LoopIR.proc)
        mem = fn.args[i].mem
        is_const = None
        if isinstance(e, LoopIR.WindowExpr):
            callee_buf = fn.args[i].name
            is_const = fn.is_const_param(callee_buf)
        return self.comp_fnarg_impl(e, mem, is_const, force_pass_by_value)

    def comp_fnarg_impl(self, e, mem, is_const, force_pass_by_value):
        """Returns InstrWindowArg or InstrNonWindowArg"""
        if isinstance(e, LoopIR.Read):
            assert not e.idx
            rtyp = self.envtyp[e.name]
            cname = self.env[e.name]
            if rtyp.is_indexable() or rtyp is T.bool or rtyp is T.stride:
                return InstrNonWindowArg(cname, e.srcinfo)
            if rtyp.is_dense_tensor():
                return InstrNonWindowArg(cname, e.srcinfo)
            elif e.name in self._scalar_refs:
                star = "*" if force_pass_by_value else ""
                return InstrNonWindowArg(f"{star}{cname}", e.srcinfo)
            elif rtyp.is_win():
                return self.comp_fnarg_window(e, mem, is_const)
            else:
                assert rtyp.is_real_scalar()
                amp = "" if force_pass_by_value else "&"
                return InstrNonWindowArg(f"{amp}{cname}", e.srcinfo)
        elif isinstance(e, LoopIR.WindowExpr):
            return self.comp_fnarg_window(e, mem, is_const)
        else:
            return InstrNonWindowArg(self.comp_e(e, op_prec["."]), e.srcinfo)

    def comp_fnarg_window(
        self, e: LoopIR.expr, encoder_mem: Type[MemWin], is_const: bool
    ):
        # Create private copy of WindowFeatures, with offsets etc. updated
        # if the input is a WindowExpr.
        if isinstance(e, LoopIR.WindowExpr):
            features = self.env_window_features[e.name].new_window(
                *self.window_expr_to_cir(e)
            )
        else:
            assert isinstance(e, LoopIR.Read)
            assert not e.idx
            features = self.env_window_features[e.name].copy()

        # Replace encoder for the features (which we have a private copy of
        # because of the new_window(...) or copy()) with encoder based on the
        # called function's window dimensionality, memory type, and constness
        # (all of which can differ subtly compared to the input).
        features._encoder = None
        if encoder_mem.has_window_encoder():
            typ = str(e.type.basetype())
            n_dims = len(e.type.shape())
            features._encoder = encoder_mem.make_window_encoder(typ, n_dims, is_const)

        # Package InstrWindowArg
        indexer_mem = features.get_mem()
        return InstrWindowArg(
            self._util_injector.with_tag(encoder_mem.name()),
            self._util_injector.with_tag(indexer_mem.name()),
            features,
            e.srcinfo,
        )

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
            return self.comp_cir(simplify_cir(stride), prec=0)

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


# Assemble includes or utility code into List[str] of lines
# from list of pairs of (required_by: str, content: str)
# where content is a header name or a cu_util blob.
# We remove exact duplicate strings.
def make_utility_lines(
    is_includes: bool,
    cu_namespace: Optional[str],
    tagged_content: List[Tuple[str, str]],
) -> List[str]:
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
        if cu_namespace:
            lines.append("")
            lines.append(f"namespace {cu_namespace} {{")
            lines.append(f"namespace exo_CudaUtil = ::{cu_namespace};")

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

    if cu_namespace:
        lines.append(f"}}  // end namespace {cu_namespace}")

    return lines


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Builds collection of C code from code requested from Memory and
# WindowEncoder (MemIncludeC and MemGlobalC). We place code in the header
# file iff it's required by a MemWin type in a public proc's interface.


@dataclass(slots=True)
class MemCodeResult:
    h_includes: List[str]
    h_code: List[str]
    c_includes: List[str]
    c_code: List[str]


@dataclass(slots=True)
class MemCodeBuilder(object):
    # Maps header names to set of MemWin types requiring it.
    header_used_by_dict: Dict[str, Set[Type[MemWin]]] = field(default_factory=dict)

    # Maps MemGlobalC.name to set of MemWin types requiring the MemGlobalC
    code_name_used_by_dict: Dict[str, Set[Type[MemWin]]] = field(default_factory=dict)

    # Maps MemGlobalC.name to MemGlobalC.code
    name_to_code_dict: Dict[str, str] = field(default_factory=dict)

    # List of MemGlobal.name in the order received.
    code_name_order: List[str] = field(default_factory=list)

    def register_memwin(self, mem: Type[MemWin]):
        glob = mem.global_()
        mem_name = mem.mangled_name()
        if isinstance(glob, str):
            glob = MemGlobalC(mem_name, glob)
        self._add_global(mem, glob)

    def register_window_encoder(self, encoder: WindowEncoder):
        depends_on = []
        sdef = encoder.define_struct(depends_on)
        glob = MemGlobalC(encoder.exo_struct_name(), sdef, tuple(depends_on))
        self._add_global(encoder.mem, glob)

    def _add_header(self, mem: Type[MemWin], item: MemIncludeC):
        headers = self.header_used_by_dict
        header_name = item.header_name
        mem_set = headers.get(header_name)
        if mem_set is None:
            mem_set = {mem}
            headers[header_name] = mem_set
        else:
            # Hacky: hide CodegenSmem from the generated used_by comments
            mem_set.add(mem.wrapped_smem_type())

    def _add_global(self, mem: Type[MemWin], item: MemGlobalC):
        # Add dependecies first
        for sub_item in item.depends_on:
            if isinstance(sub_item, MemGlobalC):
                self._add_global(mem, sub_item)
            elif isinstance(sub_item, MemIncludeC):
                self._add_include(mem, sub_item)
            else:
                assert 0, f"{mem.name()}: Unexpected type {(sub_item)}"

        name, code = item.name, item.code
        if code:
            # Empty code is ignored
            used_by = self.code_name_used_by_dict.get(name)
            if used_by is None:
                used_by = set()
                self.code_name_used_by_dict[name] = used_by

            old_code = self.name_to_code_dict.get(name)
            if old_code is None:
                # First time seeing this chunk of code
                self.name_to_code_dict[name] = code
                self.code_name_order.append(name)
            elif old_code != code:
                conflict_names = [sus.name() for sus in used_by]
                raise ValueError(
                    f"Name collision; different code with same name {name}; {mem.name()} conflicts with {conflict_names}"
                )

            used_by.add(mem.wrapped_smem_type())

    def generate_code(self, header_memwins: Set[Type[MemWin]]) -> MemCodeResult:
        """Given set of MemWin needed in the header file, generate lists of C code"""
        result = MemCodeResult([], [], [], [])

        include_pairs = sorted(self.header_used_by_dict.items(), key=lambda a: a[0])
        for header, used_by in include_pairs:
            lines = (
                result.c_includes
                if used_by.isdisjoint(header_memwins)
                else result.h_includes
            )
            for user in sorted(user.name() for user in used_by):
                include_lines.append(f"/* Required by {user.name()} */")
            include_lines.append(f'#include "{header}"')

        assert len(self.code_name_order) == len(self.code_name_used_by_dict)
        assert len(self.code_name_order) == len(self.name_to_code_dict)
        for name in self.code_name_order:
            used_by = self.code_name_used_by_dict[name]
            lines = (
                result.c_code if used_by.isdisjoint(header_memwins) else result.h_code
            )
            code = self.name_to_code_dict[name]
            header_guard = f"EXO_MEMORY_GLOBAL_{name}"
            for user in sorted(user.name() for user in used_by):
                lines.append(f"/* Required by {user} */")
            lines.append(f"#ifndef {header_guard}")
            lines.append(f"#define {header_guard}")
            lines.append(code)
            lines.append("#endif")

        return result


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
        return self._compiler.comp_fnarg_impl(
            e, mem, is_const, force_pass_by_value
        ).to_arg_strs()

    def get_barrier_usage(self, name: Sym) -> BarrierUsage:
        return self._compiler.barrier_uses[name]
