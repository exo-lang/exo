import ast as pyast
import inspect
import re
import types
from pathlib import Path
from typing import Optional, Union, List
from enum import Enum, auto

import exo.LoopIR_scheduling as scheduling
from exo.LoopIR_scheduling import SchedulingError

from .API_types import ProcedureBase
from . import LoopIR as LoopIR
from .LoopIR_compiler import run_compile, compile_to_strings
from .LoopIR_interpreter import run_interpreter
from .LoopIR_unification import DoReplace, UnificationError
from .configs import Config
from .effectcheck import InferEffects, CheckEffects
from .memory import Memory
from .parse_fragment import parse_fragment
from .pattern_match import match_pattern, get_match_no
from .prelude import *
from .new_eff import Check_Aliasing

# Moved to new file
from .proc_eqv import decl_new_proc, derive_proc, assert_eqv_proc, check_eqv_proc
from .pyparser import get_ast_from_python, Parser, get_src_locals
from .reflection import LoopIR_to_QAST
from .typecheck import TypeChecker

from . import API_cursors as C
from . import internal_cursors as IC

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Top-level decorator


def proc(f, _instr=None) -> "Procedure":
    if not isinstance(f, types.FunctionType):
        raise TypeError("@proc decorator must be applied to a function")

    body, getsrcinfo = get_ast_from_python(f)
    assert isinstance(body, pyast.FunctionDef)

    parser = Parser(
        body,
        getsrcinfo,
        func_globals=f.__globals__,
        srclocals=get_src_locals(depth=3 if _instr else 2),
        instr=_instr,
        as_func=True,
    )
    return Procedure(parser.result())


def instr(instruction):
    if not isinstance(instruction, str):
        raise TypeError("@instr decorator must be @instr(<your instuction>)")

    def inner(f):
        if not isinstance(f, types.FunctionType):
            raise TypeError("@instr decorator must be applied to a function")

        return proc(f, _instr=instruction)

    return inner


def config(_cls=None, *, readwrite=True):
    def parse_config(cls):
        if not inspect.isclass(cls):
            raise TypeError("@config decorator must be applied to a class")

        body, getsrcinfo = get_ast_from_python(cls)
        assert isinstance(body, pyast.ClassDef)

        parser = Parser(
            body,
            getsrcinfo,
            func_globals={},
            srclocals=get_src_locals(depth=2),
            as_config=True,
        )
        return Config(*parser.result(), not readwrite)

    if _cls is None:
        return parse_config
    else:
        return parse_config(_cls)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   iPython Display Object


class MarkDownBlob:
    def __init__(self, mkdwn_str):
        self.mstr = mkdwn_str

    def _repr_markdown_(self):
        return self.mstr


class FindBefore(LoopIR.LoopIR_Do):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        self.result = None
        super().__init__(proc)

    def result(self):
        return self.result

    def do_stmts(self, stmts):
        prev = None
        for s in stmts:
            if s == self.stmt:
                self.result = prev
                return
            else:
                self.do_s(s)
                prev = s


class FindDup(LoopIR.LoopIR_Do):
    def __init__(self, proc):
        self.result = False
        self.env = []
        super().__init__(proc)

    def result(self):
        return self.result

    def do_s(self, s):
        for e in self.env:
            if s is e:
                self.result = True
                print(s)
        self.env.append(s)

        super().do_s(s)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Procedure Objects


def compile_procs(proc_list, basedir: Path, c_file: str, h_file: str):
    c_data, h_data = compile_procs_to_strings(proc_list, h_file)
    (basedir / c_file).write_text(c_data)
    (basedir / h_file).write_text(h_data)


def compile_procs_to_strings(proc_list, h_file_name: str):
    assert isinstance(proc_list, list)
    assert all(isinstance(p, Procedure) for p in proc_list)
    return run_compile([p._loopir_proc for p in proc_list], h_file_name)


class Procedure(ProcedureBase):
    def __init__(
        self,
        proc,
        _provenance_eq_Procedure: "Procedure" = None,
        _forward=None,
        _mod_config=None,
    ):
        super().__init__()

        _mod_config = _mod_config or frozenset()

        if isinstance(proc, LoopIR.UAST.proc):
            proc = TypeChecker(proc).get_loopir()
            proc = InferEffects(proc).result()
            CheckEffects(proc)
            Check_Aliasing(proc)

        assert isinstance(proc, LoopIR.LoopIR.proc)

        # add this procedure into the equivalence tracking mechanism
        if _provenance_eq_Procedure:
            derive_proc(
                _provenance_eq_Procedure._loopir_proc, proc, frozenset(_mod_config)
            )
        else:
            decl_new_proc(proc)

        if _forward is None:

            def _forward(_):
                raise NotImplementedError(
                    "This forwarding function has not been implemented"
                )

        self._loopir_proc = proc
        self._provenance_eq_Procedure = _provenance_eq_Procedure
        self._forward = _forward

    def forward(self, cur: C.Cursor):
        p = self
        fwds = []
        while p is not None and p is not cur.proc():
            fwds.append(p._forward)
            p = p._provenance_eq_Procedure

        ir = cur._impl
        for fn in reversed(fwds):
            ir = fn(ir)

        return C.lift_cursor(ir, self)

    def __str__(self):
        return str(self._loopir_proc)

    def __eq__(self, other):
        if not isinstance(other, Procedure):
            return False
        return self._loopir_proc == other._loopir_proc

    def _repr_markdown_(self):
        return "```python\n" + self.__str__() + "\n```"

    def INTERNAL_proc(self):
        return self._loopir_proc

    # -------------------------------- #
    #     introspection operations
    # -------------------------------- #

    def check_effects(self):
        self._loopir_proc = InferEffects(self._loopir_proc).result()
        CheckEffects(self._loopir_proc)
        return self

    def name(self):
        return self._loopir_proc.name

    def show_effects(self):
        self._loopir_proc = InferEffects(self._loopir_proc).result()
        return str(self._loopir_proc.eff)

    def show_effect(self, stmt_pattern):
        self._loopir_proc = InferEffects(self._loopir_proc).result()
        if match := match_pattern(
            self._root(), stmt_pattern, call_depth=1, default_match_no=0
        ):
            assert len(match[0]) == 1, "Must match single statements"
            return str(match[0][0]._node.eff)
        raise SchedulingError("failed to find statement", pattern=stmt_pattern)

    def is_instr(self):
        return self._loopir_proc.instr is not None

    def get_instr(self):
        return self._loopir_proc.instr

    def args(self):
        if args := self._root()._child_block("args"):
            return C.lift_cursor(args, self)
        return []

    def body(self):
        """
        Return a BlockCursor selecting the entire body of the Procedure
        """
        block = self._root()._child_block("body")
        return C.lift_cursor(block, self)

    def find(self, pattern, many=False):
        """
        Find the most specific possible cursor for the given pattern.
        For example, a pattern matching a single assignment statement
        will return an AssignCursor, not a StmtCursor or BlockCursor.

        If the optional parameter `many` is set to True, then return a list,
        potentially containing more than one Cursor.

        In any event, if no matches are found, a SchedulingError is raised
        """
        if not isinstance(pattern, str):
            raise TypeError("expected a pattern string")
        default_match_no = None if many else 0
        raw_cursors = match_pattern(
            self._root(), pattern, call_depth=1, default_match_no=default_match_no
        )
        assert isinstance(raw_cursors, list)
        cursors = []
        for c in raw_cursors:
            c = C.lift_cursor(c, self)
            if isinstance(c, (C.BlockCursor, C.ExprListCursor)) and len(c) == 1:
                c = c[0]
            cursors.append(c)

        if not cursors:
            raise SchedulingError("failed to find matches", pattern=pattern)

        return cursors if many else cursors[0]

    def find_loop(self, pattern, many=False):
        """
        This is the same as proc.find(...), except if the supplied pattern
        is of the form 'name' or 'name #n', then it will be auto-expanded
        to 'for name in _:_' or 'for name in _:_ #n'
        """
        if not isinstance(pattern, str):
            raise TypeError("expected a pattern string")

        _name_count_re = r"^([a-zA-Z_]\w*)\s*(\#\s*[0-9]+)?$"
        results = re.search(_name_count_re, pattern)
        if results:
            name, count = results[1], (results[2] if results[2] else "")
            pattern = f"for {name} in _: _{count}"

        return self.find(pattern, many)

    def find_alloc_or_arg(self, pattern):
        _name_count_re = r"^([a-zA-Z_]\w*)\s*(\#\s*[0-9]+)?$"
        results = re.search(_name_count_re, pattern)
        if results:
            name, count = results[1], (results[2] if results[2] else "")
            for arg in self.args():
                if arg._impl._node.name.name() == name:
                    return arg

            pattern = f"{name}: _{count}"

        return self.find(pattern)

    def find_all(self, pattern):
        return self.find(pattern, many=True)

    def get_ast(self, pattern=None):
        if pattern is None:
            return LoopIR_to_QAST(self._loopir_proc).result()

        # do pattern matching
        match = match_pattern(self._root(), pattern, call_depth=1)

        # convert matched sub-trees to QAST
        assert isinstance(match, list)
        if not match:
            return None

        return [
            LoopIR_to_QAST(node._node).result() for block in match for node in block
        ]

    # ---------------------------------------------- #
    #     execution / interpretation operations
    # ---------------------------------------------- #

    def show_c_code(self):
        return MarkDownBlob("```c\n" + self.c_code_str() + "\n```")

    def c_code_str(self):
        decls, defns = compile_to_strings("c_code_str", [self._loopir_proc])
        return decls + "\n" + defns

    def compile_c(self, directory: Path, filename: str):
        compile_procs([self._loopir_proc], directory, f"{filename}.c", f"{filename}.h")

    def interpret(self, **kwargs):
        run_interpreter(self._loopir_proc, kwargs)

    # ------------------------------- #
    #     scheduling operations
    # ------------------------------- #

    def has_dup(self):
        """
        Internal check to see if there are any reference diamonds in the AST
        """
        return FindDup(self._loopir_proc).result

    def unsafe_assert_eq(self, other_proc):
        if not isinstance(other_proc, Procedure):
            raise TypeError("expected a procedure as argument")
        assert_eqv_proc(self._loopir_proc, other_proc._loopir_proc)
        return self

    def partial_eval(self, *args, **kwargs):
        if kwargs and args:
            raise ValueError("Must provide EITHER ordered OR named arguments")
        if not kwargs and not args:
            # Nothing to do if empty partial eval
            return self

        p = self._loopir_proc

        if args:
            if len(args) > len(p.args):
                raise TypeError(
                    f"expected no more than {len(p.args)} "
                    f"arguments, but got {len(args)}"
                )
            kwargs = {arg.name: val for arg, val in zip(p.args, args)}
        else:
            # Get the symbols corresponding to the names
            params_map = {sym.name.name(): sym.name for sym in p.args}
            kwargs = {params_map[k]: v for k, v in kwargs.items()}

        p = scheduling.DoPartialEval(kwargs).apply_proc(p)
        return Procedure(p)  # No provenance because signature changed

    def transpose(self, arg_cursor):
        if not (isinstance(arg_cursor, C.ArgCursor) and len(arg_cursor.shape()) == 2):
            raise TypeError("expected a 2D argument cursor")

        ir, _ = scheduling.DoRearrangeDim(arg_cursor._impl, [1, 0])
        return Procedure(ir)  # No provenance because signature changed

    def add_assertion(self, assertion, configs=None):
        if not isinstance(assertion, str):
            raise TypeError("assertion must be an Exo string")

        configs = configs or []

        p = self._loopir_proc
        assertion = parse_fragment(p, assertion, p.body[0], configs=configs)
        p = LoopIR.LoopIR.proc(
            p.name, p.args, p.preds + [assertion], p.body, p.instr, p.eff, p.srcinfo
        )
        return Procedure(p, _provenance_eq_Procedure=None)

    def is_eq(self, proc: "Procedure"):
        eqv_set = check_eqv_proc(self._loopir_proc, proc._loopir_proc)
        return eqv_set == frozenset()

    def _root(self):
        return IC.Cursor.create(self._loopir_proc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #


class ExoType(Enum):
    F32 = auto()
    F64 = auto()
    I8 = auto()
    I32 = auto()
    R = auto()
    Index = auto()
    Bool = auto()
    Size = auto()

    def is_indexable(self):
        return self in [ExoType.Index, ExoType.Size]

    def is_numeric(self):
        return self in [ExoType.F32, ExoType.F64, ExoType.I8, ExoType.I32, ExoType.R]

    def is_bool(self):
        return self == ExoType.Bool
