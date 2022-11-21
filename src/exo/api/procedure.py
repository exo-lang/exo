from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import exo.LoopIR as LoopIR
import exo.api as API
import exo.api.cursors as C
import exo.internal_cursors as IC
from exo.LoopIR_compiler import compile_to_strings
from exo.LoopIR_interpreter import run_interpreter
from exo.LoopIR_scheduling import Schedules, SchedulingError
from exo.effectcheck import InferEffects, CheckEffects
from exo.parse_fragment import parse_fragment
from exo.pattern_match import match_pattern, get_match_no, match_cursors
from exo.proc_eqv import decl_new_proc, derive_proc, assert_eqv_proc, check_eqv_proc
from exo.reflection import LoopIR_to_QAST
from exo.typecheck import TypeChecker

__all__ = ["Procedure"]


class Procedure:
    def __init__(
        self,
        proc,
        _provenance_eq_Procedure: "Procedure" = None,
        _mod_config=None,
    ):
        super().__init__()

        _mod_config = _mod_config or frozenset()

        if isinstance(proc, LoopIR.UAST.proc):
            proc = TypeChecker(proc).get_loopir()
            proc = InferEffects(proc).result()
            CheckEffects(proc)

        assert isinstance(proc, LoopIR.LoopIR.proc)

        # add this procedure into the equivalence tracking mechanism
        if _provenance_eq_Procedure:
            derive_proc(
                _provenance_eq_Procedure._loopir_proc, proc, frozenset(_mod_config)
            )
        else:
            decl_new_proc(proc)

        self._loopir_proc = proc

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
        return str(self._loopir_proc.eff)

    def show_effect(self, stmt_pattern):
        stmt = self._find_stmt(stmt_pattern)
        return str(stmt.eff)

    def is_instr(self):
        return self._loopir_proc.instr is not None

    def get_instr(self):
        return self._loopir_proc.instr

    def body(self):
        """
        Return a BlockCursor selecting the entire body of the Procedure
        """
        impl = IC.Cursor.root(self).body()
        return C.new_Cursor(impl)

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
        raw_cursors = match_cursors(
            self, pattern, call_depth=1, default_match_no=default_match_no
        )
        assert isinstance(raw_cursors, list)
        cursors = []
        for c in raw_cursors:
            c = C.new_Cursor(c)
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

    def find_all(self, pattern):
        return self.find(pattern, many=True)

    def _TEST_find_cursors(self, pattern):
        cursors = match_cursors(self, pattern, call_depth=1)
        assert isinstance(cursors, list)
        if not cursors:
            raise SchedulingError("failed to find matches", pattern=pattern)
        return cursors

    def _TEST_find_stmt(self, pattern):
        curs = self._TEST_find_cursors(pattern)
        assert len(curs) == 1
        curs = curs[0]
        if len(curs) != 1:
            raise SchedulingError(
                "pattern did not match a single statement", pattern=pattern
            )
        return curs[0]

    def get_ast(self, pattern=None):
        if pattern is None:
            return LoopIR_to_QAST(self._loopir_proc).result()
        else:
            # do pattern matching
            match_no = get_match_no(pattern)
            match = match_pattern(self, pattern, call_depth=1)

            # convert matched sub-trees to QAST
            assert isinstance(match, list)
            if len(match) == 0:
                return None
            elif isinstance(match[0], LoopIR.LoopIR.expr):
                results = [LoopIR_to_QAST(e).result() for e in match]
            elif isinstance(match[0], list):
                # statements
                assert all(
                    isinstance(s, LoopIR.LoopIR.stmt) for stmts in match for s in stmts
                )
                results = [LoopIR_to_QAST(stmts).result() for stmts in match]
            else:
                assert False, "bad case"

            # modulate the return type depending on whether this
            # was a query for a specific match or for all matches
            if match_no is None:
                return results
            else:
                assert len(results) == 1
                return results[0]

    # ---------------------------------------------- #
    #     execution / interpretation operations
    # ---------------------------------------------- #

    def show_c_code(self):
        return MarkDownBlob("```c\n" + self.c_code_str() + "\n```")

    def c_code_str(self):
        decls, defns = compile_to_strings("c_code_str", [self._loopir_proc])
        return decls + "\n" + defns

    def compile_c(self, directory: Path, filename: str):
        API.compile_procs(
            [self._loopir_proc], directory, f"{filename}.c", f"{filename}.h"
        )

    def interpret(self, **kwargs):
        run_interpreter(self._loopir_proc, kwargs)

    # ------------------------------- #
    #     scheduling operations
    # ------------------------------- #

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

        p = Schedules.DoPartialEval(p, kwargs).result()
        return Procedure(p)  # No provenance because signature changed

    def _find_stmt(
        self, stmt_pattern, call_depth=2, default_match_no: Optional[int] = 0
    ):
        stmt_lists = match_pattern(
            self, stmt_pattern, call_depth=call_depth, default_match_no=default_match_no
        )
        if len(stmt_lists) == 0 or len(stmt_lists[0]) == 0:
            raise SchedulingError("failed to find statement", pattern=stmt_pattern)
        elif default_match_no is None:
            return [s[0] for s in stmt_lists]
        else:
            return stmt_lists[0][0]

    def _find_callsite(self, call_site_pattern):
        call_stmt = self._find_stmt(call_site_pattern, call_depth=3)
        if not isinstance(call_stmt, LoopIR.LoopIR.Call):
            raise TypeError("pattern did not describe a call-site")

        return call_stmt

    def add_assertion(self, assertion, configs=[]):
        if not isinstance(assertion, str):
            raise TypeError("assertion must be an Exo string")

        p = self._loopir_proc
        assertion = parse_fragment(p, assertion, p.body[0], configs=configs)
        p = LoopIR.LoopIR.proc(
            p.name, p.args, p.preds + [assertion], p.body, p.instr, p.eff, p.srcinfo
        )
        return Procedure(p, _provenance_eq_Procedure=None)

    def is_eq(self, proc: "Procedure"):
        eqv_set = check_eqv_proc(self._loopir_proc, proc._loopir_proc)
        return eqv_set == frozenset()


class MarkDownBlob:
    def __init__(self, mkdwn_str):
        self.mstr = mkdwn_str

    def _repr_markdown_(self):
        return self.mstr
