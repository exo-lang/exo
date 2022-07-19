#import ast as pyast
import inspect
import re
#import types
from dataclasses import dataclass
from typing import Any, Optional, Union, List
import functools

from .API import Procedure
from .LoopIR_scheduling import Schedules
from .LoopIR import LoopIR, T #, UAST, LoopIR_Do
#from .LoopIR_compiler import run_compile, compile_to_strings
#from .LoopIR_interpreter import run_interpreter
#from .LoopIR_scheduling import (Schedules, name_plus_count, SchedulingError,
#                                iter_name_to_pattern,
#                                nested_iter_names_to_pattern)
#from .LoopIR_unification import DoReplace, UnificationError
from .configs import Config
from .effectcheck import InferEffects, CheckEffects
from .memory import Memory
from .parse_fragment import parse_fragment
from .pattern_match import match_pattern, get_match_no, match_cursors
from .prelude import *
## Moved to new file
#from .proc_eqv import (decl_new_proc, derive_proc,
#                       assert_eqv_proc, check_eqv_proc)
#from .pyparser import get_ast_from_python, Parser, get_src_locals
#from .reflection import LoopIR_to_QAST
#from .typecheck import TypeChecker

from .API_cursors import public_cursors as PC
from . import API_cursors

def is_subclass_obj(x, cls):
    return isinstance(x, type) and issubclass(x, cls)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Generic Definitions: Atomic Scheduling Operations and Argument Processing

@dataclass
class ArgumentProcessor:
    i           : int
    arg_name    : str
    f_name      : str

    def __init__(self):
        # see setdata below for setting of the above fields
        pass

    def err(self, message, Error=TypeError):
        raise Error(f"argument {self.i}, '{self.arg_name}' to {self.f_name}: "
                    f"{message}")

    def setdata(self, i, arg_name, f_name):
        self.i          = i
        self.arg_name   = arg_name
        self.f_name     = f_name

    def __call__(self, arg, all_args):
        raise NotImplementedError("Must Sub-class and redefine __call__")

@dataclass
class AtomicSchedulingOp:
    sig       : inspect.Signature
    arg_procs : List[ArgumentProcessor]
    func      : Any

    def __str__(self):
        return f"<AtomicSchedulingOp-{self.__name__}>"

    def __call__(self, *args, **kwargs):
        # capture the arguments according to the provided signature
        bound_args = self.sig.bind(*args, **kwargs)

        # potentially need to patch in default values...
        bargs = bound_args.arguments
        if len(self.arg_procs) != len(bargs):
            for nm in self.sig.parameters:
                if nm not in bargs:
                    default_val = self.sig.parameters[nm].default
                    assert default_val != inspect.Parameter.empty
                    kwargs[nm] = default_val
            # now re-bind the arguments with the defaults having been added
            bound_args = self.sig.bind(*args, **kwargs)
            bargs = bound_args.arguments

        # convert the arguments using the provided argument processors
        assert len(self.arg_procs) == len(bargs)
        for nm,argp in zip(bargs, self.arg_procs):
            bargs[nm] = argp(bargs[nm], bargs)

        # invoke the scheduling function with the modified arguments
        return self.func(*bound_args.args, **bound_args.kwargs)


# decorator for building Atomic Scheduling Operations in the
# remainder of this file
def sched_op(arg_procs):
    def check_ArgP(argp):
        if is_subclass_obj(argp, ArgumentProcessor):
            return argp()
        else:
            assert isinstance(argp, ArgumentProcessor)
            return argp
    # note pre-pending of ProcA
    arg_procs = [ check_ArgP(argp) for argp in ([ProcA] + arg_procs) ]

    def build_sched_op(func):
        f_name = func.__name__
        sig = inspect.signature(func)
        assert len(arg_procs) == len(sig.parameters)

        # record extra implicit information in the argument processors
        for i, (param, arg_p) in enumerate(zip(sig.parameters, arg_procs)):
            arg_p.setdata(i, param, f_name)

        atomic_op = AtomicSchedulingOp(sig, arg_procs, func)
        return functools.wraps(func)(atomic_op)

    return build_sched_op

def is_atomic_scheduling_op(x):
    return isinstance(x, AtomicSchedulingOp)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Argument Processing

class IdA(ArgumentProcessor):
    def __call__(self, arg, all_args):
        return arg

class ProcA(ArgumentProcessor):
    def __call__(self, proc, all_args):
        if not isinstance(proc, Procedure):
            self.err("expected a Procedure object")
        return proc

class MemoryA(ArgumentProcessor):
    def __call__(self, mem, all_args):
        if not is_subclass_obj(mem, Memory):
            self.err("expected a Memory subclass")
        return mem

class ConfigA(ArgumentProcessor):
    def __call__(self, config, all_args):
        if not isinstance(config, Config):
            self.err("expected a Config object")
        return config

class ConfigFieldA(ArgumentProcessor):
    def __init__(self, config_arg_name='config'):
        self.cfg_arg = config_arg_name

    def __call__(self, field, all_args):
        config = all_args[self.cfg_arg]
        if not is_valid_name(field):
            self.err("expected a valid name string")
        elif not config.has_field(field):
            self.err(f"expected '{field}' to be a field "
                     f"of config '{config.name()}'", ValueError)
        return field

class NameA(ArgumentProcessor):
    def __call__(self, name, all_args):
        if not is_valid_name(name):
            self.err("expected a valid name")
        return name

class PosIntA(ArgumentProcessor):
    def __call__(self, val, all_args):
        if not is_pos_int(val):
            self.err("expected a positive integer")
        return val

class BoolA(ArgumentProcessor):
    def __call__(self, bval, all_args):
        if not isinstance(bval, bool):
            self.err("expected a bool")
        return bval

class OptionalA(ArgumentProcessor):
    def __init__(self, arg_proc):
        if is_subclass_obj(arg_proc, ArgumentProcessor):
            arg_proc = arg_proc()
        self.arg_proc = arg_proc

    def setdata(self, i, arg_name, f_name):
        super().setdata(i, arg_name, f_name)
        self.arg_proc.setdata(i, arg_name, f_name)

    def __call__(self, opt_arg, all_args):
        if opt_arg is None:
            return opt_arg
        else:
            return self.arg_proc(opt_arg, all_args)

class ListA(ArgumentProcessor):
    def __init__(self, elem_arg_proc, list_only=False, length=None):
        if is_subclass_obj(elem_arg_proc, ArgumentProcessor):
            elem_arg_proc = elem_arg_proc()
        self.elem_arg_proc  = elem_arg_proc
        self.list_only      = list_only
        self.fixed_length   = length

    def setdata(self, i, arg_name, f_name):
        super().setdata(i, arg_name, f_name)
        self.elem_arg_proc.setdata(i, arg_name, f_name)

    def __call__(self, xs, all_args):
        if self.list_only:
            if not isinstance(xs, list):
                self.err("expected a list")
        else:
            if not isinstance(xs, (list,tuple)):
                self.err("expected a list or tuple")
        if self.fixed_length:
            if len(xs) != self.fixed_length:
                self.err(f"expected a list of length {self.fixed_length}")
        # otherwise, check the entries
        xs = [ self.elem_arg_proc(x, all_args) for x in xs ]
        return xs

class ListOrElemA(ListA):
    def __call__(self, xs, all_args):
        arg_typ = list if self.list_only else (list,tuple)
        if isinstance(xs, arg_typ):
            return super().__call__(xs)
        else:
            return [ self.elem_arg_proc(xs, all_args) ]

class InstrStrA(ArgumentProcessor):
    def __call__(self, instr, all_args):
        if not isinstance(instr, str):
            self.err("expected an instruction macro "
                     "(i.e. a string with {} escapes)")
        return instr

_name_count_re = r"^([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$"
class NameCountA(ArgumentProcessor):
    def __call__(self, name_count, all_args):
        if not isinstance(name_count, str):
            self.err("expected a string")
        results = re.search(_name_count_re, name_count)
        if not results:
            self.err("expected a name pattern of the form\n"
                     "  <ident> [# <int>]?\n"
                     "where <ident> is the name of a variable "
                     "and <int> specifies which occurrence. "
                     "(e.g. 'x #2' means 'the second occurence of x')",
                     ValueError)

        name    = results[1]
        count   = int(results[3]) if results[3] else None
        return (name,count)

class EnumA(ArgumentProcessor):
    def __init__(self, enum_vals):
        assert isinstance(enum_vals, list)
        self.enum_vals = enum_vals

    def __call__(self, arg, all_args):
        if arg not in self.enum_vals:
            vals_str = ', '.join([str(v) for v in self.enum_vals])
            self.err(f"expected one of the following values: {vals_str}",
                     ValueError)
        return arg

class TypeAbbrevA(ArgumentProcessor):   
    _shorthand = {
        'R':    T.R,
        'f32':  T.f32,
        'f64':  T.f64,
        'i8':   T.int8,
        'i32':  T.int32,
    }
    def __call__(self, typ, all_args):
        if typ in TypeAbbrevA._shorthand:
            return TypeAbbrevA._shorthand[typ]
        else:
            precisions = ", ".join([ t for t in TypeAbbrevA._shorthand ])
            self.err(f"expected one of the following strings specifying "
                     f"precision: {precisions}", ValueError)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Cursor Argument Processing

class ExprCursorA(ArgumentProcessor):
    def __init__(self, many=False):
        self.match_many = many

    def __call__(self, expr_pattern, all_args):
        if isinstance(expr_pattern, PC.ExprCursor):
            return expr_pattern
        elif isinstance(expr_pattern, PC.Cursor):
            self.err(f"expected an ExprCursor, not {type(expr_pattern)}")
        elif not isinstance(expr_pattern, str):
            self.err("expected an ExprCursor or pattern string")

        proc    = all_args["proc"]
        # TODO: Remove all need for `call_depth`
        matches = proc.find(expr_pattern, many=self.match_many)

        match   = matches[0] if self.match_many else matches
        if not isinstance(match, PC.ExprCursor):
            self.err(f"expected pattern to match an ExprCursor, "
                     f"not {type(match)}")

        return match

class StmtCursorA(ArgumentProcessor):
    def __init__(self, many=False):
        self.match_many = many

    def __call__(self, stmt_pattern, all_args):
        if isinstance(stmt_pattern, PC.StmtCursor):
            return stmt_pattern
        elif isinstance(stmt_pattern, PC.Cursor):
            self.err(f"expected an StmtCursor, not {type(stmt_pattern)}")
        elif not isinstance(stmt_pattern, str):
            self.err("expected a StmtCursor or pattern string")

        proc    = all_args["proc"]
        # TODO: Remove all need for `call_depth`
        matches = proc.find(stmt_pattern, many=self.match_many)

        match   = matches[0] if self.match_many else matches
        if not isinstance(match, PC.StmtCursor):
            self.err(f"expected pattern to match a StmtCursor, "
                     f"not {type(match)}")

        return match

class BlockCursorA(ArgumentProcessor):
    def __init__(self, many=False):
        self.match_many = many

    def __call__(self, block_pattern, all_args):
        if isinstance(block_pattern, PC.BlockCursor):
            return block_pattern
        elif isinstance(block_pattern, PC.StmtCursor):
            return block_pattern.as_block()
        elif isinstance(block_pattern, PC.Cursor):
            self.err(f"expected a StmtCursor or BlockCursor, "
                     f"not {type(block_pattern)}")
        elif not isinstance(block_pattern, str):
            self.err("expected a Cursor or pattern string")

        proc    = all_args["proc"]
        # TODO: Remove all need for `call_depth`
        matches = proc.find(block_pattern, many=self.match_many)

        match   = matches[0] if self.match_many else matches
        if isinstance(match, PC.StmtCursor):
            match = match.as_block()
        elif not isinstance(match, PC.BlockCursor):
            self.err(f"expected pattern to match a BlockCursor, "
                     f"not {type(match)}")

        return match

class GapCursorA(ArgumentProcessor):
    def __call__(self, gap_cursor, all_args):
        if not isinstance(gap_cursor, PC.GapCursor):
            self.err("expected a GapCursor")
        return gap_cursor


class AllocCursorA(StmtCursorA):
    def __call__(self, alloc_pattern, all_args):
        cursor = super().__call__(alloc_pattern, all_args)
        if not isinstance(cursor, PC.AllocCursor):
            self.err(f"expected an AllocCursor, not {type(cursor)}")
        return cursor

class WindowStmtCursorA(StmtCursorA):
    def __call__(self, alloc_pattern, all_args):
        cursor = super().__call__(alloc_pattern, all_args)
        if not isinstance(cursor, PC.WindowStmtCursor):
            self.err(f"expected a WindowStmtCursor, not {type(cursor)}")
        return cursor

class ForSeqOrIfCursorA(StmtCursorA):
    def __call__(self, cursor_pat, all_args):
        # TODO: eliminate this redundancy with the ForSeqCursorA code
        # allow for a special pattern short-hand, but otherwise
        # handle as expected for a normal statement cursor
        try:
            name, count     = NameCountA()(cursor_pat, all_args)
            count           = f"#{count}" if count is not None else ""
            cursor_pat      = f"for {name} in _: _{count}"
        except:
            pass

        cursor = super().__call__(cursor_pat, all_args)
        if not isinstance(cursor, (PC.ForSeqCursor,PC.IfCursor)):
            self.err(f"expected a ForSeqCursor or IfCursor, "
                     f"not {type(cursor)}")
        return cursor


class ForSeqCursorA(StmtCursorA):
    def __call__(self, loop_pattern, all_args):
        # allow for a special pattern short-hand, but otherwise
        # handle as expected for a normal statement cursor
        try:
            name, count     = NameCountA()(loop_pattern, all_args)
            count           = f"#{count}" if count is not None else ""
            loop_pattern    = f"for {name} in _: _{count}"
        except:
            pass

        cursor = super().__call__(loop_pattern, all_args)
        if not isinstance(cursor, PC.ForSeqCursor):
            self.err(f"expected a ForSeqCursor, not {type(cursor)}")
        return cursor


_name_name_count_re = r"^([a-zA-Z_]\w*)\s*([a-zA-Z_]\w*)\s*(\#\s*([0-9]+))?$"
class NestedForSeqCursorA(StmtCursorA):
    def __call__(self, loops_pattern, all_args):

        if isinstance(loops_pattern, PC.ForSeqCursor):
            if (len(loops_pattern.body()) != 1 or
                not isinstance(loops_pattern.body()[0], PC.ForSeqCursor)):
                self.err(f"expected the body of the outer loop "
                         f"to be a single loop, but it was a "
                         f"{loops_pattern.body()[0]}",
                         ValueError)
            cursor = loops_pattern
        elif isinstance(loops_pattern, PC.Cursor):
            self.err(f"expected a ForSeqCursor, not {type(loops_pattern)}")
        elif (isinstance(loops_pattern, str) and
              (match_result := re.search(_name_name_count_re,
                                         loops_pattern))):
            pass
            out_name    = match_result[1]
            in_name     = match_result[2]
            count       = f" #{match_result[3]}" if match_result[3] else ""
            pattern     = (f"for {out_name} in _:\n"
                           f"  for {in_name} in _: _{count}")
            cursor = super().__call__(pattern, all_args)
        else:
            self.err("expected a ForSeqCursor, pattern match string, "
                     "or 'outer_loop inner_loop' shorthand")

        return cursor


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# New Code Fragment Argument Processing


class NewExprA(ArgumentProcessor):
    def __init__(self, cursor_arg, before=True):
        self.cursor_arg = cursor_arg
        self.before     = before

    def __call__(self, expr_str, all_args):
        proc    = all_args['proc']
        cursor  = all_args[self.cursor_arg]
        if isinstance(expr_str, int):
            return LoopIR.Const(expr_str, T.int, null_srcinfo())
        elif isinstance(expr_str, float):
            return LoopIR.Const(expr_str, T.R, null_srcinfo())
        elif isinstance(expr_str, bool):
            return LoopIR.Const(expr_str, T.bool, null_srcinfo())
        elif not isinstance(expr_str, str):
            self.err("expected a string")

        # if we don't have a gap cursor, convert to a gap cursor
        if not isinstance(cursor, PC.GapCursor):
            cursor = cursor.before() if self.before else cursor.after()

        # resolve gaps down to statements in a somewhat silly way
        # TODO: improve parse_fragment to just take gaps
        if not (stmtc := cursor.after()):
            assert (stmtc := cursor.before())
        ctxt_stmt = stmtc._impl._node()

        expr = parse_fragment(proc._loopir_proc, expr_str, ctxt_stmt)

        return expr


# --------------------------------------------------------------------------- #
#  - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#                       Atomic Scheduling Operations
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#  - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * - * -
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Basic Operations

@sched_op([])
def simplify(proc):
    """
    Simplify the code in the procedure body. Tries to reduce expressions
    to constants and eliminate dead branches and loops. Uses branch
    conditions to simplify expressions inside the branches.
    """
    p = proc._loopir_proc
    p = Schedules.DoSimplify(p).result()
    return Procedure(p, _provenance_eq_Procedure=proc)

@sched_op([NameA])
def rename(proc, name):
    """
    Rename the procedure. Affects generated symbol names.

    args:
        name    - string
    """
    p = proc._loopir_proc
    p = p.update(name=name)
    return Procedure(p, _provenance_eq_Procedure=proc)

@sched_op([InstrStrA])
def make_instr(proc, instr):
    """
    Turn this procedure into an "instruction" using the provided macro-string

    args:
        name    - string representing an instruction macro
    """
    p = proc._loopir_proc
    p = p.update(instr=instr)
    return Procedure(p, _provenance_eq_Procedure=proc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Sub-procedure Operations

@sched_op([NameA, StmtCursorA])
def extract_subproc(proc, subproc_name, body_stmt):
    """
    Documentation TODO
    """
    loopir          = proc._loopir_proc
    stmt            = body_stmt._impl._node()
    passobj         = Schedules.DoExtractMethod(loopir, subproc_name, stmt)
    loopir, subproc = passobj.result(), passobj.subproc()
    return ( Procedure(loopir, _provenance_eq_Procedure=proc),
             Procedure(subproc) )


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Precision, Memory and Window Setting Operations

@sched_op([NameCountA, TypeAbbrevA])
def set_precision(proc, name, typ):
    """
    Set the precision annotation on a given buffer to the provided
    base-type precision.

    args:
        name    - string w/ optional count, e.g. "x" or "x #3"
        typ     - string representing base data type

    rewrite:
        `name : _[...]    ->    name : typ[...]`
    """
    name, count = name
    loopir  = proc._loopir_proc
    loopir  = Schedules.SetTypAndMem(loopir, name, count,
                                     basetyp=typ).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([NameCountA, BoolA])
def set_window(proc, name, is_window=True):
    """
    Set the annotation on a given buffer to indicate that it should be
    a window (True) or should not be a window (False)

    args:
        name        - string w/ optional count, e.g. "x" or "x #3"
        is_window   - boolean representing whether a buffer is a window

    rewrite when is_window = True:
        `name : R[...]    ->    name : [R][...]`
    """
    name, count = name
    loopir = proc._loopir_proc
    loopir = Schedules.SetTypAndMem(loopir, name, count,
                                    win=is_window).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([NameCountA, MemoryA])
def set_memory(proc, name, memory_type):
    """
    Set the memory annotation on a given buffer to the provided memory.

    args:
        name    - string w/ optional count, e.g. "x" or "x #3"
        mem     - new Memory object

    rewrite:
        `name : _ @ _    ->    name : _ @ mem`
    """
    name, count = name
    loopir = proc._loopir_proc
    loopir = Schedules.SetTypAndMem(loopir, name, count,
                                    mem=memory_type).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Configuration Modifying Operations

@sched_op([ExprCursorA, ConfigA, ConfigFieldA])
def bind_config(proc, var_cursor, config, field):
    """
    extract a control-value expression and write it into some
    designated field of a config

    args:
        var_cursor  - cursor or pattern pointing at the expression to
                      be bound
        config      - config object to be written into
        field       - (string) the field of `config` to be written to

    rewrite:
        Let `s[ e ]` mean a statement with control expression `e` occurring
        within it.  Then,
        `s[ e ]    ->    config.field = e ; s[ config.field ]`
    """
    e               = var_cursor._impl._node()
    cfg_f_type      = config.lookup(field)[1]
    if not isinstance(e, LoopIR.Read):
        raise ValueError("expected a cursor to a single variable Read")
    elif e.type != cfg_f_type:
        raise ValueError(f"expected type of expression to bind ({e.type}) "
                         f"to match type of Config variable ({cfg_f_type})")

    loopir          = proc._loopir_proc
    rewrite_pass    = Schedules.DoBindConfig(loopir, config, field, e)
    mod_config      = rewrite_pass.mod_eq()
    loopir          = rewrite_pass.result()

    return Procedure(loopir, _provenance_eq_Procedure=proc,
                             _mod_config=mod_config)

@sched_op([StmtCursorA])
def delete_config(proc, stmt_cursor):
    """
    delete a statement that writes to some config.field

    args:
        stmt_cursor - cursor or pattern pointing at the statement to
                      be deleted

    rewrite:
        `s1 ; config.field = _ ; s3    ->    s1 ; s3`
    """
    stmt            = stmt_cursor._impl._node()
    loopir          = proc._loopir_proc
    rewrite_pass    = Schedules.DoDeleteConfig(loopir, stmt)
    mod_config      = rewrite_pass.mod_eq()
    loopir          = rewrite_pass.result()

    return Procedure(loopir, _provenance_eq_Procedure=proc,
                             _mod_config=mod_config)


@sched_op([GapCursorA, ConfigA, ConfigFieldA, NewExprA('gap_cursor')])
def write_config(proc, gap_cursor, config, field, rhs):
    """
    insert a statement that writes a desired value to some config.field

    args:
        gap_cursor  - cursor pointing to where the new write statement
                      should be inserted
        config      - config object to be written into
        field       - (string) the field of `config` to be written to
        rhs         - (string) the expression to write into the field

    rewrite:
        `s1 ; s3    ->    s1 ; config.field = new_expr ; s3`
    """

    # TODO: just have scheduling pass take a gap cursor directly
    before = True
    if not (stmtc := gap_cursor.after()):
        assert (stmtc := gap_cursor.before())
        before = False
    stmt = stmtc._impl._node()

    loopir          = proc._loopir_proc
    rewrite_pass    = Schedules.DoConfigWrite(loopir, stmt,
                                              config, field, rhs,
                                              before=before)
    mod_config      = rewrite_pass.mod_eq()
    loopir          = rewrite_pass.result()

    return Procedure(loopir, _provenance_eq_Procedure=proc,
                             _mod_config=mod_config)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Memory and Windowing-oriented Operations

@sched_op([AllocCursorA, NewExprA('buf_cursor'),
                         NewExprA('buf_cursor'), BoolA])
def expand_dim(proc, buf_cursor, alloc_dim, indexing_expr,
                                 unsafe_disable_checks=False):
    """
    expand the number of dimensions of a buffer variable (`buf_cursor`).
    After expansion, the existing code will initially only use particular
    entries of the new dimension, chosen by the provided `indexing_expr`

    args:
        buf_cursor      - cursor pointing to the Alloc to expand
        alloc_dim       - (string) an expression for the size
                          of the new buffer dimension.
        indexing_expr   - (string) an expression to index the newly
                          created dimension with.

    rewrite:
        `x : T[...] ; s`
          ->
        `x : T[..., alloc_dim] ; s[ x[...] -> x[..., indexing_expr] ]`
    checks:
        The provided dimension size is checked for positivity and the
        provided indexing expression is checked to make sure it is in-bounds
    """
    loopir  = proc._loopir_proc
    stmt    = buf_cursor._impl._node()
    loopir  = Schedules.DoExpandDim(loopir, stmt, alloc_dim,
                                                  indexing_expr).result()
    if not unsafe_disable_checks:
        CheckEffects(loopir)

    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([AllocCursorA, PosIntA])
def lift_alloc(proc, alloc_cursor, n_lifts=1):
    """
    Lift a buffer allocation up and out of various Loops / If-statements.

    args:
        alloc_cursor    - cursor to the allocation to lift up
        n_lifts         - number of times to try to move the allocation up

    rewrite:
        `for i in _:`
        `    buf : T` <- alloc_cursor
        `    ...`
        ->
        `buf : T`
        `for i in _:`
        `    ...`
    """
    loopir  = proc._loopir_proc
    stmt    = alloc_cursor._impl._node()
    loopir  = Schedules.DoLiftAllocSimple(loopir, stmt, n_lifts).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([AllocCursorA, PosIntA,
           EnumA(['row','col']), OptionalA(PosIntA), BoolA])
def autolift_alloc(proc, alloc_cursor, n_lifts=1,
                   mode='row', size=None, keep_dims=False):
    """
    Lift a buffer allocation up and out of various Loops / If-statements.

    Has some additional special legacy behavior.  Use lift_alloc instead for
    all new code.

    args:
        alloc_cursor    - cursor to the allocation to lift up
        n_lifts         - number of times to try to move the allocation up
        mode            - whether to expand the buffer's dimensions
                          on the inner or outer position
        size            - dimension extents to expand to?
        keep_dims       - ???

    rewrite:
        `for i in _:`
        `    buf : T` <- alloc_cursor
        `    ...`
        ->
        `buf : T`
        `for i in _:`
        `    ...`
    """
    loopir  = proc._loopir_proc
    stmt    = alloc_cursor._impl._node()
    loopir  = Schedules.DoLiftAlloc(loopir, stmt, n_lifts,
                                    mode, size, keep_dims).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([AllocCursorA, AllocCursorA])
def reuse_buffer(proc, buf_cursor, replace_cursor):
    """
    reuse existing buffer (`buf_cursor`) instead of
    allocating a new buffer (`replace_cursor`).

    Old Name: data_reuse

    args:
        buf_cursor      - cursor pointing to the Alloc to reuse
        replace_cursor  - cursor pointing to the Alloc to eliminate

    rewrite:
        `x : T ; ... ; y : T ; s`
          ->
        `x : T ; ... ; s[ y -> x ]`
    checks:
        Can only be performed if the variable `x` is dead at the statement
        `y : T`.
    """
    buf_s   = buf_cursor._impl._node()
    rep_s   = replace_cursor._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoDataReuse(loopir, buf_s, rep_s).result()

    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([WindowStmtCursorA])
def inline_window(proc, winstmt_cursor):
    """
    Eliminate use of a window by inlining its definition and expanding
    it at all use-sites

    args:
        winstmt_cursor  - cursor pointing to the WindowStmt to inline

    rewrite:
        `y = x[...] ; s` -> `s[ y -> x[...] ]
    """
    stmt    = winstmt_cursor._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoInlineWindow(loopir, stmt).result()

    return Procedure(loopir, _provenance_eq_Procedure=proc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop Rewriting

@sched_op([ForSeqCursorA, PosIntA, ListA(NameA, length=2),
           EnumA(['cut','guard','cut_and_guard']), BoolA])
def divide_loop(proc, loop_cursor, div_const, new_iters,
                                   tail='guard', perfect=False):
    """
    Divide a loop into an outer and inner loop, where the inner loop
    iterates over the range 0 to `div_const`.

    Old Name: In Halide and TVM, this was called "split"

    args:
        loop_cursor     - cursor pointing to the loop to split ;
                          can also be specified using the special shorthands
                          pattern: <loop-iterator-name>
                               or: <loop-iterator-name> #<int>
        div_const       - integer > 1 specifying what to "divide by"
        new_iters       - list or tuple of two strings specifying the new
                          outer and inner iteration variable names
        tail (opt)      - specifies the strategy for handling the "remainder"
                          of the loop division (called the tail of the loop).
                          value can be "cut", "guard", or "cut_and_guard".
                          Default value: "guard"
        perfect (opt)   - Boolean (default False) that can be set to true
                          to assert that you know the remainder will always
                          be zero (i.e. there is no tail).  You will get an
                          error if the compiler cannot verify this fact itself.

    rewrite:
        divide(..., div_const=q, new_iters=['hi','lo'], tail='cut')
        `for i in seq(0,e):`
        `    s`
            ->
        `for hi in seq(0,e / q):`
        `    for lo in seq(0, q):`
        `        s[ i -> q*hi + lo ]`
        `for lo in seq(0,e - q * (e / q)):`
        `    s[ i -> q * (e / q) + lo ]
    """
    if div_const == 1:
        raise TypeError("why are you trying to split by 1?")

    stmt    = loop_cursor._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoSplit(loopir, stmt, quot=div_const,
                                hi=new_iters[0], lo=new_iters[1],
                                tail=tail,
                                perfect=perfect).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([NestedForSeqCursorA])
def reorder_loops(proc, nested_loops):
    """
    Reorder two loops that are directly nested with each other.
    This is the primitive loop reordering operation, out of which
    other reordering operations can be built.

    args:
        nested_loops    - cursor pointing to the outer loop of the
                          two loops to reorder; a pattern to find said
                          cursor with; or a 'name name' shorthand where
                          the first name is the iteration variable of the
                          outer loop and the second name is the iteration
                          variable of the inner loop.  An optional '#int'
                          can be added to the end of this shorthand to
                          specify which match you want,

    rewrite:
        `for outer in _:`
        `    for inner in _:`
        `        s`
            ->
        `for inner in _:`
        `    for outer in _:`
        `        s`
    """

    stmt    = nested_loops._impl._node()
    loopir  = Schedules.DoReorder(proc._loopir_proc, stmt).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([GapCursorA, PosIntA])
def fission(proc, gap_cursor, n_lifts=1):
    """
    fission apart the ForSeq and If statements wrapped around
    this block of statements into two copies; the first containing all
    statements before the cursor, and the second all statements after the
    cursor.

    args:
        gap_cursor          - a cursor pointing to the point in the
                              statement block that we want to fission at.
        n_lifts (optional)  - number of levels to fission upwards (default=1)

    rewrite:
        `for i in _:`
        `    s1`
        `      ` <- gap
        `    s2`
            ->
        `for i in _:`
        `    s1`
        `for i in _:`
        `    s2`
    """

    if not (stmtc := gap_cursor.before()) or not gap_cursor.after():
        raise ValueError("expected cursor to point to "
                         "a gap between statements")
    stmt    = stmtc._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoFissionAfterSimple(loopir, stmt, n_lifts).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([GapCursorA, PosIntA])
def autofission(proc, gap_cursor, n_lifts=1):
    """
    Split the enclosing ForSeq and If statements wrapped around
    this block of statements at the indicated point.

    If doing so splits a loop, this version of fission attempts
    to remove those loops as well.

    args:
        gap_cursor          - a cursor pointing to the point in the
                              statement block that we want to fission at.
        n_lifts (optional)  - number of levels to fission upwards (default=1)

    rewrite:
        `for i in _:`
        `    s1`
        `      ` <- gap
        `    s2`
            ->
        `for i in _:`
        `    s1`
        `for i in _:`
        `    s2`
    """

    if not (stmtc := gap_cursor.before()) or not gap_cursor.after():
        raise ValueError("expected cursor to point to "
                         "a gap between statements")
    stmt    = stmtc._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoFissionLoops(loopir, stmt, n_lifts).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([ForSeqOrIfCursorA, ForSeqOrIfCursorA])
def fusion(proc, stmt1, stmt2):
    """
    fuse together two loops or if-guards, provided that the loop bounds
    or guard conditions are compatible.

    args:
        stmt1, stmt2        - cursors to the two loops or if-statements
                              that are being fused

    rewrite:
        `for i in e:` <- stmt1
        `    s1`
        `for j in e:` <- stmt2
        `    s2`
            ->
        `for i in e:`
        `    s1`
        `    s2[ j -> i ]`
    or
        `if cond:` <- stmt1
        `    s1`
        `if cond:` <- stmt2
        `    s2`
            ->
        `if cond:`
        `    s1`
        `    s2`
    """
    if isinstance(stmt1, PC.IfCursor) != isinstance(stmt2, PC.IfCursor):
        raise ValueError("expected the two argument cursors to either both "
                         "point to loops or both point to if-guards")
    s1      = stmt1._impl._node()
    s2      = stmt2._impl._node()
    loopir  = proc._loopir_proc
    SCHED   = (Schedules.DoFuseIf if isinstance(stmt1, PC.IfCursor)
               else Schedules.DoFuseLoop)
    loopir  = SCHED(loopir, s1, s2,).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([ForSeqCursorA])
def remove_loop(proc, loop_cursor):
    """
    Remove the loop around some block of statements.
    This operation is allowable when the block of statements in question
    can be proven to be idempotent.

    args:
        loop_cursor     - cursor pointing to the loop to remove

    rewrite:
        `for i in _:`
        `    s`
            ->
        `s`
    """

    stmt    = loop_cursor._impl._node()
    loopir  = Schedules.DoRemoveLoop(proc._loopir_proc, stmt).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([BlockCursorA, NameA, NewExprA('block_cursor'), BoolA])
def add_loop(proc, block_cursor, iter_name, hi_expr, guard=False):
    """
    Add a loop around some block of statements.
    This operation is allowable when the block of statements in question
    can be proven to be idempotent.

    args:
        block_cursor    - cursor pointing to the block to wrap in a loop
        iter_name       - string name for the new iteration variable
        hi_expr         - string to be parsed into the upper bound expression
                          for the new loop
        guard           - Boolean (default False) signaling whether to
                          wrap the block in a `if iter_name == 0: block`
                          condition; in which case idempotency need not
                          be proven.

    rewrite:
        `s`  <--- block_cursor
        ->
        `for iter_name in hi_expr:`
        `    s`
    """

    if len(block_cursor) != 1:
        raise NotImplementedError("TODO: support blocks of size > 1")

    stmt    = block_cursor[0]._impl._node()
    loopir  = Schedules.DoAddLoop(proc._loopir_proc, stmt,
                                  iter_name, hi_expr, guard).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Guard Conditions

@sched_op([BlockCursorA, ListOrElemA(NewExprA('block_cursor'))])
def specialize(proc, block_cursor, conds):
    """
    Duplicate a statement block multiple times, with the provided
    `cond`itions indictaing when each copy should be invoked.
    Doing this allows one to then schedule differently the "specialized"
    variants of the blocks in different ways.

    If `n` conditions are given, then `n+1` specialized copies of the block
    are created (with the last copy as a "default" version).

    args:
        block_cursor    - cursor pointing to the block to duplicate/specialize
        conds           - list of strings or string to be parsed into
                          guard conditions for the 

    rewrite:
        `s`
            ->
        `if cond_0:`
        `    s`
        `elif cond_1:`
        `    s`
        ...
        `else:`
        `    s`
    """

    if len(block_cursor) != 1:
        raise NotImplementedError("TODO: support blocks of size > 1")

    stmt    = block_cursor[0]._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoSpecialize(loopir, stmt, conds).result()
    return Procedure(loopir, _provenance_eq_Procedure=proc)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Deprecated Operations

@sched_op([BlockCursorA, NewExprA('block_cursor')])
def add_unsafe_guard(proc, block_cursor, var_expr):
    """
    DEPRECATED
    This operation is deprecated, and will be removed soon.
    """
    stmt    = block_cursor._impl[0]._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoAddUnsafeGuard(loopir, stmt, var_expr).result()

    return Procedure(loopir, _provenance_eq_Procedure=proc)


@sched_op([StmtCursorA, StmtCursorA, PosIntA])
def double_fission(proc, stmt1, stmt2, n_lifts=1):
    """
    DEPRECATED
    This operation is deprecated, and will be removed soon.
    """
    s1      = stmt1._impl._node()
    s2      = stmt2._impl._node()
    loopir  = proc._loopir_proc
    loopir  = Schedules.DoDoubleFission(loopir, s1, s2, n_lifts).result()

    return Procedure(loopir, _provenance_eq_Procedure=proc)

@sched_op([ForSeqCursorA])
def bound_and_guard(self, loop):
    """
    DEPRECATED
    recommendation: replace with similar but more general primitive

    Replace
      for i in par(0, e): ...
    with
      for i in par(0, c):
        if i < e: ...
    where c is the tightest constant bound on e

    This currently only works when e is of the form x % n
    """
    stmt    = loop._impl._node()
    loopir  = Schedules.DoBoundAndGuard(self._loopir_proc, stmt).result()
    return Procedure(loopir, _provenance_eq_Procedure=self)



