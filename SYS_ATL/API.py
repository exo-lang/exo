from .prelude import *
from .LoopIR import UAST, LoopIR
from . import shared_types as T
from .typecheck import TypeChecker
from .LoopIR_compiler import Compiler, run_compile, compile_to_strings
from .LoopIR_interpreter import Interpreter, run_interpreter
from .LoopIR_scheduling import Schedules, name_str_2_symbols, name_str_2_pairs
from .effectcheck import InferEffects

from .pattern_match import match_pattern

from weakref import WeakKeyDictionary

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   proc provenance tracking

# every LoopIR.proc is either a root (not in this dictionary)
# or there is some other LoopIR.proc which is its root
_proc_root = WeakKeyDictionary()

def _proc_prov_eq(lhs, rhs):
    """ test whether two procs have the same provenance """
    lhs = lhs if lhs not in _proc_root else _proc_root[lhs]
    rhs = rhs if rhs not in _proc_root else _proc_root[rhs]
    assert lhs not in _proc_root and rhs not in _proc_root
    return lhs is rhs

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   iPython Display Object

class MarkDownBlob:
    def __init__(self, mkdwn_str):
        self.mstr = mkdwn_str

    def _repr_markdown_(self):
        return self.mstr

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   Procedure Objects
class Procedure:
    def __init__(self, proc, _testing=None, _provenance_eq_Procedure=None):
        if isinstance(proc, LoopIR.proc):
            self._loopir_proc = proc
        else:
            assert isinstance(proc, UAST.proc)

            self._uast_proc = proc
            if _testing != "UAST":
                self._loopir_proc = TypeChecker(proc).get_loopir()
                self._loopir_proc = InferEffects(self._loopir_proc).result()

        # find the root provenance
        parent = _provenance_eq_Procedure
        if parent is None:
            pass # this is a new root; done
        else:
            parent = parent._loopir_proc
            # if the provenance Procedure is not a root, find its root
            if parent in _proc_root:
                parent = _proc_root[parent]
            assert parent not in _proc_root
            # and then set this new proc's root
            _proc_root[self._loopir_proc] = parent

    def __str__(self):
        if hasattr(self,'_loopir_proc'):
            return str(self._loopir_proc)
        else:
            return str(self._uast_proc)

    def _repr_markdown_(self):
        return ("```python\n"+self.__str__()+"\n```")

    def INTERNAL_proc(self):
        return self._loopir_proc

    # -------------------------------- #
    #     introspection operations
    # -------------------------------- #

    def name(self):
        return self._loopir_proc.name

    def show_effect(self, stmt_pattern):
        stmt        = self._find_stmt(stmt_pattern)
        return str(stmt.eff)

    # ---------------------------------------------- #
    #     execution / interpretation operations
    # ---------------------------------------------- #

    def show_c_code(self):
        return MarkDownBlob("```c\n"+self.c_code_str()+"\n```")

    def c_code_str(self):
        decls, defns = compile_to_strings([self._loopir_proc])
        return defns

    def compile_c(self, directory, filename, malloc=False):
        run_compile([self._loopir_proc], directory,
                    (filename + ".c"), (filename + ".h"), malloc)

    def jit_compile(self):
        if not hasattr(self, '_cached_c_jit'):
            decls, defns = compile_to_strings([self._loopir_proc])
            self._cached_c_jit = CJit_Func(self._loopir_proc, defns)
        return self._cached_c_jit

    def interpret(self, **kwargs):
        run_interpreter(self._loopir_proc, kwargs)

    # ------------------------------- #
    #     scheduling operations
    # ------------------------------- #

    def rename(self, name):
        if not is_valid_name(name):
            raise TypeError(f"'{name}' is not a valid name")
        p = self._loopir_proc
        p = LoopIR.proc( name, p.args, p.body, p.srcinfo )
        return Procedure(p, _provenance_eq_Procedure=self)

    def split(self, split_var, split_const, out_vars, cut_tail=False):
        if type(split_var) is not str:
            raise TypeError("expected first arg to be a string")
        elif not is_pos_int(split_const):
            raise TypeError("expected second arg to be a positive integer")
        elif split_const == 1:
            raise TypeError("why are you trying to split by 1?")
        elif not isinstance(out_vars,list) and not isinstance(out_vars, tuple):
            raise TypeError("expected third arg to be a list or tuple")
        elif len(out_vars) != 2:
            raise TypeError("expected third arg list/tuple to have length 2")
        elif type(out_vars[0]) != str or type(out_vars[1]) != str:
            raise TypeError("expected third arg to be a list/tuple of two "+
                            "strings")

        split_names = name_str_2_symbols(self._loopir_proc, split_var)
        if len(split_names) == 0:
            raise TypeError(f"failed to find any symbols described by "+
                            f"'{split_var}'")
        loopir      = self._loopir_proc
        for nm in split_names:
            loopir  = Schedules.DoSplit(loopir, nm, quot=split_const,
                                        hi=out_vars[0], lo=out_vars[1],
                                        cut_tail=cut_tail).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def reorder(self, out_var, in_var):
        if type(out_var) is not str:
            raise TypeError("expected first arg to be a string")
        elif type(in_var) is not str:
            raise TypeError("expected second arg to be a string")

        reorder_pairs = name_str_2_pairs(self._loopir_proc, out_var, in_var)
        if len(reorder_pairs) == 0:
            raise TypeError(f"failed to find nested symbol pairs described "+
                            f"by '{out_var}' outside of '{in_var}'")
        loopir      = self._loopir_proc
        for out_v, in_v in reorder_pairs:
            loopir  = Schedules.DoReorder(loopir, out_v, in_v).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def unroll(self, unroll_var):
        if type(unroll_var) is not str:
            raise TypeError("expected first arg to be a string")

        unroll_names = name_str_2_symbols(self._loopir_proc, unroll_var)
        if len(unroll_names) == 0:
            raise TypeError(f"failed to find any symbols described by "+
                            f"'{unroll_var}'")
        loopir      = self._loopir_proc
        for nm in unroll_names:
            loopir  = Schedules.DoUnroll(loopir, nm).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def _find_stmt(self, stmt_pattern, call_depth=2):
        body        = self._loopir_proc.body
        stmt_lists  = match_pattern(body, stmt_pattern,
                                    call_depth=call_depth,
                                    default_match_no=0)
        if len(stmt_lists) == 0 or len(stmt_lists[0]) == 0:
            raise TypeError("failed to find statement")
        else:
            return stmt_lists[0][0]

    def _find_callsite(self, call_site_pattern):
        call_stmt   = self._find_stmt(call_site_pattern, call_depth=3)
        if type(call_stmt) is not LoopIR.Call:
            raise TypeError("pattern did not describe a call-site")

        return call_stmt

    def abstract(self, subproc, pattern):
        raise NotImplementedError("TODO: implement abstract")

    def inline(self, call_site_pattern):
        call_stmt   = self._find_callsite(call_site_pattern)

        loopir      = self._loopir_proc
        loopir      = Schedules.DoInline(loopir, call_stmt).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def call_eqv(self, other_Procedure, call_site_pattern):
        call_stmt   = self._find_callsite(call_site_pattern)

        old_proc    = call_stmt.f
        new_proc    = other_Procedure._loopir_proc
        if not _proc_prov_eq(old_proc, new_proc):
            raise TypeError("the procedures were not equivalent")

        loopir      = self._loopir_proc
        loopir      = Schedules.DoCallSwap(loopir, call_stmt,
                                                   new_proc).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def bind_expr(self, new_name, expr_pattern):
        if not is_valid_name(new_name):
            raise TypeError("expected first argument to be a valid name")
        body        = self._loopir_proc.body
        expr_list   = match_pattern(body, expr_pattern,
                                    call_depth=2, default_match_no=0)
        if len(expr_list) == 0:
            raise TypeError("failed to find expression")
        elif not isinstance(expr_list[0], LoopIR.expr):
            raise TypeError("pattern matched, but not to an expression")
        else:
            expr = expr_list[0]

        if not expr.type.is_numeric():
            raise TypeError("only numeric (not index or size) expressions "+
                            "can be targeted by bind_expr()")

        loopir      = self._loopir_proc
        loopir      = Schedules.DoBindExpr(loopir, new_name, expr).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def lift_alloc(self, alloc_site_pattern, n_lifts=1):
        if not is_pos_int(n_lifts):
            raise TypeError("expected second argument 'n_lifts' to be "+
                            "a positive integer")

        alloc_stmt  = self._find_stmt(alloc_site_pattern)
        if type(alloc_stmt) is not LoopIR.Alloc:
            raise TypeError("pattern did not describe an alloc statement")

        loopir      = self._loopir_proc
        loopir      = Schedules.DoLiftAlloc(loopir, alloc_stmt,
                                                    n_lifts).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def fission_after(self, stmt_pattern, n_lifts=1):
        if not is_pos_int(n_lifts):
            raise TypeError("expected second argument 'n_lifts' to be "+
                            "a positive integer")

        stmt        = self._find_stmt(stmt_pattern)

        loopir      = self._loopir_proc
        loopir      = Schedules.DoFissionLoops(loopir, stmt,
                                                       n_lifts).result()
        return Procedure(loopir, _provenance_eq_Procedure=self)

    def factor_out_stmt(self, name, stmt_pattern):
        if not is_valid_name(name):
            raise TypeError("expected first argument to be a valid name")
        stmt        = self._find_stmt(stmt_pattern)

        loopir          = self._loopir_proc
        passobj         = Schedules.DoFactorOut(loopir, name, stmt)
        loopir, subproc = passobj.result(), passobj.subproc()
        return ( Procedure(loopir, _provenance_eq_Procedure=self),
                 Procedure(subproc) )

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
#   JIT Compilation

import ctypes
import os
import sys
import time
import subprocess
import hashlib
import numpy as np

def _shell(cstr):
  subprocess.run(cstr, check=True, shell=True)

_HERE_DIR       = os.path.dirname(os.path.abspath(__file__))
_C_CACHE        = os.path.join(_HERE_DIR,'.sys_atl_c_cache')
if not os.path.isdir(_C_CACHE):
  os.mkdir(_C_CACHE)

def get_time(s):
  if not os.path.exists(s):
    return None
  else:
    return os.path.getmtime(s)

def comp_and_load_c_file(filename, c_code):
    hashstr     = hashlib.md5(c_code.encode('utf-8')).hexdigest()

    filename    += hashstr
    c_filename  = os.path.join(_C_CACHE, filename+".c")
    so_filename = os.path.join(_C_CACHE, filename+".so")
    comp_cmd    = (f"clang -Wall -Werror -fPIC -O3 -shared "+
                   f"-o {so_filename} {c_filename}")

    def matches_file(src, fname):
        if not os.path.isfile(fname):
            return False
        with open(fname, 'r', encoding ='utf-8') as F:
            return F.read() == src
    def write_file(src, fname):
        with open(fname, 'w', encoding ='utf-8') as F:
            F.write(src)

    # do we need to rebuild the corresponding Shared Object?
    if not matches_file(c_code, c_filename):
        write_file(c_code, c_filename)
        _shell(comp_cmd)

    # load the raw module
    module = ctypes.CDLL(so_filename)
    return module

# clean the cache when it exceeds 50MB;
# keep all files created in the last day or so
def clean_cache(size_trigger = int(50e6), clear_time_window = 86400.0):
  curr_time   = time.time()
  filenames   = [ os.path.join(_C_CACHE,f) for f in os.listdir(_C_CACHE) ]
  used_size   = sum(os.path.getsize(f) for f in filenames)

  if used_size > size_trigger:
    for file in filenames:
      mtime   = get_time(file)
      if curr_time - mtime > clear_time_window:
        os.remove(file)

# try out a single clean every time we load this module
clean_cache()

#       ----------------------------------       #

_F32_PTR = ctypes.POINTER(ctypes.c_float)

class CJit_Func:
    """ Manage JIT compilation of C code for convenience """

    def __init__(self, proc, c_code):
        self._proc      = proc

        # compile and load the c code into a dynamically linked module
        code_module     = comp_and_load_c_file(proc.name, c_code)
        self._module    = code_module

        # place some default ctypes type-checking around this...
        cfun            = getattr(self._module, proc.name)
        atyps           = []
        for fa in proc.args:
            if fa.type == T.size:
                atyps.append(ctypes.c_int)
            elif fa.type.is_numeric():
                atyps.append(_F32_PTR)
        cfun.argtypes   = atyps
        cfun.restype    = None
        self._c_func = cfun

    def __call__(self, *args, **kwargs):
        args = list(args)
        # first normalize calling convention
        argvals = []
        for fa in self._proc.args:
            if str(fa.name) in kwargs:
                argvals.append( kwargs.pop(str(fa.name)) )
            elif len(args) > 0:
                argvals.append( args.pop(0) )
            else:
                raise TypeError( "Call did not supply value for "+
                                f"argument '{fa.name}'")
        if len(args) > 0 or len(kwargs) > 0:
            raise TypeError("extraneous, unused arguments were supplied")

        # second, check those values against the procedure type signature
        size_env = dict()
        for fa, a in zip(self._proc.args, argvals):
            self._check_arg(fa, a, size_env)
            if fa.type == T.size:
                size_env[fa.name] = a

        # having succeeded then, perform the actual function call
        argvals = [ (a.ctypes.data_as(_F32_PTR)
                        if type(a) is np.ndarray else
                     a)
                    for a in argvals ]
        self._c_func(*argvals)


    def _check_arg(self, fa, a, env):
        def err(msg):
            raise TypeError(f"bad argument '{fa.name}': {msg}")
        if fa.type == T.size:
            if not is_pos_int(a):
                err(f"expected positive integer but got value '{a}' "+
                    f"of type '{type(a)}'")
        else:
            typ = fa.type
            if type(a) is not np.ndarray:
                err( "expected numpy.ndarray but got value of "+
                    f"type '{type(a)}'")
            elif a.dtype != np.float32:
                err( "expected numpy.ndarray of 32-bit floats but "+
                    f"buffer contains values of type '{a.dtype}'")

            if typ is T.R:
                if tuple(a.shape) != (1,):
                    err( "expected buffer of shape (1,) "+
                        f"but got shape {tuple(a.shape)}")
            else:
                shape = tuple(r if is_pos_int(r) else env[r]
                              for r in typ.shape())
                if shape != tuple(a.shape):
                    err(f"expected buffer of shape {shape} "+
                        f"but got shape {tuple(a.shape)}")



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
