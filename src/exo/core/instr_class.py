"""
Module for "new" class-based instr

    @instr
        class instr_name:
            def behavior(arg_a: Ta, arg_b: Tb, ...):
                # Exo code specifies instr behavior

            def instance(self, tparam...):
                # Python code configures instruction
                self.instr_format = ...

            # Each tparam (template parameter) in the instance()
            # must match a parameter in behavior(), and causes that parameter
            # to become a template parameter.

For context, the "old" instr is like

    @instr(instr_format)
    def instr_name(arg_a: Ta, arg_b: Tb, ...):
        # Exo code specifies instr behavior

"""

import ast as pyast
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple, Type

from .prelude import Sym

from .LoopIR import LoopIR, AccessInfo, InstrInfo, SubstArgs, Identifier
from .memory import DRAM
from ..frontend.pyparser import get_ast_from_python, Parser
from ..spork import actor_kinds
from ..spork.coll_algebra import standalone_thread


def proc_default_access_info(proc: LoopIR.proc):
    access_info = {}
    for arg in proc.args:
        if not arg.type.is_numeric():
            continue
        nm = arg.name.name()
        mem = DRAM if arg.mem is None else arg.mem
        access_info[nm] = AccessInfo(mem)
    return access_info


class InstrTemplateError(Exception):
    pass


def tparams_from_signature(clsname: str, tproc: LoopIR.proc, signature):
    assert isinstance(tproc, LoopIR.proc)

    tparam_syms = []
    tparam_types = []

    for i, param in enumerate(signature.parameters.values()):
        nm = param.name
        # Skip self
        if i == 0:
            assert nm == "self", f"{clsname}.instance: missing self"
            continue
        problem = None
        if param.kind.name != "POSITIONAL_OR_KEYWORD":
            problem = f"cannot be {param.kind.name} argument"
        elif param.default is not inspect._empty:
            problem = "cannot have default value"
        # Look for matching parameter in behavior() and get its Sym
        for tproc_a in tproc.args:
            if tproc_a.name.name() == nm:
                sym = tproc_a.name
                typ = tproc_a.type
                if not typ.is_indexable():
                    raise TypeError(
                        f"{clsname}.instance: parameter {nm} "
                        f"must refer to index type, not {typ}"
                    )
                break
        else:
            problem = f"does not refer to any parameter of {clsname}.behavior"

        if problem:
            raise ValueError(f"{clsname}.instance: parameter {nm} {problem}")
        tparam_syms.append(sym)
        tparam_types.append(typ)

    return tparam_syms, tparam_types


def prefill_instr_info(info: InstrInfo, proc: LoopIR.proc):
    info.c_global = ""
    info.cu_util = ""
    info.cu_includes = []
    info.coll_unit = standalone_thread
    info.actor_kind = actor_kinds.cpu
    info.access_info = proc_default_access_info(proc)
    info._formatted_tparam_kwargs = ""


def old_style_instr_info(proc: LoopIR.proc, c_instr: str, c_global: str):
    """InstrInfo from old-style @instr decorator"""
    assert isinstance(c_instr, str)
    assert isinstance(c_global, str)
    info = InstrInfo()
    prefill_instr_info(info, proc)
    info.instr_format = c_instr
    info.c_global = c_global
    return info


class InstrTemplate:
    """Templatized instruction -- call operator yields Procedure instr"""

    __slots__ = [
        "make_procedure",
        "tparam_syms",
        "tparam_types",
        "tproc",
        "info_cls",
        "cache",
    ]

    # Avoid circular modules: proc -> Procedure
    make_procedure: Callable[[object], "Procedure"]

    # Syms of tproc paremeters that are template parameters
    tparam_syms: List[Sym]

    # LoopIR types of template parameters
    tparam_types: List

    # "Template proc"; this is not an instr; this is directly parsed from
    # the user's cls.behavior Exo function.
    tproc: LoopIR.proc

    # Subtype of InstrInfo defined by the user.
    info_cls: Type[InstrInfo]

    # Cache of Procedures.
    # When we substitute template parameters, we cache the resulting Procedure
    # here indexed by a tuple of tparam values (same order as tparam_syms)
    cache: Dict[tuple, "Procedure"]

    def __init__(self, cls, make_procedure, parent_scope):
        nm = cls.__name__
        assert hasattr(cls, "behavior"), f"Missing {nm}.behavior"
        behavior_body, src_info = get_ast_from_python(cls.behavior)
        assert hasattr(cls, "instance"), f"Missing {nm}.instance"
        instance_signature = inspect.signature(cls.instance)

        parser = Parser(
            behavior_body, src_info, parent_scope=parent_scope, as_func=True
        )
        uast_tproc = parser.result().update(name=Identifier(nm))
        tproc = make_procedure(uast_tproc)._loopir_proc

        # Deduce the (Sym) names of tparams based on cls.instance
        tparam_syms, tparam_types = tparams_from_signature(
            nm, tproc, instance_signature
        )

        # The user's cls.instance function will be used to initialize InstrInfo.
        def info_init(info, **tparam_dict):
            prefill_instr_info(info, tproc)
            info.instance(**tparam_dict)
            self._postprocess_instr_info(tproc, info, tparam_dict)

            assert hasattr(info, "instr_format"), f"{nm}: missing instr_format"
            assert isinstance(info.instr_format, str), f"{nm}: missing instr_format"

        # The user-provided class gets converted to a subclass of InstrInfo.
        # Override __init__, and add __slots__ if user didn't.
        # I strongly believe in the typo-checking provided by __slots__.
        info_dict = dict(cls.__dict__)
        info_bases = [b for b in cls.__bases__ if b is not object]
        if not issubclass(cls, InstrInfo):
            info_bases.append(InstrInfo)
        if "__slots__" not in info_dict:
            info_dict["__slots__"] = []
        info_dict["__init__"] = info_init
        info_cls = type(nm, tuple(info_bases), info_dict)

        self.make_procedure = make_procedure
        self.tparam_syms = tparam_syms
        self.tparam_types = tparam_types
        self.tproc = tproc
        self.info_cls = info_cls
        self.cache = {}

    def __call__(self, **tparam_dict):
        # Try to get cached result
        tparam_values = self._tparam_values(**tparam_dict)
        procedure = self.cache.get(tparam_values)
        if procedure is not None:
            return procedure

        # Generate InstrInfo for this instanced instruction
        try:
            clsname = self.info_cls.__name__
            instr_info = self.info_cls(**tparam_dict)
        except AssertionError:
            # Avoid common Python error: using asserts to validate stuff
            # that still needs to be checked in release builds...
            raise
        except Exception as e:
            kwargs_str = self._format_tparam_kwargs(tparam_values)
            raise InstrTemplateError(
                f"Failed to instantiate {clsname}({kwargs_str}): {e}"
            ) from e

        # Convert template proc (tproc) to instanced proc (iproc) by
        #   * Substituting concrete values in place of template params (tparams)
        #   * Removing fnargs that correspond to tparams
        #   * Adding the InstrInfo; set fnarg.mem as needed from InstrInfo
        # TODO
        tproc = self.tproc
        binding = {}
        assert len(self.tparam_syms) == len(tparam_values)
        assert len(self.tparam_types) == len(tparam_values)
        for sym, v, typ in zip(self.tparam_syms, tparam_values, self.tparam_types):
            binding[sym] = LoopIR.Const(v, typ, tproc.srcinfo)
        iproc_body = SubstArgs(tproc.body, binding).result()
        iproc_args = [a for a in tproc.args if a.name not in self.tparam_syms]
        assert len(iproc_args) + len(self.tparam_syms) == len(tproc.args)
        for i, a in enumerate(iproc_args):
            iproc_args[i] = a.update(mem=instr_info.access_info[str(a.name)].mem)
        iproc_args = SubstArgs(iproc_args, binding).result()
        iproc = LoopIR.proc(
            tproc.name, iproc_args, tproc.preds, iproc_body, instr_info, tproc.srcinfo
        )

        # Build and save Procedure in cache.
        procedure = self.make_procedure(iproc)
        self.cache[tparam_values] = procedure
        return procedure

    def _loopir_proc(self, **tparam_dict):
        return self(**tparam_dict)._loopir_proc

    def _tparam_values(self, **tparam_dict):
        clsname = self.info_cls.__name__
        syms = self.tparam_syms
        tparam_values = []
        for sym in syms:
            assert isinstance(sym, Sym)
            nm = sym.name()
            v = tparam_dict.get(nm)
            if isinstance(v, int):
                tparam_values.append(v)
            elif v is None:
                raise InstrTemplateError(f"{clsname}: missing template parameter {nm}")
            else:
                raise InstrTemplateError(f"{clsname}: {nm} must be int, not {type(v)}")
        # Do this assert late as the "missing parameter"
        # message above has better clarity.
        assert len(tparam_dict) == len(syms), f"{clsname}: excess arguments"
        return tuple(tparam_values)

    def _format_tparam_kwargs(self, tparam_values):
        assert len(tparam_values) == len(self.tparam_syms)
        return ", ".join(f"{nm}={v}" for nm, v in zip(self.tparam_syms, tparam_values))

    def _postprocess_instr_info(self, proc: LoopIR.proc, info: InstrInfo, tparam_dict):
        actor_kind = info.actor_kind
        assert isinstance(actor_kind, actor_kinds.ActorKind)
        access_info = info.access_info
        clsname = self.info_cls.__name__

        for arg in proc.args:
            if not arg.type.is_numeric():
                continue
            nm = arg.name.name()
            arg_info = access_info[nm]
            if arg.mem is not None:
                assert (
                    arg.mem == arg_info.mem
                ), f"{clsname}: cannot override mem for {nm} @ {arg.mem}"

            signature = arg_info.actor_signature
            assert (
                signature in actor_kind.signatures
            ), f"{clsname}: cannot access {nm} with actor signature {signature} for actor kind {actor_kind}"

        info._formatted_tparam_kwargs = self._format_tparam_kwargs(
            self._tparam_values(**tparam_dict)
        )
