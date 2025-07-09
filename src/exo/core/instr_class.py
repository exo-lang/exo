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

            def codegen(self, args: InstrArgs) -> List[str]:
                # Each non-template param x in behavior becomes args.x of type
                # InstrWindowArg or InstrNonWindowArg. Template params will
                # be kept as their literal Python types (usually int).
                #
                # Return list of C lines
                # This is optional if you define self.instr_format

For context, the "old" instr is like

    @instr(instr_format[0])
    def instr_name(arg_a: Ta, arg_b: Tb, ...):
        # Exo code specifies instr behavior

"""

import ast as pyast
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple, Type, Set

from .prelude import Sym, SrcInfo

from .instr_info import AccessInfo, InstrInfo
from .LoopIR import LoopIR, SubstArgs, Identifier, get_writes_of_stmts
from .memory import DRAM
from ..frontend.pyparser import get_ast_from_python, Parser
from ..spork import timelines
from ..spork.coll_algebra import standalone_thread, CollUnit
from ..spork.timelines import (
    Instr_tl,
    Usage_tl,
    Sync_tl,
    cpu_in_order_instr,
    cuda_in_order_instr,
)
from .c_window import WindowFeatures, UtilInjector, WindowIndexerResult


def proc_default_access_info(proc: LoopIR.proc, write_syms: Set[Sym]):
    access_info = {}
    for arg in proc.args:
        if not arg.type.is_numeric():
            continue
        nm = arg.name.name()
        mem = DRAM if arg.mem is None else arg.mem
        access = AccessInfo()
        access.mem = mem
        access.const = arg.name not in write_syms
        access_info[nm] = access

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
    write_syms = set(x for x, _ in get_writes_of_stmts(proc.body))
    info.instr_format = None
    info.c_utils = []
    info.c_includes = []
    info.cu_utils = []
    info.cu_includes = []
    info.coll_unit = standalone_thread
    info.instr_tl = cpu_in_order_instr
    info.access_info = proc_default_access_info(proc, write_syms)
    info._tparam_dict = {}
    info._formatted_tparam_kwargs = ""


def old_style_instr_info(proc: LoopIR.proc, c_instr: str, c_global: str):
    """InstrInfo from old-style @instr decorator"""
    assert isinstance(c_instr, str)
    assert isinstance(c_global, str)
    info = OldStyleInstrInfo()
    prefill_instr_info(info, proc)
    info.instr_format = c_instr.split("\n")
    if c_global:
        info.c_utils.append(c_global)
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
        has_custom_codegen = hasattr(cls, "codegen")

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
            self._postprocess_instr_info(tproc, info, tparam_dict, has_custom_codegen)

        # The user-provided class gets converted to a subclass of InstrInfo.
        # Override __init__, and add __slots__ if user didn't.
        # I strongly believe in the typo-checking provided by __slots__.
        # Finally, add a fallback if no codegen callback was provided.
        info_dict = dict(cls.__dict__)
        info_bases = [b for b in cls.__bases__ if b is not object]
        if not issubclass(cls, InstrInfo):
            info_bases.append(InstrInfo)
        if "__slots__" not in info_dict:
            info_dict["__slots__"] = list(cls.__annotations__)
        info_dict["__init__"] = info_init
        if not has_custom_codegen:
            info_dict["codegen"] = OldStyleInstrInfo.codegen
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
        tproc = self.tproc
        binding = {}
        assert len(self.tparam_syms) == len(tparam_values)
        assert len(self.tparam_types) == len(tparam_values)
        for sym, v, typ in zip(self.tparam_syms, tparam_values, self.tparam_types):
            binding[sym] = LoopIR.Const(v, typ, tproc.srcinfo)
        iproc_preds = SubstArgs(tproc.preds, binding).result()
        iproc_body = SubstArgs(tproc.body, binding).result()
        iproc_args = [a for a in tproc.args if a.name not in self.tparam_syms]
        assert len(iproc_args) + len(self.tparam_syms) == len(tproc.args)
        for i, a in enumerate(iproc_args):
            if (access := instr_info.access_info.get(str(a.name))) is not None:
                iproc_args[i] = a.update(mem=access.mem)
        iproc_args = SubstArgs(iproc_args, binding).result()
        iproc = LoopIR.proc(
            tproc.name, iproc_args, iproc_preds, iproc_body, instr_info, tproc.srcinfo
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

    def _postprocess_instr_info(
        self, proc: LoopIR.proc, info: InstrInfo, tparam_dict, has_custom_codegen: bool
    ):
        clsname = self.info_cls.__name__
        has_instr_format = info.instr_format is not None
        if not has_custom_codegen:
            assert has_instr_format, f"{clsname}: missing instr_format or codegen()"
        if has_instr_format:
            assert isinstance(info.instr_format, list), clsname
            assert all(isinstance(line, str) for line in info.instr_format), clsname
        assert all(isinstance(s, str) for s in info.c_utils), clsname
        assert all(isinstance(s, str) for s in info.c_includes), clsname
        assert all(isinstance(s, str) for s in info.cu_utils), clsname
        assert all(isinstance(s, str) for s in info.cu_includes), clsname
        assert isinstance(info.coll_unit, CollUnit), clsname

        # instr_tl (L^i) must be Instr_tl typed
        instr_tl = info.instr_tl
        assert not isinstance(
            instr_tl, Sync_tl
        ), f"{clsname}: use {instr_tl}_instr, if it exists"
        assert isinstance(instr_tl, Instr_tl), clsname
        access_info = info.access_info

        for arg in proc.args:
            if not arg.type.is_numeric():
                continue
            nm = arg.name.name()
            arg_info = access_info[nm]
            if arg.mem is not None and arg.mem is not DRAM:
                assert (
                    arg.mem == arg_info.mem
                ), f"{clsname}: cannot override mem for {nm} @ {arg.mem.name()}"

            # Set usage_tl (L^u) if not explicitly given
            if not isinstance(arg_info.usage_tl, Usage_tl):
                assert arg_info.usage_tl is None
                try:
                    arg_info.usage_tl = arg_info.mem.default_usage_tl(instr_tl)
                except Exception as e:
                    raise ValueError(
                        f"{nm} @ {arg.mem.name()} needs explicit usage_tl"
                    ) from e
            usage_tl = arg_info.usage_tl

            # Set up ext_instr_tl (L_X^i) and ext_usage_tl (L_X^u) if not given.
            # Currently they are just {L^i} and {L^u}.
            # We may change this default later for certain Qual_tl(L^i, L^u).
            if not arg_info.ext_instr_tl:
                arg_info.ext_instr_tl = [instr_tl]
            assert all(isinstance(tl, Instr_tl) for tl in arg_info.ext_instr_tl)
            assert instr_tl in arg_info.ext_instr_tl

            if not arg_info.ext_usage_tl:
                arg_info.ext_usage_tl = [usage_tl]
            assert all(isinstance(tl, Usage_tl) for tl in arg_info.ext_usage_tl)
            assert usage_tl in arg_info.ext_usage_tl

            # Non-in-order instructions must set the OOO flag explicitly
            if instr_tl not in (cpu_in_order_instr, cuda_in_order_instr):
                assert (
                    arg_info.out_of_order is not None
                ), f"{clsname}: need out_of_order flag for {nm} @ {arg.mem.name()}"

        info._tparam_dict = tparam_dict
        info._formatted_tparam_kwargs = self._format_tparam_kwargs(
            self._tparam_values(**tparam_dict)
        )


@dataclass(slots=True)
class InstrWindowArg:
    _encoder_utils: UtilInjector
    _indexer_utils: UtilInjector
    _features: WindowFeatures
    _srcinfo: SrcInfo

    # The _special args are hacky: for @instr, we don't ever convert
    # Memory to a SpecialWindow, but we re-use this object in the
    # compiler to implement such conversions, to avoid code divergence.

    def __str__(self):
        return self.get_window()

    def get_window(self, _special=False) -> str:
        features = self._features
        encoder = features.get_encoder()

        # Check intact packed dimensions
        mem = features.get_mem()
        packed_tensor_shape = features.packed_tensor_shape()
        assert features.n_packed_dims() == len(packed_tensor_shape)
        for i, c in enumerate(packed_tensor_shape):
            features.get_packed_offset(i).exo_expect_int(0)
            sz = features.get_packed_interval_size(i)
            if sz is None:
                raise ValueError(
                    f"{features.get_raw_name()} must not have point expressions for packed dimensions (last {features.n_packed_dims()})"
                )
            sz.exo_expect_int(c)

        # Conditionally forbid dimensionality change
        can_change_dim = (
            encoder.supports_special_dim_change()
            if _special
            else encoder.supports_dim_change()
        )
        if not can_change_dim:
            if any(
                features.get_array_interval_size(i) is None
                for i in range(features.n_array_dims())
            ):
                raise ValueError(
                    f"{features.get_raw_name()} must not have point expressions for array dimensions"
                )

        do_encode = encoder.encode_special_window if _special else encoder.encode_window
        return str(do_encode(self._encoder_utils, features))

    def get_separate_dataptr(self, _special=False) -> str:
        features = self._features
        if _special:
            do_encode = features.get_encoder().encode_special_separate_dataptr
        else:
            do_encode = features.get_encoder().encode_separate_dataptr
        return str(do_encode(self._encoder_utils, features))

    def separate_dataptr(self):
        return self._features.separate_dataptr()

    def get_raw_name(self) -> str:
        return self._features.get_raw_name()

    def index(self, *idxs) -> str:
        new_features = self._features.new_window(
            idxs, [None] * len(idxs), self._srcinfo
        )
        indexed = self._features.get_indexer().index(
            self._indexer_utils, self._features
        )
        assert isinstance(indexed, WindowIndexerResult)
        return indexed.code

    def to_arg_strs(self):
        if self.separate_dataptr():
            return [self.get_separate_dataptr(), self.get_window()]
        else:
            return [self.get_window()]

    def to_strides_as_packed(self):
        return self._features.interval_array_strides_as_packed()

    def srcinfo(self):
        return self._srcinfo


@dataclass(slots=True)
class InstrNonWindowArg:
    # This could be expanded later...
    _code: str
    _srcinfo: SrcInfo

    def __str__(self):
        return self._code

    def separate_dataptr(self):
        return False

    def to_arg_strs(self):
        return [self._code]

    def srcinfo(self):
        return self._srcinfo


@dataclass(slots=True)
class InstrArgs:
    _exo_args_dict: Dict[str, InstrWindowArg | InstrNonWindowArg]

    def __init__(self, the_dict):
        self._exo_args_dict = the_dict

    def __getattr__(self, attr):
        assert not attr.startswith(
            "exo_"
        ), "exo_ prefix not allowed for arg name (or typo)"
        assert not attr.startswith(
            "_exo_"
        ), "_exo_ prefix not allowed for arg name (or typo)"
        return self._exo_args_dict[attr]

    def __iter__(self):
        return iter(self._exo_args_dict)

    def items(self):
        return self._exo_args_dict.items()


class OldStyleInstrInfo(InstrInfo):
    __slots__ = []

    def codegen(self, args: InstrArgs) -> List[str]:
        """Translate args to dictionary then use instr_format.format"""
        d = dict()
        for name, value in args.items():
            if isinstance(value, InstrWindowArg):
                mem = self.access_info[name].mem
                if mem.has_window_encoder():
                    d[name] = str(value)
                if mem.has_window_indexer():
                    d[name + "_data"] = value.index()
                d[name + "_int"] = value.get_raw_name()
            else:
                # Non-window; Exo 1 defines {name}_data; unclear why.
                s_value = str(value)
                d[name] = s_value
                d[name + "_data"] = s_value
        return [line.format(**d) for line in self.instr_format]
