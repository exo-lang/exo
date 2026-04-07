"""Module for "new" class-based instruction template (InstrTemplate)

Each instruction template is parameterized with

* Control value template parameters (int).
  These substitute concrete values for control parameters in the instr behavior.

* Precision template parameters (Exo ScalarInfo)
  These substitute for R (generic Num) in the type signature of runtime parameters.

* Extra template parameters, any user-defined object type.
  These may be used to customize the instr in freeform ways.

An InstrTemplate is callable with the template parameters as
keyword arguments, yielding a concrete instr Procedure object.

    @instr
    class instr_name(InstrInfo):
        def behavior(arg_a: Ta, arg_b: Tb, ...):
            # Exo code specifies instr behavior
            #
            # When the InstrTemplate is instantiated, any parameters
            # matching the name of a control value template parameter
            # is deleted; the remaining parameters are runtime parameters.
            #
            # Each parameter of base type R (generic Num) will cause
            # the InstrTemplate to take a precision template parameter
            # of the same name; this substitutes for R for the concrete instr.
            #
            # IMPORTANT: somewhat broken if defined in a base class, since
            # the parser runs ``at the wrong time'' (using locals in scope
            # when the subclass is defined, not the parent class).

        def instance(self, control_tparams..., *, extra_tparams...):
            # Python code configures instruction
            #
            # control_tparam are named control value template parameters
            # extra_tparams are named precision template parameters
            #
            # Precision parameters are not here, they may be accessed by
            # * self.access_info[<param_name>: str].scalar_info
            # * args.<param_name>.get_scalar_info()
            #
            self.instr_tl = ...

        # Each control_tparam name must match a parameter of behavior()
        # Each extra_tparam name must NOT match a parameter of behavior()
        # The precision template

        def codegen(self, args: InstrArgs) -> List[str]:
            # Each runtime param x in behavior becomes args.x of type
            # InstrWindowArg or InstrNonWindowArg, with precision
            # given by args.x.scalar_info.
            # Template parameters y (except for precision template parameters)
            # are kept as their literal Python types (often int) as args.y
            # XXX default arguments aren't correctly passed.
            #
            # Return list of C lines
            # codegen() is optional if you define self.instr_format

        valid_num_types: Set[Tuple[ScalarInfo]], or object with __contains__
        # This is not needed if no generic Num parameters exist.
        #
        # The tuple of substituted precision template parameters
        # (ordered by the order the corresponding runtime parameters in behavior())
        # must be in valid_num_types. e.g. an mma(accum, A, B) supporting
        # f32 accum and bf16/f16/f32 A and B can have
        # {(f32, bf16, bf16), (f32, f16, f16), (f32, f32, f32)}
        #
        # ScalarInfo objects are defined in the exo.scalars module.
        # Or use ScalarInfo.same() to accept any tuple of identical types.

For context, the "old" instr is like

    @instr(instr_format[0])
    def instr_name(arg_a: Ta, arg_b: Tb, ...):
        # Exo code specifies instr behavior

Within Exo object code, the syntax for calling an InstrTemplate
contains both positional and keyword parameters.
The positional parameters are the runtime parameters,
and the keyword parameters are the template parameters, e.g.

    Sm90_mma_async(
        D_rmem[...], A_rmem[...], B_rmem[...],  # Runtime parameters
        M=64, N=256,  # Control value template parameters
        A=f16, B=f16, C=f32,  # Precision template parameters
    )

"""

import ast as pyast
import inspect
from dataclasses import dataclass
from typing import Callable, Optional, Dict, List, Tuple, Type, Set

from .prelude import Sym, SrcInfo, ScalarInfo

from .cir import CIR_Wrapper
from .instr_info import AtomicityInfo, AccessInfo, InstrInfo
from .LoopIR import (
    LoopIR,
    SubstArgs,
    ReplacePrecision,
    Identifier,
    get_writes_of_stmts,
    get_reads_of_stmts,
)
from .memory import MemWin, DRAM, BarrierMechanism
from ..frontend.pyparser import get_ast_from_python, Parser
from ..spork import timelines
from ..spork.coll_algebra import standalone_thread, CollUnit
from ..spork.timelines import (
    Instr_tl,
    Qual_tl,
    cpu_in_order_instr,
)
from .c_window import WindowFeatures, UtilInjector, WindowIndexerResult


def proc_default_access_info(
    proc: LoopIR.proc, write_syms: Set[Sym], read_syms: Set[Sym]
):
    access_info = {}
    for arg in proc.args:
        if not arg.type.is_numeric():
            continue
        nm = arg.name.name()
        mem = DRAM if arg.mem is None else arg.mem
        access = AccessInfo()
        access.mem = mem
        access.const = arg.name not in write_syms
        access.write_only = arg.name not in read_syms
        basetype = arg.type.basetype()
        if not isinstance(basetype, LoopIR.Num):
            access.scalar_info = ScalarInfo(basetype)
        access_info[nm] = access

    return access_info


class InstrTemplateError(Exception):
    pass


def tparams_from_signature(clsname: str, tproc: LoopIR.proc, signature):
    assert isinstance(tproc, LoopIR.proc)

    control_tparam_syms = []
    control_tparam_types = []
    extra_tparam_names = []
    extra_tparam_defaults = {}

    for i, param in enumerate(signature.parameters.values()):
        nm = param.name
        # Skip self
        if i == 0:
            assert nm == "self", f"{clsname}.instance: missing self"
            continue
        problem = None
        if param.kind.name == "POSITIONAL_OR_KEYWORD":
            is_control_param = True
        elif param.kind.name == "KEYWORD_ONLY":
            is_control_param = False
        else:
            problem = f"cannot be {param.kind.name} argument"
        if param.default is not inspect._empty:
            if is_control_param:
                problem = "cannot have default value"
            else:
                extra_tparam_defaults[str(nm)] = param.default
        # Look for matching parameter in behavior() and get its Sym
        for tproc_a in tproc.args:
            if tproc_a.name.name() == nm:
                if not is_control_param:
                    problem = (
                        f"name conflict with {clsname}.behavior parameter "
                        f"(note, move before * if intended)"
                    )
                sym = tproc_a.name
                typ = tproc_a.type
                if typ.is_numeric():
                    raise TypeError(
                        f"{clsname}.instance: parameter {nm} "
                        f"must refer to control type, not {typ} "
                        f"(Precision parameters are passed implicitly in self.access_info)"
                    )
                break
        else:
            if is_control_param:
                problem = (
                    f"does not refer to any parameter of {clsname}.behavior "
                    f"(note, move after *, i.e. make keyword-only, if intended)"
                )

        if problem:
            raise ValueError(f"{clsname}.instance: parameter {nm} {problem}")
        if is_control_param:
            control_tparam_syms.append(sym)
            control_tparam_types.append(typ)
        else:
            extra_tparam_names.append(str(nm))

    return (
        control_tparam_syms,
        control_tparam_types,
        extra_tparam_names,
        extra_tparam_defaults,
    )


def prefill_instr_info(info: InstrInfo, proc: LoopIR.proc):
    const_dict = proc.get_cached_const_param_dict()
    write_syms = set(x for x, const in const_dict.items() if not const)
    read_syms = set(x for x, _ in get_reads_of_stmts(proc.body, include_reduce=True))
    info.instr_format = None
    info.c_utils = []
    info.c_includes = []
    info.cu_utils = []
    info.cu_includes = []
    info.coll_unit = standalone_thread
    info.instr_tl = cpu_in_order_instr
    info.access_info = proc_default_access_info(proc, write_syms, read_syms)
    info.barrier_mechanism = None
    info.barrier_coll_units = ()
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


class ProcCallGen:
    __slots__ = []

    def ProcCallGen_behavior(self) -> LoopIR.proc:
        raise NotImplementedError

    def ProcCallGen_make_call(self, args: List[LoopIR.expr], srcinfo) -> LoopIR.Call:
        raise NotImplementedError


class InstrTemplateBase(ProcCallGen):
    __slots__ = []

    def partial(self, **tparam_dict):
        """Partial evaluation"""
        return PartialInstrTemplate(self, tparam_dict)


@dataclass(slots=True)
class InstrTemplate(InstrTemplateBase):
    """Templatized instruction -- call operator yields Procedure instr"""

    # Avoid circular modules: proc -> Procedure
    make_procedure: Callable[[object], "Procedure"]

    # "Template proc"; this is not an instr; this is directly parsed from
    # the user's cls.behavior Exo function.
    tproc: LoopIR.proc

    # Syms of tproc parameters that also name precision template parameters
    prec_tparam_syms: List[Sym]

    # Syms of tproc parameters that are control value template parameters
    control_tparam_syms: List[Sym]

    # LoopIR types of control value template parameters
    control_tparam_types: List[LoopIR.type]

    # Extra template parameters, named by str, not Sym, since they don't
    # correspond to any tproc (behavior) parameters.
    extra_tparam_names: List[str]
    extra_tparam_defaults: Dict[str, object]

    # Subtype of InstrInfo defined by the user.
    info_cls: Type[InstrInfo]

    # Cache of Procedures.
    # When we substitute template parameters, we cache the resulting Procedure
    # here indexed by a tuple of tparam values
    # (order = prec_tparam_syms + control_tparam_syms + extra_tparam_names)
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

        prec_tparam_syms = [
            fa.name for fa in tproc.args if isinstance(fa.type.basetype(), LoopIR.Num)
        ]

        if prec_tparam_syms:
            # fmt: off
            assert hasattr(cls, "valid_num_types"), f"Missing {nm}.valid_num_types (consider ScalarInfo.same())"

        # Deduce the names of tparams based on cls.instance
        (
            control_tparam_syms,
            control_tparam_types,
            extra_tparam_names,
            extra_tparam_defaults,
        ) = tparams_from_signature(nm, tproc, instance_signature)

        # The user's cls.instance function will be used to initialize InstrInfo.
        def info_init(info, **tparam_dict):
            prefill_instr_info(info, tproc)
            if self.prec_tparam_syms:
                instance_params = self._check_prec_parameters(info, tparam_dict, cls)
            else:
                instance_params = tparam_dict
            info.instance(**instance_params)
            self._postprocess_instr_info(tproc, info, tparam_dict, has_custom_codegen)

        # The user-provided class gets converted to a subclass of InstrInfo.
        # Override __init__, and add __slots__.
        # I strongly believe in the typo-checking provided by __slots__.
        # Finally, add a fallback if no codegen callback was provided.
        info_dict = dict(cls.__dict__)
        info_bases = [b for b in cls.__bases__ if b is not object]
        if not issubclass(cls, InstrInfo):
            info_bases.append(InstrInfo)
        assert "__slots__" not in info_dict, f"{cls.__name__}: use annotations"
        info_dict["__slots__"] = list(cls.__annotations__)
        info_dict["__init__"] = info_init
        if not has_custom_codegen:
            info_dict["codegen"] = OldStyleInstrInfo.codegen
        info_cls = type(nm, tuple(info_bases), info_dict)

        self.make_procedure = make_procedure
        self.tproc = tproc
        self.prec_tparam_syms = prec_tparam_syms
        self.control_tparam_syms = control_tparam_syms
        self.control_tparam_types = control_tparam_types
        self.extra_tparam_names = extra_tparam_names
        self.extra_tparam_defaults = extra_tparam_defaults
        self.info_cls = info_cls
        self.cache = {}

    def __call__(self, **tparam_dict):
        # NB see also partial(...)
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

        n_prec = len(self.prec_tparam_syms)
        n_control = len(self.control_tparam_syms)
        n_extras = len(self.extra_tparam_names)
        assert len(tparam_values) == n_prec + n_control + n_extras

        prec_tparam_values = tparam_values[:n_prec]
        control_tparam_values = tparam_values[n_prec : n_prec + n_control]

        # Convert template proc (tproc) to instanced proc (iproc) by
        #   * Substituting concrete values in place of template params (tparams)
        #   * Removing fnargs that correspond to control value template parameters
        #   * Rewriting fnargs using precision template parameters
        #   * Adding the InstrInfo; set fnarg.mem as needed from InstrInfo
        tproc = self.tproc
        iproc_args = tproc.args
        iproc_preds = tproc.preds
        iproc_body = tproc.body
        if control_tparam_values:
            assert len(self.control_tparam_syms) == n_control
            assert len(self.control_tparam_types) == n_control
            binding = {
                sym: LoopIR.Const(v, typ, tproc.srcinfo)
                for sym, v, typ in zip(
                    self.control_tparam_syms,
                    control_tparam_values,
                    self.control_tparam_types,
                )
            }
            iproc_preds = SubstArgs(iproc_preds, binding).result()
            iproc_body = SubstArgs(iproc_body, binding).result()
            iproc_args = [
                a for a in iproc_args if a.name not in self.control_tparam_syms
            ]
            iproc_args = SubstArgs(iproc_args, binding).result()
        if prec_tparam_values:
            assert len(self.prec_tparam_syms) == len(prec_tparam_values)
            prec_rewrites = {
                sym: scalar_info.loopir
                for sym, scalar_info in zip(self.prec_tparam_syms, prec_tparam_values)
            }
            iproc_args = ReplacePrecision(iproc_args, prec_rewrites).result()
            iproc_preds = ReplacePrecision(iproc_preds, prec_rewrites).result()
            iproc_body = ReplacePrecision(iproc_body, prec_rewrites).result()
        for i, a in enumerate(iproc_args):
            if (access := instr_info.access_info.get(str(a.name))) is not None:
                iproc_args[i] = a.update(mem=access.mem)
        assert len(iproc_args) + len(self.control_tparam_syms) == len(tproc.args)
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
        """Convert kwargs dict into tuple of template parameter values

        The args are ordered to correspond to the concatenation of
          * prec_tparam_syms
          * control_tparam_syms
          * extra_tparam_syms
        The last two are the same order as args in the instance(...) function.

        """

        clsname = self.info_cls.__name__
        tparam_values = []
        for sym in self.prec_tparam_syms:
            assert isinstance(sym, Sym)
            nm = sym.name()
            try:
                # Have to cast to ScalarInfo in case the caller provided "f32" or such.
                # Have to do this before caching, else we could get duplicate instrs.
                tmp = tparam_dict[nm]
                scalar_info = ScalarInfo(tmp)
            except KeyError:
                raise InstrTemplateError(f"{clsname}: missing template parameter {nm}")
            except Exception as e:
                raise InstrTemplateError(
                    f"{clsname}: expected {nm}: ScalarInfo, not {tmp}"
                ) from e
            tparam_values.append(scalar_info)
        for sym in self.control_tparam_syms:
            assert isinstance(sym, Sym)
            nm = sym.name()
            v = tparam_dict.get(nm)
            if isinstance(v, int):
                tparam_values.append(v)
            elif v is None:
                raise InstrTemplateError(f"{clsname}: missing template parameter {nm}")
            else:
                raise InstrTemplateError(f"{clsname}: {nm} must be int, not {type(v)}")
        extras = self.extra_tparam_names
        num_defaults = 0
        for nm in extras:
            try:
                try:
                    v = tparam_dict[nm]
                except KeyError:
                    v = self.extra_tparam_defaults[nm]
                    num_defaults += 1
            except KeyError:
                raise InstrTemplateError(f"{clsname}: missing template parameter {nm}")
            tparam_values.append(v)

        # Do this assert late as the "missing parameter"
        # message above has better clarity.
        # fmt: off
        num_formal = len(self.prec_tparam_syms) + len(self.control_tparam_syms) + len(extras)
        assert len(tparam_dict) + num_defaults == num_formal, f"{clsname}: excess template parameters {tparam_dict}"
        # fmt: on
        return tuple(tparam_values)

    def _format_tparam_kwargs(self, tparam_values):
        all_names = (
            self.prec_tparam_syms + self.control_tparam_syms + self.extra_tparam_names
        )
        assert len(tparam_values) == len(all_names)
        return ", ".join(f"{nm}={v!r}" for nm, v in zip(all_names, tparam_values))

    def _postprocess_instr_info(
        self, proc: LoopIR.proc, info: InstrInfo, tparam_dict, has_custom_codegen: bool
    ):
        # =====================================================================
        # If anything in this code fails, it's almost certainly the fault
        # of the @instr author, not the end user using the instr.
        # =====================================================================

        clsname = self.info_cls.__name__
        has_instr_format = info.instr_format is not None
        # fmt: off
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
        assert info.barrier_mechanism is None or issubclass(info.barrier_mechanism, BarrierMechanism), clsname
        assert all(isinstance(unit, CollUnit) for unit in info.barrier_coll_units), clsname

        instr_tl = info.instr_tl.as_instr_tl()
        info.instr_tl = instr_tl
        access_info = info.access_info
        # fmt: on

        for arg in proc.args:
            if not arg.type.is_numeric():
                continue
            nm = arg.name.name()
            arg_info = access_info[nm]
            if arg.mem is not None and arg.mem is not DRAM:
                # fmt: off
                mem = arg.mem
                assert mem == arg_info.mem, f"{clsname}: cannot override mem for {nm} @ {arg.mem.name()}"
                # fmt: on
            else:
                mem = arg_info.mem
            assert issubclass(mem, MemWin)

            # Non-in-order instructions must set the OOO flag explicitly
            if arg_info.out_of_order is None:
                # fmt: off
                assert ("_in_order" in str(instr_tl)), \
                    f"{clsname}: need out_of_order flag for {nm} @ {mem.name()}"
                # fmt: on
                arg_info.out_of_order = False

            if arg_info.atomicity is not None:
                atomicity = arg_info.atomicity
                # fmt: off
                assert isinstance(atomicity, AtomicityInfo), f"{clsname}, {nm}"
                assert all(isinstance(q, Qual_tl) for q in atomicity.qual_tl_list), f"{clsname}, {nm}"
                # fmt: on

            # Distributed memory configuration checks
            # fmt: off
            for i, unit in enumerate(arg_info.distributed_coll_units):
                extent = arg.type.hi[i]
                extent_is_template_param = (
                    isinstance(extent, LoopIR.Read) and str(extent.name) in tparam_dict
                )
                assert isinstance(unit, CollUnit), clsname
                assert isinstance(arg.type, LoopIR.Tensor), clsname
                assert i < len(arg.type.hi), clsname
                assert isinstance(extent, LoopIR.Const) or extent_is_template_param
                assert (instr_tl != cpu_in_order_instr
                    ), f"{clsname} can't have CPU distributed memory"
            if arg_info.distributed_coll_units:
                assert isinstance(arg_info.access_by_owner_only, bool
                ), f"{clsname} must set access_by_owner_only for distributed memory args explicitly"
            # fmt: on

        info._tparam_dict = tparam_dict
        info._formatted_tparam_kwargs = self._format_tparam_kwargs(
            self._tparam_values(**tparam_dict)
        )

    def _check_prec_parameters(self, info: InstrInfo, tparam_dict: dict, cls: type):
        """Precision template parameters helper.

        * Update InstrInfo with ScalarInfo precision template parameters,
        * Delete precision template parameters from a copy of the dict
          (instance(...) forbids them)
        * Check valid_num_types is OK with the precision template parameters

        """
        assert self.prec_tparam_syms
        valid_num_types = cls.valid_num_types
        str_tparam_names = [str(nm) for nm in self.prec_tparam_syms]
        # Have to cast to ScalarInfo in case the caller provided "f32" or such.
        key = tuple(ScalarInfo(tparam_dict[s]) for s in str_tparam_names)
        if key not in valid_num_types:
            precs = ", ".join(f"{a}={v}" for (a, v) in zip(self.prec_tparam_syms, key))
            raise TypeError(
                f"{cls.__name__}: unsupported precision {precs}; valid: {valid_num_types}"
            )
        instance_params = tparam_dict.copy()  # not deepcopy
        for a, k in zip(str_tparam_names, key):
            del instance_params[a]
            info.access_info[a].scalar_info = k
        return instance_params

    def ProcCallGen_behavior(self) -> LoopIR.proc:
        return self.tproc

    def ProcCallGen_make_call(
        self, args: List[LoopIR.expr], srcinfo, partial_tparams: dict = {}
    ) -> LoopIR.Call:
        """Callback from LoopIR unification.

        The unification wants to call the behavior function with the given arguments.
        We need to filter the arguments into template and runtime parameters,
        instantiate a real proc with the template parameters, and generate
        a call stmt using the real proc and the runtime parameters.

        """
        clsname = self.info_cls.__name__
        tproc = self.tproc
        assert len(args) == len(tproc.args)

        # Separate destinations for template and runtime parameters.
        control_dict = {str(nm): None for nm in self.control_tparam_syms}
        tparam_dict = {}
        prec_names = {str(nm) for nm in self.prec_tparam_syms}
        control_names = {str(nm) for nm in self.control_tparam_syms}
        call_args = []
        for a, fa in zip(args, tproc.args):
            strnm = str(fa.name)
            if strnm in control_names:
                # TODO not sure if Unification will generate stuff like `2 + 0`
                if not isinstance(a, LoopIR.Const):
                    InstrTemplateError(
                        f"{clsname}: non-constant control value template parameter {strnm}={a}"
                    )
                tparam_dict[strnm] = int(a.val)
                # Not a runtime parameter; skip call_args.append(a)
            elif strnm in prec_names:
                tparam_dict[strnm] = a.type.basetype()  # Cast to ScalarInfo internally
                call_args.append(a)
            else:
                call_args.append(a)

        # Add in the template parameters given by the user.
        for k, v in partial_tparams.items():
            real_value = tparam_dict.get(k, v)
            # fmt: off
            assert v == real_value, f"{clsname}: deduced {k}={real_value} mismatches InstrTemplate.partial({k}={v})"
            tparam_dict[k] = real_value

        # Generate Call
        api_proc = self(**tparam_dict)
        call_proc = api_proc.ProcCallGen_behavior()
        assert len(call_proc.args) == len(call_args)
        return LoopIR.Call(call_proc, call_args, None, srcinfo)


# Note, we don't use functools.partial because the LoopIR
# unification needs the ProcCallGen base class to work.
@dataclass(slots=True)
class PartialInstrTemplate(InstrTemplateBase):
    _from: InstrTemplateBase
    _partial_tparams: dict

    def __call__(self, **tparam_dict):
        merged = self._partial_tparams | tparam_dict
        return self._from(**merged)

    def ProcCallGen_behavior(self) -> LoopIR.proc:
        return self._from.ProcCallGen_behavior()

    def ProcCallGen_make_call(
        self, args: List[LoopIR.expr], srcinfo, partial_tparams: dict = {}
    ) -> LoopIR.Call:
        merged = self._partial_tparams | partial_tparams
        return self._from.ProcCallGen_make_call(args, srcinfo, merged)


@dataclass(slots=True)
class InstrWindowArg:
    _encoder_utils: UtilInjector
    _indexer_utils: UtilInjector
    _features: WindowFeatures
    _scalar_info_input: object
    _srcinfo: SrcInfo

    def __post_init__(self):
        # Check intact packed dimensions
        # We documented we did this in MemWin.packed_tensor_shape
        features = self._features
        packed_tensor_shape = features.packed_tensor_shape()
        assert features.n_packed_dims() == len(packed_tensor_shape)
        for i, c in enumerate(packed_tensor_shape):
            sz = features.get_packed_interval_size(i)
            if sz is None:
                raise ValueError(
                    f"{features.get_raw_name()} must not have point expressions for packed dimensions (last {features.n_packed_dims()})"
                )
            features.get_packed_offset(i).exo_expect_int(0)
            sz.exo_expect_int(c)

    def __str__(self):
        return self._get_window_impl()

    def get_scalar_info(self) -> ScalarInfo:
        return ScalarInfo(self._scalar_info_input)

    def get_window(self) -> str:
        return self._get_window_impl()

    def __getitem__(self, pos) -> str:
        """Array indexing used to encode window struct to sub-window.

        Currently only support slices with explicit lo and hi, e.g. win[lo:hi].
        lo and hi must be of int or CIR_Wrapper type.

        See index(), index_ptr() to use the memory's WindowIndexer
        instead of WindowEncoder.

        """
        if not isinstance(pos, tuple):
            pos = (pos,)
        offsets = []
        interval_sizes = []
        for coord in pos:
            if isinstance(coord, slice):
                assert coord.start is not None, "Exo @instr supports only lo:hi slices"
                assert coord.stop is not None, "Exo @instr supports only lo:hi slices"
                assert coord.step is None, "Exo @instr supports only lo:hi slices"
                assert not isinstance(coord.start, str)
                assert not isinstance(coord.stop, str)
                offsets.append(coord.start)
                interval_sizes.append(coord.stop - coord.start)
            else:
                assert 0, "Expected slice"
                # assert not isinstance(coord, str)
                # offsets.append(coord)
                # interval_sizes.append(None)
        return self._get_window_impl(
            _special=False, _offsets=offsets, _interval_sizes=interval_sizes
        )

    def get_separate_dataptr(self, _special=False) -> str:
        features = self._features
        if _special:
            do_encode = features.get_encoder().encode_special_separate_dataptr
        else:
            do_encode = features.get_encoder().encode_separate_dataptr
        return str(do_encode(self._encoder_utils, features))

    def separate_dataptr(self) -> bool:
        return self._features.separate_dataptr()

    def get_raw_name(self) -> str:
        return self._features.get_raw_name()

    def get_raw_dataptr(self) -> str:
        return str(self._features.get_dataptr())

    def index_result(self, *idxs, **kwargs) -> WindowIndexerResult:
        new_features = self._features.new_window(
            idxs, [None] * len(idxs), self._srcinfo
        )
        indexed = self._features.get_indexer().index(
            self._indexer_utils,
            new_features,
            **kwargs,
        )
        assert isinstance(indexed, WindowIndexerResult)
        return indexed

    def index(self, *idxs, **kwargs) -> str:
        """Give expression for C++ reference to window[*idxs].

        Missing indices are implicitly 0.
        Any keyword arguments given are passed-through to
        the underlying MemWin type's window_indexer.
        """
        r = self.index_result(*idxs, **kwargs)
        return f"({r.code})[0]" if r.is_ptr else r.code

    def index_ptr(self, *idxs, **kwargs) -> str:
        """Give expression for pointer to window[*idxs], similar to index(...)"""
        r = self.index_result(*idxs, **kwargs)
        return r.code if r.is_ptr else f"(&{r.code})"

    def to_arg_strs(self) -> List[str]:
        if self.separate_dataptr():
            return [self.get_separate_dataptr(), self.get_window()]
        else:
            return [self.get_window()]

    def to_strides_as_packed(self):
        return self._features.interval_array_strides_as_packed()

    def srcinfo(self) -> SrcInfo:
        return self._srcinfo

    # The _special args are hacky: for @instr, we don't ever convert
    # Memory to a SpecialWindow, but we re-use this object in the
    # compiler to implement such conversions, to avoid code divergence.

    def _get_window_impl(self, _special=False, _offsets=(), _interval_sizes=()) -> str:
        features = self._features.new_window(_offsets, _interval_sizes, self._srcinfo)
        encoder = features.get_encoder()

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

    def _compiler_encode_special_window(self, special):
        return self._get_window_impl(_special=special)


@dataclass(slots=True)
class InstrNonWindowArg:
    # This could be expanded later...
    _code: str
    _is_ptr: bool
    _defaults_to_ptr: bool
    _scalar_info_input: object
    _srcinfo: SrcInfo

    def __str__(self):
        """Backwards-compatibility hack"""
        return self.index_ptr() if self._defaults_to_ptr else self.index()

    def index(self) -> str:
        """For compatibility with InstrWindowArg"""
        code = self._code
        return f"({code})[0]" if self._is_ptr else code

    def index_ptr(self) -> str:
        """For compatibility with InstrWindowArg"""
        code = self._code
        return code if self._is_ptr else f"(&{code})"

    def separate_dataptr(self) -> bool:
        """For compatibility with InstrWindowArg"""
        return False

    def to_arg_strs(self) -> List[str]:
        return [str(self)]

    def get_scalar_info(self) -> ScalarInfo:
        # Lazy evaluate because not all types have a valid ScalarInfo.
        return ScalarInfo(self._scalar_info_input)

    def srcinfo(self) -> SrcInfo:
        return self._srcinfo


@dataclass(slots=True)
class InstrArgs:
    _exo_args_dict: Dict[str, object]
    _compiler: object

    def __getattr__(self, attr):
        if attr.startswith("exo_"):
            assert (
                attr == "exo_barrier" or attr == "exo_clusterDim"
            ), "exo_ prefix not allowed for arg name"
        assert not attr.startswith("_exo_"), "_exo_ prefix not allowed for arg name"
        return self._exo_args_dict[attr]

    def __iter__(self):
        return iter(self._exo_args_dict.items())

    def exo_wrap_cir(self, n) -> CIR_Wrapper:
        return self._compiler.wrap_cir(n, "(from InstrArgs.exo_wrap_cir)")


class OldStyleInstrInfo(InstrInfo):
    __slots__ = []

    def codegen(self, args: InstrArgs) -> List[str]:
        """Translate args to dictionary then use instr_format.format"""
        d = dict()
        for name, value in args:
            if isinstance(value, InstrWindowArg):
                mem = self.access_info[name].mem
                if mem.has_window_encoder():
                    # Must be parenthesized in case this is used as a macro
                    # parameter (commas in the struct will cause problems)
                    d[name] = f"({str(value)})"
                if mem.has_window_indexer():
                    d[name + "_data"] = value.index()
                d[name + "_int"] = value.get_raw_name()
            else:
                # Non-window; Exo 1 defines {name}_data; unclear why.
                # value should have been initialized with self.comp_e(e, op_prec["."])
                # to avoid extra parens (which new-style instrs rely on).
                s_value = str(value)
                d[name] = s_value
                d[name + "_data"] = s_value
        return [line.format(**d) for line in self.instr_format]
