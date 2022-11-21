from __future__ import annotations

import ast as pyast
import inspect
import types

import exo.api as API
from exo.configs import Config
from exo.pyparser import get_ast_from_python, Parser, get_src_locals

__all__ = ["proc", "instr", "config"]


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Top-level decorator


def proc(f, _instr=None) -> "API.Procedure":
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
    return API.Procedure(parser.result())


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
