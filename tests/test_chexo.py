from __future__ import annotations
from exo.core.prelude import Sym

from exo.rewrite.chexo import (
    TypeVisitor,
    get_used_config_fields,
    get_free_variables,
    collect_path_constraints,
    collect_arg_size_constraints,
)
from exo.rewrite.constraint_solver import ConstraintMaker
from exo import proc, config
from exo.core.memory import StaticMemory


def stringify_dict(d):
    def check_tuple(x):
        if isinstance(x, tuple):
            return f"({', '.join(str(x_item) for x_item in x)})"
        else:
            return str(x)

    return "\n".join(
        sorted(f"{check_tuple(k)}: {check_tuple(v)}" for k, v in d.items())
    )


def test_type_visitor(golden):
    @proc
    def foo(a: size, b: f32[a]):
        for i in seq(0, a):
            c: f32 @ StaticMemory
            c = b[i] * 2

    type_visitor = TypeVisitor()
    type_visitor.visit(foo._loopir_proc)
    types = stringify_dict(type_visitor.type_map)
    mems = stringify_dict(type_visitor.mem_map)
    assert golden == f"Types:\n{types}\nMems:\n{mems}"


def test_get_used_config_fields(golden):
    @config
    class TestConfig:
        a: f32
        b: size
        c: f32

    @proc
    def foo(a: f32):
        TestConfig.c = a
        a = TestConfig.a

    used_configs = get_used_config_fields(foo._loopir_proc.body)
    assert golden == stringify_dict(used_configs)


def test_free_variables(golden):
    @proc
    def foo(a: size, b: f32[a]):
        for i in seq(0, a):
            c: f32 @ StaticMemory
            c = b[i] * 2

    type_visitor = TypeVisitor()
    type_visitor.visit(foo._loopir_proc)
    free_vars = get_free_variables(
        type_visitor.type_map,
        type_visitor.mem_map,
        [cursor._impl._node for cursor in foo.find("c: _").as_block().expand()],
    )
    assert golden == stringify_dict(free_vars)


def test_path_constraints(golden):
    @proc
    def foo(a: size, b: f32[a]):
        for i in seq(0, a):
            if 2 * i < a:
                b[i] = 0
            else:
                for j in seq(0, a):
                    b[j] = b[i]

    type_visitor = TypeVisitor()
    type_visitor.visit(foo._loopir_proc)
    cm = ConstraintMaker(type_visitor.type_map)
    assert (
        golden
        == collect_path_constraints(foo.find("b[j] = b[i]")._impl, cm).pretty_print()
    )


def test_arg_size_constraints(golden):
    @proc
    def foo(a: size, b: size, c: f32[a * 2, b + 1]):
        pass

    type_visitor = TypeVisitor()
    type_visitor.visit(foo._loopir_proc)
    cm = ConstraintMaker(type_visitor.type_map)
    constraints = collect_arg_size_constraints(foo._loopir_proc.args, cm)
    assert golden == constraints.pretty_print()
