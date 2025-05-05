from __future__ import annotations
from exo.core.prelude import Sym

from exo.rewrite.constraint_solver import ConstraintMaker, DisjointConstraint
from exo.core.LoopIR import T
from exo import proc
from exo.rewrite.chexo import TypeVisitor
from exo.backend.LoopIR_transpiler import Transpiler, CoverageArgs


def get_coverage_args(p) -> CoverageArgs:
    p_type = TypeVisitor()
    p_type.visit(p._loopir_proc)
    cm = ConstraintMaker(p_type.type_map)
    return CoverageArgs(cm)


def test_matmul(golden):
    Sym._unq_count = 1

    @proc
    def matmul(N: size, M: size, K: size, a: f32[N, K], b: f32[K, M], c: f32[N, M]):
        for i in seq(0, N):
            for j in seq(0, M):
                for k in seq(0, K):
                    c[i, j] += a[i, k] * b[k, j]

    assert golden == Transpiler(matmul._loopir_proc).get_javascript_template().template


def test_matmul_coverage(golden):
    Sym._unq_count = 1

    @proc
    def matmul(N: size, M: size, K: size, a: f32[N, K], b: f32[K, M], c: f32[N, M]):
        for i in seq(0, N):
            for j in seq(0, M):
                for k in seq(0, K):
                    c[i, j] += a[i, k] * b[k, j]

    assert (
        golden
        == Transpiler(matmul._loopir_proc, get_coverage_args(matmul))
        .get_javascript_template()
        .template
    )


def test_window_coverage(golden):
    Sym._unq_count = 1

    @proc
    def foo(a: i32[16]):
        a_win = a[1:8]
        a[3] = 2
        a_win[2] = 3

    assert (
        golden
        == Transpiler(foo._loopir_proc, get_coverage_args(foo))
        .get_javascript_template()
        .template
    )


def test_variable_length_array_coverage(golden):
    Sym._unq_count = 1

    @proc
    def foo(n: size):
        assert n > 2
        for i in seq(2, n):
            b: i32[i]
            b[i - 1] = 0
            b[i - 2] = 1

    assert (
        golden
        == Transpiler(foo._loopir_proc, get_coverage_args(foo))
        .get_javascript_template()
        .template
    )


def test_nested_control_flow_coverage(golden):
    Sym._unq_count = 1

    @proc
    def foo(n: size, b: f32):
        for i in seq(0, n):
            if i < n / 2:
                b = 2
            else:
                b = 3
            if i == n - 1:
                b += 1
            else:
                b += 2

    assert (
        golden
        == Transpiler(foo._loopir_proc, get_coverage_args(foo))
        .get_javascript_template()
        .template
    )
