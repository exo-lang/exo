# from __future__ import annotations
# import pytest
# from exo import proc, DRAM, Procedure, config
# from exo.stdlib.scheduling import *
# from exo.libs.externs import sin, intmin

from exo.rewrite.dataflow import D, V, ASubs
from exo.rewrite.approximation import Strategy1
from exo.rewrite.canonicalize import _canon_dir
from exo.rewrite.internal_analysis import *
from exo.rewrite.lift_to_smt import *

import sympy as sm

# Symbols we will reuse
x, y, z = sm.symbols("x y z")
_env = {x: Sym("x"), y: Sym("y"), z: Sym("z")}

###############################################################################
# lift_to_smt_a                                                                #
###############################################################################


def test_lift_to_smt_a_arithmetic(golden):
    e = x + 2 * y
    aexpr = lift_to_smt_a(e, _env)

    assert isinstance(aexpr, A.expr)
    assert aexpr.type is T.R
    assert str(aexpr) == golden


def test_lift_to_smt_a_relational(golden):
    e = sm.Eq(x, y + 1)
    aexpr = lift_to_smt_a(e, _env)

    assert isinstance(aexpr, A.expr)
    assert aexpr.type is T.bool
    assert str(aexpr) == golden


def test_lift_to_smt_a_logical(golden):
    e = sm.And(x > 0, y < 0)
    aexpr = lift_to_smt_a(e, _env)

    assert isinstance(aexpr, A.expr)
    assert aexpr.type is T.bool
    # Top‑level node should be an "and" BinOp or the helper AAnd result
    if isinstance(aexpr, A.BinOp):
        assert aexpr.op == "and"
    assert str(aexpr) == golden


###############################################################################
# lift_to_smt_val                                                              #
###############################################################################


def test_lift_to_smt_val_arrayvar(golden):
    aname = A.Var(Sym("tmp"), T.R, null_srcinfo())
    arr_val = D.ArrayVar(name=sm.Symbol("A"), idx=[sm.Integer(0), sm.Integer(1)])

    expr = lift_to_smt_val(aname, arr_val, _env)

    assert isinstance(expr, A.expr)
    assert expr.type is T.bool
    # Should be == between aname and some Var
    assert isinstance(expr, A.BinOp) and expr.op == "=="
    assert str(expr) == golden


def test_lift_to_smt_val_scalarexpr(golden):
    aname = A.Var(Sym("tmp"), T.R, null_srcinfo())
    scalar = D.ScalarExpr(poly=x + 2)

    expr = lift_to_smt_val(aname, scalar, _env)

    assert isinstance(expr, A.expr)
    assert expr.type is T.bool
    # RHS should be lifted arithmetic expression
    assert lift_to_smt_a(x + 2, _env).type is T.R
    assert str(expr) == golden


###############################################################################
# lift_to_smt_n                                                                #
###############################################################################


def _make_leaf(val):
    """Helper to manufacture a Leaf node"""
    return D.Leaf(v=val, sample={})


def test_lift_to_smt_n_leaf(golden):
    aname_sym = Sym("leaf_test")
    arr_val = D.ArrayVar(name=sm.Symbol("B"), idx=[sm.Integer(0)])
    leaf = _make_leaf(arr_val)

    expr = lift_to_smt_n(aname_sym, leaf, _env)

    assert isinstance(expr, A.expr)
    assert expr.type is T.bool
    assert str(expr) == golden


def test_lift_to_smt_n_linsplit(golden):
    # Build leaves
    lv1 = _make_leaf(D.ArrayVar(name=sm.Symbol("C"), idx=[sm.Integer(0)]))
    lv2 = _make_leaf(D.ArrayVar(name=sm.Symbol("C"), idx=[sm.Integer(1)]))

    # Guards
    g1 = sm.Eq(x, 0)
    g2 = sm.Eq(x, 1)

    # Cells (first match wins)
    c1 = D.Cell(eq=g1, tree=lv1)
    c2 = D.Cell(eq=g2, tree=lv2)

    split = D.LinSplit(cells=[c1, c2])

    expr = lift_to_smt_n(Sym("arr_elem"), split, _env)

    assert isinstance(expr, A.expr)
    assert expr.type is T.bool
    # Expect top‑level Select
    assert isinstance(expr, A.Select)
    assert str(expr) == golden


def test_smt():
    v1 = sm.Eq(x, 3)
    v2 = sm.Eq(y, 4)
    e1 = lift_to_smt_a(v1, _env)
    e2 = lift_to_smt_a(v2, _env)
    ir1_k = A.Var(_env[x], T.int, null_srcinfo())
    ir2_k = A.Var(_env[y], T.int, null_srcinfo())
    is_unchanged = AImplies(AAnd(e1, e2), AEq(ir1_k, ir2_k))
    print(is_unchanged)
    slv = SMTSolver(verbose=False)
    is_ok = slv.verify(is_unchanged)
    print(is_ok)
