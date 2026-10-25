"""Known bugs with WindowStmt (derived windows); SKIPPED BY DEFAULT.

Run with EXO_RUN_BUGGY_WINDOWSTMT_TESTS=1 to see them fail.

Background (David Zhao Akeley + Claude, 2026-10-25)
---------------------------------------------------

Each test asserts the *correct* behavior, so each test currently fails.
They are skipped so the bugs are documented in the open without leaving
red tests that future agents feel obligated to fix (or get stuck on).
Please don't fix these (or un-skip them) unless updating WindowStmt is
directly relevant to the current task.

Derived windows (`w = x[...]`, LoopIR.WindowStmt, typed as T.Window) alias
the underlying tensor. See spork/docs/SpecialWindow.tex (legacy doc) for
the original complaints: rewrites need to be aliasing-aware everywhere,
which is an "O(n^2)" feature-interaction cost and a source of blind spots.
Second-order windows (window of a WindowStmt window) were fixed by
https://github.com/exo-lang/exo/pull/762, but re-checking turned up the
bugs here.

The fix is deliberately deferred: we may later adopt some hybrid scheme
for WindowStmt / SpecialWindow (e.g. a rewrite that inserts the
Sm90_tensorMap WindowStmt, which is currently only possible in object
code; see the TODO in spork/docs/spork_b/Tma.tex), and that decision
affects how these should be fixed.

Bugs:

1. inline_window gives each rewritten WindowExpr use the *inlined
   window's* T.Window type, rather than building a new one
   (create_window_type). So as_tensor (shape) is wrong whenever the use has
   different dimensionality, giving invalid C; and src_buf is wrong
   (an intermediate window, not the underlying tensor) when the use is
   the rhs of another WindowStmt, tripping an assert in mem_analysis.

2. stage_mem ignores accesses through derived windows aliasing the staged
   buffer; it stages x while writes through `w = x[...]` bypass the
   staging buffer, and the result computes a different answer.

3. stage_mem cannot use a plain WindowStmt window as its source
   ("Must find the symbol in env"), because extract_env only records
   WindowStmts with a SpecialWindow. Staging from Sm90_tensorMap works.

Also found at the same time, in DoStageMem's WindowExpr path (not
WindowStmt-specific, but same family of window-handling bugs):

4. If the staged window has point dimensions, rewrite_win keeps them
   (rewrite_idx correctly drops them), giving a malformed WindowExpr that
   crashes in new_eff.

5. A WindowExpr passed to a call only marks the staged buffer as read,
   never as written, so writes made by the callee are never stored back.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from exo import proc
from exo.core.LoopIR import LoopIR
from exo.stdlib.scheduling import *

pytestmark = pytest.mark.skipif(
    os.environ.get("EXO_RUN_BUGGY_WINDOWSTMT_TESTS") != "1",
    reason="Known WindowStmt bugs, see file docstring; set EXO_RUN_BUGGY_WINDOWSTMT_TESTS=1",
)


@proc
def take_add(n: size, w: [f32][n]):
    for i in seq(0, n):
        w[i] += 1.0


@proc
def take_copy(n: size, w: [f32][n], y: f32[n]):
    for i in seq(0, n):
        y[i] = w[i]


def window_stmts(p):
    return {
        str(s.name): s.rhs.type
        for s in p._loopir_proc.body
        if isinstance(s, LoopIR.WindowStmt)
    }


def stage_mem_or_reject(p, *args):
    """Rejecting the rewrite is also an acceptable fix; returns None then."""
    try:
        return stage_mem(p, *args)
    except SchedulingError:
        return None


def assert_same_result(compiler, p0, p1, *shapes):
    """Run p0 and p1 on identical random inputs and compare all buffers."""
    p0 = rename(p0, "buggy_orig")
    p1 = rename(p1, "buggy_staged")
    lib = compiler.compile([p0, p1])
    rng = np.random.default_rng(0)
    bufs = [rng.random(s, dtype=np.float32) for s in shapes]
    bufs0 = [b.copy() for b in bufs]
    bufs1 = [b.copy() for b in bufs]
    lib.buggy_orig(None, *bufs0)
    lib.buggy_staged(None, *bufs1)
    for b0, b1 in zip(bufs0, bufs1):
        np.testing.assert_array_equal(b0, b1)


# Bug 1


def test_inline_window_call_arg_type(compiler):
    @proc
    def foo(x: f32[8, 16]):
        w = x[1:8, 0:16]
        take_add(8, w[0, 0:8])

    p = inline_window(foo, foo.find("w = _"))
    arg_typ = p._loopir_proc.body[0].args[1].type
    # Currently [f32][8 - 1, 16] (w's shape), so the C passes a 2D window
    # struct where take_add expects a 1D one.
    assert len(arg_typ.as_tensor.shape()) == 1
    compiler.compile(p, compile_only=True)


def test_inline_window_dependent_window_stmt(compiler):
    @proc
    def foo(x: f32[8, 16]):
        w = x[1:8, 0:16]
        v = w[2, 4:12]
        u = v[1:5]
        for i in seq(0, 4):
            u[i] = 2.0

    p = inline_window(foo, foo.find("v = _"))
    u_typ = window_stmts(p)["u"]
    # Currently src_buf=w and as_tensor=[f32][12 - 4] (v's type).
    assert str(u_typ.src_buf) == "x"
    assert len(u_typ.as_tensor.shape()) == 1
    compiler.compile(p, compile_only=True)  # Currently asserts in mem_analysis


# Bug 2


def test_stage_mem_ignores_alias_writes(compiler):
    @proc
    def foo(x: f32[8, 16]):
        w = x[1:8, 0:16]
        for i in seq(0, 4):
            x[3, 4 + i] = 1.0
            w[2, 4 + i] += 5.0  # Same location as x[3, 4 + i]

    # Currently accepted: w's writes bypass the staging buffer and are then
    # clobbered by the store-back (x[3, 4] ends up 1 instead of 6).
    p = stage_mem_or_reject(foo, foo.find_loop("i"), "x[3, 4:8]", "t")
    if p is not None:
        assert_same_result(compiler, foo, p, (8, 16))


# Bug 3


def test_stage_mem_from_plain_window_stmt(compiler):
    @proc
    def foo(x: f32[8, 16]):
        w = x[1:8, 0:16]
        for i in seq(0, 16):
            w[2, i] += 1.0

    # Currently AssertionError("Must find the symbol in env")
    p = stage_mem(foo, foo.find_loop("i"), "w[2, 0:16]", "t")
    assert_same_result(compiler, foo, p, (8, 16))


# Bug 4


def test_stage_mem_point_dim_window_expr(compiler):
    @proc
    def foo(x: f32[8, 4], y: f32[4]):
        take_copy(4, x[2, 0:4], y)

    # Currently AssertionError in new_eff: the call's window keeps the
    # point dim (2 indices into the 1D staging buffer t).
    p = stage_mem(foo, foo.body()[0], "x[2, 0:4]", "t")
    assert len(p._loopir_proc.body[-1].args[1].idx) == 1
    assert_same_result(compiler, foo, p, (8, 4), (4,))


# Bug 5


def test_stage_mem_call_window_write_back(compiler):
    @proc
    def foo(x: f32[8, 4]):
        take_add(4, x[2, 0:4])

    # Currently accepted with a load into t but no store back to x.
    p = stage_mem_or_reject(foo, foo.body()[0], "x[0:8, 0:4]", "t")
    if p is not None:
        assert_same_result(compiler, foo, p, (8, 4))
