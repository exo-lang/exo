from __future__ import annotations

import pytest

from exo.new_eff import *

from exo import proc, config, DRAM, SchedulingError
from exo.stdlib.scheduling import *
from exo.platforms.x86 import *


print()
print("Dev Tests for new_eff.py")


def test_debug_let_and_mod():
    N = AInt(Sym("N"))
    j = AInt(Sym("j"))
    i = AInt(Sym("i"))
    x = AInt(Sym("x"))

    F = A.ForAll(
        i.name,
        A.Let(
            [x.name],
            [A.Let([j.name], [AInt(64) * i], N + j, T.index, j.srcinfo)],
            AEq(x % AInt(64), AInt(0)),
            T.bool,
            x.srcinfo,
        ),
        T.bool,
        i.srcinfo,
    )

    slv = SMTSolver(verbose=True)

    slv.verify(F)


def test_reorder_stmts_fail():
    @proc
    def foo(N: size, x: R[N]):
        x[0] = 3.0
        x[0] = 4.0

    with pytest.raises(SchedulingError, match="do not commute"):
        foo = reorder_stmts(foo, "x[0] = 3.0 ; x[0] = 4.0")


def test_reorder_alloc_fail():
    @proc
    def foo(N: size, x: R[N]):
        y: R
        y = 4.0
        x[0] = y

    with pytest.raises(SchedulingError, match="do not commute"):
        foo = reorder_stmts(foo, "y : R ; y = 4.0")


def test_reorder_loops_success(golden):
    @proc
    def foo(N: size, x: R[N, N]):
        for i in seq(0, N):
            for j in seq(0, N):
                x[i, j] = x[i, j] * 2.0

    foo = reorder_loops(foo, "i j")
    assert str(foo) == golden


def test_reorder_loops_fail():
    @proc
    def foo(N: size, x: R[N, N]):
        for i in seq(0, N):
            for j in seq(0, N):
                x[i, j] = x[j, i] * 2.0

    with pytest.raises(SchedulingError, match="cannot be reordered"):
        foo = reorder_loops(foo, "i j")


def test_alloc_success(golden):
    @proc
    def foo(N: size, x: R[N, N]):
        for i in seq(0, N):
            for j in seq(0, N):
                tmp: R
                tmp = x[i, j] * 2.0
                x[i, j] = tmp

    foo = reorder_loops(foo, "i j")
    assert str(foo) == golden


def test_reorder_loops_requiring_seq(golden):
    # the stencil pattern here looks like
    #     o     o
    #       \   |
    #         \ V
    #     o --> x
    #
    # so it isn't safe to _reverse_
    # the iteration order, but it is safe
    # to reorder the loops

    @proc
    def foo(N: size, x: R[N, N]):
        for i in seq(0, N):
            for j in seq(0, N):
                if i > 0 and j > 0:
                    x[i, j] += (
                        -1.0 / 3.0 * (x[i - 1, j] + x[i - 1, j - 1] + x[i, j - 1])
                    )

    foo = reorder_loops(foo, "i j")
    assert str(foo) == golden


def test_reorder_loops_4pt_stencil_succeed(golden):
    # Also, a 4-point stencil being
    # used in e.g. a Gauss-Seidel scheme can be reordered

    @proc
    def foo(N: size, x: R[N, N]):
        for i in seq(0, N):
            for j in seq(0, N):
                if 0 < i < N - 1 and 0 < j < N - 1:
                    x[i, j] += (
                        -1.0
                        / 4.0
                        * (x[i - 1, j] + x[i + 1, j] + x[i, j - 1] + x[i, j + 1])
                    )

    foo = reorder_loops(foo, "i j")
    assert str(foo) == golden


def test_reorder_loops_failing_seq():
    # But if we do the stencil over the 4 diagonals, then it's not safe

    @proc
    def foo(N: size, x: R[N, N]):
        for i in seq(0, N):
            for j in seq(0, N):
                if 0 < i < N - 1 and 0 < j < N - 1:
                    x[i, j] += (
                        -1.0
                        / 4.0
                        * (
                            x[i - 1, j - 1]
                            + x[i - 1, j + 1]
                            + x[i + 1, j - 1]
                            + x[i + 1, j + 1]
                        )
                    )

    with pytest.raises(SchedulingError, match="cannot be reordered"):
        foo = reorder_loops(foo, "i j")


# Should add a test that shows that READing something in an assertion
# does in fact count as a READ effect for the procedure, but not
# for its body.  This can probably distinguish whether certain
# rewrites are allowed or not.


def test_delete_config_basic(golden):
    @config
    class CFG:
        a: index
        b: size

    @proc
    def foo(N: size, x: R[N]):
        CFG.a = 3
        for i in seq(0, N):
            x[i] = x[i] + 1.0

    foo = delete_config(foo, "CFG.a = _")
    assert str(foo) == golden


def test_delete_config_subproc_basic(golden):
    @config
    class CFG:
        a: index
        b: size

    @proc
    def do_config():
        CFG.a = 3
        CFG.b = 5

    @proc
    def foo(N: size, x: R[N]):
        do_config()
        for i in seq(0, N):
            x[i] = x[i] + 1.0

    foo = delete_config(foo, "do_config()")
    assert str(foo) == golden


def test_delete_config_fail():
    @config
    class CFG:
        a: index
        b: size

    @proc
    def foo(N: size, x: R[N]):
        CFG.a = 3
        for i in seq(0, N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    with pytest.raises(
        SchedulingError, match="Cannot change configuration value of CFG_a"
    ):
        foo = delete_config(foo, "CFG.a = _")


def test_delete_config_subproc_fail():
    @config
    class CFG:
        a: index
        b: size

    @proc
    def do_config():
        CFG.a = 3
        CFG.b = 5

    @proc
    def foo(N: size, x: R[N]):
        do_config()
        for i in seq(0, N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    with pytest.raises(
        SchedulingError, match="Cannot change configuration value of CFG_a"
    ):
        foo = delete_config(foo, "do_config()")


def test_delete_config_bc_shadow(golden):
    @config
    class CFG:
        a: index
        b: size

    @proc
    def foo(N: size, x: R[N]):
        CFG.a = 34
        CFG.a = 3
        for i in seq(0, N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    foo = delete_config(foo, "CFG.a = _ #0")
    assert str(foo) == golden


def test_delete_config_bc_redundant(golden):
    @config
    class CFG:
        a: index
        b: size

    @proc
    def foo(N: size, x: R[N]):
        CFG.a = 3
        CFG.a = 3
        for i in seq(0, N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    foo = delete_config(foo, "CFG.a = _ #1")
    assert str(foo) == golden


def test_delete_config_fail_bc_not_redundant():
    @config
    class CFG:
        a: index
        b: size

    @proc
    def foo(N: size, x: R[N]):
        CFG.a = 34
        CFG.a = 3
        for i in seq(0, N):
            if i < CFG.a:
                x[i] = x[i] + 1.0

    with pytest.raises(
        SchedulingError, match="Cannot change configuration value of CFG_a"
    ):
        foo = delete_config(foo, "CFG.a = _ #1")


def test_scal():
    @proc
    def sscal(n: size, alpha: f32 @ DRAM, x: [f32][n] @ DRAM):
        assert stride(x, 0) == 1
        reg0: f32[8] @ AVX2
        if n / 8 > 0:
            mm256_broadcast_ss_scalar(reg0[0:8], alpha)
        reg1: f32[8] @ AVX2
        reg2: f32[8] @ AVX2
        for io in seq(0, n / 8):
            mm256_loadu_ps(reg1[0:8], x[8 * io + 0 : 8 * io + 8])
            mm256_mul_ps(reg2[0:8], reg0[0:8], reg1[0:8])
            mm256_storeu_ps(x[8 * io + 0 : 8 * io + 8], reg2[0:8])
        for ii in seq(0, n % 8):
            x[ii + n / 8 * 8] = alpha * x[ii + n / 8 * 8]

    stmt_c = sscal.find("mm256_mul_ps(_)")

    with pytest.raises(SchedulingError):
        sscal = reorder_stmts(sscal, stmt_c.expand(1, 0))


def test_alias_check():
    """This fails when we have an aliasing in a proc"""

    @proc
    def foo(N: size, x: [f32][N], y: [f32][N]):
        for i in seq(0, N):
            x[i] = y[i]

    with pytest.raises(SchedulingError, match="Cannot Pass the same buffer"):

        @proc
        def bar(N: size, x: [f32][N]):
            foo(N, x, x)
