from __future__ import annotations
import pytest
from exo import proc, DRAM, Procedure, config
from exo.stdlib.scheduling import *
from exo.libs.externs import sin

from exo.rewrite.dataflow import D, V, ASubs
from exo.rewrite.approximation import Strategy1
from exo.rewrite.canonicalize import _canon_dir

import sympy as sm


def _leaf(value=V.Top(), sample=None):
    """Create the common leaf pattern quickly."""
    if sample is None:
        sample = {}
    return D.Leaf(D.SubVal(V.ValConst(value)), sample)


# ──────────────────────────────────────────────────────────────────────
# 1) purely numeric substitution
# ──────────────────────────────────────────────────────────────────────
def test_asubs_numeric():
    """{x: 5, y: 7} should literally overwrite the sample coordinates."""
    from sympy.abc import x, y
    import sympy as sm

    leaf = _leaf(sample={x: 0, y: 1})
    cad = D.abs([x, y], [x + y], leaf)

    new_tree = ASubs(cad.tree, {x: sm.Integer(5), y: sm.Integer(7)}).result()

    assert new_tree.sample == {x: 5, y: 7}  # ← sample updated
    assert new_tree.v == leaf.v  # payload unchanged
    # polynomial inside cad (x + y) → 5 + 7
    assert new_tree.v is leaf.v  # immutability check


def test_asubs_translation():
    """
    Original guard:   y < 0 | y == 0 | y > 0
    After Σ:          y - 1 < 0  …  ⇒   y < 1, etc.
    Samples should move forward by +1.
    """
    from sympy.abc import x, y
    import sympy as sm

    cells = [
        D.Cell(y < 0, _leaf(sample={y: -1})),
        D.Cell(sm.Eq(y, 0), _leaf(sample={y: 0})),
        D.Cell(y > 0, _leaf(sample={y: 1})),
    ]
    cad = D.abs([y], [y], D.LinSplit(cells))

    new_tree = ASubs(cad.tree, {y: y - 1}).result()
    new_cells = new_tree.cells  # unpack LinSplit

    # guards
    lhs = lambda r: sm.simplify(r.lhs - r.rhs)
    assert lhs(new_cells[0].eq) == lhs(y < 1)
    assert lhs(new_cells[1].eq) == lhs(sm.Eq(y, 1))
    assert lhs(new_cells[2].eq) == lhs(y > 1)

    # samples
    assert new_cells[0].tree.sample == {y: 0}
    assert new_cells[1].tree.sample == {y: 1}
    assert new_cells[2].tree.sample == {y: 2}


def test_asubs_mixed():
    from sympy.abc import x, y
    import sympy as sm

    leaf = _leaf(sample={x: 1, y: 2})
    cad = D.abs([x, y], [x - y], leaf)

    sigma = {x: x + 2, y: sm.Integer(3)}
    new_tree = ASubs(cad.tree, sigma).result()

    # Guard polynomial  (x - y)  →  (x + 2) - 3 = x - 1
    assert sm.simplify((cad.poly[0]).xreplace(sigma) - (x - 1)) == 0

    # Sample:  x + 2 = 1  ⇒  x = -1 ;  y → 3
    assert new_tree.sample == {x: -1, y: 3}


def _make_strategy():
    @proc
    def foo():
        pass

    return Strategy1(foo.dataflow()[0])


def test_issubsetof_basic():
    from sympy.abc import x, y

    a1 = D.abs([y], [], D.Leaf(D.SubVal(V.Top()), {y: 1}))
    a2 = D.abs(
        [y],
        [y],
        D.LinSplit(
            [
                D.Cell(y < 0, D.Leaf(D.SubVal(V.Top()), sample={y: -1})),
                D.Cell(sm.Eq(y, 0), D.Leaf(D.SubVal(V.Top()), sample={y: 0})),
                D.Cell(y > 0, D.Leaf(D.SubVal(V.Top()), sample={y: 1})),
            ]
        ),
    )
    strat = _make_strategy()
    assert strat.issubsetof(a1, a2)
    assert strat.issubsetof(a2, a1)


def test_issubsetof_or():
    from sympy.abc import x, y

    t1 = D.LinSplit(
        [
            D.Cell(y < 0, _leaf(value=1, sample={y: -1})),
            D.Cell(sm.Eq(y, 0), _leaf(value=1, sample={y: 0})),
            D.Cell(y > 0, _leaf(value=3, sample={y: 1})),
        ]
    )
    t2 = D.LinSplit(
        [
            D.Cell(sm.Or(y < 0, sm.Eq(y, 0)), _leaf(value=1, sample={y: -1})),
            D.Cell(y > 0, _leaf(value=3, sample={y: 1})),
        ]
    )
    a1 = D.abs([], [], t1)
    a2 = D.abs([], [], t2)
    strat = _make_strategy()
    assert strat.issubsetof(a1, a2)


def test_issubsetof_false():
    from sympy.abc import x, y

    gens = [x, y]
    strat = _make_strategy()
    a1 = D.abs(gens, [], D.Leaf(D.SubVal(V.ValConst(1)), {x: 1, y: 1}))
    a2 = D.abs(gens, [], D.Leaf(D.SubVal(V.ValConst(2)), {x: 1, y: 1}))
    assert not strat.issubsetof(a1, a2)


def test_widening(golden):
    from sympy.abc import x, y

    a = D.abs(
        [y],
        [y],
        D.LinSplit(
            [
                D.Cell(y < 0, D.Leaf(D.SubVal(V.Top()), sample={y: -1})),
                D.Cell(sm.Eq(y, 0), D.Leaf(D.SubVal(V.ValConst(1)), sample={y: 0})),
                D.Cell(y > 0, D.Leaf(D.SubVal(V.Top()), sample={y: 1})),
            ]
        ),
    )
    strat = _make_strategy()
    assert str(strat.abs_widening(a, a, 0, y)) == golden


def test_simple(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        z = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple1_5(golden):
    @proc
    def foo(x: R[2]):
        x[1] = 3.0
        x[0] = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple1_7(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        pass

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple_ref(golden):
    @proc
    def foo(n: size, x: R[3], y: R[3]):
        x[0] = y[0]
        if n < 3:
            x[n] = y[n]
        pass

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple2(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        x[0] = 2.0
        x[1] = 3.0
        x[2] = 5.0
        x[0] = 12.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple_stmts(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        z = 2.0

    d_ir, stmts, _ = foo.dataflow(foo.find("z = _ ; z = _"))

    assert (
        str(_canon_dir(d_ir)) + "\n\n" + "\n".join([str(s[0]) for s in stmts]) == golden
    )


@pytest.mark.skip()
def test_simple_stmts2(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        pass

    d_ir, stmts, _ = foo.dataflow(foo.find("if n < 3: _"))

    assert (
        str(_canon_dir(d_ir)) + "\n\n" + "\n".join([str(s[0]) for s in stmts]) == golden
    )


def test_simple3(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        x[2] = 5.0
        x[0] = 12.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple4(golden):
    @proc
    def foo(n: size, dst: f32[n]):
        for i in seq(0, n):
            dst[i] = 0.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple5(golden):
    @proc
    def foo(dst: f32[3], src: f32[3]):
        for i in seq(0, 3):
            dst[i] = src[i]

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_simple6(golden):
    @proc
    def foo(n: size, dst: f32[n], src: f32[n]):
        for i in seq(0, n):
            dst[i] = src[i]

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_print(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        z = 4.2
        z = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_print_new(golden):
    @proc
    def foo(z: R[3]):
        for i in seq(0, 3):
            z[i] = 3.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_print_0(golden):
    @proc
    def foo(z: R[3]):
        z[0] = 1.0
        for i in seq(0, 3):
            z[i] = 3.0
        z[2] = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_print_1(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R[3]):
        z[0] = x[0] * y[2]
        for i in seq(0, 3):
            z[i] = 3.0
        z[2] = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_print_3(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 4.0
        z = 0.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_print_5(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = 3.0
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_sliding_window_debug(golden):
    @proc
    def foo(dst: i8[30]):
        for i in seq(0, 10):
            for j in seq(0, 20):
                dst[i] = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


def test_sliding_window_const_guard(golden):
    @proc
    def foo(dst: i8[30]):
        for i in seq(0, 10):
            for j in seq(0, 20):
                if i == 0 or j == 19:
                    dst[i + j] = 2.0

    assert str(_canon_dir(foo.dataflow()[0])) == golden


@pytest.mark.skip()
def test_sliding_window_print():
    @proc
    def foo(n: size, m: size, dst: i8[n + m]):
        for i in seq(0, n):
            for j in seq(0, m):
                dst[i + j] = 2.0

    print(foo.dataflow()[0])


@pytest.mark.skip()
def test_multi_dim_print():
    @proc
    def foo(n: size, dst: i8[n, n]):
        for i in seq(0, n):
            for j in seq(1, n):
                dst[i, n - j] = 2.0

    print(foo.dataflow()[0])


# TODO: Currently add_unsafe_guard lacks analysis, but we should be able to analyze this
@pytest.mark.skip()
def test_sliding_window(golden):
    @proc
    def foo(n: size, m: size, dst: i8[n + m], src: i8[n + m]):
        for i in seq(0, n):
            for j in seq(0, m):
                dst[i + j] = src[i + j]

    foo = add_unsafe_guard(foo, "dst[_] = src[_]", "i == 0 or j == m - 1")

    assert str(foo.dataflow()[0]) == golden


# TODO: fission should be able to handle this
@pytest.mark.skip()
def test_fission_fail():
    @proc
    def foo(n: size, dst: i8[n + 1], src: i8[n + 1]):
        for i in seq(0, n):
            dst[i] = src[i]
            dst[i + 1] = src[i + 1]

    with pytest.raises(SchedulingError, match="Cannot fission"):
        foo = fission(foo, foo.find("dst[i] = _").after())
        print(foo)


# TODO: This is unsafe, lift_alloc should give an error
@pytest.mark.skip()
def test_lift_alloc_unsafe(golden):
    @proc
    def foo():
        for i in seq(0, 10):
            a: i8[11] @ DRAM
            a[i] = 1.0
            a[i + 1] += 1.0

    foo = lift_alloc(foo, "a : _")

    assert str(foo.dataflow()[0]) == golden


# TODO: We are not supporting this AFAIK but should keep this example in mind
@pytest.mark.skip()
def test_reduc(golden):
    @proc
    def foo(n: size, a: f32, c: f32):
        tmp: f32[n]
        for i in seq(0, n):
            for j in seq(0, 4):
                tmp[i] = a
                a = tmp[i] + 1.0
        for i in seq(0, n):
            c += tmp[i]  # some use of tmp

    assert str(foo.dataflow()[0]) == golden


# Below are Configuration sanity checking tests


def new_config_f32():
    @config
    class ConfigAB:
        a: f32
        b: f32

    return ConfigAB


def new_control_config():
    @config
    class ConfigControl:
        i: index
        s: stride
        b: bool

    return ConfigControl


def test_config_1(golden):
    ConfigAB = new_config_f32()

    @proc
    def foo(x: f32, y: f32):
        ConfigAB.a = ConfigAB.a
        ConfigAB.a = 1.0
        ConfigAB.b = 3.0
        x = ConfigAB.a
        ConfigAB.b = ConfigAB.a

    assert str(foo.dataflow()[0]) == golden


def test_config_2(golden):
    ConfigAB = new_config_f32()

    @proc
    def foo(x: f32, y: f32):
        ConfigAB.a = 1.0
        ConfigAB.b = 3.0
        for i in seq(0, 10):
            x = ConfigAB.a
            ConfigAB.b = ConfigAB.a
        ConfigAB.a = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_config_3(golden):
    CTRL = new_control_config()

    @proc
    def foo(n: size):
        for i in seq(0, n):
            if CTRL.i == 2:
                CTRL.i = 4
            if n == n - 1:
                CTRL.i = 3

    assert str(foo.dataflow()[0]) == golden


def test_config_4(golden):
    CTRL = new_control_config()

    @proc
    def foo(n: size, src: [i8][n]):
        assert stride(src, 0) == CTRL.s
        pass

    assert str(foo.dataflow()[0]) == golden


# Below are function inlining tests


def test_function(golden):
    @proc
    def bar():
        for i in seq(0, 10):
            A: i8
            A = 0.3

    @proc
    def foo(n: size, src: [i8][n]):
        bar()

    assert str(foo.dataflow()[0]) == golden


def test_window_stmt(golden):
    @proc
    def foo(n: size, src: [i8][20]):
        tmp = src[0:10]
        for i in seq(0, 10):
            tmp[i] = 1.0

    assert str(foo.dataflow()[0]) == golden


def test_config_function(golden):
    ConfigAB = new_config_f32()

    @proc
    def bar(z: f32):
        z = 3.0
        ConfigAB.a = 2.0

    @proc
    def foo(x: f32):
        ConfigAB.a = 1.0
        bar(x)
        ConfigAB.b = x

    assert str(foo.dataflow()[0]) == golden


def test_usub(golden):
    @proc
    def foo(n: size, tmp: R[n]):
        x: R
        for i in seq(0, n - n + (n / 1)):
            x = -1.0
            tmp[i] = -x

    assert str(foo.dataflow()[0]) == golden


def test_usub2(golden):
    @config
    class CFG:
        a: index

    @proc
    def foo(N: size, x: R[N]):
        CFG.a = N - 1
        CFG.a = -N
        CFG.a = (3 % 1) + 0
        CFG.a = -1 + N
        for i in seq(0, N):
            x[i] = x[CFG.a] + 1.0

    assert str(foo.dataflow()[0]) == golden


def test_builtin(golden):
    @config
    class CFG:
        a: f32

    @proc
    def foo(n: index, x: f32):
        CFG.a = sin(x)
        CFG.a = sin(3.0)
        CFG.a = -sin(4.0)
        CFG.a = 3.0 * 2.0
        CFG.a = 3.0 - 2.0
        CFG.a = 3.0 / 2.0
        CFG.a = 3.0

    assert str(foo.dataflow()[0]) == golden


def test_bool(golden):
    @config
    class CFG:
        a: bool

    @proc
    def foo(n: index, x: f32):
        CFG.a = 3 > 2
        CFG.a = 3 < 2
        CFG.a = 3 >= 2
        CFG.a = 3 <= 2
        CFG.a = 3 == 2
        CFG.a = 3 == 3
        CFG.a = 3 == 3 or 2 == 1
        CFG.a = 3 == 3 and 2 == 1

    assert str(foo.dataflow()[0]) == golden


def test_builtin_true(golden):
    @config
    class CFG:
        a: f32

    @proc
    def foo(x: f32):
        CFG.a = sin(3.0)
        CFG.a = -CFG.a
        x = CFG.a

    print(foo.dataflow()[0])
    assert str(foo.dataflow()[0]) == golden


def test_simple_call(golden):
    @proc
    def barbar(z: f32):
        z = 0.2

    @proc
    def bar(z: f32):
        z = 1.2
        barbar(z)

    @proc
    def foo(z: R):
        z = 4.2
        bar(z)
        z = 2.0
        barbar(z)

    assert str(foo.dataflow()[0]) == golden


def test_simple_call_window(golden):
    @proc
    def barbar(z: f32[2]):
        z[0] = 0.2

    @proc
    def bar(z: f32[5]):
        z[0] = 1.2
        barbar(z[2:4])

    @proc
    def foo(z: f32[10]):
        z[0] = 4.2
        bar(z[1:6])
        z[2] = 2.0
        barbar(z[8:10])

    assert str(foo.dataflow(foo.find("z = _ #0"))[0]) == golden


def test_simple_scalar(golden):
    @proc
    def foo(N: size, x: i8, src: i8[N]):
        x = 3.0
        for k in seq(0, N):
            x = x * x
            if k == 0:
                x = 0.0
            else:
                x = src[k]

    assert str(foo.dataflow()[0]) == golden


def test_arrays(golden):
    @proc
    def foo(n: size, m: size, dst: i8[n + m] @ DRAM, src: i8[n + m] @ DRAM):
        for i in seq(0, n):
            for j in seq(0, m):
                if i == 0 or j == m - 1:
                    dst[i + j] = src[i + j]
                    dst[0] = 2.0
                    dst[i] = 1.0

    print(foo.dataflow()[0])
    assert str(foo.dataflow()[0]) == golden


def test_arrays2(golden):
    @proc
    def foo(n: size, m: size, dst: i8[n + m] @ DRAM, src: i8[n + m] @ DRAM):
        for i in seq(0, n):
            for j in seq(0, m):
                dst[i + j] = src[i + j]

    assert str(foo.dataflow()[0]) == golden


# TODO: make  configs able to depend on iteration variables and unskip this test
@pytest.mark.skip()
def test_config_5(golden):
    @config
    class CFG:
        a: index

    @proc
    def foo(x: f32[10]):
        CFG.a = 0
        for i in seq(0, 10):
            CFG.a = i
            CFG.a = i + 1  # THIS
            x[CFG.a] = 0.2

    assert str(foo.dataflow()[0]) == golden


def test_function_1(golden):
    @proc
    def bar(dst: f32[8]):
        for i in seq(0, 8):
            dst[i] += 2.0

    @proc
    def foo(n: size, x: f32[n]):
        assert n > 10
        tmp: f32[11]
        tmp[10] = 3.0
        bar(tmp[0:8])
        for i in seq(0, n):
            if i < 11:
                x[i] = tmp[i]
            x[i] += 1.0

    assert str(foo.dataflow()[0]) == golden


def test_reduc_1(golden):
    @proc
    def foo(N: size, dst: f32[N], src: f32[N]):
        dst[0] = 1.0
        for i in seq(0, N - 1):
            if i == 1:
                dst[i] = dst[i - 1] - src[i]
            dst[i] += src[i]
            dst[i + 1] = 3.0

    assert str(foo.dataflow()[0]) == golden


def test_reduc_2(golden):
    @proc
    def foo(K: size, x: f32, dst: f32[K]):
        x = 3.0
        for k in seq(0, K):
            x += dst[k]
        x = x + 1.0

    assert str(foo.dataflow()[0]) == golden


def test_config_assert(golden):
    @config
    class CFG:
        a: index

    @proc
    def foo(N: size, x: f32[N]):
        assert CFG.a < N
        for i in seq(0, N):
            if CFG.a == 3:
                CFG.a = 2
                for j in seq(0, CFG.a):
                    x[j] = 2.0

    print(foo.dataflow()[0])
    assert str(foo.dataflow()[0]) == golden


def test_sliding(golden):
    @proc
    def blur(g: R[100] @ DRAM, inp: R[102] @ DRAM):
        f: R[101] @ DRAM
        for i in seq(0, 100):
            for j in seq(0, 101):
                f[j] = inp[j] + inp[j + 1]
            g[i] = f[i] + f[i + 1]

    print(blur.dataflow()[0])


def test_sliding2(golden):
    @proc
    def blur(N: size, y: R[N]):
        x: R[N + 1]
        for i in seq(0, N):
            x[i + 1] = y[i]
        for i in seq(0, N):
            x[i] = y[i]

    print(blur.dataflow()[0])


def test_reverse():
    @proc
    def foo(N: size, x: R[N]):
        for i in seq(1, N):
            x[N - i] = 3.0
            x[i] = 1.0

    print(foo.dataflow()[0])


# sanity checked 02/04/2025
def test_reverse_x_11(golden):
    @proc
    def foo(x: R[11]):
        for i in seq(0, 11):
            x[10 - i] = 3.0
            x[i] = 1.0

    # should be:
    # 0 - 4 : 3.0
    # 5 - 10 : 1.0
    res = foo.dataflow()[0]
    print(res)
    assert str(res) == golden


# sanity checked 02/04/2025
def test_reverse_x_10(golden):
    @proc
    def foo(x: R[11]):
        for i in seq(0, 10):
            x[10 - i] = 3.0
            x[i] = 1.0

    # should be:
    # 0 : 1.0
    # 1 - 4 : 3.0
    # 5 - 9 : 1.0
    # 10 : 3.0
    res = foo.dataflow()[0]
    print(res)
    assert str(res) == golden


# sanity checked 02/04/2025
def test_reverse_x_10_lo(golden):
    @proc
    def foo(x: R[11]):
        for i in seq(3, 9):
            x[10 - i] = 3.0
            x[i] = 1.0

    # should be:
    # 0 - 1 : x[d0]
    # 2 - 4 : 3.0
    # 5 - 8 : 1.0
    # 9 -10 : x[d0]
    res = foo.dataflow()[0]
    print(res)
    assert str(res) == golden


# sanity checked 02/04/2025
def test_two_loops(golden):
    @proc
    def foo(x: f32[10]):
        for i in seq(0, 10):
            x[i] = 0.0
        for i in seq(0, 10):
            x[i] = 0.0

    res = foo.dataflow()[0]
    print(res)
    assert str(res) == golden


# sanity checked 02/04/2025
def test_super_simple(golden):
    @proc
    def foo(x: f32[5]):
        for i in seq(0, 5):
            x[i] = 0.0

    res = foo.dataflow()[0]
    print(res)
    assert str(res) == golden


def test_super_simple2():
    @proc
    def foo(x: f32[10], y: f32[5]):
        for i in seq(0, 5):
            x[2 * i] = y[i]

    res = foo.dataflow()[0]
    print(res)


# ---------------------------------
# Various failing examples below
# ---------------------------------

# FIXME: This is sound but currently imprecise. This is a limtitation of the current design. We might need to color the hyyperplanes as well as the cells.
def test_orig_array():
    @proc
    def foo(x: R[10]):
        for i in seq(0, 10):
            x[i] = x[i]

    print(foo.dataflow()[0])


def test_simple_read():
    @proc
    def foo(x: f32[5], y: f32[5]):
        for i in seq(0, 5):
            x[i] = y[i]

    res = foo.dataflow()[0]
    print(res)


def test_simple_read_nest_same():
    @proc
    def foo(x: f32[5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[j] = x[j]

    res = foo.dataflow()[0]
    print(res)


def test_simple_read_nest():
    @proc
    def foo(x: f32[5], y: f32[5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i] = y[j]

    res = foo.dataflow()[0]
    print(res)


def test_add_unsafe_guard():
    @proc
    def foo(dst: R[30], src: R[30]):
        for i in seq(0, 20):
            for j in seq(0, 10):
                dst[i + j] = src[i + j]

    res = foo.dataflow()[0]
    print(res)


def test_add_unsafe_guard2():
    @proc
    def foo(dst: R[30], src: R[30]):
        for i in seq(0, 20):
            for j in seq(0, 10):
                if i == 0 or j == 9:
                    dst[i + j] = src[i + j]

    res = foo.dataflow()[0]
    print(res)


def test_array_rhs():
    @proc
    def foo(x: R[10]):
        for i in seq(0, 10):
            x[i] = 3.0 + 2.0

    print(foo.dataflow()[0])


# Incorrect due to the lack of partitioning
# FIXME: There's a bug in the ssa translation, x[i, i] should be x[i, d0].
def test_reverse_const():
    @proc
    def foo(x: R[10], y: R[10]):
        for i in seq(1, 10):
            x[10 - i] = 3.0
            y[i] = x[i]

    print(foo.dataflow()[0])


# Current analysis output is incorrect on this example. We'll need to run the fixpoint on y's loopstart as well
def test_reverse2():
    @proc
    def foo(N: size, x: R[N], y: R[N]):
        for i in seq(1, N):
            x[N - i] = 3.0
            y[i] = x[i]

    print(foo.dataflow()[0])


def test_reverse3():
    @proc
    def foo(N: size, x: R[N], y: R[N]):
        for i in seq(1, N):
            x[N - i] = y[i]
            y[i] = x[i]

    print(foo.dataflow()[0])


def test_mod():
    @proc
    def foo(N: size, x: R[N]):
        for i in seq(1, N):
            x[i] = 1.0
            if i % 3 == 0:
                x[i - 1] = 2.0

    print(foo.dataflow()[0])


def test_diagonal():
    @proc
    def foo(x: R[20]):
        for i in seq(0, 20):
            x[i] = 3.0
            x[20 - i - 1] = 1.0
            if i - 5 >= 0:
                x[i - 3] = x[i - 5]

    print(foo.dataflow()[0])


def test_diagonal2():
    @proc
    def foo(x: R[5]):
        for i in seq(0, 5):
            x[i] = 3.0
            x[5 - i - 1] = 1.0
            if i - 1 >= 0:
                x[i] = x[i - 1]

    print(foo.dataflow()[0])


def test_diagonal3():
    @proc
    def foo(x: R[5]):
        for i in seq(0, 5):
            x[i] = 3.0
            x[5 - i - 1] = 1.0
            if i - 2 >= 0:
                x[i] = x[i - 2]

    print(foo.dataflow()[0])
