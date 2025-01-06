from __future__ import annotations
import pytest
from exo import proc, DRAM, Procedure, config
from exo.stdlib.scheduling import *
from exo.dataflow import (
    D,
    substitute,
    sub_aexpr,
    partition,
    V,
    abs_simplify,
    widening,
    cvt_vdom,
    cvt_val,
    adom_to_aexpr,
)
from exo.prelude import Sym


def test_cvt():
    t = cvt_vdom(V.ValConst(3.0))

    i = Sym("i")
    d = Sym("d")
    N = Sym("N")
    vi = D.Var(i)
    vd = D.Var(d)
    vN = D.Var(N)

    sx_1 = Sym("x_1")
    sx = Sym("x")
    x_1 = D.ArrayConst(sx_1, [D.Add(vi, D.Const(-1)), vd])
    x_2 = D.ArrayConst(sx_1, [D.Add(vi, D.Const(3)), vd])

    print(cvt_val(x_1))
    print(cvt_val(x_2))


def test_widening2():
    i = Sym("i")
    d = Sym("d")
    N = Sym("N")
    vi = D.Var(i)
    vd = D.Var(d)
    vN = D.Var(N)

    sx_1 = Sym("x_1")
    sx = Sym("x")
    x_1 = D.ArrayConst(sx_1, [D.Add(vi, D.Const(-1)), vd])
    x = D.ArrayConst(sx, [vd])
    eq2 = D.Add(D.Add(vi, D.Const(-1)), D.Mult(-1, vd))
    bot = D.Leaf(D.SubVal(V.Bot()))
    eq3 = D.Add(D.Add(D.Add(vi, D.Mult(-1, D.Const(1))), vd), D.Mult(-1, vN))
    tree_eq3 = D.AffineSplit(
        eq3, D.Leaf(x_1), D.Leaf(D.SubVal(V.ValConst(3.0))), D.Leaf(x_1)
    )
    x_1_tree = D.AffineSplit(
        vi,
        bot,
        D.Leaf(x),
        D.AffineSplit(eq2, tree_eq3, D.Leaf(D.SubVal(V.ValConst(1.0))), tree_eq3),
    )
    x_1_abs = D.abs([i, N, d], x_1_tree)

    print(x_1_abs)
    widened_x_1 = partition(sx_1, x_1_abs)
    print(widened_x_1)

    i_minus = D.Add(vi, D.Const(-1))
    x_1_i_1 = abs_simplify(sub_aexpr(vi, i_minus, widened_x_1))
    print(x_1_i_1)

    after = substitute(x_1, x_1_i_1.tree, widened_x_1)
    after = abs_simplify(after)
    after = abs_simplify(widening(sx_1, after))
    after = abs_simplify(after)
    print(after)
    print(adom_to_aexpr(sx_1, after))


def test_widening():
    i = Sym("i")
    d = Sym("d")
    vi = D.Var(i)
    vd = D.Var(d)

    sx_2 = Sym("x_2")
    x_2_ = D.ArrayConst(sx_2, [D.Minus(vi, D.Const(1)), vd])
    y = D.ArrayConst(Sym("y"), [vi, vi])
    x_3_tree = D.AffineSplit(
        D.Add(D.Minus(vi, vd), D.Const(1)), D.Leaf(x_2_), D.Leaf(y), D.Leaf(x_2_)
    )
    print()
    print(x_3_tree)

    x_3 = D.ArrayConst(Sym("x_3"), [vi, vd])
    x_2_tree = D.AffineSplit(D.Minus(vi, vd), D.Leaf(x_3), D.Leaf(y), D.Leaf(x_3))
    x_2_abs = D.abs([i, d], x_2_tree)
    print(x_2_abs)

    new_x_2 = substitute(x_3, x_3_tree, x_2_abs)
    print(new_x_2)

    widened_x_2 = widening(sx_2, new_x_2)
    print(widened_x_2)


def test_substitute_mod():
    i = Sym("i")
    d = Sym("d")
    vi = D.Var(i)
    vd = D.Var(d)
    x_3 = D.ArrayConst(Sym("x_3"), [vi, vd])
    x_4 = D.ArrayConst(Sym("x_4"), [vi, vd])
    x_1 = D.ArrayConst(Sym("x_1"), [vi, vd])
    x_2_tree = D.ModSplit(vi, 3, D.Leaf(x_3), D.Leaf(x_4))
    x_2_abs = D.abs([i, d], x_2_tree)
    x_4_tree = D.AffineSplit(
        D.Minus(vi, D.Add(vd, D.Const(1))),
        D.Leaf(x_3),
        D.Leaf(D.ValConst(2.0)),
        D.Leaf(x_3),
    )
    new_x_2 = substitute(x_4, x_4_tree, x_2_abs)
    x_3_tree = D.AffineSplit(
        D.Minus(vi, vd), D.Leaf(x_1), D.Leaf(D.ValConst(1.0)), D.Leaf(x_1)
    )

    new_new_x_2 = substitute(x_3, x_3_tree, new_x_2)
    print(x_2_tree)
    print(x_4_tree)
    print(x_3_tree)
    print(new_x_2)
    print(new_new_x_2)

    i_minus = D.Minus(vi, D.Const(1))
    final_x2 = sub_aexpr(vi, i_minus, new_new_x_2)
    print(final_x2)

    x = D.ArrayConst(Sym("x"), [vd])
    x_2_minus = D.ArrayConst(Sym("x_2"), [i_minus, vd])
    x_1_tree = D.AffineSplit(vi, D.Leaf(D.Bot()), D.Leaf(x), D.Leaf(x_2_minus))
    x_1_abs = D.abs([i, d], x_1_tree)
    print(x_1_tree)
    final_x_1 = substitute(x_2_minus, final_x2.tree, x_1_abs)
    print(final_x_1)


def test_substitute():
    i = Sym("i")
    d = Sym("d")
    vi = D.Var(i)
    vd = D.Var(d)

    x_1 = D.ArrayConst(Sym("x_1"), [vi, vd])
    y = D.ArrayConst(Sym("y"), [vi, vi])
    x_3_tree = D.AffineSplit(
        D.Add(D.Minus(vi, vd), D.Const(1)), D.Leaf(x_1), D.Leaf(y), D.Leaf(x_1)
    )
    print()
    print(x_3_tree)

    x_3 = D.ArrayConst(Sym("x_3"), [vi, vd])
    x_2_tree = D.AffineSplit(D.Minus(vi, vd), D.Leaf(x_3), D.Leaf(y), D.Leaf(x_3))
    x_2_abs = D.abs([i, d], x_2_tree)
    print(x_2_abs)

    new_x_2 = substitute(x_3, x_3_tree, x_2_abs)
    print(new_x_2)


def test_abs_pprint():
    x = Sym("x")
    y = Sym("y")
    vx = D.Var(x)
    vy = D.Var(y)
    one = D.Const(1)
    print(one)
    print(vx)
    add = D.Add(vx, vy)
    mul = D.Mult(2, vx)
    addmul = D.Add(vx, D.Mult(4, vx))
    print(add)
    print(mul)
    print(addmul)

    print(D.Top())
    print(D.Bot())
    print(D.ValConst(4.0))
    print(D.ArrayConst(Sym("a"), [vx, vy]))
    print(D.ArrayConst(Sym("b"), []))

    i = D.Var(Sym("i"))
    d = D.Var(Sym("d"))
    p1 = D.Minus(i, d)

    four = D.ValConst(4.0)
    leaf = D.Leaf(four)
    print(leaf)
    print(D.AffineSplit(p1, leaf, leaf, leaf))

    p2 = D.Add(p1, D.Const(1))
    a1 = D.ArrayConst(Sym("y"), [i, i])
    a2 = D.ArrayConst(Sym("x"), [i, d])
    n = D.AffineSplit(
        p1,
        D.AffineSplit(p2, D.Leaf(a2), D.Leaf(a1), D.Leaf(a2)),
        D.Leaf(a1),
        D.Leaf(a2),
    )
    print(n)

    mod = D.ModSplit(p1, 3, n, leaf)
    print(mod)

    a = D.abs([x, y], mod, True)
    print(a)

    # def substitute(var : D.ArrayConst, term : D.node, src : D.abs):
    print("--- try substitute ---")
    print("var: ", a1)
    print("term: ", n)
    print("src: ", a)
    new_tree = substitute(a1, n, a)
    print(new_tree)


def test_simple(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_simple2(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        x[0] = 2.0
        x[1] = 3.0
        x[2] = 5.0
        x[0] = 12.0

    assert str(foo.dataflow()[0]) == golden


def test_simple_stmts(golden):
    @proc
    def foo(z: R, x: R[3]):
        z = 4.2
        z = 2.0

    d_ir, stmts = foo.dataflow(foo.find("z = _ ; z = _"))

    assert str(d_ir) + "".join([str(s) for s in stmts]) == golden


def test_simple_stmts2(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        pass

    d_ir, stmts = foo.dataflow(foo.find("if n < 3: _"))

    assert str(d_ir) + "".join([str(s) for s in stmts]) == golden


def test_simple3(golden):
    @proc
    def foo(z: R, n: size, x: R[3]):
        z = 4.2
        x[0] = 2.0
        if n < 3:
            x[n] = 3.0
        x[2] = 5.0
        x[0] = 12.0

    print(foo.dataflow()[0])
    assert str(foo.dataflow()[0]) == golden


def test_print(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        z = 4.2
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_print_0(golden):
    @proc
    def foo(z: R[3]):
        z[0] = 1.0
        for i in seq(0, 3):
            z[i] = 3.0
        z[2] = 2.0

    print(foo.dataflow()[0])
    assert str(foo.dataflow()[0]) == golden


def test_print_new(golden):
    @proc
    def foo(z: R[3]):
        for i in seq(0, 3):
            z[i] = 3.0

    print(foo.dataflow()[0])


def test_print_1(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R[3]):
        z[0] = x[0] * y[2]
        for i in seq(0, 3):
            z[i] = 3.0
        z[2] = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_print_2(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_print_3(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 4.0
        z = 0.0

    assert str(foo.dataflow()[0]) == golden


def test_print_4(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]
        if 0 == 0:
            z = 3.0
        else:
            z = 3.0
        z = 0.0

    assert str(foo.dataflow()[0]) == golden


def test_print_5(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = 3.0
        for i in seq(0, 3):
            z = 3.0
        z = 2.0

    assert str(foo.dataflow()[0]) == golden


def test_sliding_window_debug():
    @proc
    def foo(dst: i8[30]):
        for i in seq(0, 10):
            for j in seq(0, 20):
                dst[i] = 2.0

    print(foo.dataflow()[0])


def test_sliding_window_debug2():
    @proc
    def foo(dst: i8[30]):
        for i in seq(0, 10):
            for j in seq(0, 20):
                dst[j] = 2.0

    print(foo.dataflow()[0])


def test_sliding_window_print():
    @proc
    def foo(n: size, m: size, dst: i8[n + m]):
        for i in seq(0, n):
            for j in seq(0, m):
                dst[i + j] = 2.0

    print(foo.dataflow()[0])


def test_multi_dim_print():
    @proc
    def foo(n: size, dst: i8[n, n]):
        for i in seq(0, n):
            for j in seq(1, n):
                dst[i, n - j] = 2.0

    print(foo.dataflow()[0])


# TODO: Currently add_unsafe_guard lacks analysis, but we should be able to analyze this
def test_sliding_window(golden):
    @proc
    def foo(n: size, m: size, dst: i8[n + m], src: i8[n + m]):
        for i in seq(0, n):
            for j in seq(0, m):
                dst[i + j] = src[i + j]

    foo = add_unsafe_guard(foo, "dst[_] = src[_]", "i == 0 or j == m - 1")

    assert str(foo.dataflow()[0]) == golden


# TODO: fission should be able to handle this
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


def test_absval_init(golden):
    @proc
    def foo1(n: size, dst: f32[n]):
        for i in seq(0, n):
            dst[i] = 0.0

    @proc
    def foo2(n: size, dst: f32[n], src: f32[n]):
        for i in seq(0, n):
            dst[i] = src[i]

    assert str(foo1.dataflow()[0]) + str(foo2.dataflow()[0]) == golden


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

    print(foo.dataflow()[0])
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


def test_mod():
    @proc
    def foo(N: size, x: R[N]):
        for i in seq(1, N):
            x[i] = 1.0
            if i % 3 == 0:
                x[i - 1] = 2.0

    print(foo.dataflow()[0])
