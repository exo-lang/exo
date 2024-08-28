from __future__ import annotations

import pytest

from exo import ParseFragmentError, proc, DRAM, Procedure, config
from exo.libs.memories import GEMM_SCRATCH
from exo.stdlib.scheduling import *
from exo.platforms.x86 import *
from exo.API_types import *


def test_commute(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] * y[2]

    assert str(commute_expr(foo, "x[0] * y[_]")) == golden


def test_commute2():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] + y[0] + x[1] + y[1]

    with pytest.raises(SchedulingError, match="failed to find matches"):
        # TODO: Currently, expression pattern matching fails to find
        # 'y[0]+x[1]' because LoopIR.BinOp is structured as (x[0], (y[0], (x[1], y[1]))).
        # I think pattern matching should be powerful to find this.
        commute_expr(foo, "y[0] + x[1]")


def test_commute3(golden):
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = (x[0] + y[0]) * (x[1] + y[1] + y[2])

    assert str(commute_expr(foo, "(x[_] + y[_]) * (x[_] + y[_] + y[_])")) == golden


def test_commute4():
    @proc
    def foo(x: R[3], y: R[3], z: R):
        z = x[0] - y[2]

    with pytest.raises(TypeError, match="can commute"):
        commute_expr(foo, "x[0] - y[_]")


def test_left_reassociate_expr_1(golden):
    @proc
    def foo(a: f32, b: f32, c: f32):
        b = a + (b + c)

    foo = left_reassociate_expr(foo, "_ + _")
    foo = commute_expr(foo, [foo.find("_ + _")])
    assert str(foo) == golden


def test_left_reassociate_expr_2(golden):
    @proc
    def foo(a: f32, b: f32, c: f32):
        b = (a * b) * (b * c)

    foo = left_reassociate_expr(foo, "_ * _")
    foo = commute_expr(foo, [foo.find("_ * _")])
    assert str(foo) == golden


def test_reassociate_then_fold(golden):
    @proc
    def foo(a: f32, b: f32, c: f32):
        b = a + (b + c)

    foo = commute_expr(foo, [foo.find("_ + _ #1")])
    foo = left_reassociate_expr(foo, "_ + _")
    foo = commute_expr(foo, [foo.find("_ + _")])
    foo = fold_into_reduce(foo, "_ = _")
    assert str(foo) == golden


def test_left_reassociate_expr_fail_1():
    @proc
    def foo(a: f32, b: f32, c: f32):
        b = a - (b + c)

    with pytest.raises(TypeError, match="got -"):
        foo = left_reassociate_expr(foo, "_ - _")


def test_left_reassociate_expr_fail_2():
    @proc
    def foo(a: f32, b: f32, c: f32):
        b = a + (b - c)

    with pytest.raises(TypeError, match="same binary operation as the expression"):
        foo = left_reassociate_expr(foo, "_ + _")


def test_product_loop(golden):
    @proc
    def foo(n: size):
        x: R[n, 30]
        for i in seq(0, n):
            for j in seq(0, 30):
                x[i, j] = 0.0

    assert str(mult_loops(foo, "i j", "ij")) == golden


def test_product_loop2(golden):
    @proc
    def foo(n: size, x: R[n, 30]):
        for i in seq(0, n):
            for j in seq(0, 30):
                x[i, j] = 0.0

    assert str(mult_loops(foo, "i j", "ij")) == golden


def test_product_loop3():
    @proc
    def foo(n: size, m: size):
        x: R[n, m]
        for i in seq(0, n):
            for j in seq(0, m):
                x[i, j] = 0.0

    with pytest.raises(
        SchedulingError, match="expected the inner loop to have a constant bound"
    ):
        mult_loops(foo, "i j", "ij")


def test_product_loop4(golden):
    @proc
    def foo(n: size, x: R[n]):
        for i in seq(0, n):
            for j in seq(0, 30):
                x[i] = 0.0

    assert str(mult_loops(foo, "i j", "ij")) == golden


def test_product_loop5(golden):
    @proc
    def foo(n: size, m: size, x: R[n, 100]):
        assert m < n
        x2 = x[0:m, 0:30]
        for i in seq(0, m):
            for j in seq(0, 30):
                x2[i, j] = 0.0

    assert str(mult_loops(foo, "i j", "ij")) == golden


def test_product_loop_nonzero_lo():
    @proc
    def foo(n: size, x: R[n, 30]):
        for i in seq(1, n):
            for j in seq(0, 30):
                x[i, j] = 0.0

    with pytest.raises(SchedulingError, match="expected the inner and outer loops"):
        mult_loops(foo, "i j", "ij")


def test_delete_pass(golden):
    @proc
    def foo(x: R):
        pass
        x = 0.0

    assert str(delete_pass(foo)) == golden
    assert str(delete_pass(delete_pass(foo))) == golden

    @proc
    def foo(x: R):
        for i in seq(0, 16):
            for j in seq(0, 2):
                pass
        x = 0.0

    assert str(delete_pass(foo)) == golden
    assert str(delete_pass(delete_pass(foo))) == golden

    @proc
    def foo(x: R):
        for i in seq(0, 16):
            pass
            for j in seq(0, 2):
                pass
                pass
            pass
        x = 0.0

    assert str(delete_pass(foo)) == golden
    assert str(delete_pass(delete_pass(foo))) == golden


def test_delete_pass_1(golden):
    @proc
    def foo(x: R):
        for i in seq(0, 16):
            x = 1.0
            pass
            for j in seq(0, 2):
                pass
                pass
            pass
        x = 0.0

    assert str(delete_pass(foo)) == golden


def test_add_loop(golden):
    @proc
    def foo(x: R):
        x = 1.0
        x = 2.0
        x = 3.0

    foo = add_loop(foo, "x = 2.0", "i", 5)
    assert str(foo) == golden


def test_add_loop1(golden):
    @proc
    def foo():
        x: R
        x = 0.0

    assert str(add_loop(foo, "x = _", "i", 10)) == golden


def test_add_loop2(golden):
    @proc
    def foo():
        x: R
        x = 0.0

    assert str(add_loop(foo, "x = _", "i", 10, guard=True)) == golden


def test_add_loop3(golden):
    @proc
    def foo(n: size, m: size):
        x: R
        x = 0.0

    assert str(add_loop(foo, "x = _", "i", "n+m", guard=True)) == golden


def test_add_loop4_fail():
    @proc
    def foo():
        x: R
        x = 0.0

    with pytest.raises(
        TypeError, match="argument 4, 'guard' to add_loop: expected a bool"
    ):
        add_loop(foo, "x = _", "i", 10, guard=100)


def test_add_loop5_fail():
    @proc
    def foo(x: R):
        x += 1.0
        x += 2.0
        x += 3.0

    with pytest.raises(SchedulingError, match="The statement at .* is not idempotent"):
        add_loop(foo, "x += 2.0", "i", 5)


def test_add_loop6_runs_fail():
    @proc
    def foo(x: R):
        x = 1.0
        x = 2.0
        x = 3.0

    with pytest.raises(
        SchedulingError,
        match="The expression 0 is not " "guaranteed to be greater than 0",
    ):
        add_loop(foo, "x = 2.0", "i", 0)


# Should fix this test with program analysis
@pytest.mark.skip
def test_add_loop5():
    @proc
    def foo(n: size, m: size):
        x: R
        x = 0.0

    with pytest.raises(Exception, match="bound expression should be positive"):
        add_loop(foo, "x = _", "i", "n-m", guard=True)


def test_proc_equal():
    def make_foo():
        @proc
        def foo(n: size, m: size):
            assert m == 1 and n == 1
            y: R[10]
            y[10 * m - 10 * n + 2 * n] = 2.0

        return foo

    foo = make_foo()
    foo2 = Procedure(foo.INTERNAL_proc())

    assert foo != 3  # check that wrong type doesn't crash, but returns false
    assert foo == foo  # reflexivity
    assert foo == foo2  # same underlying LoopIR.proc
    assert foo2 == foo  # symmetric

    # Coincidentally identical procs created from scratch should not be
    # considered equal.
    assert foo != make_foo()


def test_simplify(golden):
    @proc
    def foo(n: size, m: size):
        x: R[n, 16 * (n + 1) - n * 16, (10 + 2) * m - m * 12 + 10]
        for i in seq(0, 4 * (n + 2) - n * 4 + n * 5):
            pass
        y: R[10]
        y[n * 4 - n * 4 + 1] = 0.0

    assert str(simplify(foo)) == golden


def test_simplify2(golden):
    @proc
    def foo(
        A: i8[32, 64] @ DRAM,
        B: i8[16, 128] @ DRAM,
        C: i32[32, 32] @ DRAM,
        ko: size,
        ji_unroll: size,
        ii_unroll: size,
    ):
        for io in seq(0, 1):
            for jo in seq(0, 1):
                Btile1: i8[
                    16 * (ko + 1) - 16 * ko,
                    128 * jo
                    + 64 * (ji_unroll + 1 + 1)
                    - (128 * jo + 64 * (ji_unroll + 1)),
                ] @ DRAM
                Btile0: i8[
                    16 * (ko + 1) - 16 * ko,
                    128 * jo + 64 * (ji_unroll + 1) - (128 * jo + 64 * ji_unroll),
                ] @ DRAM
                Atile0: i8[
                    32 * io + 16 * (ii_unroll + 1) - (32 * io + 16 * ii_unroll),
                    64 * (ko + 1) - 64 * ko,
                ] @ DRAM
                Atile1: i8[
                    32 * io
                    + 16 * (ii_unroll + 1 + 1)
                    - (32 * io + 16 * (ii_unroll + 1)),
                    64 * (ko + 1) - 64 * ko,
                ] @ DRAM

    assert str(simplify(foo)) == golden


def test_simplify3(golden):
    @proc
    def foo(n: size, m: size):
        assert m == 1 and n == 1
        y: R[10]
        y[10 * m - 10 * n + 2 * n] = 2.0

    assert str(simplify(foo)) == golden


def test_simplify4(golden):
    @proc
    def bar():
        for i in seq(0, 3):
            for j in seq(0, i * 16 + 16 - i * 16):
                pass

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_loop_bounds(golden):
    @proc
    def foo(n: size):
        for i in seq(2 + 5 + n, 9 + 8 + n):
            pass

    assert str(simplify(foo)) == golden


def test_simplify_nested_div(golden):
    @proc
    def foo(n: size):
        x: f32
        for i in seq(0, (n / 4) / 3 / 2):
            x = 0.0

    assert str(simplify(foo)) == golden


def test_simplify_nested_div_2(golden):
    @proc
    def foo(n: size):
        x: f32
        for ii in seq(0, n):
            for i in seq(0, (ii + n / 4 * 4) / 8):
                x = 0.0

    assert str(simplify(foo)) == golden


def test_pattern_match():
    @proc
    def foo(N1: size, M1: size, K1: size, N2: size, M2: size, K2: size):
        x: R[N1, M1, K1]
        x: R[N2, M2, K2]

    res1 = rearrange_dim(foo, "x : _", [2, 1, 0])
    res1 = rearrange_dim(res1, "x : _ #1", [2, 1, 0])
    res2 = rearrange_dim(foo, "x : R[N1, M1, K1]", [2, 1, 0])

    assert str(res1) != str(res2)


def test_fission_after_simple(golden):
    @proc
    def foo(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    @proc
    def bar(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

        for k in seq(0, 30):
            for l in seq(0, 100):
                x: i8
                x = 4.0
                y: f32
                y = 1.1

    cases = [
        fission(foo, foo.find("x = _").after(), n_lifts=2),
        fission(bar, bar.find("x = _").after(), n_lifts=2),
    ]

    assert "\n".join(map(str, cases)) == golden


def test_fission_after_simple_fail():
    @proc
    def foo():
        for i in seq(0, 4):
            x: i8
            x = 0.0
            y: i8
            y = 1.0

    with pytest.raises(SchedulingError, match="Can only lift past a for loop"):
        fission(foo, foo.find("x = 0.0").after(), n_lifts=2)


def test_resize_dim(golden):
    @proc
    def foo():
        x: i8[10]
        for i in seq(1, 9):
            x[i] = 1.0

    foo = resize_dim(foo, "x", 0, 19, 1)
    assert str(simplify(foo)) == golden

    with pytest.raises(SchedulingError, match="The buffer x is accessed out-of-bounds"):
        foo = resize_dim(foo, "x", 0, 7, 1)
    with pytest.raises(SchedulingError, match="The buffer x is accessed out-of-bounds"):
        foo = resize_dim(foo, "x", 0, 7, 2)


def test_resize_dim_2(golden):
    @proc
    def foo(n: size):
        assert n > 4
        x: i8[n]
        for i in seq(2, n - 1):
            x[i] = 1.0

    foo = resize_dim(foo, "x", 0, "n-3", 2)
    assert str(simplify(foo)) == golden

    with pytest.raises(SchedulingError, match="The buffer x is accessed out-of-bounds"):
        foo = resize_dim(foo, "x", 0, "n-4", 2)
    with pytest.raises(SchedulingError, match="The buffer x is accessed out-of-bounds"):
        foo = resize_dim(foo, "x", 0, "n-4", 3)


def test_resize_dim_3(golden):
    @proc
    def foo(n: size):
        x: i8[n + 4]
        for i in seq(n + 1, n + 3):
            x[i] = 1.0

    foo = resize_dim(foo, "x", 0, 2, "n + 1")
    assert str(foo) == golden


def test_resize_dim_4(golden):
    @proc
    def bar(A: [i8][3]):
        for i in seq(0, 3):
            A[i] = 0.0

    @proc
    def foo1():
        x: i8[10]
        for i in seq(3, 6):
            bar(x[i : i + 3])

    @proc
    def foo2():
        x: i8[10, 10]
        for i in seq(3, 6):
            bar(x[i, i : i + 3])

    foo1 = resize_dim(foo1, "x", 0, 6, "2")
    foo2 = resize_dim(foo2, "x", 0, 15, "2")

    assert str(foo1) + "\n" + str(foo2) == golden


def test_resize_dim_5(golden):
    @proc
    def foo():
        x: i8[8]
        for i in seq(1, 8):
            x[i] = 1.0

    assert str(resize_dim(foo, "x", 0, 10, -1)) == golden
    with pytest.raises(SchedulingError, match="The buffer x is accessed out-of-bounds"):
        foo = resize_dim(foo, "x", 0, 7, 0)


def test_rearrange_dim(golden):
    @proc
    def foo(N: size, M: size, K: size, x: i8[N, M, K]):
        a: i8[N, M, K]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[n, m, k] = x[n, m, k]

    @proc
    def bar(N: size, M: size, K: size, x: i8[N, M, K]):
        a: i8[N, M, K]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[n, m, k] = x[n, m, k]

        a: i8[M, K, N]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[m, k, n] = x[n, m, k]

    @proc
    def baz(N: size, M: size, x: i8[N, M]):
        a: i8[N, M]
        for n in seq(0, N):
            for m in seq(0, M):
                a[n, m] = x[n, m]

    foo = rearrange_dim(foo, "a : i8[_]", [1, 2, 0])
    bar = rearrange_dim(bar, "a : i8[_]", [1, 0, 2])
    bar = rearrange_dim(bar, "a : i8[_] #1", [1, 0, 2])
    baz = rearrange_dim(baz, "a : i8[_]", [1, 0])
    cases = [foo, bar, baz]

    assert "\n".join(map(str, cases)) == golden


def test_rearrange_dim_2(golden):
    @proc
    def bar(s: stride):
        pass

    @proc
    def foo():
        a: i8[10, 10]
        for i in seq(0, 10):
            for j in seq(0, 10):
                a[i, j] = a[j, i]
                bar(stride(a, 1))

    foo = rearrange_dim(foo, "a : _", [1, 0])
    assert str(foo) == golden


def test_rearrange_dim_fail():
    @proc
    def foo(N: size, M: size, K: size, x: i8[N, M, K]):
        a: i8[N, M, K]
        for n in seq(0, N):
            for m in seq(0, M):
                for k in seq(0, K):
                    a[n, m, k] = x[n, m, k]

    perm = [1, 1, 0]
    for p in (perm, perm + [2]):
        with pytest.raises(ValueError, match="was not a permutation of"):
            rearrange_dim(foo, "a : i8[_]", p)


def test_rearrange_dim_fail2():
    @proc
    def bar(m: size, a: [i8][m, m]):
        a[0, 0] += 1.0

    @proc
    def foo1():
        a: i8[10, 10]
        bar(10, a[0:10, 0:10])

    with pytest.raises(SchedulingError, match="Cannot permute buffer "):
        rearrange_dim(foo1, "a : i8[_]", [1, 0])

    @proc
    def foo2():
        a: i8[10, 10]
        x = a[0:10, 0:10]

    with pytest.raises(SchedulingError, match="windows is not currently supported"):
        rearrange_dim(foo2, "a : i8[_]", [1, 0])


def test_remove_loop(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x = a

    @proc
    def bar(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    cases = [
        remove_loop(foo, "for i in _:_"),
        remove_loop(bar, "for i in _:_"),
    ]

    assert "\n".join(map(str, cases)) == golden


def test_remove_loop_fail(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x += a

    with pytest.raises(SchedulingError, match="The statement at .* is not idempotent"):
        remove_loop(foo, "for i in _:_")


def test_remove_loop_deterministic(golden):
    @proc
    def foo(M: size, N: size, K: size, A: f32[M, N]):
        for k in seq(0, K / 4):
            for i in seq(0, M):
                for j in seq(0, N):
                    A[i, j] = 1.0

    # An older Z3 version caused check within remove_loop
    # to fail non-deterministically (return an unknwon result).
    # This test make sure that over a few runs, it always passes.
    for i in range(10):
        assert str(remove_loop(foo, "k")) == golden


def test_sink_alloc_simple_for_loop(golden):
    @proc
    def foo():
        a: i8[10] @ DRAM
        for i in seq(0, 10):
            pass

    foo = sink_alloc(foo, foo.find("a : _"))
    assert str(foo) == golden


def test_sink_alloc_simple_if_stmt(golden):
    @proc
    def foo():
        a: i8[10] @ DRAM
        if 1 < 10:
            a[1] = 0.0

    foo = sink_alloc(foo, foo.find("a : _"))
    assert str(foo) == golden


def test_sink_alloc_when_if_has_else(golden):
    @proc
    def foo():
        a: i8[10] @ DRAM
        if 1 < 10:
            a[1] = 0.0
        else:
            a[1] = 1.0

    foo = sink_alloc(foo, foo.find("a : _"))
    assert str(foo) == golden


def test_sink_alloc_fail_because_accesses_outside_scope():
    @proc
    def foo():
        a: i8[10] @ DRAM
        for i in seq(0, 10):
            pass
        a[0] = 0.0

    with pytest.raises(SchedulingError, match="Cannot sink allocation"):
        foo = sink_alloc(foo, foo.find("a : _"))


def test_lift_alloc_simple(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    bar = lift_alloc(bar, "tmp_a : _", n_lifts=2)
    assert str(bar) == golden


def test_lift_alloc_simple2(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    bar = lift_alloc(bar, "tmp_a : _", n_lifts=2)
    assert str(bar) == golden


def test_lift_alloc_simple3(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for k in seq(0, n):
            for i in seq(0, n):
                for j in seq(0, n):
                    tmp_a: i8
                    tmp_a = A[i]

    bar = lift_alloc(bar, "tmp_a : _", n_lifts=3)
    assert str(bar) == golden


def test_lift_alloc_simple_empty_body(golden):
    @proc
    def bar():
        for i in seq(0, 4):
            tmp: i8

    bar = lift_alloc(bar, "tmp: _")
    assert str(bar) == golden


def test_lift_alloc_simple_fv_error():
    @proc
    def bar(n: size, A: i8[n]):
        for k in seq(0, n):
            for i in seq(0, n):
                for j in seq(0, n):
                    tmp_a: i8[k + 1]
                    tmp_a[k] = A[i]

    with pytest.raises(SchedulingError, match="Cannot lift allocation statement"):
        lift_alloc(bar, "tmp_a : _", n_lifts=3)


def test_lift_alloc_simple_error():
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    with pytest.raises(SchedulingError, match="specified lift level"):
        lift_alloc(bar, "tmp_a : _", n_lifts=3)


def test_autolift_alloc_error():
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8
                tmp_a = A[i]

    with pytest.raises(SchedulingError, match="specified lift level 3 is"):
        autolift_alloc(bar, "tmp_a : _", n_lifts=3)


def test_expand_dim1_bad_scope():
    @proc
    def foo(n: size, m: size, x: i8):
        a: i8
        for i in seq(0, n):
            for j in seq(0, m):
                x = a

    with pytest.raises(ParseFragmentError, match="i not found in"):
        expand_dim(foo, "a : i8", "n", "i")  # should be error


def test_expand_dim2_bad_scope():
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            a: i8
            for j in seq(0, m):
                x = a

        for i in seq(0, n):
            for k in seq(0, m):
                pass

    with pytest.raises(ParseFragmentError, match="k not found in"):
        foo = expand_dim(foo, "a : i8", "n", "k")  # should be error


def test_expand_dim3(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                pass

        for i in seq(0, n):
            a: i8
            for j in seq(0, m):
                x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    foo = expand_dim(foo, "a : i8", "n", "i")  # did it pick the right i?
    assert foo.c_code_str() == golden


def test_expand_dim4(golden):
    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                pass

        for q in seq(0, 30):
            for i in seq(0, n):
                for j in seq(0, m):
                    a: i8
                    x = a

        for i in seq(0, n):
            for j in seq(0, m):
                pass

    with pytest.raises(
        SchedulingError,
        match="The expression 10 - 20 is not guaranteed to be greater than 0",
    ):
        expand_dim(foo, "a : i8", "10-20", "10")  # this is not fine

    with pytest.raises(
        SchedulingError,
        match="The expression n - m is not guaranteed to be greater than 0",
    ):
        expand_dim(foo, "a : i8", "n - m", "i")  # out of bounds

    with pytest.raises(ParseFragmentError, match="not found in"):
        expand_dim(foo, "a : i8", "hoge", "i")  # does not exist

    with pytest.raises(SchedulingError, match="The buffer a is accessed out-of-bounds"):
        expand_dim(foo, "a : i8", "n", "i-j")  # bound check should fail

    cases = [
        expand_dim(foo, "a : i8", "n", "i"),  # did it pick the right i?
        expand_dim(foo, "a : i8", "40 + 1", "10"),  # this is fine
        expand_dim(foo, "a : i8", "n + m", "i"),  # fine
        expand_dim(foo, "a : i8", "n", "n-1"),
    ]

    assert "\n".join(map(str, cases)) == golden


def test_expand_dim5(golden):
    @proc
    def foo(n: size, x: i8):
        for i in seq(0, n):
            a: i8
            a = x

    foo = expand_dim(foo, "a : i8", "n", "i")
    assert str(foo) == golden


def test_expand_dim6(golden):
    @proc
    def bar(m: size, a: i8[m]):
        a[0] += 1.0

    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                a: i8[m]
                a[j] = a[j] + 1.0
                a[j] += 1.0
                bar(m, a[0:m])

    foo = expand_dim(foo, "a : _", "n", "i")
    assert str(foo) == golden


def test_expand_dim7():
    @proc
    def bar(m: size, a: i8[m]):
        a[0] += 1.0

    @proc
    def foo(n: size, m: size, x: i8):
        for i in seq(0, n):
            for j in seq(0, m):
                a: i8[m]
                a[j] = a[j] + 1.0
                bar(m, a)

    with pytest.raises(
        SchedulingError, match="support for passing windows to scalar arguments"
    ):
        foo = expand_dim(foo, "a : _", "n", "i")


def test_pattern_matching_id_in_scheduling_ops(golden):
    @proc
    def bar(n: size, ret: i8):
        reg: i8[n]
        for i in seq(0, n):
            ret += reg[i] + 1.0

    bar = bind_expr(bar, "1.0", "reg")
    scalar_reg = bar.find("reg : _ #1")
    bar = expand_dim(bar, scalar_reg, "n", "i")
    assert str(bar) == golden


def test_divide_dim_1(golden):
    @proc
    def foo(n: size, m: size, A: R[n + m + 12]):
        x: R[n, 12, m]
        for i in seq(0, n):
            for j in seq(0, 12):
                for k in seq(0, m):
                    x[i, j, k] = A[i + j + k]

    foo = divide_dim(foo, "x", 1, 4)
    assert str(foo) == golden


def test_divide_dim_2(golden):
    @proc
    def foo(n: size, m: size, A: R[n + m + 12]):
        x: R[n, 12 * m, m]
        for i in seq(0, n):
            for j in seq(0, 12):
                for k in seq(0, m):
                    x[i, j, k] = A[i + j + k]

    foo = simplify(divide_dim(foo, "x", 1, 4))
    assert str(foo) == golden


def test_divide_dim_3(golden):
    @proc
    def foo(n: size, m: size):
        x: R[n, ((m + 7) / 8) * 8, m]
        for i in seq(0, n):
            for j in seq(0, m):
                for k in seq(0, m):
                    x[i, j, k] = 2.0

    for i in range(2, -1, -1):
        foo = divide_dim(foo, "x", i, 1)
    foo = simplify(divide_dim(foo, "x", 2, 8))
    assert str(foo) == golden


def test_divide_dim_fail_1():
    @proc
    def foo(n: size, m: size, A: R[n + m + 12]):
        x: R[n, 12, m]
        for i in seq(0, n):
            for j in seq(0, 12):
                for k in seq(0, m):
                    x[i, j, k] = A[i + j + k]

    with pytest.raises(ValueError, match="out-of-bounds"):
        divide_dim(foo, "x", 3, 4)

    with pytest.raises(SchedulingError, match="cannot perfectly divide"):
        divide_dim(foo, "x", 1, 5)


def test_divide_dim_fail_2():
    @proc
    def foo(n: size, m: size):
        x: R[n, 3 * m, m]

    with pytest.raises(SchedulingError, match="cannot perfectly divide"):
        for i in range(3):
            divide_dim(foo, "x", i, 15)


def test_mult_dim_1(golden):
    @proc
    def foo(n: size, m: size, A: R[n + m + 12]):
        x: R[n, m, 4]
        for i in seq(0, n):
            for j in seq(0, m):
                for k in seq(0, 4):
                    x[i, j, k] = A[i + j + k]

    foo = mult_dim(foo, "x", 0, 2)
    assert str(foo) == golden


def test_mult_dim_fail_1():
    @proc
    def foo(n: size, m: size, A: R[n + m + 12]):
        x: R[n, m, 4]
        for i in seq(0, n):
            for j in seq(0, m):
                for k in seq(0, 4):
                    x[i, j, k] = A[i + j + k]

    with pytest.raises(ValueError, match="out-of-bounds"):
        mult_dim(foo, "x", 3, 4)

    with pytest.raises(ValueError, match="by itself"):
        mult_dim(foo, "x", 2, 2)

    with pytest.raises(SchedulingError, match="Cannot multiply with non-literal"):
        mult_dim(foo, "x", 0, 1)


def test_delete_buffer(golden):
    @proc
    def foo():
        a: i8[10]

    foo = delete_buffer(foo, foo.find("a: _"))
    assert str(foo) == golden


def test_delete_buffer_fail():
    @proc
    def foo():
        a: i8[10]
        a[0] = 1.0

    with pytest.raises(
        SchedulingError,
        match="The variable a can potentially be used after the statement",
    ):
        foo = delete_buffer(foo, foo.find("a : _"))


def test_reuse_buffer(golden):
    @proc
    def foo(a: f32 @ DRAM, b: f32 @ DRAM):
        aa: f32
        bb: f32
        aa = a
        bb = b

        c: f32
        c = aa + bb
        b = c

    foo = reuse_buffer(foo, "bb:_", "c:_")
    assert str(foo) == golden


def test_reuse_buffer2(golden):
    @proc
    def bar(a: f32):
        pass

    @proc
    def foo():
        bb: f32
        c: f32
        bar(c)

    foo = reuse_buffer(foo, "bb:_", "c:_")
    assert str(foo) == golden


def test_reuse_buffer_loop_fail():
    @proc
    def foo(a: f32 @ DRAM, b: f32 @ DRAM):
        aa: f32
        bb: f32
        aa = a
        bb = b

        c: f32
        for i in seq(0, 10):
            c = aa + bb
        b = c

    with pytest.raises(
        SchedulingError, match="The variable bb can potentially be used after"
    ):
        foo = reuse_buffer(foo, "bb:_", "c:_")


def test_fold_buffer_loop_simple(golden):
    @proc
    def foo(N: size):
        assert N > 4
        x: i8[N]
        for i in seq(0, N - 4):
            for j in seq(i, i + 4):
                x[j] = 1.0

    with pytest.raises(
        SchedulingError,
        match="Buffer folding failed because access window of iteration",
    ):
        foo = resize_dim(foo, foo.find("x: _"), 0, 2, 0, fold=True)

    foo = resize_dim(foo, foo.find("x: _"), 0, 3, 0, fold=True)
    foo = simplify(foo)
    assert str(foo) == golden


# TODO: In general, the current fold buffer analysis cannot handle non-constant width windows.
# There are some limited situations where it works (e.g. if the non-constant loop is the only
# statement in the body). However, if there's any context around it, the check will fail
# conservatively. For example, the following example's buffer should theoretically be foldable.
# If we want to support such transformations, we need index analysis which leverages SMT solvers
# for index comparisons.

#     @proc
#     def foo(N: size):
#         x: i8[N]
#         x[0] = 1.0
#         for i in seq(0, N):
#             x[i] = 0.0


def test_fold_buffer_sequential_stmts(golden):
    @proc
    def foo():
        x: i8[10]
        x[0] = 0.0
        x[2] = 0.0
        x[3] = 0.0
        x[2] = 0.0
        x[7] = 0.0
        x[5] = 0.0

    with pytest.raises(
        SchedulingError, match="Buffer folding failed because access window of x\[5\]"
    ):
        foo = resize_dim(foo, foo.find("x: _"), 0, 2, 0, fold=True)

    foo = resize_dim(foo, foo.find("x: _"), 0, 3, 0, fold=True)
    assert str(simplify(foo)) == golden


def test_fold_buffer_within_stmt(golden):
    @proc
    def foo():
        x: i8[10]
        x[1] = x[3] + x[0]

    with pytest.raises(SchedulingError, match="Buffer folding failed because RHS"):
        foo = resize_dim(foo, foo.find("x: _"), 0, 3, 0, fold=True)

    foo = resize_dim(foo, foo.find("x: _"), 0, 4, 0, fold=True)
    assert str(simplify(foo)) == golden


def test_fold_buffer_if_stmt(golden):
    @proc
    def foo(condition: bool):
        x: i8[10]
        x[2] = 0.0
        if condition:
            x[1] = 0.0
            x[5] = 0.0
        else:
            for i in seq(2, 5):
                x[i] = 1.0
                x[i - 1] = 2.0
                x[i - 2] = 2.0
        x[3] = 0.0

    with pytest.raises(
        SchedulingError,
        match="Buffer folding failed because access window of x\[i \- 2\]",
    ):
        foo = resize_dim(foo, foo.find("x: _"), 0, 2, 0, fold=True)

    foo = resize_dim(foo, foo.find("x: _"), 0, 3, 0, fold=True)
    assert str(simplify(foo)) == golden


def test_fold_buffer_blur(golden):
    @proc
    def blur(H: size, W: size, inp: i8[H + 2, W], out: i8[H, W]):
        assert H % 32 == 0
        assert W > 32
        for io in seq(0, H / 32):
            blur_x: i8[34, W]
            for ii in seq(0, 2):
                for j in seq(0, W - 2):
                    blur_x[ii, j] = (
                        inp[io * 32 + ii, j]
                        + inp[io * 32 + ii, j + 1]
                        + inp[io * 32 + ii, j + 2]
                    )
            for ii in seq(0, 32):
                for j in seq(0, W - 2):
                    blur_x[ii + 2, j] = (
                        inp[io * 32 + ii, j]
                        + inp[io * 32 + ii, j + 1]
                        + inp[io * 32 + ii, j + 2]
                    )
                for j in seq(0, W - 2):
                    out[32 * io + ii, j] = (
                        blur_x[ii, j] + blur_x[ii + 1, j] + blur_x[ii + 2, j]
                    )

    blur = resize_dim(blur, blur.find("blur_x: _"), 0, 3, 0, fold=True)
    assert str(simplify(blur)) == golden


def test_fold_buffer_unsharp(golden):
    @proc
    def exo_unsharp_base(
        W: size,
        H: size,
        output: f32[3, H, W] @ DRAM,
        input: f32[3, H + 6, W + 6] @ DRAM,
    ):
        assert H % 32 == 0
        for y in par(0, H / 32):
            gray: f32[38, 6 + W] @ DRAM
            ratio: f32[1, W] @ DRAM
            blur_y: f32[1, 6 + W] @ DRAM
            for yi in seq(0, 6):
                for x in seq(0, 6 + W):
                    gray[yi, x] = (
                        input[0, yi + 32 * y, x]
                        + input[1, yi + 32 * y, x]
                        + input[2, yi + 32 * y, x]
                    )
            for y_i in seq(0, 32):
                for x in seq(0, 6 + W):
                    gray[6 + y_i, x] = (
                        input[0, 6 + y_i + 32 * y, x]
                        + input[1, 6 + y_i + 32 * y, x]
                        + input[2, 6 + y_i + 32 * y, x]
                    )
                for x in seq(0, 6 + W):
                    blur_y[0, x] = (
                        gray[3 + y_i, x]
                        + gray[2 + y_i, x]
                        + gray[4 + y_i, x]
                        + gray[1 + y_i, x]
                        + gray[5 + y_i, x]
                        + gray[y_i, x]
                        + gray[6 + y_i, x]
                    )
                for x in seq(0, W):
                    ratio[0, x] = (
                        gray[3 + y_i, 3 + x]
                        - (
                            blur_y[0, 3 + x]
                            + blur_y[0, 2 + x]
                            + blur_y[0, 4 + x]
                            + blur_y[0, 1 + x]
                            + blur_y[0, 5 + x]
                            + blur_y[0, x]
                            + blur_y[0, 6 + x]
                        )
                    ) / gray[3 + y_i, 3 + x]
                for c in seq(0, 3):
                    for x in seq(0, W):
                        output[c, y_i + 32 * y, x] = (
                            ratio[0, x] * input[c, 3 + y_i + 32 * y, 3 + x]
                        )

    foo = resize_dim(
        exo_unsharp_base, exo_unsharp_base.find("gray: _"), 0, 8, 0, fold=True
    )

    assert str(simplify(foo)) == golden


def test_fuse_loop(golden):
    @proc
    def foo(n: size, x: R[n]):
        y: R[n]
        for i in seq(0, n):
            y[i] = x[i]
        for j in seq(0, n):
            x[j] = y[j] + 1.0

    foo = fuse(foo, "for i in _:_", "for j in _:_")
    assert str(foo) == golden


def test_fuse_loop2(golden):
    @proc
    def foo(n: size, x: R[n]):
        assert n > 3
        y: R[n]
        for i in seq(3, n):
            y[i] = x[i]
        for j in seq(3, n):
            x[j] = y[j] + 1.0

    foo = fuse(foo, "for i in _:_", "for j in _:_")
    assert str(foo) == golden


def test_fuse_loop_fail():
    @proc
    def foo(n: size, x: R[n + 1]):
        y: R[n + 1]
        y[0] = x[0]
        for i in seq(0, n):
            y[i + 1] = x[i]
        for j in seq(0, n):
            x[j + 1] = y[j + 1] + 1.0

    with pytest.raises(SchedulingError, match="Cannot fission loop"):
        fuse(foo, "for i in _:_", "for j in _:_")


def test_fuse_loop_commute_config(golden):
    @config
    class CFG:
        j: index

    @proc
    def foo(n: size, x: R[n]):
        y: R[n]
        for i in seq(0, n):
            CFG.j = 0
        for j in seq(0, n):
            CFG.j = 0

    foo = fuse(foo, "for i in _:_", "for j in _:_")
    assert str(foo) == golden


def test_fuse_if(golden):
    @proc
    def foo(x: R, a: index, b: index):
        if a == b:
            x += 1.0
        if a - b == 0:
            x += 2.0
        else:
            x += 3.0

    foo = fuse(foo, "if a==b:_", "if _==0: _")
    assert str(foo) == golden


def test_fuse_if_fail():
    @proc
    def foo(x: R, a: index, b: index):
        if a == b:
            x += 1.0
        if a + b == 0:
            x += 2.0

    with pytest.raises(SchedulingError, match="Expressions are not equivalent"):
        fuse(foo, "if a==b:_", "if _==0: _")


def test_divide_with_recompute(golden):
    @proc
    def foo(n: size, A: i8[n + 3]):
        assert n % 4 == 0
        for i in seq(0, n + 3):
            A[i] = 1.0

    foo = divide_with_recompute(foo, foo.find_loop("i"), "n/4", 4, ["io", "ii"])
    foo = rewrite_expr(foo, "n % 4", 0)
    foo = simplify(foo)
    assert str(foo) == golden


def test_divide_with_recompute_fail_not_idempotent():
    @proc
    def foo(n: size, A: i8[n + 3]):
        for i in seq(0, n + 3):
            A[i] += 1.0

    with pytest.raises(SchedulingError, match="The statement at .* is not idempotent"):
        foo = divide_with_recompute(foo, foo.find_loop("i"), "n/4", 4, ["io", "ii"])


def test_divide_with_recompute_fail_outer_hi_too_big():
    @proc
    def foo(n: size, A: i8[n + 3]):
        for i in seq(0, n + 3):
            A[i] = 1.0

    with pytest.raises(
        SchedulingError, match=r"outer_hi \* outer_stride exceeds loop's hi n \+ 3"
    ):
        foo = divide_with_recompute(
            foo, foo.find_loop("i"), "(n + 4) / 4", 4, ["io", "ii"]
        )


def test_simple_divide_loop(golden):
    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            tmp[i] = A[i]

    bar = divide_loop(bar, "i", 4, ["io", "ii"], tail="guard")
    assert str(bar) == golden


def test_divide_loop_perfect(golden):
    @proc
    def foo(n: size, A: i8[n]):
        assert n % 4 == 0
        for i in seq(0, n):
            A[i] = 1.0

    foo = divide_loop(foo, "i", 4, ["io", "ii"], perfect=True)
    assert str(foo) == golden


def test_divide_loop_perfect2(golden):
    @proc
    def foo(n: size, A: i8[n]):
        assert n % 4 == 0
        for i in seq(0, n):
            A[i] = 0.2

    foo = divide_loop(foo, "i", 4, ["io", "ii"], perfect=True)
    foo = stage_mem(foo, "for ii in _:_", "A[io*4:io*4+4]", "tile")
    assert str(simplify(foo)) == golden


def test_divide_loop_perfect3(golden):
    @proc
    def foo(m: size, n: size, A: R[m, n]):
        assert n % 4 == 0 and m % 8 == 0
        for i in seq(0, m):
            for j in seq(0, n):
                A[i, j] = 0.2

    foo = divide_loop(foo, "i", 8, ["io", "ii"], perfect=True)
    foo = divide_loop(foo, "j", 4, ["jo", "ji"], perfect=True)
    assert str(simplify(foo)) == golden


def test_divide_loop_perfect_fail():
    @proc
    def foo(n: size, A: i8[n]):
        assert n % 6 == 0
        for i in seq(0, n):
            A[i] = 1.0

    with pytest.raises(SchedulingError, match="cannot perfectly divide"):
        foo = divide_loop(foo, "i", 4, ["io", "ii"], perfect=True)


def test_divide_loop_cut_and_guard(golden):
    @proc
    def foo(x: i8[1]):
        pass

    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            tmp[i] = A[i]
            foo(tmp[i : i + 1])

    bar = divide_loop(bar, "i", 4, ["io", "ii"], tail="cut_and_guard")
    assert str(bar) == golden


def test_divide_loop_fail_nonzero_lo():
    @proc
    def bar():
        for i in seq(1, 8):
            pass

    with pytest.raises(
        SchedulingError, match="expected the lower bound of the loop to be zero"
    ):
        bar = divide_loop(bar, "i", 4, ["io", "ii"], tail="guard")


def test_divide_loop_by_1_guard(golden):
    @proc
    def bar(n: size):
        for i in seq(0, n):
            pass

    bar1 = divide_loop(bar, bar.find_loop("i"), 1, ("io", "ii"), tail="guard")
    bar2 = simplify(bar1)

    assert f"{bar1}\n{bar2}" == golden


def test_divide_loop_by_1_cut(golden):
    @proc
    def bar(n: size):
        for i in seq(0, n):
            pass

    bar1 = divide_loop(bar, bar.find_loop("i"), 1, ("io", "ii"), tail="cut")
    bar2 = simplify(bar1)

    assert f"{bar1}\n{bar2}" == golden


def test_simple_reorder(golden):
    @proc
    def bar(n: size, m: size, A: i8[n, m]):
        tmp: i8[n, m]
        for i in seq(0, n):
            for j in seq(0, m):
                tmp[i, j] = A[i, j]

    bar = reorder_loops(bar, "i j")
    assert str(bar) == golden


def test_simple_reorder2(golden):
    @proc
    def bar(n: size, m: size, A: i8[n, m]):
        assert n > 5
        assert m > 7
        tmp: i8[n, m]
        for i in seq(4, n):
            for j in seq(2, m):
                tmp[i, j] = A[i, j]

    bar = reorder_loops(bar, "i j")
    assert str(bar) == golden


def test_reorder_loops(golden):
    @proc
    def bar(n: size, A: i8[n, n]):
        for i in seq(0, n):
            for j in seq(0, (-1 - i + n)):
                A[i, j] = 0.0

    with pytest.raises(
        SchedulingError,
        match="inner loop's lo or hi depends on outer loop's iteration variable",
    ):
        bar = reorder_loops(bar, bar.find("for i in _:_"))


def test_reorder_stmts(golden):
    @proc
    def bar(g: R[100] @ DRAM):
        f: R[101] @ DRAM
        for i in seq(0, 100):
            f[i] = 1.0
        f[100] = 1.0
        for i in seq(0, 100):
            g[i] = f[i] + f[i + 1]

    bar = reorder_stmts(bar, "for i in _:_ ;\nf[_] = _")
    assert str(bar) == golden


def test_merge_writes_all_4_cases(golden):
    @proc
    def bar(x: R[4], y: R[4]):
        for i in seq(0, 10):
            if i < 5:
                tmp: R[4]
                tmp[0] = x[0]
                tmp[0] = y[0]
                tmp[1] = x[1]
                tmp[1] += y[1]
                tmp[2] += x[2]
                tmp[2] = y[2]
                tmp[3] += x[3]
                tmp[3] += y[3]

    bar = merge_writes(bar, "tmp[0] = x[0]; tmp[0] = y[0]")
    bar = merge_writes(bar, "tmp[1] = x[1]; tmp[1] += y[1]")
    bar = merge_writes(bar, "tmp[2] += x[2]; tmp[2] = y[2]")
    bar = merge_writes(bar, "tmp[3] += x[3]; tmp[3] += y[3]")
    assert str(bar) == golden


def test_merge_writes_consecutively(golden):
    @proc
    def bar(w: R, x: R, y: R, z: R):
        z = w
        z += x
        z += y
        w = x

    bar = merge_writes(bar, "z = w; z += x")
    bar = merge_writes(bar, "z = w + x; z += y")
    assert str(bar) == golden


def test_merge_writes_array_indexing(golden):
    @proc
    def bar(x: R[3], y: R[3], z: R):
        for i in seq(0, 3):
            for j in seq(0, 3):
                if i < 2:
                    tmp: R[4, 4]
                    tmp[i + j, j] = x[i]
                    tmp[i + j, j] += y[j]

    bar = merge_writes(bar, "tmp[i+j, j] = x[i]; tmp[i+j, j] += y[j]")
    assert str(bar) == golden


def test_merge_writes_type_check(golden):
    @proc
    def bar(y: f32):
        x: f32 @ DRAM
        x = 0.0
        x += y

    bar = merge_writes(bar, "x = 0.0; x += y")
    assert str(bar) == golden


def test_merge_writes_second_rhs_depends_on_first_lhs():
    @proc
    def bar(x: R[5], y: R[3]):
        for i in seq(0, 3):
            if i > 0:
                x[2 * i - 1] = x[i] + y[i]
                x[2 * i - 1] += x[i]

    with pytest.raises(
        SchedulingError, match="expected the right hand side of the second statement"
    ):
        bar = merge_writes(bar, "x[2*i-1] = x[i] + y[i]; x[2*i-1] += x[i]")


def test_merge_writes_wrong_type_error():
    @proc
    def bar(x: R, y: R):
        for i in seq(0, 10):
            y = x
            if i > 5:
                y = x

    with pytest.raises(
        ValueError, match="expected two consecutive assign/reduce statements"
    ):
        bar = merge_writes(bar, "y = x; _")


def test_merge_writes_different_lhs_error():
    @proc
    def bar(x: R, y: R):
        x = y
        y += x

    with pytest.raises(
        ValueError,
        match="expected the two statements' left hand sides to have the same name & type",
    ):
        bar = merge_writes(bar, "x = y; y += x")


def test_merge_writes_different_lhs_arrays_error():
    @proc
    def bar(x: R[3], y: R):
        x[0] = y
        x[1] += y

    with pytest.raises(
        SchedulingError, match="expected the left hand side's indices to be the same."
    ):
        bar = merge_writes(bar, "x[0] = y; x[1] += y")


def test_merge_writes_different_lhs_arrays_error():
    @proc
    def bar(x: R[3, 3], y: R):
        z = x[0:2, 0:2]
        for i in seq(0, 2):
            z[i, 1] = y
            z[i + 1, 1] += y

    with pytest.raises(
        SchedulingError, match="expected the left hand side's indices to be the same."
    ):
        bar = merge_writes(bar, "z[i, 1] = y; z[i+1, 1] += y")


def test_split_write(golden):
    @proc
    def bar(x: i8):
        x = 1 + 2
        x += 3 + 4

    bar = split_write(bar, bar.body()[0])
    bar = split_write(bar, bar.body()[2])
    assert str(bar) == golden


def test_split_write_then_merge():
    @proc
    def bar(x: i8):
        x = 1 + 2
        x += 3 + 4

    start_bar = bar
    assign = bar.body()[0]
    reduce = bar.body()[1]
    bar = split_write(bar, assign)
    bar = split_write(bar, reduce)
    bar = merge_writes(bar, assign.as_block())
    bar = merge_writes(bar, reduce.as_block())
    assert str(bar) == str(start_bar)


def test_split_write_fail():
    @proc
    def bar(x: i8):
        x = 1 * 2
        x += 4

    for s in bar.body():
        with pytest.raises(
            SchedulingError,
            match="Expected the rhs of the statement to be an addition.",
        ):
            bar = split_write(bar, s)


def test_fold_into_reduce_1(golden):
    @proc
    def bar(result: f32):
        result = result + 1.0

    bar = fold_into_reduce(bar, bar.find("result = _"))
    assert str(bar) == golden


def test_fold_into_reduce_2(golden):
    @proc
    def bar(m: size, n: size, a: f32[m, n], x: f32):
        for i in seq(0, m):
            for j in seq(0, n):
                a[i, j] = a[i, j] + (x * x)

    bar = fold_into_reduce(bar, bar.find("a[_] = _"))
    assert str(bar) == golden


def test_fold_into_reduce_fail_1():
    @proc
    def bar(m: size, n: size, a: f32[m, n], x: f32):
        for i in seq(0, m):
            for j in seq(0, n):
                a[i, j] = a[i, j] * x

    with pytest.raises(
        SchedulingError, match="The rhs of the assignment must be an add."
    ):
        bar = fold_into_reduce(bar, bar.find("a[_] = _"))


def test_fold_into_reduce_fail_1():
    @proc
    def bar(m: size, n: size, a: f32[m, n], x: f32):
        for i in seq(0, m):
            for j in seq(0, n):
                a[i, j] = a[i, j]

    with pytest.raises(
        SchedulingError, match="The rhs of the assignment must be an add."
    ):
        bar = fold_into_reduce(bar, bar.find("a[_] = _"))


def test_fold_into_reduce_fail_3():
    @proc
    def bar(m: size, n: size, a: f32[m, n + 1], x: f32):
        for i in seq(0, m):
            for j in seq(0, n):
                a[i, j] = a[i, j + 1] + x

    with pytest.raises(
        SchedulingError,
        match="The lhs of the addition is not a read to the lhs of the assignment.",
    ):
        bar = fold_into_reduce(bar, bar.find("a[_] = _"))


def test_fold_into_reduce_fail_4():
    @proc
    def bar(m: size, n: size, a: f32[m, n], x: f32):
        for i in seq(0, m):
            for j in seq(0, n):
                a[i, j] = (x + a[i, j]) + x

    with pytest.raises(
        SchedulingError,
        match="The lhs of the addition is not a read to the lhs of the assignment.",
    ):
        bar = fold_into_reduce(bar, bar.find("a[_] = _"))


def test_inline_assign(golden):
    @proc
    def foo(n: size, y: i8[n]):
        for i in seq(0, n):
            x: i8[5]
            x[1] = 1.0
            y[i] = x[1] + x[2]
            a: i8
            a = x[1]

    foo = inline_assign(foo, foo.find("x = _"))
    assert str(foo) == golden


def test_inline_assign_scalar(golden):
    @proc
    def foo(b: f32):
        a: f32
        a = 1.0
        b = a

    foo = inline_assign(foo, foo.find("a = _"))
    assert str(foo) == golden


def test_inline_assign_fail():
    @proc
    def foo(n: size, y: i8[n]):
        for i in seq(0, n):
            x: i8[5]
            x[1] = 1.0
            x[2] = 2.0
            y[i] = x[1] + x[2]

    # Current check can't reason about indices, so check too strict
    with pytest.raises(SchedulingError, match="Cannot inline assign"):
        foo = inline_assign(foo, foo.find("x = _"))


def test_simple_unroll(golden):
    @proc
    def bar(A: i8[10]):
        tmp: i8[10]
        for i in seq(0, 10):
            tmp[i] = A[i]

    bar = unroll_loop(bar, "i")
    assert str(bar) == golden


def test_simple_unroll2(golden):
    @proc
    def bar(A: i8[10]):
        tmp: i8[10]
        for i in seq(3, 10):
            tmp[i] = A[i]

    bar = unroll_loop(bar, "i")
    assert str(bar) == golden


def test_simple_unroll3():
    @proc
    def bar(m: size, A: i8[10]):
        assert m < 10
        tmp: i8[10]
        for i in seq(m, 10):
            tmp[i] = A[i]

    with pytest.raises(
        SchedulingError, match="expected loop 'i' to have constant bounds"
    ):
        bar = unroll_loop(bar, "i")


def test_simple_inline(golden):
    @proc
    def foo(x: i8, y: i8, z: i8):
        z = x + y

    @proc
    def bar(n: size, src: i8[n], dst: i8[n]):
        for i in seq(0, n):
            tmp_src1: i8
            tmp_src2: i8
            tmp_src1 = src[i]
            tmp_src2 = src[i]
            tmp_dst: i8
            tmp_dst = dst[i]
            foo(tmp_src1, tmp_src2, tmp_dst)

    bar = inline(bar, "foo(_)")
    assert str(bar) == golden


def test_simple_partial_eval(golden):
    @proc
    def bar(n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            tmp[i] = A[i]

    bar = bar.partial_eval(10)
    assert str(bar) == golden


def test_bool_partial_eval(golden):
    @proc
    def bar(b: bool, n: size, A: i8[n]):
        tmp: i8[n]
        for i in seq(0, n):
            if b == True:
                tmp[i] = A[i]

    bar = bar.partial_eval(False)
    assert str(bar) == golden


def test_transpose(golden):
    @proc
    def bar(m: size, n: size, A: i8[m, n]):
        for i in seq(0, m):
            for j in seq(0, n):
                A[i, j] += 1.0

    bar = bar.transpose(bar.args()[2])
    assert str(bar) == golden


def test_simple_typ_and_mem(golden):
    @proc
    def bar(n: size, A: R[n]):
        A[0] += 1.0

    A = bar.args()[1]
    bar = set_precision(bar, A, "i32")
    bar = set_memory(bar, A, GEMM_SCRATCH)
    bar = set_window(bar, "A", True)

    assert str(bar) == golden

    A_assign = bar.find("A[_] += _")
    assert str(A_assign._impl._node.type) == "i32"


def test_simple_typ_and_mem_2(golden):
    @proc
    def bar(n: size):
        A: R[n]
        A[0] += 1.0

    A = bar.find("A : _")
    bar = set_precision(bar, A, "i32")
    bar = set_memory(bar, A, GEMM_SCRATCH)

    assert str(bar) == golden

    A_assign = bar.find("A[_] += _")
    assert str(A_assign._impl._node.type) == "i32"


def test_set_precision_for_tensors_and_windows():
    @proc
    def bar(n: size, x: [i8][n]):
        pass

    @proc
    def foo(n: size, y: i8[n]):
        assert n > 1
        a = y[0 : n - 1]
        bar(n, y)
        bar(n - 1, y[0 : n - 1])

    assign = foo.find("a = _")
    call1 = foo.find("bar(_)")
    call2 = foo.find("bar(_)")
    foo = set_precision(foo, "y", "f32")
    foo = set_window(foo, "y", True)
    assert (
        str(foo.forward(assign)._impl._node.rhs.type)
        == "Window(src_type=f32[n], as_tensor=[f32][n - 1], src_buf=y, idx='[0:n - 1]')"
    )
    assert str(foo.forward(call1)._impl._node.args[1].type) == "f32[n]"
    assert str(foo.forward(call2)._impl._node.args[1].type) == "f32[n]"


def test_set_precision_api_type(golden):
    @proc
    def bar(n: size, x: R[n]):
        pass

    bar = set_precision(bar, bar.args()[1], ExoType.F32)
    assert str(bar) == golden


def test_set_precision_illegal_precision_value():
    @proc
    def bar(n: size, x: R[n]):
        pass

    with pytest.raises(
        ValueError,
        match="expected an instance of <enum 'ExoType'> or one of the following strings",
    ):
        bar = set_precision(bar, bar.args()[1], "Z")


def test_set_precision_illegal_precision_type():
    @proc
    def bar(n: size, x: R[n]):
        pass

    with pytest.raises(
        TypeError,
        match="expected an instance of <enum 'ExoType'> or <class 'str'> specifying the precision",
    ):
        bar = set_precision(bar, bar.args()[1], bar)


def test_rewrite_expr(golden):
    @proc
    def foo(n: size):
        assert n % 4 == 2
        for i in seq(0, 4 + n % 4):
            pass

    bar = rewrite_expr(foo, "n % 4", "2")
    assert str(simplify(bar)) == golden

    bar = rewrite_expr(foo, "4 + n % 4", "6")
    assert str(bar) == golden


def test_rewrite_expr_2(golden):
    @proc
    def foo(n: size):
        for i in seq(0, n % 4):
            pass

    bar = rewrite_expr(foo, "n % 4", "n - n/4 * 4")
    assert str(simplify(bar)) == golden


def test_rewrite_expr_fail():
    @proc
    def foo(n: size):
        for i in seq(0, n):
            pass

    with pytest.raises(SchedulingError, match="Expressions are not equivalent:"):
        bar = rewrite_expr(foo, "n", "n + n%4")


def test_simple_bind_expr(golden):
    @proc
    def bar(n: size, x: i8[n], y: i8[n], z: i8[n]):
        for i in seq(0, n):
            z[i] = x[i] + y[i]

    bar = bind_expr(bar, "x[_] + y[_]", "z_tmp")
    assert str(bar) == golden


def test_bind_expr_diff_indices(golden):
    @proc
    def bar(n: size, x: i8[n], y: i8[n], z: i8[n]):
        for i in seq(0, n - 1):
            w: i8[n]
            x[i] = x[i] - y[i]
            w[i + 1] = x[i] + y[i] + 1.0
            x[i] = y[i]
            w[i] = x[i] + y[i] + 1.0

    bar = bind_expr(bar, bar.find("x[i]+y[i]+1.0"), "tmp")
    assert str(bar) == golden


def test_bind_lhs(golden):
    @proc
    def myfunc_cpu(inp: i32[1, 1, 16] @ DRAM, out: i32[1, 1, 16] @ DRAM):
        for ii in seq(0, 1):
            for jj in seq(0, 1):
                for kk in seq(0, 16):
                    out[ii, jj, kk] += out[ii, jj, kk] + inp[ii, jj, kk]
                    out[ii, jj, kk] = out[ii, jj, kk] * inp[ii, jj, kk]

    myfunc_cpu = bind_expr(myfunc_cpu, myfunc_cpu.find("inp[_]", many=True), "inp_ram")

    with pytest.raises(SchedulingError, match="Unsafe to bind all"):
        myfunc_cpu = bind_expr(
            myfunc_cpu, myfunc_cpu.find("out[_]", many=True), "out_ram"
        )

    myfunc_cpu = bind_expr(myfunc_cpu, myfunc_cpu.find("out[_]"), "out_ram")
    assert str(myfunc_cpu) == golden


def test_bind_cursor_arg(golden):
    @proc
    def foo(a: R):
        a = 1.0

    foo = bind_expr(foo, foo.find("1.0"), "const")
    assert str(foo) == golden


def test_bind_expr_cse(golden):
    @proc
    def foo(a: i8, b: i8, c: i8):
        b = 2.0 * a
        for i in seq(0, 5):
            c += 2.0 * a
            a = 2.0 * a

    with pytest.raises(SchedulingError, match="Unsafe to bind"):
        foo = bind_expr(foo, foo.find("2.0 * a", many=True), "two_times_a")
        print(foo)

    # Safe to just bind the one outside the loop
    foo = bind_expr(foo, foo.find("2.0 * a"), "two_times_a")
    assert str(foo) == golden


def test_bind_expr_cse_2(golden):
    @proc
    def foo(x: i8[5], y: i8[5]):
        for i in seq(0, 5):
            x[i] = 2.0
        for i in seq(0, 5):
            y[i] = 2.0

    foo = bind_expr(foo, foo.find("2.0", many=True), "two")
    assert str(foo) == golden


def test_simple_lift_alloc(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            tmp_a: i8
            tmp_a = A[i]

    bar = autolift_alloc(bar, "tmp_a : _", n_lifts=1, keep_dims=True)
    assert str(bar) == golden


def test_simple_fission(golden):
    @proc
    def bar(n: size, A: i8[n], B: i8[n], C: i8[n]):
        for i in seq(0, n):
            C[i] += A[i]
            C[i] += B[i]

    bar = autofission(bar, bar.find("C[_] += A[_]").after())
    assert str(bar) == golden


@pytest.mark.skip()
def test_partition():
    @proc
    def bar(n: size, A: i8[n], pad: size):
        assert n > pad
        for i in seq(0, n):
            tmp = A[i]

        for i in seq(0, pad):
            tmp = A[i]
        for i in seq(pad, n - pad):
            tmp = A[i]
        for i in seq(n - pad, n):
            tmp = A[i]


def test_fission(golden):
    @proc
    def bar(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1

    bar = autofission(bar, bar.find("x = _").after(), n_lifts=2)
    assert str(bar) == golden


def test_fission2():
    @proc
    def bar(n: size, m: size):
        for i in seq(0, n):
            for j in seq(0, m):
                x: f32
                x = 0.0
                y: f32
                y = 1.1
                y = x

    with pytest.raises(SchedulingError, match="Will not fission here"):
        autofission(bar, bar.find("x = _").after(), n_lifts=2)


def test_lift(golden):
    @proc
    def bar(A: i8[16, 10]):
        for i in seq(0, 10):
            a: i8[16]
            for k in seq(0, 16):
                a[k] = A[k, i]

    bar = autolift_alloc(
        bar, "a: i8[_]", n_lifts=1, mode="col", size=20, keep_dims=True
    )
    assert str(bar) == golden


def test_unify1(golden):
    @proc
    def bar(n: size, src: R[n, n], dst: R[n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i, j] = y[i, j]

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify2(golden):
    @proc
    def bar(n: size, src: [R][n, n], dst: [R][n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[12, 12], y: R[12, 12]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i + 3, j + 1] = y[i + 5, j + 2]

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify3(golden):
    @proc
    def simd_add4(dst: [R][4], a: [R][4], b: [R][4]):
        for i in seq(0, 4):
            dst[i] = a[i] + b[i]

    @proc
    def foo(n: size, z: R[n], x: R[n], y: R[n]):
        assert n % 4 == 0

        for i in seq(0, n / 4):
            for j in seq(0, 4):
                z[4 * i + j] = x[4 * i + j] + y[4 * i + j]

    foo = replace(foo, "for j in _ : _", simd_add4)
    assert str(foo) == golden


def test_unify4(golden):
    @proc
    def bar(n: size, src: [R][n], dst: [R][n]):
        for i in seq(0, n):
            if i < n - 2:
                dst[i] = src[i] + src[i + 1]

    @proc
    def foo(x: R[50, 2], y: R[50, 2]):
        for j in seq(0, 50):
            if j < 48:
                y[j, 1] = x[j, 0] + x[j + 1, 0]

    foo = replace(foo, "for j in _ : _", bar)
    assert str(foo) == golden


def test_unify5(golden):
    @proc
    def bar(n: size, src: R[n, n], dst: R[n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp: f32
                tmp = src[i, j]
                dst[i, j] = tmp

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                c: f32
                c = y[i, j]
                x[i, j] = c

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify6(golden):
    @proc
    def load(
        n: size,
        m: size,
        src: [i8][n, m],
        dst: [i8][n, 16],
    ):
        assert n <= 16
        assert m <= 16

        for i in seq(0, n):
            for j in seq(0, m):
                dst[i, j] = src[i, j]

    @proc
    def bar(K: size, A: [i8][16, K] @ DRAM):

        for k in seq(0, K / 16):
            a: i8[16, 16] @ DRAM
            for i in seq(0, 16):
                for k_in in seq(0, 16):
                    a[i, k_in] = A[i, 16 * k + k_in]

    bar = replace(bar, "for i in _:_", load)
    assert str(bar) == golden


# Unused arguments
def test_unify7(golden):
    @proc
    def bar(unused_b: bool, n: size, src: R[n, n], dst: R[n, n], unused_m: index):
        for i in seq(0, n):
            for j in seq(0, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(0, 5):
            for j in seq(0, 5):
                x[i, j] = y[i, j]

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify8(golden):
    @proc
    def bar(n: size, m: size, src: R[n, n], dst: R[n, n]):
        assert m < n
        for i in seq(m, n):
            for j in seq(m, n):
                dst[i, j] = src[i, j]

    @proc
    def foo(x: R[5, 5], y: R[5, 5]):
        for i in seq(3, 5):
            for j in seq(3, 5):
                x[i, j] = y[i, j]

    foo = replace(foo, "for i in _ : _", bar)
    assert str(foo) == golden


def test_unify9(golden):
    @proc
    def bar(dst: [f32][8], src: [f32][8], bound: size):
        for i in seq(0, 8):
            if i < bound:
                dst[i] = src[i]

    @proc
    def foo(n: size, m: size, x: f32[n]):
        assert n - m >= 1
        assert n - m <= 8
        y: f32[8]
        for i in seq(0, 8):
            if i + m < n:
                y[i] = x[i]

    foo = replace(foo, foo.find_loop("i"), bar)
    assert str(simplify(foo)) == golden


def test_unify10(golden):
    @proc
    def bar(dst: [f32][8], src: [f32][8], bound: size):
        for i in seq(0, 8):
            if i < bound:
                dst[i] = src[i]

    @proc
    def foo(n: size, m: size, x: f32[n]):
        assert n - m >= 1
        assert n - m <= 8
        y: f32[8]
        for i in seq(0, 8):
            if i + m <= n:
                y[i] = x[i]

    foo = replace(foo, foo.find_loop("i"), bar)
    assert str(simplify(foo)) == golden


def test_unify11(golden):
    @proc
    def bar(dst: [f32][8], src: [f32][8], bound: size):
        for i in seq(0, 8):
            if i < bound:
                dst[i] = src[i]

    @proc
    def foo(n: size, m: size, x: f32[n]):
        assert n - m >= 1
        assert n - m <= 8
        y: f32[8]
        for i in seq(0, 8):
            if m > n + i:
                y[i] = x[i]

    foo = replace(foo, foo.find_loop("i"), bar)
    assert str(simplify(foo)) == golden


def test_unify12(golden):
    @proc
    def bar(dst: [f32][8], src: [f32][8], bound: size):
        for i in seq(0, 8):
            if i < bound:
                dst[i] = src[i]

    @proc
    def foo(n: size, m: size, x: f32[n]):
        assert n - m >= 1
        assert n - m <= 8
        y: f32[8]
        for i in seq(0, 8):
            if m >= n + i:
                y[i] = x[i]

    foo = replace(foo, foo.find_loop("i"), bar)
    assert str(simplify(foo)) == golden


def test_inline_window(golden):
    @proc
    def foo(n: size, m: size, x: R[n, m]):
        assert n > 4
        assert m > 4
        y = x[2 : n - 2, 1 : m - 3]

        for i in seq(0, n - 4):
            for j in seq(0, m - 4):
                a: R
                a = x[i, j] * y[i, j]
                y[i, j] = a + x[i + 1, j + 1]

    foo = inline_window(foo, "y = _")
    assert str(foo) == golden


def test_inline_window2(golden):
    @proc
    def bar(s: stride):
        x: R
        x = 0.0

    @proc
    def foo(n: size, m: size, k: size, x: R[n, m, k, 10]):
        y = x[0, :, :, 0]
        y[0, 0] = 0.0
        bar(stride(y, 1))

    foo = inline_window(foo, "y = _")
    assert str(simplify(foo)) == golden


def test_inline_window3(golden):
    @proc
    def inner_memset(x: [R][16] @ DRAM):
        for i in seq(0, 16):
            x[i] = 0.0

    @proc
    def memset(n: size, x: [R][n] @ DRAM):
        assert n % 16 == 0
        res: R
        for io in seq(0, n / 16):
            x_1 = x[16 * io : 16 * io + 16]
            inner_memset(x_1)
            res += x_1[0]

    memset = inline_window(memset, "x_1 = _")
    assert str(simplify(memset)) == golden


def test_lift_if_second_statement_in_then_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            if m > 12:
                x[0] = 1.0
                if i < 10:
                    x[i] = 2.0

    with pytest.raises(
        SchedulingError, match="expected if statement to be directly nested in parent"
    ):
        foo = lift_if(foo, "if i < 10: _")


def test_lift_if_second_statement_in_else_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            if m > 12:
                pass
            else:
                x[0] = 1.0
                if i < 10:
                    x[i] = 2.0

    with pytest.raises(
        SchedulingError, match="expected if statement to be directly nested in parent"
    ):
        foo = lift_if(foo, "if i < 10: _")


def test_lift_if_second_statement_in_for_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            x[0] = 1.0
            if m > 12:
                pass

    with pytest.raises(
        SchedulingError, match="expected if statement to be directly nested in parent"
    ):
        foo = lift_if(foo, "if m > 12: _")


def test_lift_if_too_high_error():
    @proc
    def foo(m: size, x: R[m], j: size):
        for i in seq(0, m):
            if j < 10:
                x[i] = 2.0

    with pytest.raises(
        SchedulingError, match=r"Cannot lift scope of top-level statement"
    ):
        foo = lift_if(foo, "if j < 10: _", n_lifts=2)


def test_lift_if_dependency_error():
    @proc
    def foo(m: size, x: R[m]):
        for i in seq(0, m):
            if i < 10:
                x[i] = 2.0

    with pytest.raises(
        SchedulingError, match=r"if statement depends on iteration variable"
    ):
        foo = lift_if(foo, "if i < 10: _")


def test_lift_if_past_if(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        assert i > 0
        if i < n:
            if i < 10:
                x[i] = 1.0

    foo = lift_if(foo, "if i < 10: _")
    assert str(foo) == golden


def test_lift_if_past_for(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if i < 10:
                x[j] = 1.0

    foo = lift_if(foo, "if i < 10: _")
    assert str(foo) == golden


def test_lift_if_halfway(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if n > 20:
                if i < 10:
                    x[j] = 1.0

    foo = lift_if(foo, "if i < 10: _")
    assert str(foo) == golden


def test_lift_if_past_if_then_for(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if n > 20:
                if i < 10:
                    x[j] = 1.0

    foo = lift_if(foo, "if i < 10: _", n_lifts=2)
    assert str(foo) == golden


def test_lift_if_middle(golden):
    @proc
    def foo(n: size, x: R[n], i: index):
        for j in seq(0, n):
            if n > 20:
                if i < 10:
                    x[j] = 1.0

    foo = lift_if(foo, "if n > 20: _")
    assert str(foo) == golden


def test_lift_if_with_else_past_if(golden):
    @proc
    def foo(n: size, x: R[n], i: size):
        assert n > 10
        if i < 10:
            if n > 20:
                x[i] = 1.0
            else:
                x[i] = 2.0

    foo = lift_if(foo, "if n > 20: _")
    assert str(foo) == golden


def test_lift_if_with_else_past_if_with_else(golden):
    @proc
    def foo(n: size, x: R[n], i: size):
        assert n > 10
        assert i < n
        if i < 10:
            if n > 20:
                x[i] = 1.0
            else:
                x[i] = 2.0
        else:
            x[i] = 3.0

    foo = lift_if(foo, "if n > 20: _")
    assert str(foo) == golden


def test_lift_if_with_pass_body(golden):
    @proc
    def foo(n: size):
        if 10 < n:
            if n < 20:
                pass

    foo = lift_if(foo, "if n < 20: _")
    assert str(foo) == golden


def test_lift_if_with_pass_body_and_else(golden):
    @proc
    def foo(n: size):
        if 10 < n:
            if n < 20:
                pass
            else:
                pass

    foo = lift_if(foo, "if n < 20: _")
    assert str(foo) == golden


def test_lift_if_in_else_branch_of_parent(golden):
    @proc
    def foo(n: size, x: R[n]):
        if 10 < n:
            x[0] = 1.0
        else:
            if n < 20:
                x[0] = 2.0
            else:
                x[0] = 3.0

    foo = lift_if(foo, "if n < 20: _")
    assert str(foo) == golden


def test_lift_if_in_full_nest(golden):
    @proc
    def foo(n: size, x: R[n]):
        if 10 < n:
            if n < 15:
                x[0] = 1.0
            else:
                x[0] = 2.0
        else:
            if n < 20:
                x[0] = 3.0
            else:
                x[0] = 4.0

    foo = lift_if(foo, "if n < 20: _")
    assert str(foo) == golden


def test_lift_scope(golden):
    @proc
    def foo(n: size, x: R[n, n]):
        for j in seq(0, n):
            if j < 10:
                for i in seq(0, n):
                    x[i, j] = 1.0

    foo = lift_scope(foo, "for i in _: _")
    assert str(foo) == golden


def test_lift_scope_lift_for_when_outer_if_has_noelse_error(golden):
    @proc
    def foo(n: size, x: R[n]):
        if n < 10:
            for i in seq(0, n):
                x[i] = 1.0
        else:
            pass

    with pytest.raises(
        SchedulingError, match="cannot lift for loop when if has an orelse clause"
    ):
        foo = lift_scope(foo, "for i in _: _")


def test_stage_mem(golden):
    # This test stages a buffer being accumulated in
    # on a per-tile basis
    @proc
    def sqmat(n: size, A: R[n, n], B: R[n, n]):
        assert n % 4 == 0
        for i in seq(0, n / 4):
            for j in seq(0, n / 4):
                for k in seq(0, n / 4):
                    for ii in seq(0, 4):
                        for jj in seq(0, 4):
                            for kk in seq(0, 4):
                                A[4 * i + ii, 4 * j + jj] += (
                                    B[4 * i + ii, 4 * k + kk]
                                    * B[4 * k + kk, 4 * j + jj]
                                )

    sqmat = stage_mem(sqmat, "for k in _: _", "A[4*i:4*i+4, 4*j:4*j+4]", "Atile")
    assert str(simplify(sqmat)) == golden


def test_stage_mem_point(golden):
    @proc
    def matmul(n: size, A: R[n, n], B: R[n, n], C: R[n, n]):
        for i in seq(0, n):
            for j in seq(0, n):
                for k in seq(0, n):
                    C[i, j] += A[i, k] * B[k, j]

    matmul = stage_mem(matmul, "for k in _:_", "C[i, j]", "res")
    assert str(simplify(matmul)) == golden


def test_fail_stage_mem():
    # This test fails to stage the buffer B
    # because it's not just being read in a single way
    # therefore the bounds check will fail
    @proc
    def sqmat(n: size, A: R[n, n], B: R[n, n]):
        assert n % 4 == 0
        for i in seq(0, n / 4):
            for j in seq(0, n / 4):
                for k in seq(0, n / 4):
                    for ii in seq(0, 4):
                        for jj in seq(0, 4):
                            for kk in seq(0, 4):
                                A[4 * i + ii, 4 * j + jj] += (
                                    B[4 * i + ii, 4 * k + kk]
                                    * B[4 * k + kk, 4 * j + jj]
                                )

    with pytest.raises(
        SchedulingError,
        match="Buffer has accesses which are neither fully within nor disjoint from the window",
    ):
        sqmat = stage_mem(sqmat, "for ii in _: _", "B[4*i:4*i+4, 4*k:4*k+4]", "Btile")


def test_stage_mem_recursive(golden):
    @proc
    def recursive(n: size, y: R[n] @ DRAM, x: R[n] @ DRAM):
        assert n > 2
        assert (-2 + n) % 4 == 0
        for io in seq(0, (-2 + n) / 4):
            y[2 + 4 * io] = y[1 + 4 * io] + y[4 * io] + x[4 * io]
            y[3 + 4 * io] = (
                y[1 + 4 * io] + y[4 * io] + x[4 * io] + y[1 + 4 * io] + x[1 + 4 * io]
            )
            y[4 + 4 * io] = (
                y[1 + 4 * io]
                + y[4 * io]
                + x[4 * io]
                + y[1 + 4 * io]
                + x[1 + 4 * io]
                + (y[1 + 4 * io] + y[4 * io] + x[4 * io])
                + x[2 + 4 * io]
            )
            y[5 + 4 * io] = (
                y[1 + 4 * io]
                + y[4 * io]
                + x[4 * io]
                + y[1 + 4 * io]
                + x[1 + 4 * io]
                + (y[1 + 4 * io] + y[4 * io] + x[4 * io])
                + x[2 + 4 * io]
                + (
                    y[1 + 4 * io]
                    + y[4 * io]
                    + x[4 * io]
                    + y[1 + 4 * io]
                    + x[1 + 4 * io]
                )
                + x[3 + 4 * io]
            )

    block = recursive.find_loop("io").body()
    recursive = stage_mem(recursive, block, "y[2+4*io : 6+4*io]", "y_tmp")
    assert str(simplify(recursive)) == golden


def test_stage_mem_twice(golden):
    # This test now finds a way to stage the buffer B twice
    @proc
    def sqmat(n: size, A: R[n, n], B: R[n, n]):
        assert n % 4 == 0
        for i in seq(0, n / 4):
            for j in seq(0, n / 4):
                for k in seq(0, n / 4):
                    for ii in seq(0, 4):
                        for jj in seq(0, 4):
                            for kk in seq(0, 4):
                                A[4 * i + ii, 4 * j + jj] += (
                                    B[4 * i + ii, 4 * k + kk]
                                    * B[4 * k + kk, 4 * j + jj]
                                )

    sqmat = bind_expr(sqmat, "B[4*i+ii,4*k+kk]", "B1")
    sqmat = expand_dim(sqmat, "B1 : _", "4", "kk")
    sqmat = expand_dim(sqmat, "B1 : _", "4", "ii")
    sqmat = autolift_alloc(sqmat, "B1 : _", n_lifts=3)
    sqmat = autofission(sqmat, sqmat.find("B1[_] = _").after(), n_lifts=3)
    sqmat = stage_mem(sqmat, "for ii in _: _ #1", "B[4*k:4*k+4, 4*j:4*j+4]", "B2")
    assert str(simplify(sqmat)) == golden


def test_stage_mem_accum(golden):
    # This test stages a buffer being accumulated in
    # on a per-tile basis
    @proc
    def sqmat(n: size, A: R[n, n], B: R[n, n]):
        assert n % 4 == 0
        for i in seq(0, n / 4):
            for j in seq(0, n / 4):
                for k in seq(0, n / 4):
                    for ii in seq(0, 4):
                        for jj in seq(0, 4):
                            for kk in seq(0, 4):
                                A[4 * i + ii, 4 * j + jj] += (
                                    B[4 * i + ii, 4 * k + kk]
                                    * B[4 * k + kk, 4 * j + jj]
                                )

    sqmat = stage_mem(
        sqmat, "for k in _: _", "A[4*i:4*i+4, 4*j:4*j+4]", "Atile", accum=True
    )
    assert str(simplify(sqmat)) == golden


def test_stage_mem_accum2(golden):
    @proc
    def accum(out: R[4, 16, 16], w: R[16], im: R[16]):
        for k in seq(0, 4):
            for i in seq(0, 16):
                for j in seq(0, 16):
                    out[k, i, j] += w[j] * im[i]

    accum = stage_mem(accum, "for i in _:_", "out[k, 0:16, 0:16]", "o")

    assert str(simplify(accum)) == golden


def get_1D_memcpy_tiled():
    @proc
    def memcpy(n: size, x: f32[n], y: f32[n]):
        for i in seq(0, n):
            x[i] = y[i]

    loop = memcpy.find_loop("i")
    memcpy = divide_loop(memcpy, loop, 4, ("io", "ii"), tail="guard")

    return memcpy


def get_2D_mempcpy_1D_tiled():
    @proc
    def memcpy_2D(m: size, n: size, x: f32[m, n], y: f32[m, n]):
        for i in seq(0, m):
            for j in seq(0, n):
                x[i, j] = y[i, j]

    j_loop = memcpy_2D.find_loop("j")
    memcpy_2D = divide_loop(memcpy_2D, j_loop, 4, ("jo", "ji"), tail="guard")

    return memcpy_2D


def get_2D_mempcpy_2D_tiled():
    @proc
    def memcpy_2D(m: size, n: size, x: f32[m, n], y: f32[m, n]):
        for i in seq(0, m):
            for j in seq(0, n):
                x[i, j] = y[i, j]

    i_loop = memcpy_2D.find_loop("i")
    j_loop = memcpy_2D.find_loop("j")
    memcpy_2D = divide_loop(memcpy_2D, j_loop, 4, ("jo", "ji"), tail="guard")
    memcpy_2D = divide_loop(memcpy_2D, i_loop, 7, ("io", "ii"), tail="guard")

    memcpy_2D = lift_scope(memcpy_2D, j_loop)
    memcpy_2D = lift_scope(memcpy_2D, j_loop)

    return memcpy_2D


def test_stage_mem_out_of_bounds_load_1D(golden):
    memcpy = get_1D_memcpy_tiled()
    memcpy = stage_mem(memcpy, memcpy.find_loop("ii"), "y[4 * io:4 * io + 4]", "yReg")
    memcpy = simplify(memcpy)

    assert str(memcpy) == golden


def test_stage_mem_out_of_bounds_load_2D_one_cond(golden):
    memcpy_2D = get_2D_mempcpy_1D_tiled()
    memcpy_2D = stage_mem(
        memcpy_2D, memcpy_2D.find_loop("ji"), "y[i, 4 * jo:4 * jo + 4]", "yReg"
    )
    memcpy_2D = simplify(memcpy_2D)

    assert str(memcpy_2D) == golden


def test_stage_mem_out_of_bounds_load_2D_two_conds(golden):
    memcpy_2D = get_2D_mempcpy_2D_tiled()
    memcpy_2D = stage_mem(
        memcpy_2D,
        memcpy_2D.find_loop("ii"),
        "y[7 * io: 7 * io + 7, 4 * jo:4 * jo + 4]",
        "yReg",
    )
    memcpy_2D = simplify(memcpy_2D)

    assert str(memcpy_2D) == golden


def test_stage_mem_out_of_bounds_store_1D(golden):
    memcpy = get_1D_memcpy_tiled()
    memcpy = stage_mem(memcpy, memcpy.find_loop("ii"), "x[4 * io:4 * io + 4]", "xReg")
    memcpy = simplify(memcpy)

    assert str(memcpy) == golden


def test_stage_mem_out_of_bounds_reduction(golden):
    @proc
    def axpy(n: size, x: f32[n], y: f32[n]):
        for i in seq(0, n):
            y[i] += x[i]

    axpy = divide_loop(axpy, axpy.find_loop("i"), 5, ("io", "ii"), tail="guard")
    axpy = stage_mem(axpy, axpy.find_loop("ii"), "y[5*io:5*io+5]", "yReg")
    axpy = simplify(axpy)

    assert str(axpy) == golden


def test_stage_mem_out_of_bound_reduction_accum(golden):
    @proc
    def axpy(n: size, x: f32[n], y: f32[n]):
        for i in seq(0, n):
            y[i] += x[i]

    axpy = divide_loop(axpy, axpy.find_loop("i"), 5, ("io", "ii"), tail="guard")
    axpy = stage_mem(axpy, axpy.find_loop("ii"), "y[5*io:5*io+5]", "yReg", accum=True)
    axpy = simplify(axpy)

    assert str(axpy) == golden


def test_stage_mem_out_of_bound_block(golden):
    @proc
    def axpy(n: size, x: f32[n], y: f32[n]):
        for i in seq(0, n):
            y[i] += x[i]

    axpy = divide_loop(axpy, axpy.find_loop("i"), 5, ("io", "ii"), tail="guard")
    axpy = stage_mem(axpy, axpy.find_loop("io").body(), "x[5*io:5*io+5]", "xReg")
    axpy = simplify(axpy)

    assert str(axpy) == golden


def test_stage_mem_out_of_bound_point(golden):
    @proc
    def foo(n: size, m: size, x: f32[n], y: f32[n]):
        assert m >= n
        for i in seq(0, m):
            if i < n:
                y[i] = x[i]

    foo = stage_mem(foo, foo.find_loop("i").body(), "x[i]", "tmp")
    assert str(foo) == golden


def test_new_expr_multi_vars(golden):
    @proc
    def bar(n: size, arr: R[n] @ DRAM):
        for i in seq(0, n):
            tmp: R @ DRAM
            tmp = 1.0
            arr[i] = tmp
        i: R @ DRAM
        i = 1.0

    bar = expand_dim(bar, "tmp : _", "n", "i")


def test_formatted_expr_1(golden):
    @proc
    def bar(n: size, arr: R[n] @ DRAM):
        for i in seq(0, n):
            tmp: R
            tmp = 1.0
            arr[i] = tmp

    alloc_stmt = bar.find("tmp : _")
    seq_for_hi = alloc_stmt.parent().hi()
    seq_for_iter = alloc_stmt.parent().name()
    bar = expand_dim(
        bar, "tmp : _", FormattedExprStr("_ + 1", seq_for_hi), str(seq_for_iter)
    )
    assert str(bar) == golden


def test_formatted_expr_2(golden):
    @proc
    def bar(n: size, m: size, arr: R[n, m] @ DRAM):
        for i in seq(0, n):
            for j in seq(0, m):
                tmp: R
                tmp = 1.0
                arr[i, j] = tmp

    alloc_stmt = bar.find("tmp : _")
    seq_i = alloc_stmt.parent()
    seq_o = seq_i.parent()
    new_dim = FormattedExprStr("(_ + 1) * (1 + _)", seq_o.hi(), seq_i.hi())
    indexing_expr = FormattedExprStr(f"{seq_o.name()} * _ + {seq_i.name()}", seq_i.hi())

    bar = expand_dim(bar, "tmp : _", new_dim, indexing_expr)

    assert str(bar) == golden


def test_formatted_expr_3(golden):
    @proc
    def foo(n: size, x: f32[n]):
        assert n >= 10
        for i in seq(0, n - 2):
            x[i] = 0.0

    loop_cursor = foo.find_loop("i")
    foo = cut_loop(foo, loop_cursor, FormattedExprStr("_ - 1", loop_cursor.hi()))
    assert str(simplify(foo)) == golden


def test_formatted_expr_errors_1():
    @proc
    def bar(n: size, arr: R[n] @ DRAM):
        for i in seq(0, n):
            tmp: R
            tmp = 1.0
            arr[i] = tmp

    with pytest.raises(
        ParseFragmentError, match="String contains more holes than expressions provided"
    ):
        expand_dim(bar, "tmp : _", FormattedExprStr("1 + _"), "i")  # should be error

    with pytest.raises(
        ParseFragmentError, match="String contains more holes than expressions provided"
    ):
        alloc_stmt = bar.find("tmp : _")
        seq_for_hi = alloc_stmt.parent().hi()
        expand_dim(
            bar, "tmp : _", FormattedExprStr("1 + _ + _", seq_for_hi), "i"
        )  # should be error

    with pytest.raises(ParseFragmentError, match="String cannot contain holes"):
        expand_dim(bar, "tmp : _", "1 + _", "i")  # should be error

    with pytest.raises(
        TypeError,
        match="Cursor provided to fill a hole must be a ExprCursor",
    ):
        alloc_stmt = bar.find("tmp : _")
        expand_dim(
            bar, "tmp : _", FormattedExprStr("1 + _", alloc_stmt), "i"
        )  # should be error


def test_formatted_expr_errors_2():
    @proc
    def bar(n: size, arr: R[n] @ DRAM):
        for i in seq(0, n):
            tmp: R
            tmp = 1.0
            arr[i] = tmp
        i: R
        i = 1.0
        j: R
        j = i

    with pytest.raises(ParseFragmentError, match="not found in current environment"):
        assign_stmt = bar.find("j = i")
        i_type_R_read_cursor = assign_stmt.rhs()

        alloc_stmt = bar.find("tmp : _")
        seq_for_hi = alloc_stmt.parent().hi()
        seq_for_iter = alloc_stmt.parent().name()
        expand_dim(
            bar, "tmp : _", "n", FormattedExprStr("_", i_type_R_read_cursor)
        )  # should be error


def test_simplify_index_div(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 4 + 2 * j + 3) / 3] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div1(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 4 + 2 * j + 3) / 2] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div2(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 4 + 2 * j + 24) / 2] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div3(golden):
    @proc
    def bar(N: size, x: R[N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 4):
                x[(io * 4 + ii) / 4] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div4(golden):
    @proc
    def bar(N: size, x: R[N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 4):
                x[(io * 4 + ii + 8) / 4] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div5(golden):
    @proc
    def bar(N: size, x: R[N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 4):
                x[(io * 5 + ii + 1) / 5] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div6(golden):
    @proc
    def bar(N: size):
        for i in seq(0, N):
            for j in seq(0, 4):
                if (i * 4 + j) / 16 > 0:
                    pass

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div_fail(golden):
    @proc
    def bar(N: size, x: R[1 + N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 4):
                x[(io * 4 + ii + 1) / 4] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div_fail1(golden):
    @proc
    def bar(N: size, x: R[1 + N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 5):
                x[(io * 4 + ii) / 4] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_div_fail2(golden):
    @proc
    def bar(N: size, x: R[2 * N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 5):
                x[(N + ii + 4 * io) / 2] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_mod(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 4 + 2 * j + 3) % 5] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_mod1(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 4 + 2 * j + 3) % 3] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_mod2(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 4 + 2 * j + 3) % 2] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_mod3(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[(i * 4 + i * 3 + 2 * j + 3) % 2] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_mod4(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[((i * 6 + i * 4 + 5 * j + 9) % 5 + 2 * i + 2) / 2] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_mod5(golden):
    @proc
    def bar(N: size, x: R[N]):
        assert N >= 1
        assert N % 4 == 0
        for io in seq(0, N / 4):
            for ii in seq(0, 4):
                x[(io * 4 + ii) % 4] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_index_nested_div_mod(golden):
    @proc
    def bar(x: R[1000]):
        for i in seq(0, 4):
            for j in seq(0, 5):
                x[
                    (
                        (
                            (((i * 4 + i * 12 + 2 * j + 2) % 2) + (20 * i + 40 * j) / 5)
                            / 2
                        )
                        + (3 + 8 * j)
                    )
                    % 4
                ] = 1.0

    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_div_mod_staging(golden):
    @proc
    def bar(x: R[64], y: R[64], out: R[64]):
        for i in seq(0, 64):
            out[i] = x[i] * y[i]

    sc = bar.find("for i in _:_")
    bar = divide_loop(bar, sc, 4, ("io", "ii"), tail="cut")
    bar = stage_mem(bar, sc, "x[0:64]", "xReg")
    bar = simplify(bar)
    assign_loop_sc = bar.forward(sc).prev()
    bar = divide_loop(bar, assign_loop_sc, 4, ("io", "ii"), tail="cut")
    bar = divide_dim(bar, "xReg", 0, 4)
    bar = simplify(bar)
    assert str(bar) == golden


def test_simplify_with_window_stmts():
    @proc
    def foo(n: size):
        x: i8[1]
        x_window = x[0 + 0 : 1 + 0]
        x_window[0] = 0.0

    return simplify(foo)


def test_simplify_logical(golden):
    @proc
    def foo(n: size):
        if (n > 0 and True) or (False and n == 1):
            pass
        if n > 4 or True:
            pass

    foo = simplify(foo)
    assert str(foo) == golden


def test_cut_loop_syrk(golden):
    @proc
    def SYRK(
        M: size,
        K: size,
        A: [f32][M, K] @ DRAM,
        A_t: [f32][M, K] @ DRAM,
        C: [f32][M, M] @ DRAM,
    ):
        assert M >= 1
        assert K >= 1
        assert stride(A, 1) == 1
        assert stride(A_t, 1) == 1
        assert stride(C, 1) == 1
        for io in seq(0, M / 4):
            for ii in seq(0, 4):
                for j in seq(0, 4 * io + ii + 1):
                    for k in seq(0, K):
                        C[4 * io + ii, j] += A[4 * io + ii, k] * A_t[j, k]

    SYRK = cut_loop(SYRK, SYRK.find_loop("j"), 1)
    SYRK = shift_loop(SYRK, SYRK.find_loop("j #1"), 0)
    assert str(simplify(SYRK)) == golden


def test_cut_loop1():
    @proc
    def foo(n: size):
        for i in seq(0, n):
            x: R
            x = 0.0

    with pytest.raises(SchedulingError, match="Expected `cut_point` <= `hi`"):
        foo = cut_loop(foo, foo.find_loop("i"), 3)


def test_cut_loop2(golden):
    @proc
    def foo(n: size):
        assert n > 3
        for i in seq(0, n):
            x: R
            x = 0.0

    foo = cut_loop(foo, foo.find_loop("i"), 3)
    assert str(simplify(foo)) == golden


def test_cut_loop3():
    @proc
    def foo(n: size):
        for i in seq(0, n):
            x: R
            x = 0.0

    with pytest.raises(SchedulingError, match="Expected `lo` <= `cut_point`"):
        foo = cut_loop(foo, foo.find_loop("i"), -3)


def test_cut_loop_nonzero_lo(golden):
    @proc
    def foo(n: size):
        assert n >= 5
        x: R[n]
        for i in seq(3, n):
            x[i] = 0.0

    with pytest.raises(SchedulingError, match="Expected `lo` <= `cut_point`"):
        foo = cut_loop(foo, foo.find_loop("i"), 2)

    foo = cut_loop(foo, foo.find_loop("i"), 5)
    assert str(simplify(foo)) == golden


def test_cut_loop_nonzero_lo2(golden):
    @proc
    def foo(n: size, m: size):
        assert m >= 5
        assert m <= 8
        assert n >= 9
        assert n > m + 1
        x: R[n]
        for i in seq(m, n):
            x[i] = 0.0

    with pytest.raises(SchedulingError, match="Expected `lo` <= `cut_point`"):
        foo = cut_loop(foo, foo.find_loop("i"), 6)

    foo = cut_loop(foo, foo.find_loop("i"), 9)
    assert str(simplify(foo)) == golden


def test_cut_loop_at_lo(golden):
    @proc
    def foo():
        for i in seq(3, 5):
            x: f32

    foo = cut_loop(foo, foo.find_loop("i"), 3)
    assert str(foo) == golden


def test_cut_loop_at_hi(golden):
    @proc
    def foo():
        for i in seq(3, 5):
            x: f32

    foo = cut_loop(foo, foo.find_loop("i"), 5)
    assert str(foo) == golden


def test_cut_loop_by_expr(golden):
    @proc
    def foo(n: size, x: f32[n]):
        assert n >= 1
        for i in seq(0, n):
            x[i] = 0.0

    loop_cursor = foo.find_loop("i")
    foo = cut_loop(foo, loop_cursor, "n / 2")
    assert str(foo) == golden


def test_cut_loop_by_expr1(golden):
    @proc
    def foo(n: size, x: f32[n]):
        assert n >= 1
        for i in seq(0, n):
            x[i] = 0.0

    loop_cursor = foo.find_loop("i")
    foo = cut_loop(foo, loop_cursor, FormattedExprStr("_ - 1", loop_cursor.hi()))
    assert str(foo) == golden


def test_cut_loop_by_expr2(golden):
    @proc
    def foo(n: size, m: size):
        assert n > m
        x: R[n]
        for i in seq(m, n):
            x[i] = 0.0

    loop_cursor = foo.find_loop("i")
    foo = cut_loop(foo, loop_cursor, FormattedExprStr("_ + 1", loop_cursor.lo()))
    assert str(simplify(foo)) == golden


def test_shift_loop(golden):
    @proc
    def foo(n: size, x: f32[n] @ DRAM):
        for i in seq(0, n):
            x[i] = 0.0

    foo = shift_loop(foo, foo.find_loop("i"), 1)
    assert str(simplify(foo)) == golden


def test_shift_loop_to_negative_lo():
    @proc
    def foo(n: size, x: f32[n] @ DRAM):
        for i in seq(0, n):
            x[i] = 0.0

    with pytest.raises(SchedulingError, match="Expected 0 <= `new_lo`"):
        foo = shift_loop(foo, foo.find_loop("i"), -3)

    with pytest.raises(SchedulingError, match="Expected 0 <= `new_lo`"):
        foo = shift_loop(foo, foo.find_loop("i"), "n-3")


def test_shift_loop_by_expr(golden):
    @proc
    def foo(n: size, x: f32[n + 1] @ DRAM):
        for i in seq(0, n):
            x[i + 1] = 0.0

    foo = shift_loop(foo, foo.find_loop("i"), "n + 2")
    assert str(simplify(foo)) == golden


def test_shift_loop_nonzero_lo(golden):
    @proc
    def foo(n: size, m: size, x: f32[n + 1] @ DRAM):
        assert n >= m
        for i in seq(m, n):
            x[i] = 0.0

    foo = shift_loop(foo, foo.find_loop("i"), 4)
    assert str(simplify(foo)) == golden


def test_cut_then_shift_loop(golden):
    @proc
    def foo(n: size, m: size, x: f32[20] @ DRAM):
        assert n >= m
        for i in seq(2, 20):
            x[i] = 0.0

    loop_cursor = foo.find_loop("i")
    foo = cut_loop(foo, loop_cursor, 10)
    foo = shift_loop(foo, foo.find_loop("i"), 5)
    foo = shift_loop(foo, foo.forward(loop_cursor).next(), 0)
    assert str(simplify(foo)) == golden


def test_join_loops_body_match(golden):
    # TODO: add write_config, stride_expr, read_config
    @proc
    def do_nothing(x: [i8][2, 1]):
        pass

    @proc
    def foo(n: size, x: i8[n + 1]):
        for i in seq(0, n):
            x[i] = 0.0
            x[i] += -(1.0 + x[i])
            for j in seq(0, 1):
                if i == j:
                    pass
            a: i8[4, 2]
            y = a[1:3, 1:2]
            do_nothing(y)
        for i in seq(n, n + 1):
            x[i] = 0.0
            x[i] += -(1.0 + x[i])
            for j in seq(0, 1):
                if i == j:
                    pass
            a: i8[4, 2]
            y = a[1:3, 1:2]
            do_nothing(y)

    foo = join_loops(foo, foo.find_loop("i"), foo.find_loop("i #1"))
    assert str(foo) == golden


def test_join_loops_equiv_but_diff_bounds(golden):
    @proc
    def foo(n: size, x: i8[4]):
        assert n % 4 == 2
        for i in seq(0, 2):
            x[i] = 0.0
        for i in seq(n % 4, 4):
            x[i] = 0.0

    foo = join_loops(foo, foo.find_loop("i"), foo.find_loop("i #1"))
    assert str(foo) == golden


def test_join_loops_fail_type_match():
    @proc
    def foo():
        for i in seq(0, 2):
            x: i8[4]
            x[i] = 0.0
        for i in seq(2, 4):
            x: f32[4]
            x[i] = 0.0

    with pytest.raises(SchedulingError, match=""):
        foo = join_loops(foo, foo.find_loop("i"), foo.find_loop("i #1"))


def test_join_loops_fail_equal_bounds():
    @proc
    def foo(n: size, x: i8[n + 2]):
        for i in seq(0, n):
            x[i] = 0.0
        for i in seq(n + 1, n + 2):
            x[i] = 0.0

    with pytest.raises(
        SchedulingError,
        match=r"expected the first loop upper bound n to be the same as the second loop lower bound n \+ 1",
    ):
        foo = join_loops(foo, foo.find_loop("i"), foo.find_loop("i #1"))


def test_replace_once(golden):
    @proc
    def bar(src: f32[8] @ DRAM):
        dst: f32[8] @ AVX2
        for i in seq(0, 8):
            dst[i] = src[i]
        for i in seq(0, 8):
            src[i] = dst[i]

    bar = replace_once(bar, [mm256_loadu_ps])
    assert str(bar) == golden


def test_mem_aware_replace(golden):
    @proc
    def bar(src: f32[8] @ DRAM):
        dst: f32[8] @ AVX2
        for i in seq(0, 8):
            dst[i] = src[i]
        for i in seq(0, 8):
            src[i] = dst[i]

    bar = call_site_mem_aware_replace(bar, "for i in _:_", mm256_loadu_ps)
    bar = call_site_mem_aware_replace(bar, "for i in _:_", mm256_storeu_ps)
    assert str(bar) == golden


def test_mem_aware_replace_fail():
    @proc
    def bar(src: f32[8] @ DRAM):
        dst: f32[8] @ AVX2
        for i in seq(0, 8):
            dst[i] = src[i]
        for i in seq(0, 8):
            src[i] = dst[i]

    with pytest.raises(MemoryError, match="failed due to memory type mismatch"):
        bar = call_site_mem_aware_replace(bar, "for i in _:_", mm256_storeu_ps)


def test_replace_all_unambiguous(golden):
    @proc
    def bar(src: f32[8] @ DRAM):
        dst: f32[8] @ AVX2
        for i in seq(0, 8):
            dst[i] = src[i]
        for i in seq(0, 8):
            src[i] = dst[i]

    bar = replace_all(bar, [mm256_loadu_ps, mm256_storeu_ps])
    assert str(bar) == golden


def test_replace_all_arch(golden):
    @proc
    def bar(src: f32[8] @ DRAM):
        dst: f32[8] @ AVX2
        for i in seq(0, 8):
            dst[i] = src[i]
        for i in seq(0, 8):
            src[i] = dst[i]

    arch = [mm256_storeu_ps, mm256_mul_ps, mm256_loadu_ps]
    bar = replace_all(bar, arch)
    assert str(bar) == golden


def test_replace_all_length_mismatch(golden):
    @proc
    def bar(x: i8):
        x = 1.0
        x += 1.0

    @proc
    def foo(x: i8):
        x = 1.0
        x += 1.0
        x = 1.0

    foo = replace_all(foo, [bar])
    assert str(foo) == golden


def test_eliminate_dead_code(golden):
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 > -1:
                x = 0.0
                a: R
                a = x
            else:
                b: R
                b = x

    assert str(eliminate_dead_code(foo, "if _:_ #0")) == golden


def test_eliminate_dead_code2(golden):
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 < -1:
                x = 0.0
                a: R
                a = x
            else:
                b: R
                b = x

    assert str(eliminate_dead_code(foo, "if _:_ #0")) == golden


def test_eliminate_dead_code3(golden):
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 > -1:
                x = 0.0
                a: R
                a = x

    assert str(eliminate_dead_code(foo, "if _:_ #0")) == golden


def test_eliminate_dead_code4(golden):
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i + 3 < -1:
                x = 0.0
                a: R
                a = x

    assert str(eliminate_dead_code(foo, "if _:_ #0")) == golden


def test_eliminate_dead_code5():
    @proc
    def foo():
        x: f32 @ DRAM
        for i in seq(0, 8):
            if i < 4:
                x = 0.0

    with pytest.raises(
        SchedulingError,
        match="If condition isn't always True or always False",
    ):
        eliminate_dead_code(foo, "if _:_ #0")


def test_eliminate_dead_code6():
    @proc
    def foo(n: size):
        for i in seq(0, n):
            pass

    with pytest.raises(
        SchedulingError,
        match="Loop condition isn't always False",
    ):
        eliminate_dead_code(foo, foo.find_loop("i"))


def test_eliminate_dead_code7(golden):
    @proc
    def foo(n: size):
        for i in seq(0, n):
            x: f32

    foo = specialize(foo, foo.find_loop("i"), "0 < n")
    foo = eliminate_dead_code(foo, foo.find_loop("i #1"))

    assert str(foo) == golden


def test_eliminate_dead_code8(golden):
    @proc
    def foo(n: size):
        for i in seq((7 + n) / 8 * 8, (7 + n) / 8 * 8):
            pass

    foo = eliminate_dead_code(foo, foo.find_loop("i"))
    assert str(foo) == golden


def test_eliminate_dead_code9(golden):
    @proc
    def foo(n: size):
        for i in seq(0 + (7 + n) / 8 * 8, ((7 + n) / 8 * 8 + 7) / 8 * 8):
            pass

    foo = eliminate_dead_code(foo, foo.find_loop("i"))
    assert str(foo) == golden


def test_lift_reduce_constant_1(golden):
    @proc
    def foo():
        x: R @ DRAM
        x = 0.0
        for i in seq(0, 8):
            x += 3.0 * 2.0

    assert str(lift_reduce_constant(foo, "x = 0.0; _")) == golden


def test_lift_reduce_constant_2(golden):
    @proc
    def foo():
        x: R[2] @ DRAM
        y: R[8] @ DRAM
        x[0] = 0.0
        for i in seq(0, 8):
            for j in seq(0, 8):
                y[i] += y[i] * 2.0
            for k in seq(0, 8):
                x[0] += 3.0 * y[i]
            x[0] += 3.0 * y[i]

    assert str(lift_reduce_constant(foo, "x[0] = 0.0; _")) == golden


def test_lift_reduce_constant_3(golden):
    @proc
    def foo():
        x: R @ DRAM
        y: R[8] @ DRAM
        for j in seq(0, 8):
            x = 0.0
            for i in seq(0, 8):
                x += y[j] * 2.0

    assert str(lift_reduce_constant(foo, "x = 0.0; _")) == golden


def test_lift_reduce_constant_bad_1():
    @proc
    def foo():
        x: R[2] @ DRAM
        y: R[8] @ DRAM
        x[0] = 0.0
        for i in seq(0, 8):
            x[0] += 3.0 * y[i]
            x[1] += 2.0 * y[i]

    with pytest.raises(
        SchedulingError,
        match="cannot lift constant because there are other operations on the same buffer that may interfere",
    ):
        lift_reduce_constant(foo, "x[0] = 0.0; _")


def test_lift_reduce_constant_bad_2():
    @proc
    def foo():
        x: R[4] @ DRAM
        y: R[8] @ DRAM
        for j in seq(0, 2):
            x[2 * j + 1] = 0.0
            x[j + 2] = 0.0
            for i in seq(0, 8):
                x[2 * j + 1] += 3.0 * y[i]
                x[j + 2] += 2.0 * y[i]

    with pytest.raises(
        SchedulingError,
        match="cannot lift constant because there are other operations on the same buffer that may interfere",
    ):
        lift_reduce_constant(foo, "x[j+2] = 0.0; _")


def test_lift_reduce_constant_bad_3():
    @proc
    def foo():
        x: R[2] @ DRAM
        y: R[8] @ DRAM
        x[0] = 0.0
        for i in seq(0, 8):
            x[0] += 3.0 * y[i]
            x[0] += 2.0 * y[i]

    with pytest.raises(
        SchedulingError, match="cannot lift constant because the reduces to buffer x"
    ):
        lift_reduce_constant(foo, "x[0] = 0.0; _")


def test_lift_reduce_constant_bad_4():
    @proc
    def foo():
        x: R[2] @ DRAM
        y: R[8] @ DRAM
        x[0] = 0.0
        for i in seq(0, 8):
            y[i] = x[0]
            x[0] += 2.0 * y[i]

    with pytest.raises(
        SchedulingError,
        match="cannot lift constant because the buffer is read in the loop body",
    ):
        lift_reduce_constant(foo, "x[0] = 0.0; _")


def test_lift_reduce_constant_bad_5():
    @proc
    def foo():
        x: R[2] @ DRAM
        y: R[8] @ DRAM
        x[0] = 0.0
        for i in seq(0, 8):
            y[i] = 2.0 * y[i]

    with pytest.raises(
        SchedulingError, match="cannot lift constant because did not find a reduce"
    ):
        lift_reduce_constant(foo, "x[0] = 0.0; _")


def test_lift_reduce_constant_bad_6():
    @proc
    def foo():
        x: R[2] @ DRAM
        y: R[8] @ DRAM
        x[0] = 0.0
        for i in seq(0, 8):
            x[0] += y[1] * 2.0
            x[0] += y[2] * 2.0

    with pytest.raises(
        SchedulingError, match="cannot lift constant because the reduces to buffer x"
    ):
        lift_reduce_constant(foo, "x[0] = 0.0; _")


def test_lift_reduce_constant_bad_7():
    @proc
    def foo():
        x: R @ DRAM
        y: R[8] @ DRAM
        for j in seq(0, 8):
            x = 0.0
            for i in seq(0, 8):
                y[1] = 0.0
                x += y[j] * 2.0

    with pytest.raises(
        SchedulingError,
        match="cannot lift constant because it is a buffer that is written",
    ):
        lift_reduce_constant(foo, "x[0] = 0.0; _")


def test_lift_reduce_constant_bad_8():
    @proc
    def write(x: R[8] @ DRAM):
        x[0] = 1.0

    @proc
    def foo():
        x: R @ DRAM
        y: R[8] @ DRAM
        for j in seq(0, 8):
            x = 0.0
            for i in seq(0, 8):
                write(y)
                x += y[j] * 2.0

    with pytest.raises(
        NotImplementedError,
        match="unsupported stmt type",
    ):
        lift_reduce_constant(foo, "x = 0.0; _")


def test_lift_reduce_constant_bad_9():
    @proc
    def dot(n: size, x: f32[n], y: f32[n]):
        dot: R
        dot = 0.0
        for i in seq(0, n):
            dot += y[i] * x[i]

    with pytest.raises(
        SchedulingError,
        match="y\[i\] depends on the variable i which is defined within the loop",
    ):
        dot = lift_reduce_constant(dot, dot.find_loop("i").expand(1, 0))


def test_lift_reduce_constant_bad_10():
    @proc
    def dot(n: size, x: f32[n], y: f32[n]):
        dot: R
        dot = 0.0
        for i in seq(0, n):
            for j in seq(0, n):
                dot += y[j] * x[i]

    with pytest.raises(
        SchedulingError,
        match="y\[j\] depends on the variable j which is defined within the loop",
    ):
        dot = lift_reduce_constant(dot, dot.find_loop("i").expand(1, 0))


def test_lift_reduce_constant_bad_11():
    @proc
    def dot(n: size, x: f32[n], y: f32[n]):
        dot: R
        dot = 0.0
        for i in seq(0, n):
            a: f32
            dot += a * (y[i] * x[i])

    with pytest.raises(
        SchedulingError,
        match="a depends on the variable a which is defined within the loop",
    ):
        dot = lift_reduce_constant(dot, dot.find_loop("i").expand(1, 0))


def test_specialize(golden):
    @proc
    def foo(x: f32[4] @ DRAM):
        for i in seq(0, 4):
            x[i] += 1.0

    foo = specialize(foo, "x[i] += 1.0", [f"i == {i}" for i in range(4)])
    assert str(foo) == golden


def test_specialize_sizes(golden):
    @proc
    def gemm(
        M: size,
        N: size,
        K: size,
        C: f32[M, N] @ DRAM,
        A: f32[M, K] @ DRAM,
        B: f32[K, N] @ DRAM,
        alpha: f32,
    ):
        for i in seq(0, M):
            for j in seq(0, N):
                for k in seq(0, K):
                    C[i, j] += alpha * A[i, k] * B[k, j]

    foo = specialize(gemm, "for i in _:_", [f"N <= {x}" for x in [64, 128, 512]])
    foo = simplify(foo)
    assert str(foo) == golden


def test_specialize_blocks(golden):
    @proc
    def foo(n: size, a: f32):
        b: f32
        a = 1.0
        a = 2.0
        b = 1.2

    body = foo.body()
    foo = specialize(foo, body, ["n > 0"])
    assert str(foo) == golden


def test_specialize_data():
    @proc
    def gemm(
        M: size,
        N: size,
        K: size,
        C: f32[M, N] @ DRAM,
        A: f32[M, K] @ DRAM,
        B: f32[K, N] @ DRAM,
        alpha: f32,
    ):
        for i in seq(0, M):
            for j in seq(0, N):
                for k in seq(0, K):
                    C[i, j] += alpha * A[i, k] * B[k, j]

    with pytest.raises(
        SchedulingError,
        match="Invalid specialization condition",
    ):
        specialize(gemm, "for i in _:_", [f"alpha == {x}" for x in [0.0, 1.0, -1.0]])


def test_specialize_alloc_fails():
    @proc
    def foo(n: size):
        a: f32
        a = 1.0

    with pytest.raises(
        SchedulingError,
        match="Block contains allocations",
    ):
        foo = specialize(foo, foo.body()[0], "n > 0")


def test_extract_subproc(golden):
    @proc
    def foo():
        x: R @ DRAM
        y: R[8] @ DRAM
        for j in seq(0, 8):
            x = 0.0
            for i in seq(0, 8):
                x += y[j] * 2.0

    _, new_no_asserts = extract_subproc(
        foo, "for i in _:_", "fooooo", include_asserts=False
    )
    foo, new = extract_subproc(foo, "for i in _:_", "fooooo")
    assert f"{foo}\n{new_no_asserts}\n{new}" == golden


def test_extract_subproc2(golden):
    @proc
    def foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        for i in seq(0, 8):
            x[i, 0] += 2.0

    foo, new = extract_subproc(foo, "for i in _:_", "fooooo")
    assert (str(foo) + "\n" + str(new)) == golden


def test_extract_subproc3(golden):
    @proc
    def foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        assert M >= 2
        if N < 10 and M < 4:
            for i in seq(0, 8):
                x[i, 0] += 2.0
        else:
            for i in seq(0, 8):
                x[i, 0] += 1.0

    foo, foo_if = extract_subproc(foo, "for i in _:_", "foo_if")
    foo, foo_else = extract_subproc(foo, "for i in _:_", "foo_else")
    assert f"{foo}\n{foo_if}\n{foo_else}" == golden


def test_extract_subproc4(golden):
    @proc
    def foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        x[0, 0] = 0.0
        for i in seq(0, 8):
            x[i, 0] += 2.0

    foo, new = extract_subproc(foo, foo.body(), "fooooo")
    assert (str(foo) + "\n" + str(new)) == golden


def test_extract_subproc5(golden):
    @proc
    def foo(x: f32[8], y: f32[8]):
        reg: f32[8] @ AVX2
        for i in seq(0, 8):
            reg[i] = x[i]
        for i in seq(0, 8):
            y[i] = reg[i]

    foo, new = extract_subproc(foo, foo.body()[1:], "fooooo")
    assert (str(foo) + "\n" + str(new)) == golden


def test_extract_subproc6(golden):
    @proc
    def foo(x: [f32][8], y: [f32][8]):
        assert stride(x, 0) == 1
        assert stride(y, 0) == 1
        reg: f32[8] @ AVX2
        for i in seq(0, 8):
            reg[i] = x[i]

    foo, new = extract_subproc(foo, foo.body()[1:], "fooooo")
    assert (str(foo) + "\n" + str(new)) == golden


def test_extract_subproc7(golden):
    @proc
    def gemv(m: size, n: size, alpha: R, beta: R, A: [R][m, n], x: [R][n], y: [R][m]):
        assert stride(A, 1) == 1

        for i in seq(0, m):
            y[i] = y[i] * beta
            for j in seq(0, n):
                y[i] += alpha * x[j] * A[i, j]

    gemv = fission(gemv, gemv.find("y[_] = _").after())
    gemv = reorder_loops(gemv, gemv.find_loop("i #1"))
    gemv, new = extract_subproc(gemv, gemv.find_loop("i #1"), "fooooo")
    assert (str(gemv) + "\n" + str(new)) == golden


def test_unroll_buffer(golden):
    @proc
    def bar(n: size, A: i8[n]):
        for i in seq(0, n):
            for j in seq(0, n):
                tmp_a: i8[5, 2]
                tmp_a[0, 1] = A[i]
                tmp_a[0, 1] = A[i]
                tmp_a[1, 0] = A[i]

    bar0 = unroll_buffer(bar, "tmp_a : _", 0)
    bar1 = unroll_buffer(bar, "tmp_a : _", 1)
    assert str(bar0) + "\n" + str(bar1) == golden


def test_unroll_buffer1(golden):
    @proc
    def foo(src: i8[2], dst: i8[2]):
        src[0] = dst[0]
        src[1] = dst[1]

    @proc
    def bar(n: size, A: i8[n]):
        assert n > 10
        for i in seq(0, n - 4):
            for j in seq(0, n):
                tmp_a: i8[4, 2, 2]
                foo(tmp_a[0, 0, :], A[i : i + 2])
                foo(tmp_a[0, 1, :], A[i + 2 : i + 4])

    bar = unroll_buffer(bar, "tmp_a : _", 1)
    assert str(bar) == golden


def test_unroll_buffer2():
    @proc
    def bar(n: size, A: i8[n]):
        tmp_a: i8[10]
        for i in seq(0, n):
            for j in seq(0, 10):
                tmp_a[j] = A[i]
        tmp_b: i8[10]
        for j in seq(0, 10):
            tmp_b[j] = tmp_a[j]

    with pytest.raises(
        SchedulingError,
        match="Expected a constant buffer access",
    ):
        bar = unroll_buffer(bar, "tmp_a : _", 0)


def test_unroll_buffer3():
    @proc
    def bar(n: size, A: i8[n]):
        tmp_a: i8[n]
        for i in seq(0, n):
            tmp_a[i] = A[i]

    with pytest.raises(
        SchedulingError,
        match="Expected a constant buffer dimension",
    ):
        bar = unroll_buffer(bar, "tmp_a : _", 0)


def test_unroll_buffer4():
    @proc
    def bar(n: size, A: i8[n]):
        tmp_a: i8
        for i in seq(0, n):
            tmp_a = A[i]

    with pytest.raises(
        SchedulingError,
        match="Cannot unroll a scalar buffer",
    ):
        bar = unroll_buffer(bar, "tmp_a : _", 0)


def test_unroll_buffer5():
    @proc
    def foo(B: i8[2]):
        B[0] = 0.0

    @proc
    def bar(n: size, A: i8[n]):
        tmp_a: i8[10]
        foo(tmp_a[0:2])

    with pytest.raises(
        SchedulingError,
        match="Cannot unroll a buffer at a dimension used as a window",
    ):
        bar = unroll_buffer(bar, "tmp_a : _", 0)


def test_unroll_buffer6(golden):
    @proc
    def foo():
        a: f32[2]
        b: f32[2]
        a[0] = b[0]
        a[1] = b[1]

    foo = unroll_buffer(foo, "a : _", 0)
    foo = unroll_buffer(foo, "b : _", 0)
    assert str(foo) == golden


def test_parallelize_loop(golden):
    @proc
    def foo(A: i8[10]):
        for i in seq(0, 10):
            A[i] = 1.0

    foo = parallelize_loop(foo, foo.find_loop("i"))
    assert str(foo) == golden


def test_stage_mem_should_fail():
    @proc
    def foo(x: i8[10, 10, 10]):
        for i in seq(0, 10):
            x[i, i, i] = 1.0

    with pytest.raises(SchedulingError, match="Buffer has accesses"):
        foo = stage_mem(foo, foo.find_loop("i"), "x[0:10, 0:2, 0:10]", "x_tmp")


def test_stage_mem_should_fail2():
    @proc
    def foo(x: i8[10, 10, 10]):
        y: i8
        for i in seq(0, 10):
            y = x[i, i, i]

    with pytest.raises(SchedulingError, match="Buffer has accesses"):
        foo = stage_mem(foo, foo.find_loop("i"), "x[0:10, 0, 0:10]", "x_tmp")


def test_stage_mem_should_fail3():
    @proc
    def foo(x: i8[10, 10, 10], y: i8[10]):
        for i in seq(0, 10):
            x[i, i, i] = 0.0

    with pytest.raises(SchedulingError, match="Cannot stage"):
        foo = stage_mem(foo, foo.find_loop("i"), "y[0:10]", "y_tmp")


def test_stage_mem_okay(golden):
    @proc
    def foo(x: i8[10, 10, 10]):
        y: i8
        for i in seq(0, 10):
            x[i, 0, i] = 1.0
            y = x[2, 0, 3]

    foo = stage_mem(foo, foo.find_loop("i"), "x[0:10, 0, 0:10]", "x_tmp")
    assert str(simplify(foo)) == golden


def test_stage_mem_should_fail4():
    @proc
    def foo(N: size, A: i8[N, N]):
        sum_: i8
        for i in seq(0, N):
            for jo in seq(0, N / 4):
                for ji in seq(0, 4):
                    A[i, 4 * jo + ji] = 0.0
                    y = A[i : i + 1, 4 * jo + ji]
                    sum_ += y[0]

    with pytest.raises(SchedulingError, match="Existing WindowExpr"):
        foo = stage_mem(foo, "for ji in _:_", "A[i, 4*jo:4*jo+4]", "tile")


def test_stage_mem_should_fail5():
    @proc
    def foo(N: size, A: i8[N, N]):
        for i in seq(0, N):
            for jo in seq(0, N / 4):
                for ji in seq(0, 4):
                    A[i, 4 * jo + ji] = 0.0
                    y = A[i : i + 4, 4 * jo + ji]

    with pytest.raises(SchedulingError, match="Buffer has accesses"):
        foo = stage_mem(foo, "for ji in _:_", "A[i-1:i+1, 4*jo:4*jo+4]", "tile")


def test_stage_mem_asum(golden):
    @proc
    def asum(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
        result = 0.0
        for i in seq(0, n):
            result += select(0.0, x[i], x[i], -x[i])

    asum = stage_mem(asum, "result += _", "x[i]", "tile")
    assert str(simplify(asum)) == golden


def test_stage_mem_reduce(golden):
    @proc
    def foo(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
        result = 0.0
        for i in seq(0, n):
            result += x[i] + x[i]

    foo = stage_mem(foo, "result += _", "x[i]", "tile")
    assert str(simplify(foo)) == golden


def test_stage_mem_assign(golden):
    @proc
    def foo(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
        result = 0.0
        for i in seq(0, n):
            result = x[i] + x[i]

    foo = stage_mem(foo, "result = _ #1", "x[i]", "tile")
    assert str(simplify(foo)) == golden


def test_stage_mem_assign2(golden):
    @proc
    def foo(n: size, x: [f32][n] @ DRAM, result: f32 @ DRAM):
        result = 0.0
        for i in seq(0, n):
            result = x[i]
            result = x[i]

    foo = stage_mem(foo, foo.find("result = _ #1").expand(0, 1), "x[i]", "tile")
    assert str(simplify(foo)) == golden


def test_stage_mem_reduce2(golden):
    @proc
    def foo(x: f32[30], result: f32):
        result = 0.0
        for i in seq(0, 30):
            x[i] = result

    foo = stage_mem(foo, foo.body(), "result", "tmp")
    assert str(simplify(foo)) == golden


def test_insert_noop_call(golden):
    @proc
    def foo(n: size, x: i8[n], locality_hint: size):
        assert locality_hint >= 0
        assert locality_hint < 8
        pass

    foo = insert_noop_call(
        foo, foo.find("pass").before(), prefetch, ["x[1:2]", "locality_hint"]
    )
    assert str(foo) == golden


def test_insert_noop_call_bad_args():
    @proc
    def foo(n: size, x: i8[n], locality_hint: size):
        pass

    with pytest.raises(TypeError, match="Function argument count mismatch"):
        insert_noop_call(foo, foo.find("pass").before(), prefetch, [])

    with pytest.raises(SchedulingError, match="Function argument type mismatch"):
        insert_noop_call(
            foo, foo.find("pass").before(), prefetch, ["n", "locality_hint"]
        )


def test_old_lift_alloc_config(golden):
    @config
    class CFG:
        cfg: i8

    @proc
    def bar(n: size, A: i8[n]):
        assert n > 4

        CFG.cfg = A[0]
        win_stmt = A[0:4]
        for i in seq(0, n):
            tmp_a: i8
            tmp_a = A[i]
        A[0] = CFG.cfg

    bar = autolift_alloc(bar, "tmp_a : _", keep_dims=True)
    assert str(bar) == golden
