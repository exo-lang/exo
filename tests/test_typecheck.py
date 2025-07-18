from __future__ import annotations

import pytest

from exo import proc, config
from exo.core.LoopIR import LoopIR
from exo.libs.memories import GEMM_SCRATCH
from exo.frontend.pyparser import ParseError
from exo.libs.externs import *


# --- Typechecking tests ---


def new_config_ld():
    @config
    class ConfigLoad:
        scale: f32
        src_stride: stride

    return ConfigLoad


def test_size0():
    with pytest.raises(
        ParseError, match="Cannot allocate an intermediate value of type"
    ):

        @proc
        def foo(x: size):
            size: size
            pass


def test_stride1():
    ConfigLoad = new_config_ld()

    @proc
    def foo(n: size, x: R[n]):
        ConfigLoad.src_stride = stride(x, 0)


def test_seq1():
    @proc
    def foo():
        for i in seq(0, 10):
            pass


def test_loop1():
    @proc
    def foo():
        for i in seq(0, 10):
            for j in seq(0, i):
                pass


def test_fresh1():
    with pytest.raises(
        TypeError, match="unable to disambiguate assignment to undefined variable"
    ):

        @proc
        def foo(n: size, A: R[n] @ GEMM_SCRATCH):
            for i in seq(0, n):
                tmp = A[i]
                A[i] = tmp


def test_fresh2():
    with pytest.raises(
        TypeError, match="unable to disambiguate assignment to undefined variable"
    ):

        @proc
        def foo(n: size, A: R[n] @ GEMM_SCRATCH):
            for i in seq(0, n):
                tmp = 0.0
                A[i] = tmp


def test_sin1():
    @proc
    def sin_proc(x: f32):
        y: f32
        y = sin(x)


def test_sin2():
    @proc
    def sin_proc(x: f32):
        y: f32
        if False:
            y = sin(x)


def test_bool1():
    @proc
    def bool(b: bool):
        assert b == True

        x: f32
        if b == True:
            x = 0.0


def test_bool2():
    @proc
    def bool(a: bool, b: bool):
        x: f32
        if a == b:
            x = 0.0


def test_bool3():
    @proc
    def bool(a: bool, b: bool):
        x: f32
        if False:
            x = 0.0


def test_bool4():
    @proc
    def bool(a: bool, b: bool):
        x: f32
        if a:
            x = 0.0


def test_badpred():
    with pytest.raises(TypeError, match="Errors occurred during typechecking"):

        @proc
        def badpred(m: size):
            assert m + 1
            assert 10
            tmp: R[m]


def test_badaccess():
    with pytest.raises(TypeError, match="expected access of variable"):

        @proc
        def badaccess(m: size, x: R[m]):
            res: R[m]
            for i in seq(0, m):
                res[i] = x[i, 3]


def test_badaccess2():
    with pytest.raises(TypeError, match="expected lvalue access of variable"):

        @proc
        def badaccess2(m: size, x: R[m]):
            res: R[m, m]
            for i in seq(0, m):
                res[i] = x[i]


def test_badaccess3():
    with pytest.raises(TypeError, match="cannot assign/reduce to"):

        @proc
        def badaccess3(m: size, n: index, x: R):
            n = x


def test_badaccess4():
    with pytest.raises(TypeError, match="cannot assign/reduce a"):

        @proc
        def badaccess4():
            x: R
            for i in seq(0, 10):
                x = i


def test_pass():
    @proc
    def p(x: R[10]):
        pass

    return p

    @proc
    def p():
        pass

    return p


def test_if1():
    with pytest.raises(TypeError, match="expected a bool expression"):

        @proc
        def hoge():
            if 4:
                pass


def test_if2():
    @proc
    def hoge():
        if 1 == 0:
            pass
        else:
            x: R
            pass


def test_par2():
    with pytest.raises(TypeError, match="expected loop bound to be indexable"):

        @proc
        def hoge(x: R):
            for i in seq(0, x):
                pass


def test_call_pass1():
    @proc
    def hoge(y: R):
        pass

    # integer scalar now coerces to R if it is an environment where R is required
    with pytest.raises(
        TypeError, match="expected scalar arguments to be simply variable names for now"
    ):

        @proc
        def huga():
            pass
            x: R
            hoge(3 + x)


def test_call_read_size1():
    @proc
    def hoge(y: size):
        pass

    with pytest.raises(TypeError, match="expected size or index type expression"):

        @proc
        def foo(x: R):
            hoge(x)


def test_call_index_read1():
    @proc
    def hoge(y: index):
        pass

    with pytest.raises(TypeError, match="expected size or index type expression"):

        @proc
        def foo(x: R):
            hoge(x)


def test_call_tensor1_read1():
    @proc
    def hoge(n: size, m: size, y: f64[n, m]):
        pass

    with pytest.raises(TypeError, match="expected argument of type"):

        @proc
        def foo(n: size, m: size, x: R[m, n, 10]):
            hoge(n, m, x)


def test_call_tensor2_read1():
    @proc
    def hoge(y: f64):
        pass

    msg = "expected scalar arguments to be simply variable names for now"
    with pytest.raises(TypeError, match=msg):

        @proc
        def foo():
            y: R
            x: R
            hoge(x + y)


def test_const_bool():
    with pytest.raises(TypeError, match="cannot assign/reduce a 'bool' type value"):

        @proc
        def hoge(x: R):
            x = True


def test_usub():
    @proc
    def hoge(x: R):
        x = -x


def test_usub2():
    with pytest.raises(TypeError, match="cannot negate expression of type "):

        @proc
        def hoge(x: R[1]):
            x = -x


def test_usub3():
    @proc
    def foo(x: f32):
        x = -1 + x


def test_binop1():
    with pytest.raises(TypeError, match="cannot negate expression of type "):

        @proc
        def hoge(x: R[1]):
            x = -x + 3.0


def test_binop2():
    with pytest.raises(TypeError, match="expected 'bool' argument to logical op"):

        @proc
        def hoge():
            if (1 == 1) and 3:
                pass


def test_binop3():
    with pytest.raises(TypeError, match="expected 'bool' argument to logical op"):

        @proc
        def hoge():
            if 3 and (1 == 1):
                pass


@pytest.mark.skip()
def test_binop4():
    with pytest.raises(TypeError, match='using "==" for boolean not supported.'):

        @proc
        def hoge():
            if (0 == 1) == (1 == 1):
                pass


def test_binop5():
    with pytest.raises(
        TypeError, match="expected 'index' or 'size' argument to comparison op"
    ):

        @proc
        def hoge():
            if 1 < (1 == 1):
                pass


def test_binop6():
    with pytest.raises(
        TypeError, match="expected 'index' or 'size' argument to comparison op"
    ):

        @proc
        def hoge():
            if (1 == 1) < 0:
                pass


def test_binop7():
    # integer scalar now coerces to R if it is an environment where R is required
    @proc
    def hoge(x: R):
        x = x + 8


def test_binop8():
    with pytest.raises(TypeError, match="cannot compute modulus of"):

        @proc
        def hoge(x: R):
            x = x % 8.0


def test_binop9():
    @proc
    def hoge(x: f64):
        x = x + 8.0


def test_binop10():
    @proc
    def hoge(x: i8):
        x = x + 8.0


def test_binop11():
    with pytest.raises(TypeError, match="cannot perform arithmetic on 'bool' values"):

        @proc
        def hoge(x: i8):
            x = (1 == 0) + (0 == 1)


def test_binop12():
    with pytest.raises(
        TypeError, match="cannot divide or modulo by a non-constant value"
    ):

        @proc
        def hoge(x: size, y: size):
            if (x / y) > 0:
                pass


def test_binop13():
    with pytest.raises(
        TypeError, match="cannot divide or modulo by zero or a negative value"
    ):

        @proc
        def hoge(x: size, y: size):
            if (x / -3) > 0:
                pass


def test_binop13_2():
    with pytest.raises(
        TypeError, match="cannot divide or modulo by zero or a negative value"
    ):

        @proc
        def hoge(x: size, y: size):
            if (x / 0) > 0:
                pass


def test_binop14():
    @proc
    def hoge(x: size, y: size):
        if (4 * x) > 0:
            pass


def test_binop15():
    with pytest.raises(
        TypeError, match="cannot multiply two non-constant indexing/sizing expressions"
    ):

        @proc
        def hoge(x: size, y: size):
            if (y * x) > 0:
                pass


def test_binop16():
    @proc
    def foo(x: f32, y: f32):
        x = y + (1 * 4)


def test_binop17():
    # attempt to coerce to R should not allow mod on R scalars
    with pytest.raises(TypeError, match="cannot compute modulus of 'R' values"):

        @proc
        def foo(x: i32, y: i32):
            x = y + (1 % 4)


def test_binop18():
    # do not coerce if adding to control value
    with pytest.raises(TypeError, match="expected scalar type"):

        @proc
        def foo(x: i32, y: i32, n: size):
            x = y * (n + 1)


def test_binop19():
    # make sure constant 10 coerces to indexable int type in mod expression
    @proc
    def foo(n: size, m: size):
        if n == m % 10:
            pass


def test_binop20():
    # make sure constant 10 coerces to indexable int type in comparison operations
    @proc
    def foo(n: size):
        if n == 10:
            pass


def test_proj_bad():
    msg = "type-shape of calling argument may not equal the required type-shape"

    @proc
    def dot(m: size, x: R[1, 1], y: R[m]):
        huga: R
        pass

    with pytest.raises(TypeError, match=msg):

        @proc
        def proj(n: size, x: R[100, 10, 1], y: R[10, n]):
            dot(n, x[1], y[0])


def test_numeric_type_mismatch():
    with pytest.raises(TypeError, match="but got type size"):

        @proc
        def bar(n: R):
            pass

        @proc
        def foo(n: size):
            bar(n)


def test_wrong_arg_count():
    with pytest.raises(TypeError, match="got 3 arguments"):

        @proc
        def bar(n: R):
            pass

        @proc
        def foo(m: size, n: size, k: size):
            bar(m, n, k)


def test_window_dim():
    with pytest.raises(Exception, match="expected 2 indices"):

        @proc
        def bar(t: [f32][4, 8]):
            pass

        @proc
        def foo(t: f32[8, 8]):
            bar(t[4:])


def test_window_of_window():
    # fmt: off
    @proc
    def foo(i: index, dense_tensor: f32[64, 64], window_parameter: [f32][32, 128]):
        assert(i < 16)
        dt_window_1a = dense_tensor[i:, 16:]
        dt_window_2a = dt_window_1a[4, 0:]      # dense_tensor[i+4, 16:]
        dt_window_1b = dense_tensor[i, 10:]
        dt_window_2b = dt_window_1b[2*i:]       # dense_tensor[i, 10+2*i:]
        wp_window_1a = window_parameter[i:, 3:]
        wp_window_2a = wp_window_1a[0:16, i]    # window_parameter[i:i+16,3+i]
        wp_window_3a = wp_window_2a[1:]         # window_parameter[i+1:i+16,3+i]
    # fmt: on

    str_to_type = {}
    str_to_sym = {}

    loopir = foo._loopir_proc
    for fnarg in loopir.args:
        str_to_sym[str(fnarg.name)] = fnarg.name
        str_to_type[str(fnarg.name)] = fnarg.type
    for stmt in loopir.body:
        if isinstance(stmt, LoopIR.WindowStmt):
            str_to_type[str(stmt.name)] = stmt.rhs.type
            str_to_sym[str(stmt.name)] = stmt.name

    i_sym = str_to_sym["i"]
    dt_sym = str_to_sym["dense_tensor"]
    wp_sym = str_to_sym["window_parameter"]

    # Check window shapes correct
    as_tensor_str = lambda nm: str(str_to_type[nm].as_tensor)
    assert as_tensor_str("dt_window_1a") == "[f32][64 - i, 64 - 16]"
    assert as_tensor_str("dt_window_2a") == "[f32][64 - 16]"
    assert as_tensor_str("dt_window_1b") == "[f32][64 - 10]"
    assert as_tensor_str("dt_window_2b") == "[f32][64 - 10 - 2 * i]"
    assert as_tensor_str("wp_window_1a") == "[f32][32 - i, 128 - 3]"
    assert as_tensor_str("wp_window_2a") == "[f32][16]"
    assert as_tensor_str("wp_window_3a") == "[f32][16 - 1]"

    # Check accurate tracking of source tensor type and source tensor alias (Sym)
    src_type_str = lambda nm: str(str_to_type[nm].src_type)
    for nm in ("dt_window_1a", "dt_window_2a", "dt_window_1b", "dt_window_2b"):
        assert src_type_str(nm) == "f32[64, 64]"  # dense tensor[...]
        assert str_to_type[nm].src_buf is dt_sym
    for nm in ("wp_window_1a", "wp_window_2a", "wp_window_3a"):
        assert src_type_str(nm) == "[f32][32, 128]"  # [window parameter][...]
        assert str_to_type[nm].src_buf is wp_sym

    # Check windowing expression relative to aliased source tensor is correct
    def check_idx(nm, lo_or_pt0, optional_hi0, lo_or_pt1, optional_hi1):
        # We capture the slicing expressions as strings, and evaluate with i=5
        # so as not to test the simplification of indexing expressions.
        idx0, idx1 = str_to_type[nm].idx
        _eval = lambda thing: eval(str(thing), {"i": 5})
        if optional_hi0 is None:
            assert isinstance(idx0, LoopIR.Point)
            assert _eval(idx0.pt) == _eval(lo_or_pt0)
        else:
            assert isinstance(idx0, LoopIR.Interval)
            assert _eval(idx0.lo) == _eval(lo_or_pt0)
            assert _eval(idx0.hi) == _eval(optional_hi0)
        if optional_hi1 is None:
            assert isinstance(idx1, LoopIR.Point)
            assert _eval(idx1.pt) == _eval(lo_or_pt1)
        else:
            assert isinstance(idx1, LoopIR.Interval)
            assert _eval(idx1.lo) == _eval(lo_or_pt1)
            assert _eval(idx1.hi) == _eval(optional_hi1)

    check_idx("dt_window_2a", "i+4", None, "16", "64")
    check_idx("dt_window_2b", "i", None, "10 + 2*i", "64")
    check_idx("wp_window_2a", "i", "i+16", "3+i", None)
    check_idx("wp_window_3a", "i+1", "i+16", "3+i", None)
