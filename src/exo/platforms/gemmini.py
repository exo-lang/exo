from __future__ import annotations

from exo import proc, instr, DRAM, config, QAST
from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
from exo.stdlib.scheduling import *


def set_prec_mem(p, bufname, precision, memory):
    p = set_memory(p, bufname, memory)
    p = set_precision(p, bufname, precision)
    return p


def check_eqv(p_new, p_old):
    if str(p_new) != str(p_old):
        print(p_old)
        print(p_new)
        raise ValueError("Mismatch between old and new versions")


old_split = repeat(divide_loop)
old_reorder = repeat(reorder_loops)
old_unroll = repeat(unroll_loop)


def old_fission_after(proc, stmt_pattern, n_lifts=1):
    def find_stmts(p):
        return [c.after() for c in p.find_all(stmt_pattern)]

    return loop_hack(autofission, find_stmts)(proc, n_lifts)


def old_lift_alloc(proc, stmt_pat, n_lifts=1, mode="row", size=None, keep_dims=True):
    def find_stmts(p):
        return p.find_all(stmt_pat)

    return loop_hack(autolift_alloc, find_stmts)(proc, n_lifts, mode, size, keep_dims)


def split_fission_dim(conv):
    conv = old_split(conv, "ocol", 16, ["ocol_o", "ocol_i"], tail="cut_and_guard")
    conv = old_split(conv, "och", 16, ["och_o", "och_i"], perfect=True)
    conv = old_split(conv, "kch", 16, ["kch_o", "kch_i"], perfect=True)
    conv = old_reorder(conv, "ocol_i och_o")
    conv = old_lift_alloc(conv, "res : _", n_lifts=3)
    conv = old_fission_after(conv, "res[_] = _", n_lifts=3)
    conv = old_fission_after(conv, "for krow in _:_", n_lifts=3)
    conv = old_reorder(conv, "och_i krow")
    conv = old_reorder(conv, "och_i kcol")
    conv = old_reorder(conv, "och_i kch_o")
    conv = old_reorder(conv, "ocol_i krow")
    conv = old_reorder(conv, "ocol_i kcol")
    conv = old_reorder(conv, "ocol_i kch_o")
    conv = old_reorder(conv, "och_o krow")
    conv = simplify(conv)
    conv = old_lift_alloc(conv, "i_s : _", n_lifts=6)
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=1)
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=1, mode="col")
    conv = old_reorder(conv, "och_o kcol")
    conv = old_reorder(conv, "och_o kch_o")
    conv = old_lift_alloc(conv, "w_s : _", n_lifts=3)
    conv = old_fission_after(conv, "w_s = _", n_lifts=5)
    conv = old_fission_after(conv, "i_s = _", n_lifts=5)

    return conv


def replace_div_part(conv):
    conv = replace(conv, "for ocol_i in _:_ #0", ld_acc_i32_vector)
    conv = old_reorder(conv, "och_i kch_i")
    conv = old_reorder(conv, "och_o kch_i")
    conv = replace(conv, "for kch_i in _:_ #0", ld_i8_block_id1)
    conv = old_reorder(conv, "kch_o ocol_i")
    conv = replace(conv, "for ocol_i in _:_ #0", ld_i8_block_id2)
    conv = old_reorder(conv, "kch_i och_i")
    conv = replace(conv, "for ocol_i in _:_ #0", matmul_acc_i8)
    conv = replace(conv, "for ocol_i in _:_ #0", st_acc_i8)

    return conv


def replace_mod_part(conv):
    conv = replace(conv, "for ocol_i in _:_ #0", ld_acc_i32_vector)
    conv = old_reorder(conv, "och_i kch_i")
    conv = replace(conv, "for kch_i in _:_ #0", ld_i8_block_id1)
    conv = replace(conv, "for ocol_i in _:_ #0", ld_i8_block_id2)
    conv = old_reorder(conv, "kch_i och_i")
    conv = replace(conv, "for ocol_i in _:_ #0", matmul_acc_i8)
    conv = replace(conv, "for ocol_i in _:_ #0", st_acc_i8)

    return conv


def matmul_tile(gemmini):
    gemmini = divide_loop(gemmini, "j", 4, ["jo", "ji"], perfect=True)
    gemmini = divide_loop(gemmini, "i", 8, ["io", "i"], perfect=True)
    gemmini = divide_loop(gemmini, "io", 2, ["ioo", "io"], perfect=True)
    gemmini = old_reorder(gemmini, "i jo")
    gemmini = old_reorder(gemmini, "io jo")
    return gemmini


def inline_lift_config(gemmini):
    # part of scheduling count, 25
    gemmini = call_eqv(gemmini, "zero_acc_i32(_, _, _)", zero_acc_i32_v2)
    gemmini = inline(gemmini, "zero_acc_i32_v2(_, _, _)")
    gemmini = inline_window(gemmini, "dst = res[_]")
    gemmini = lift_config(gemmini, "config_zero()")

    gemmini = call_eqv(gemmini, "ld_i8_block_id1(_)", ld_i8_block_id1_v2)
    gemmini = inline(gemmini, "ld_i8_block_id1_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = A[_]")
    gemmini = inline_window(gemmini, "dst = a[_]")
    gemmini = lift_config(gemmini, "config_ld_i8_id1()")

    gemmini = call_eqv(gemmini, "ld_i8_block_id2(_)", ld_i8_block_id2_v2)
    gemmini = inline(gemmini, "ld_i8_block_id2_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = B[_]")
    gemmini = inline_window(gemmini, "dst = b[_]")
    gemmini = lift_config(gemmini, "config_ld_i8_id2()")

    gemmini = call_eqv(gemmini, "matmul_acc_i8(_, _, _, _, _)", matmul_acc_i8_v2)
    gemmini = inline(gemmini, "matmul_acc_i8_v2(_, _, _, _, _)")
    gemmini = inline_window(gemmini, "A = a[_]")
    gemmini = inline_window(gemmini, "B = b[_]")
    gemmini = inline_window(gemmini, "C = res[_]")
    gemmini = lift_config(gemmini, "config_matmul()")

    gemmini = call_eqv(gemmini, "st_acc_i8(_, _, _, _, _, _)", st_acc_i8_v2)
    gemmini = inline(gemmini, "st_acc_i8_v2(_, _, _, _, _, _)")
    gemmini = inline_window(gemmini, "src = res[_]")
    gemmini = inline_window(gemmini, "dst = C[_]")
    gemmini = lift_config(gemmini, "config_st_acc_i8(_)")
    return gemmini


def replace_gemmini_calls(gemmini):
    gemmini = replace(gemmini, "for i_in in _:_ #0", zero_acc_i32)
    gemmini = replace(gemmini, "for i_in in _:_ #0", ld_i8_block_id1)
    gemmini = replace(gemmini, "for ki in _:_ #0", ld_i8_block_id2)
    gemmini = replace(gemmini, "for i_in in _:_ #0", matmul_acc_i8)
    gemmini = replace(gemmini, "for i_in in _:_ #0", st_acc_i8)
    return gemmini


def fission_inner_blocks(gemmini):
    gemmini = divide_loop(gemmini, "k", 64, ["ko", "k"], perfect=True)
    gemmini = divide_loop(gemmini, "k", 16, ["k", "ki"], perfect=True)
    gemmini = old_lift_alloc(gemmini, "a : i8", n_lifts=3)
    gemmini = old_lift_alloc(gemmini, "a : _ #0", n_lifts=1, mode="col")
    gemmini = old_lift_alloc(gemmini, "a : _", n_lifts=2)
    gemmini = old_reorder(gemmini, "ki j_in_o")
    gemmini = old_reorder(gemmini, "ki j_in_i")
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=2)
    gemmini = old_lift_alloc(gemmini, "b : i8", n_lifts=1, mode="col")
    gemmini = old_lift_alloc(gemmini, "b : _", n_lifts=3)
    gemmini = old_fission_after(gemmini, "a[_] = _", n_lifts=5)
    gemmini = old_fission_after(gemmini, "b[_] = _", n_lifts=5)
    gemmini = old_reorder(gemmini, "j_in_i i_in")
    gemmini = old_reorder(gemmini, "ki i_in")
    gemmini = old_reorder(gemmini, "k i_in")
    gemmini = old_reorder(gemmini, "j_in_i ki")
    gemmini = old_reorder(gemmini, "j_in_o ki")
    gemmini = old_reorder(gemmini, "j_in_i i_in")
    return gemmini


def fission_outer_blocks(gemmini):
    gemmini = old_fission_after(gemmini, "res[_] = 0.0 #0", n_lifts=3)
    gemmini = old_fission_after(gemmini, "for k in _:_ #0", n_lifts=3)
    gemmini = old_reorder(gemmini, "j_in_i j_in_o")
    gemmini = old_reorder(gemmini, "i_in k")
    gemmini = old_reorder(gemmini, "j_in_i k")
    gemmini = old_reorder(gemmini, "j_in_o k")
    return gemmini


def tile_outer_loops(gemmini):
    gemmini = divide_loop(gemmini, "i", 16, ["i", "i_in"], perfect=True)
    gemmini = old_reorder(gemmini, "i_in j")
    gemmini = divide_loop(gemmini, "j", 64, ["j", "j_in"], perfect=True)
    gemmini = divide_loop(gemmini, "j_in", 16, ["j_in_o", "j_in_i"], perfect=True)
    gemmini = old_reorder(gemmini, "j_in_o j_in_i")

    return gemmini


class QAST_Do:
    def __init__(self, proc):
        self.proc = proc

        # [ self.do_fnarg(a) for a in self.proc.args ]
        [self.do_e(p) for p in self.proc.assertions]
        self.do_stmts(self.proc.body)

    def do_stmts(self, stmts):
        [self.do_s(b) for b in stmts]

    def do_s(self, s):
        if type(s) is QAST.Assign or type(s) is QAST.Reduce:
            [self.do_e(e) for e in s.idx]
            self.do_e(s.rhs)
        elif type(s) is QAST.WriteConfig:
            self.do_e(s.rhs)
        elif type(s) is QAST.For:
            self.do_e(s.lo)
            self.do_e(s.hi)
            self.do_stmts(s.body)
        elif type(s) is QAST.If:
            self.do_e(s.cond)
            self.do_stmts(s.body)
            if len(s.orelse) > 0:
                self.do_stmts(s.orelse)
        elif type(s) is QAST.Pass:
            pass
        elif type(s) is QAST.Alloc:
            pass
        elif type(s) is QAST.Call:
            [self.do_e(e) for e in s.args]
        elif type(s) is QAST.WindowStmt:
            self.do_e(s.rhs)
        else:
            assert False, "bad case"

    def do_w_access(self, w):
        if type(w) is QAST.Interval:
            self.do_e(w.lo)
            self.do_e(w.hi)
        elif type(w) is QAST.Point:
            self.do_e(w.pt)

    def do_e(self, e):
        if type(e) is QAST.Read:
            [self.do_e(ei) for ei in e.idx]
        elif type(e) is QAST.Const:
            pass
        elif type(e) is QAST.USub:
            self.do_e(e.arg)
        elif type(e) is QAST.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        elif type(e) is QAST.BuiltIn:
            [self.do_e(ei) for ei in e.args]
        elif type(e) is QAST.WindowExpr:
            [self.do_w_access(w) for w in e.idx]
        elif type(e) is QAST.StrideExpr:
            pass
        elif type(e) is QAST.ReadConfig:
            pass
        else:
            assert False, "bad case"


class CanFissionLoop(QAST_Do):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        self.result = False
        super().__init__(proc)

    def result(self):
        return self.result

    def do_s(self, s):
        if type(s) is QAST.For:
            assert len(s.body) > 0
            if s.body[0] == self.stmt:
                self.result = True

        super().do_s(s)


class CanFissionIf(QAST_Do):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        self.result = None
        super().__init__(proc)

    def result(self):
        return self.result

    def do_s(self, s):
        if type(s) is QAST.If:
            assert len(s.body) > 0
            if s.body[0] == self.stmt:
                self.result = str(s)
            elif len(s.orelse) > 0 and s.orelse[0] == self.stmt:
                self.result = str(s)

        super().do_s(s)


class CanReorder(QAST_Do):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        self.result = None
        super().__init__(proc)

    def result(self):
        return self.result

    def do_stmts(self, stmts):
        prev = None
        for b in stmts:
            if b == self.stmt and prev is not None:
                self.result = str(prev)
            else:
                self.do_s(b)
                prev = b


def lift_config(conv, string, nth=0):
    stmt = conv.get_ast(string)
    stmt = stmt[nth]  # Get the match

    while True:
        proc = conv.get_ast()
        fission_loop = CanFissionLoop(proc, stmt).result
        reorder = CanReorder(proc, stmt).result
        if fission_loop:
            conv = old_fission_after(conv, string)
        elif reorder is not None:
            conv = reorder_stmts(conv, conv.find(string).expand(1, 0))
            # conv = conv.reorder_before(string)
        else:
            break

    return conv


def inline_vector(conv):
    conv = call_eqv(conv, "ld_acc_i32_vector(_)", ld_acc_i32_vector_v2)
    conv = inline(conv, "ld_acc_i32_vector_v2(_)")
    conv = inline_window(conv, "src = bias[_]")
    conv = inline_window(conv, "dst = res[_]")
    return conv


def inline_ld_id1(conv):
    conv = call_eqv(conv, "ld_i8_block_id1(_)", ld_i8_block_id1_s2_v2)
    conv = inline(conv, "ld_i8_block_id1_s2_v2(_)")
    conv = inline_window(conv, "src = weights[_]")
    conv = inline_window(conv, "dst = w_s[_]")
    return conv


def inline_matmul(conv):
    conv = call_eqv(conv, "matmul_acc_i8(_)", matmul_acc_i8_v2)
    conv = inline(conv, "matmul_acc_i8_v2(_)")
    conv = inline_window(conv, "A = i_s[_]")
    conv = inline_window(conv, "B = w_s[_]")
    conv = inline_window(conv, "C = res[_]")
    return conv


def inline_st(conv):
    conv = call_eqv(conv, "st_acc_i8(_)", st_acc_i8_s2_v2)
    conv = inline(conv, "st_acc_i8_s2_v2(_)")
    conv = inline_window(conv, "src = res[_]")
    conv = inline_window(conv, "dst = output[_]")
    return conv


# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #


@instr("{dst}[0] = ACC_SCALE({src}[0], {scale}[0]);")
def acc_scale(src: i32, dst: f32, scale: f32):
    dst = scale * src


@config
class ConfigLoad:
    src_stride: stride


@config
class ConfigLoad_id1:
    src_stride: stride


@config
class ConfigLoad_id2:
    src_stride: stride


@instr("gemmini_extended3_config_ld({src_stride}, 1.0f, 0, 0);\n")
def config_ld_i8(src_stride: stride):
    ConfigLoad.src_stride = src_stride


@instr("gemmini_extended3_config_ld({src_stride}, 1.0f, 0, 1);\n")
def config_ld_i8_id1(src_stride: stride):
    ConfigLoad_id1.src_stride = src_stride


@instr("gemmini_extended3_config_ld({src_stride}, 1.0f, 0, 2);\n")
def config_ld_i8_id2(src_stride: stride):
    ConfigLoad_id2.src_stride = src_stride


@proc
def ld_i8_prototype(
    n: size,
    m: size,
    src: [i8][n, m] @ DRAM,
    dst: [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i, j]


def make_do_ld_i8(name, instr_str, assert_str=None, add_pass=False, p=ld_i8_prototype):
    p = make_instr(p, instr_str)
    if assert_str:
        p = p.add_assertion(
            assert_str, configs=[ConfigLoad, ConfigLoad_id1, ConfigLoad_id2]
        )
    if add_pass:
        p = insert_pass(p, p.find("for i in _: _").before())
    p = rename(p, name)
    return p


do_ld_i8 = make_do_ld_i8(
    "do_ld_i8",
    ("gemmini_extended_mvin( " "&{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );"),
    "stride(src,0) == ConfigLoad.src_stride",
)
do_ld_i8_id1 = make_do_ld_i8(
    "do_ld_i8_id1",
    ("gemmini_extended_mvin2( " "&{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );"),
    "stride(src,0) == ConfigLoad_id1.src_stride",
)
do_ld_i8_id2 = make_do_ld_i8(
    "do_ld_i8_id2",
    ("gemmini_extended_mvin3( " "&{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );"),
    "stride(src,0) == ConfigLoad_id2.src_stride",
)

_gemm_ld_i8 = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, 1.0f, 0, 0);\n"
    "gemmini_extended_mvin( "
    "&{src_data}, ((uint64_t) &{dst_data}), {m}, {n} );"
)
ld_i8 = make_do_ld_i8(
    "ld_i8",
    _gemm_ld_i8,
    add_pass=True,
)

_gemm_ld_i8_block = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, "
    + "1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( &{src_data}, "
    + "((uint64_t) &{dst_data}), 16*{m}, {n} );"
)


@instr(_gemm_ld_i8_block)
def ld_i8_block(
    n: size,
    m: size,
    src: [i8][n, 16 * m] @ DRAM,
    dst: [i8][m, n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 4
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            for k in seq(0, 16):
                dst[j, i, k] = src[i, 16 * j + k]


def make_do_ld_i8_block(name, instr_str, p=ld_i8_block):
    p = rename(p, name)
    p = make_instr(p, instr_str)
    return p


_do_gemm_ld_i8_block_id1 = (
    "gemmini_extended_mvin2( &{src_data}, " + "((uint64_t) &{dst_data}), 16*{m}, {n} );"
)
_do_gemm_ld_i8_block_id2 = (
    "gemmini_extended_mvin3( &{src_data}, " + "((uint64_t) &{dst_data}), 16*{m}, {n} );"
)
do_ld_i8_block_id1 = make_do_ld_i8_block("do_ld_i8_block_id1", _do_gemm_ld_i8_block_id1)
do_ld_i8_block_id2 = make_do_ld_i8_block("do_ld_i8_block_id2", _do_gemm_ld_i8_block_id2)


_gemm_ld_i8_block_id1 = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, "
    + "1.0f, 0, 1);\n"
    + "gemmini_extended_mvin2( &{src_data}, "
    + "((uint64_t) &{dst_data}), 16*{m}, {n} );"
)
_gemm_ld_i8_block_id2 = (
    "gemmini_extended4_config_ld({src}.strides[0]*1, "
    + "1.0f, 0, {n}, 2);\n"
    + "gemmini_extended_mvin3( &{src_data}, "
    + "((uint64_t) &{dst_data}), 16*{m}, {n} );"
)


def make_ld_i8_block(name, ld_id, p=ld_i8_block, stride_val=None):
    if ld_id == 1:
        ConfigLoad = ConfigLoad_id1
        do_ld_i8_block = do_ld_i8_block_id1
        config_ld_i8 = config_ld_i8_id1
        write_stride = "ConfigLoad_id1.src_stride = _"
        _gemm_ld_i8_block = _gemm_ld_i8_block_id1
    else:
        assert ld_id == 2
        ConfigLoad = ConfigLoad_id2
        do_ld_i8_block = do_ld_i8_block_id2
        config_ld_i8 = config_ld_i8_id2
        write_stride = "ConfigLoad_id2.src_stride = _"
        _gemm_ld_i8_block = _gemm_ld_i8_block_id2

    if stride_val:
        p = write_config(p, p.body().before(), ConfigLoad, "src_stride", stride_val)
        p = replace(p, "for i in _:_", do_ld_i8_block)
        p = replace(p, write_stride, config_ld_i8)

    p = rename(p, name)
    p = make_instr(p, _gemm_ld_i8_block)
    return p


ld_i8_block_id1 = make_ld_i8_block("ld_i8_block_id1", 1)
ld_i8_block_id1_v2 = make_ld_i8_block(
    "ld_i8_block_id1_v2", 1, stride_val="stride(src,0)"
)
ld_i8_block_id1_s2_v2 = make_ld_i8_block(
    "ld_i8_block_id1_s2_v2", 1, stride_val="stride(src,0)"
)

ld_i8_block_id2 = make_ld_i8_block("ld_i8_block_id2", 2)
ld_i8_block_id2_v2 = make_ld_i8_block(
    "ld_i8_block_id2_v2", 2, stride_val="stride(src,0)"
)
ld_i8_block_id2_s2_v2 = make_ld_i8_block(
    "ld_i8_block_id2_s2_v2", 2, stride_val="stride(src,0)"
)


_gemm_zero_block_id2 = (
    "gemmini_extended4_config_ld(0, 1.0f, 0, {n}, 2);\n"
    + "gemmini_extended_mvin3( 0, ((uint64_t) &{dst_data}), 16*{m}, {n} );"
)


@instr(_gemm_zero_block_id2)
def zero_block_id2(
    n: size,
    m: size,
    dst: [i8][m, n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 4
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            for k in seq(0, 16):
                dst[j, i, k] = 0.0


_gemm_ld_i8_id1 = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, 1.0f, 0, 1);\n"
    + "gemmini_extended_mvin2( &{src_data}, "
    + "((uint64_t) &{dst_data}), {m}, {n} );"
)
_gemm_ld_i8_id2 = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, 1.0f, 0, 2);\n"
    + "gemmini_extended_mvin3( &{src_data}, "
    + "((uint64_t) &{dst_data}), {m}, {n} );"
)


def make_ld_i8(name, ld_id=0, p=ld_i8, sval=None):
    if ld_id == 0:
        cfg = ConfigLoad
        do_load = do_ld_i8
        cfg_load = config_ld_i8
        wr_stride = "ConfigLoad.src_stride = _"
        instr_str = _gemm_ld_i8
    elif ld_id == 1:
        cfg = ConfigLoad_id1
        do_load = do_ld_i8_id1
        cfg_load = config_ld_i8_id1
        wr_stride = "ConfigLoad_id1.src_stride = _"
        instr_str = _gemm_ld_i8_id1
    else:
        assert ld_id == 2
        cfg = ConfigLoad_id2
        do_load = do_ld_i8_id2
        cfg_load = config_ld_i8_id2
        wr_stride = "ConfigLoad_id2.src_stride = _"
        instr_str = _gemm_ld_i8_id2

    if sval:
        p = write_config(p, p.body().before(), cfg, "src_stride", sval)
        # p = p.configwrite_after('pass', cfg, 'src_stride', sval)
        p = replace(p, "for i in _:_", do_load)
        p = replace(p, wr_stride, cfg_load)

    p = delete_pass(p)
    p = rename(p, name)
    p = make_instr(p, instr_str)
    return p


ld_i8_id1 = make_ld_i8("ld_i8_id1", 1)
ld_i8_id1_v2 = make_ld_i8("ld_i8_id1_v2", 1, sval="stride(src, 0)")
ld_i8_id1_s2_v2 = make_ld_i8("ld_i8_id1_s2_v2", 1, sval="stride(src, 0)")

ld_i8_id2 = make_ld_i8("ld_i8_id2", 2)
ld_i8_id2_v2 = make_ld_i8("ld_i8_id2_v2", 2, sval="stride(src, 0)")
ld_i8_id2_s2_v2 = make_ld_i8("ld_i8_id2_s2_v2", 2, sval="stride(src, 0)")

ld_i8_v2 = make_ld_i8("ld_i8_v2", 0, sval="stride(src,0)")

# finally remove the pass from ld_i8
ld_i8 = make_ld_i8("ld_i8", 0)


_gemm_ld_i8_stride_2 = (
    "gemmini_extended3_config_ld({src}.strides[0]*2, "
    + "1.0f, 0, 1);\n"
    + "gemmini_extended_mvin2( &{src_data}, "
    + "((uint64_t) &{dst_data}), {m}, {n} );"
)


@instr(_gemm_ld_i8_stride_2)
def ld_i8_s2(
    n: size,
    m: size,
    src: [i8][n * 2 - 1, m] @ DRAM,
    dst: [i8][n, 16] @ GEMM_SCRATCH,
):
    assert 0 < n and n <= 16
    assert 0 < m and m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i * 2, j]


_gemm_config_ld_i8_id1 = (
    "gemmini_extended3_config_ld({src_stride}*2, " + "1.0f, 0, 1);\n"
)


@instr(_gemm_config_ld_i8_id1)
def config_ld_i8_s2_id1(src_stride: stride):
    ConfigLoad_id1.src_stride = src_stride


_do_gemm_ld_i8_stride_2 = (
    "gemmini_extended_mvin2( &{src_data}, " + "((uint64_t) &{dst_data}), {m}, {n} );"
)


@instr(_do_gemm_ld_i8_stride_2)
def do_ld_i8_s2_id1(
    n: size,
    m: size,
    src: [i8][n * 2 - 1, m] @ DRAM,
    dst: [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i * 2, j]


_gemm_ld_i8_vec = (
    "gemmini_extended3_config_ld(1, 1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( &{src_data}, "
    + "((uint64_t) &{dst_data}), 16, 1);"
)


@instr(_gemm_ld_i8_vec)
def ld_i8_vector(
    src: [i8][16] @ DRAM,
    dst: [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16

    for i in seq(0, 16):
        dst[i] = src[i]


_do_gemm_ld_i8_vec = (
    "gemmini_extended_mvin( &{src_data}, ((uint64_t) &{dst_data}), 16, 1);"
)


@instr(_do_gemm_ld_i8_vec)
def do_ld_i8_vector(
    src: [i8][16] @ DRAM,
    dst: [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16

    for i in seq(0, 16):
        dst[i] = src[i]


# in order to load i8 values into the i32 accumulator memory,
# we must specify `shrunk=1` (3rd param of ..._config_ld)
_gemm_ld_acc_i8 = (
    "gemmini_extended3_config_ld({src}.strides[0]*1, "
    + "1.0f, 1, 0);\n"
    + "gemmini_extended_mvin( &{src_data}, "
    + "((uint32_t) &{dst_data}), {m}, {n} );"
)
ld_acc_i8 = rename(ld_i8, "ld_acc_i8")
ld_acc_i8 = set_prec_mem(ld_acc_i8, "dst", "i32", GEMM_ACCUM)
ld_acc_i8 = make_instr(ld_acc_i8, _gemm_ld_acc_i8)


def new_config_ld_acc():
    @config
    class ConfigLoadAcc:
        stride_set: bool

    return ConfigLoadAcc


ConfigLoadAcc = new_config_ld_acc()

_gemm_ld_acc_i32 = (
    "gemmini_extended3_config_ld({src}.strides[0]*4, "
    + "1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( ((uint64_t) &{src_data}), "
    + "((uint32_t) &{dst_data}), {m}, {n} );"
)


@instr(_gemm_ld_acc_i32)
def ld_acc_i32(
    n: size,
    m: size,
    src: [i32][n, m] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i, j]


_gemm_do_ld_acc_i32 = (
    "gemmini_extended_mvin( ((uint64_t) &{src_data}), "
    + "((uint32_t) &{dst_data}), {m}, {n} );"
)


@instr(_gemm_do_ld_acc_i32)
def do_ld_acc_i32(
    n: size,
    m: size,
    src: [i32][n, m] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i, j]


_gemm_config_ld_acc_i32_vector = "gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"


@instr(_gemm_config_ld_acc_i32_vector)
def config_ld_acc_i32_vector(stride_set: bool):
    ConfigLoadAcc.stride_set = stride_set


_gemm_ld_acc_i32_vec = (
    "gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( ((uint64_t) &{src_data}), ((uint32_t) &{dst_data}), 16, {n} );"
)


@instr(_gemm_ld_acc_i32_vec)
def ld_acc_i32_vector(
    n: size,
    src: [i32][1, 16] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, n):
        for j in seq(0, 16):
            dst[i, j] = src[0, j]


_do_gemm_ld_acc_i32_vec = "gemmini_extended_mvin( ((uint64_t) &{src_data}), ((uint32_t) &{dst_data}), 16, {n} );"


@instr(_do_gemm_ld_acc_i32_vec)
def do_ld_acc_i32_vector(
    n: size,
    src: [i32][1, 16] @ DRAM,
    dst: [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in seq(0, n):
        for j in seq(0, 16):
            dst[i, j] = src[0, j]


def make_ld_acc_i32_vector(name, p=ld_acc_i32_vector):
    p = rename(p, name)
    p = write_config(p, p.body().before(), ConfigLoadAcc, "stride_set", "True")
    # p = p.configwrite_after('pass', ConfigLoadAcc, 'stride_set', 'True')
    p = replace(p, "for i in _:_", do_ld_acc_i32_vector)
    p = replace(p, "ConfigLoadAcc.stride_set = _", config_ld_acc_i32_vector)
    p = make_instr(p, _gemm_ld_acc_i32_vec)
    return p


ld_acc_i32_vector_v2 = make_ld_acc_i32_vector("ld_acc_i32_vector_v2")


_gemm_st_i8 = (
    "gemmini_extended_config_st({dst}.strides[0]*1, 0, 1.0f);\n"
    + "gemmini_extended_mvout( "
    + "((uint64_t) &{dst_data}), (uint32_t) &{src_data}, {m}, {n} );"
)


@instr(_gemm_st_i8)
def st_i8(n: size, m: size, src: [i8][n, 16] @ GEMM_SCRATCH, dst: [i8][n, m] @ DRAM):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i, j]


@proc
def clamp(src: f32, dst: i8):
    l: f32
    h: f32
    l = -128.0
    h = 127.0
    dst = select(h, src, h, src)
    dst = select(src, l, l, dst)


def new_config_st():
    @config
    class ConfigStore:
        scale: f32
        dst_stride: stride
        act: bool

    return ConfigStore


ConfigStore = new_config_st()

_gemm_st_acc_i8 = (
    "gemmini_extended_config_st({dst}.strides[0]*1, {act}, {scale}[0]);\n"
    + "gemmini_extended_mvout( ((uint64_t) &{dst_data}), (uint32_t) &{src_data}, {m}, {n} );"
)


@instr(_gemm_st_acc_i8)
def st_acc_i8(
    n: size,
    m: size,
    scale: f32,
    act: bool,
    src: [i32][n, 16] @ GEMM_ACCUM,
    dst: [i8][n, m] @ DRAM,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            src_tmp: i32
            src_tmp = src[i, j]
            tmp: f32
            acc_scale(src_tmp, tmp, scale)
            tmp2: i8
            clamp(tmp, tmp2)
            if act == True:
                tmp2 = relu(tmp2)
            dst[i, j] = tmp2


_gemm_config_st_acc_i8 = (
    "gemmini_extended_config_st({dst_stride}, {act}, {scale}[0]);\n"
)


@instr(_gemm_config_st_acc_i8)
def config_st_acc_i8(scale: f32, dst_stride: stride, act: bool):
    ConfigStore.scale = scale
    ConfigStore.dst_stride = dst_stride
    ConfigStore.act = act


_gemm_st_acc_i8 = "gemmini_extended_mvout( ((uint64_t) &{dst_data}), (uint32_t) &{src_data}, {m}, {n} );"


@instr(_gemm_st_acc_i8)
def do_st_acc_i8(
    n: size, m: size, src: [i32][n, 16] @ GEMM_ACCUM, dst: [i8][n, m] @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            src_tmp: i32
            src_tmp = src[i, j]
            tmp: f32
            acc_scale(src_tmp, tmp, ConfigStore.scale)
            tmp2: i8
            clamp(tmp, tmp2)
            if ConfigStore.act == True:
                tmp2 = relu(tmp2)
            dst[i, j] = tmp2


def make_st_acc_i8_v2(p=st_acc_i8):
    p = rename(st_acc_i8, "st_acc_i8_v2")
    p = bind_config(p, "scale", ConfigStore, "scale")
    p = reorder_stmts(p, "tmp : _ ; ConfigStore.scale = _")
    p = reorder_stmts(p, "src_tmp = _ ; ConfigStore.scale = _")
    p = reorder_stmts(p, "src_tmp : _ ; ConfigStore.scale = _")
    p = old_fission_after(p, "ConfigStore.scale = _", n_lifts=2)
    p = write_config(
        p,
        p.find("ConfigStore.scale = _").after(),
        ConfigStore,
        "dst_stride",
        "stride(dst, 0)",
    )
    # p = p.configwrite_after('ConfigStore.scale = _', ConfigStore, 'dst_stride', 'stride(dst, 0)')
    p = bind_config(p, "act", ConfigStore, "act")
    p = reorder_stmts(p, "clamp(_) ; ConfigStore.act = _")
    p = reorder_stmts(p, "tmp2 : _ ; ConfigStore.act = _")
    p = reorder_stmts(p, "acc_scale(_) ; ConfigStore.act = _")
    p = reorder_stmts(p, "tmp : _ ; ConfigStore.act = _")
    p = reorder_stmts(p, "src_tmp = _ ; ConfigStore.act = _")
    p = reorder_stmts(p, "src_tmp : _ ; ConfigStore.act = _")
    p = old_fission_after(p, "ConfigStore.act = _", n_lifts=2)
    p = replace(p, "for i in _:_", do_st_acc_i8)
    p = replace(
        p,
        "ConfigStore.scale = _ ;" "ConfigStore.dst_stride = _ ;" "ConfigStore.act = _",
        config_st_acc_i8,
    )
    return p


st_acc_i8_v2 = make_st_acc_i8_v2()


def make_st_acc_i8_s2_v2(p=st_acc_i8):
    p = rename(p, "st_acc_i8_s2_v2")
    p = bind_config(p, "scale", ConfigStore, "scale")
    p = reorder_stmts(p, "tmp : _ ; ConfigStore.scale = _")
    p = reorder_stmts(p, "src_tmp = _ ; ConfigStore.scale = _")
    p = reorder_stmts(p, "src_tmp : _ ; ConfigStore.scale = _")
    p = old_fission_after(p, "ConfigStore.scale = _", n_lifts=2)
    p = write_config(
        p,
        p.find("ConfigStore.scale = _").after(),
        ConfigStore,
        "dst_stride",
        "stride(dst, 0)",
    )
    # p = p.configwrite_after('ConfigStore.scale = _', ConfigStore, 'dst_stride', 'stride(dst, 2)')
    p = bind_config(p, "act", ConfigStore, "act")
    p = reorder_stmts(p, "clamp(_) ; ConfigStore.act = _")
    p = reorder_stmts(p, "tmp2 : _ ; ConfigStore.act = _")
    p = reorder_stmts(p, "acc_scale(_) ; ConfigStore.act = _")
    p = reorder_stmts(p, "tmp : _ ; ConfigStore.act = _")
    p = reorder_stmts(p, "src_tmp = _ ; ConfigStore.act = _")
    p = reorder_stmts(p, "src_tmp : _ ; ConfigStore.act = _")
    p = old_fission_after(p, "ConfigStore.act = _", n_lifts=2)
    p = replace(p, "for i in _:_", do_st_acc_i8)
    p = replace(
        p,
        "ConfigStore.scale = _ ;" "ConfigStore.dst_stride = _ ;" "ConfigStore.act = _",
        config_st_acc_i8,
    )
    return p


st_acc_i8_s2_v2 = make_st_acc_i8_s2_v2()

_gemm_st_acc_i32 = (
    "gemmini_extended_config_st({dst}.strides[0]*4, 0, 1.0f);\n"
    + "gemmini_extended_mvout( ((uint64_t) &{dst_data}), "
    + "((uint32_t) &{src_data} | 0x20000000), {m}, {n} );"
)


@instr(_gemm_st_acc_i32)
def st_acc_i32(
    n: size, m: size, src: [i32][n, 16] @ GEMM_ACCUM, dst: [i32][n, m] @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = src[i, j]


_gemm_config_zero = "gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"


@instr(_gemm_config_zero)
def config_zero():
    ConfigLoad.src_stride = 0


_gemm_do_zero = "gemmini_extended_mvin( 0, ((uint64_t) &{dst_data})," + "{m}, {n} );"


@instr(_gemm_do_zero)
def do_zero_i8(
    n: size,
    m: size,
    dst: [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = 0.0


_gemm_zero = (
    "gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( 0, ((uint64_t) &{dst_data}),"
    + "{m}, {n} );"
)


@instr(_gemm_zero)
def zero_i8(
    n: size,
    m: size,
    dst: [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    pass

    for i in seq(0, n):
        for j in seq(0, m):
            dst[i, j] = 0.0


def make_zero_i8_v2(p=zero_i8):
    p = rename(p, "zero_i8_v2")
    p = write_config(p, p.body().before(), ConfigLoad, "src_stride", "0")
    # p = p.configwrite_after('pass', ConfigLoad, 'src_stride', '0')
    p = replace(p, "for i in _:_", do_zero_i8)
    p = replace(p, "ConfigLoad.src_stride = _", config_zero)
    return p


zero_i8_v2 = make_zero_i8_v2()

do_zero_acc_i32 = rename(do_zero_i8, "do_zero_acc_i32")
do_zero_acc_i32 = set_prec_mem(do_zero_acc_i32, "dst", "i32", GEMM_ACCUM)
do_zero_acc_i32 = make_instr(do_zero_acc_i32, _gemm_do_zero)
zero_acc_i32 = rename(zero_i8, "zero_acc_i32")
zero_acc_i32 = set_prec_mem(zero_acc_i32, "dst", "i32", GEMM_ACCUM)
zero_acc_i32 = make_instr(zero_acc_i32, _gemm_zero)


def make_zero_acc_i32_v2(p=zero_acc_i32):
    p = rename(p, "zero_acc_i32_v2")
    p = write_config(p, p.body().before(), ConfigLoad, "src_stride", "0")
    # p = p.configwrite_after('pass', ConfigLoad, 'src_stride', '0')
    p = replace(p, "for i in _:_", do_zero_acc_i32)
    p = replace(p, "ConfigLoad.src_stride = _", config_zero)
    return p


zero_acc_i32_v2 = make_zero_acc_i32_v2()


def del_and_zero(p):
    p = delete_pass(p)
    p = make_instr(p, _gemm_zero)
    return p


zero_i8 = del_and_zero(zero_i8)
zero_i8_v2 = del_and_zero(zero_i8_v2)
zero_acc_i32 = del_and_zero(zero_acc_i32)
zero_acc_i32_v2 = del_and_zero(zero_acc_i32_v2)


_gemm_zero_vec = (
    "gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"
    + "gemmini_extended_mvin( 0, ((uint64_t) &{dst_data}),"
    + "16, 1 );"
)


@instr(_gemm_zero_vec)
def zero_i8_vector(
    dst: [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16
    pass

    for i in seq(0, 16):
        dst[i] = 0.0


_do_gemm_zero_vec = "gemmini_extended_mvin( 0, ((uint64_t) &{dst_data})," + "16, 1 );"


@instr(_do_gemm_zero_vec)
def do_zero_i8_vector(
    dst: [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16

    for i in seq(0, 16):
        dst[i] = 0.0


def make_zero_i8_vector_v2(p=zero_i8_vector):
    p = rename(p, "zero_i8_vector_v2")
    p = write_config(p, p.body().before(), ConfigLoad, "src_stride", "0")
    # p = p.configwrite_after('pass', ConfigLoad, 'src_stride', '0')
    p = replace(p, "for i in _:_", do_zero_i8_vector)
    p = replace(p, "ConfigLoad.src_stride = _", config_zero)
    p = delete_pass(p)
    p = make_instr(p, _gemm_zero_vec)
    return p


zero_i8_vector_v2 = make_zero_i8_vector_v2()

zero_i8_vector = delete_pass(zero_i8_vector)
zero_i8_vector = make_instr(zero_i8_vector, _gemm_zero_vec)


def new_config_matmul():
    @config
    class ConfigMatmul:
        done: bool

    return ConfigMatmul


ConfigMatmul = new_config_matmul()

_gemm_config_matmul = "gemmini_extended_config_ex(WS, 0, 0, 0, 1, 0, 0);\n"


@instr(_gemm_config_matmul)
def config_matmul():
    ConfigMatmul.done = True


_gemm_matmul = (
    "gemmini_extended_preload("
    + "(uint32_t)(&{B_data}), (uint32_t)(&{C_data}), "
    + "{M}, {K}, "
    + "{M}, {N}"
    + ");\n"
    + "gemmini_extended_compute_preloaded("
    + "(uint32_t)(&{A_data}), ~((uint32_t)0), "
    + "{K}, {N}, "
    + "16, 16"
    + ");"
)


@instr(_gemm_config_matmul + _gemm_matmul)
def matmul_i8(
    N: size,
    M: size,
    K: size,
    A: [i8][N, 16] @ GEMM_SCRATCH,
    B: [i8][K, 16] @ GEMM_SCRATCH,
    C: [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    pass
    for i in seq(0, N):
        for j in seq(0, M):
            C[i, j] = 0.0
            for k in seq(0, K):
                a: i32
                b: i32

                a = A[i, k]
                b = B[k, j]

                C[i, j] += a * b


@instr(_gemm_matmul)
def do_matmul_i8(
    N: size,
    M: size,
    K: size,
    A: [i8][N, 16] @ GEMM_SCRATCH,
    B: [i8][K, 16] @ GEMM_SCRATCH,
    C: [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in seq(0, N):
        for j in seq(0, M):
            C[i, j] = 0.0
            for k in seq(0, K):
                a: i32
                b: i32

                a = A[i, k]
                b = B[k, j]

                C[i, j] += a * b


def make_matmul_i8_v2(p=matmul_i8):
    p = rename(p, "matmul_i8_v2")
    p = write_config(p, p.body().before(), ConfigMatmul, "done", "True")
    # p = p.configwrite_after('pass', ConfigMatmul, 'done', 'True')
    p = replace(p, "for i in _:_", do_matmul_i8)
    p = replace(p, "ConfigMatmul.done = True", config_matmul)
    p = delete_pass(p)
    p = make_instr(p, _gemm_matmul)
    return p


matmul_i8_v2 = make_matmul_i8_v2()

matmul_i8 = delete_pass(matmul_i8)
matmul_i8 = make_instr(matmul_i8, _gemm_config_matmul + _gemm_matmul)


_gemm_matmul_acc = (
    "gemmini_extended_preload("
    + "(uint32_t)(&{B_data}), (uint32_t)(&{C_data}) | 0x40000000, "
    + "{M}, {K}, "
    + "{M}, {N}"
    + ");\n"
    + "gemmini_extended_compute_preloaded("
    + "(uint32_t)(&{A_data}), ~((uint32_t)0), "
    + "{K}, {N}, "
    + "16, 16"
    + ");"
)


@instr(_gemm_matmul_acc)
def matmul_acc_i8(
    N: size,
    M: size,
    K: size,
    A: [i8][N, 16] @ GEMM_SCRATCH,
    B: [i8][K, 16] @ GEMM_SCRATCH,
    C: [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    pass
    for i in seq(0, N):
        for j in seq(0, M):
            for k in seq(0, K):
                a: i32
                b: i32

                a = A[i, k]
                b = B[k, j]

                C[i, j] += a * b


@instr(_gemm_matmul_acc)
def do_matmul_acc_i8(
    N: size,
    M: size,
    K: size,
    A: [i8][N, 16] @ GEMM_SCRATCH,
    B: [i8][K, 16] @ GEMM_SCRATCH,
    C: [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in seq(0, N):
        for j in seq(0, M):
            for k in seq(0, K):
                a: i32
                b: i32

                a = A[i, k]
                b = B[k, j]

                C[i, j] += a * b


def make_matmul_acc_i8_v2(p=matmul_acc_i8):
    p = rename(p, "matmul_acc_i8_v2")
    p = write_config(p, p.body().before(), ConfigMatmul, "done", "True")
    # p = p.configwrite_after('pass', ConfigMatmul, 'done', 'True')
    p = replace(p, "for i in _:_", do_matmul_acc_i8)
    p = replace(p, "ConfigMatmul.done = True", config_matmul)
    p = delete_pass(p)
    p = make_instr(p, _gemm_matmul_acc)
    return p


matmul_acc_i8_v2 = make_matmul_acc_i8_v2()

matmul_acc_i8 = delete_pass(matmul_acc_i8)
matmul_acc_i8 = make_instr(matmul_acc_i8, _gemm_config_matmul + _gemm_matmul_acc)

# --------------------------------------------------------------------------- #
#
# --------------------------------------------------------------------------- #
