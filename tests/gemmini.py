from __future__ import annotations

from SYS_ATL import proc, instr, DRAM, config, QAST
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM

class QAST_Do():
    def __init__(self, proc):
        self.proc = proc

        #[ self.do_fnarg(a) for a in self.proc.args ]
        [ self.do_e(p) for p in self.proc.assertions ]
        self.do_stmts(self.proc.body)

    def do_stmts(self, stmts):
        [ self.do_s(b) for b in stmts ]

    def do_s(self, s):
        if type(s) is QAST.Assign or type(s) is QAST.Reduce:
            [ self.do_e(e) for e in s.idx ]
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
            [ self.do_e(e) for e in s.args ]
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
            [ self.do_e(ei) for ei in e.idx ]
        elif type(e) is QAST.Const:
            pass
        elif type(e) is QAST.USub:
            self.do_e(e.arg)
        elif type(e) is QAST.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        elif type(e) is QAST.BuiltIn:
            [ self.do_e(ei) for ei in e.args ]
        elif type(e) is QAST.WindowExpr:
            [ self.do_w_access(w) for w in e.idx ]
        elif type(e) is QAST.StrideExpr:
            pass
        elif type(e) is QAST.ReadConfig:
            pass
        else:
            assert False, "bad case"

class CanFission(QAST_Do):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        self.result = False
        super().__init__(proc)

    def result(self):
        return self.result

    def do_s(self, s):
        if type(s) is QAST.If:
            assert len(s.body) > 0
            if s.body[0] == self.stmt:
                self.result = True
        elif type(s) is QAST.For:
            assert len(s.body) > 0
            if s.body[0] == self.stmt:
                self.result = True

        super().do_s(s)

class CanReorder(QAST_Do):
    def __init__(self, proc, stmt):
        self.stmt = stmt
        self.result = False
        super().__init__(proc)

    def result(self):
        return self.result

    def do_stmts(self, stmts):
        is_first = True
        for b in stmts:
            if b == self.stmt and not is_first:
                self.result = True
            else:
                self.do_s(b)
                is_first = False

def lift_config(conv, string):
    stmt = conv.get_ast(string)
    assert len(stmt) == 1
    stmt = stmt[0][0] # Get the match

    while True:
        proc = conv.get_ast()
        fission = CanFission(proc, stmt).result
        reorder = CanReorder(proc, stmt).result
        if fission:
            conv = conv.fission_after(string)
        elif reorder:
            conv = conv.reorder_before(string)
        else:
            break

    return conv

def conv_replace_s1(conv):
    conv = conv.replace(ld_acc_i32_vector, 'for och_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.replace(ld_i8, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8_vector, 'for kch_i in _:_ #0')
    conv = conv.replace(zero_i8_vector, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8, 'for ocol_i in _:_ #1')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #1')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #1')

    conv = conv.replace(ld_acc_i32_vector, 'for och_i in _:_ #0')
    conv = conv.reorder('och_i', 'kch_i')
    conv = conv.replace(ld_i8, 'for kch_i in _:_ #0')
    conv = conv.replace(ld_i8, 'for ocol_i in _:_ #2')
    conv = conv.replace(ld_i8_vector, 'for kch_i in _:_ #0')
    conv = conv.replace(zero_i8_vector, 'for kch_i in _:_ #0')
    conv = conv.reorder('kch_i', 'och_i')
    conv = conv.replace(matmul_acc_i8, 'for ocol_i in _:_ #2')
    conv = conv.replace(st_acc_i8, 'for ocol_i in _:_ #2')

    conv = conv.set_memory('res', GEMM_ACCUM)
    conv = conv.set_memory('i_s', GEMM_SCRATCH)
    conv = conv.set_memory('w_s', GEMM_SCRATCH)

    return conv

def conv_basic_opt(conv, if1, if2, if3, if4, if5):
    conv = conv.split('ocol', 16, ['ocol_o', 'ocol_i'], tail='cut_and_guard')
    conv = conv.split('och', 16, ['och_o', 'och_i'], perfect=True)
    conv = conv.split('kch', 16, ['kch_o', 'kch_i'], perfect=True)
    conv = conv.reorder('ocol_i', 'och_o')
    conv = conv.lift_alloc('res : _', n_lifts=2)
    conv = conv.fission_after('res[_] = _', n_lifts=2)
    conv = conv.fission_after('for krow in _:_', n_lifts=2)
    conv = conv.reorder('och_i', 'krow')
    conv = conv.reorder('och_i', 'kcol')
    conv = conv.reorder('och_i', 'kch_o')
    conv = conv.reorder('ocol_i', 'krow')
    conv = conv.reorder('ocol_i', 'kcol')
    conv = conv.reorder('ocol_i', 'kch_o')
    conv = conv.lift_alloc('i_s : _', n_lifts=4)
    conv = conv.lift_if(if1, n_lifts=3)
    conv = conv.lift_alloc('w_s : _', n_lifts=1)
    conv = conv.lift_alloc('w_s : _', n_lifts=1, mode='col')
    conv = conv.lift_alloc('w_s : _', n_lifts=2)
    conv = conv.fission_after('w_s = _', n_lifts=3)
    conv = conv.fission_after(if2, n_lifts=3)
    conv = conv.fission_after(if3, n_lifts=3)
    conv = conv.partition_loop('ocol_i #1', 1)
    conv = conv.unroll('ocol_i #1')
    conv = conv.simplify()
    conv = conv.lift_if(if4, n_lifts=1)
    conv = conv.lift_if(if5, n_lifts=1)

    return conv

# --------------------------------------------------------------------------- #
#   Instructions
# --------------------------------------------------------------------------- #

@instr("{dst}[0] = ACC_SCALE({src}[0], {scale}[0]);")
def acc_scale(src : i32, dst : f32, scale : f32):
    pass

def new_config_ld():
    @config
    class ConfigLoad:
        src_stride : stride

    return ConfigLoad

def new_config_ld_id1():
    @config
    class ConfigLoad_id1:
        src_stride : stride

    return ConfigLoad_id1

def new_config_ld_id2():
    @config
    class ConfigLoad_id2:
        src_stride : stride

    return ConfigLoad_id2

ConfigLoad = new_config_ld()
ConfigLoad_id1 = new_config_ld_id1()
ConfigLoad_id2 = new_config_ld_id2()

_gemm_config_ld_i8   = ("gemmini_extended3_config_ld({src_stride}, "+
                        "1.0f, 0, 0);\n")
@instr(_gemm_config_ld_i8)
def config_ld_i8(
    src_stride : stride
):
    ConfigLoad.src_stride = src_stride

_gemm_config_ld_i8_id1 = ("gemmini_extended3_config_ld({src_stride}, "+
                        "1.0f, 0, 1);\n")
@instr(_gemm_config_ld_i8_id1)
def config_ld_i8_id1(
    src_stride : stride
):
    ConfigLoad_id1.src_stride = src_stride

_gemm_config_ld_i8_id2 = ("gemmini_extended3_config_ld({src_stride}, "+
                          "1.0f, 0, 2);\n")
@instr(_gemm_config_ld_i8_id2)
def config_ld_i8_id2(
    src_stride : stride
):
    ConfigLoad_id2.src_stride = src_stride


_gemm_do_ld_i8   = ("gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_do_ld_i8)
def do_ld_i8(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == ConfigLoad.src_stride

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_do_ld_i8_id1 = ("gemmini_extended_mvin2( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_do_ld_i8_id1)
def do_ld_i8_id1(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1
    #assert stride(src, 0) == ConfigLoad_id1.src_stride

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_do_ld_i8_id2 = ("gemmini_extended_mvin3( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_do_ld_i8_id2)
def do_ld_i8_id2(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1
    #assert stride(src, 0) == ConfigLoad_id2.src_stride

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_ld_i8   = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                 "1.0f, 0, 0);\n"+
                 "gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_i8)
def ld_i8(
    n     : size,
    m     : size,
    src   : [i8][n, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    pass

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

ld_i8_v2 = ld_i8.rename("ld_i8_v2")
ld_i8_v2 = ld_i8_v2.configwrite_after('pass', ConfigLoad, 'src_stride', 'stride(src, 0)')
ld_i8_v2 = ld_i8_v2.replace(do_ld_i8, 'for i in _:_')
ld_i8_v2 = ld_i8_v2.replace(config_ld_i8, 'ConfigLoad.src_stride = _')


_gemm_ld_i8_id1 = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                 "1.0f, 0, 1);\n"+
                 "gemmini_extended_mvin2( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
ld_i8_id1 = ld_i8.rename("ld_i8_id1").make_instr(_gemm_ld_i8_id1)

_gemm_ld_i8_id2 = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                 "1.0f, 0, 2);\n"+
                 "gemmini_extended_mvin3( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
ld_i8_id2 = ld_i8.rename("ld_i8_id2").make_instr(_gemm_ld_i8_id2)

ld_i8_id1_v2 = ld_i8_id1.rename("ld_i8_id1_v2")
ld_i8_id1_v2 = ld_i8_id1_v2.configwrite_after('pass', ConfigLoad_id1, 'src_stride', 'stride(src, 0)')
ld_i8_id1_v2 = ld_i8_id1_v2.replace(do_ld_i8_id1, 'for i in _:_')
ld_i8_id1_v2 = ld_i8_id1_v2.replace(config_ld_i8_id1, 'ConfigLoad_id1.src_stride = _')
ld_i8_id1_v2 = ld_i8_id1_v2.delete_pass().make_instr(_gemm_ld_i8)

ld_i8_id1_s2_v2 = ld_i8_id1.rename("ld_i8_id1_s2_v2")
ld_i8_id1_s2_v2 = ld_i8_id1_s2_v2.configwrite_after('pass', ConfigLoad_id1, 'src_stride', 'stride(src, 2)')
ld_i8_id1_s2_v2 = ld_i8_id1_s2_v2.replace(do_ld_i8_id1, 'for i in _:_')
ld_i8_id1_s2_v2 = ld_i8_id1_s2_v2.replace(config_ld_i8_id1, 'ConfigLoad_id1.src_stride = _')
ld_i8_id1_s2_v2 = ld_i8_id1_s2_v2.delete_pass().make_instr(_gemm_ld_i8)

ld_i8_id2_v2 = ld_i8_id2.rename("ld_i8_id2_v2")
ld_i8_id2_v2 = ld_i8_id2_v2.configwrite_after('pass', ConfigLoad_id2, 'src_stride', 'stride(src, 0)')
ld_i8_id2_v2 = ld_i8_id2_v2.replace(do_ld_i8_id2, 'for i in _:_')
ld_i8_id2_v2 = ld_i8_id2_v2.replace(config_ld_i8_id2, 'ConfigLoad_id2.src_stride = _')
ld_i8_id2_v2 = ld_i8_id2_v2.delete_pass().make_instr(_gemm_ld_i8)

ld_i8    = ld_i8.delete_pass().make_instr(_gemm_ld_i8)


_gemm_ld_i8_stride_2 = ("gemmini_extended3_config_ld({src}.strides[0]*2, "+
                        "1.0f, 0, 1);\n"+
                        "gemmini_extended_mvin2( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_i8_stride_2)
def ld_i8_s2(
    n     : size,
    m     : size,
    src   : [i8][n*2-1, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert 0 < n and n <= 16
    assert 0 < m and m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i*2,j]

_gemm_config_ld_i8_id1 = ("gemmini_extended3_config_ld({src_stride}*2, "+
                          "1.0f, 0, 1);\n")
@instr(_gemm_config_ld_i8_id1)
def config_ld_i8_s2_id1(
    src_stride : stride
):
    ConfigLoad_id1.src_stride = src_stride

_do_gemm_ld_i8_stride_2 = ("gemmini_extended_mvin2( {src}.data, "+
                              "((uint64_t) {dst}.data), {m}, {n} );")
@instr(_do_gemm_ld_i8_stride_2)
def do_ld_i8_s2_id1(
    n     : size,
    m     : size,
    src   : [i8][n*2-1, m] @ DRAM,
    dst   : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i*2,j]

_gemm_ld_i8_vec = ("gemmini_extended3_config_ld(1, 1.0f, 0, 0);\n"+
                   "gemmini_extended_mvin( {src}.data, "+
                              "((uint64_t) {dst}.data), 16, 1);")
@instr(_gemm_ld_i8_vec)
def ld_i8_vector(
    src   : [i8][16] @ DRAM,
    dst   : [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16

    for i in par(0, 16):
        dst[i] = src[i]

_do_gemm_ld_i8_vec = ("gemmini_extended_mvin( {src}.data, ((uint64_t) {dst}.data), 16, 1);")
@instr(_do_gemm_ld_i8_vec)
def do_ld_i8_vector(
    src   : [i8][16] @ DRAM,
    dst   : [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16

    for i in par(0, 16):
        dst[i] = src[i]





# in order to load i8 values into the i32 accumulator memory,
# we must specify `shrunk=1` (3rd param of ..._config_ld)
_gemm_ld_acc_i8 = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                   "1.0f, 1, 0);\n"+
                   "gemmini_extended_mvin( {src}.data, "+
                                "((uint32_t) {dst}.data), {m}, {n} );")
ld_acc_i8 = (ld_i8.rename('ld_acc_i8')
                  .set_precision('dst', 'i32')
                  .set_memory('dst', GEMM_ACCUM)
                  .make_instr(_gemm_ld_acc_i8))


def new_config_ld_acc():
    @config
    class ConfigLoadAcc:
        stride_set : bool

    return ConfigLoadAcc
ConfigLoadAcc = new_config_ld_acc()

_gemm_ld_acc_i32   = ("gemmini_extended3_config_ld({src}.strides[0]*4, "+
                      "1.0f, 0, 0);\n"+
                      "gemmini_extended_mvin( ((uint64_t) {src}.data), "+
                               "((uint32_t) {dst}.data), {m}, {n} );")
@instr(_gemm_ld_acc_i32)
def ld_acc_i32(
    n     : size,
    m     : size,
    src   : [i32][n, m] @ DRAM,
    dst   : [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_do_ld_acc_i32   = ("gemmini_extended_mvin( ((uint64_t) {src}.data), "+
                               "((uint32_t) {dst}.data), {m}, {n} );")
@instr(_gemm_do_ld_acc_i32)
def do_ld_acc_i32(
    n     : size,
    m     : size,
    src   : [i32][n, m] @ DRAM,
    dst   : [i32][n, 16] @ GEMM_ACCUM,
):
    assert n <= 16
    assert m <= 16
    assert stride(src, 1) == 1
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = src[i,j]

_gemm_config_ld_acc_i32_vector = ("gemmini_extended3_config_ld(4, 1.0f, 0, 0);\n")
@instr(_gemm_config_ld_acc_i32_vector)
def config_ld_acc_i32_vector(
    stride_set : bool
):
    ConfigLoadAcc.stride_set = stride_set

_gemm_ld_acc_i32_vec   = ("gemmini_extended3_config_ld(4, 1.0f, 0, 0);\n"+
                          "gemmini_extended_mvin( ((uint64_t) &{src_data}), "+
                               "((uint32_t) {dst}.data), 16, 1 );")
@instr(_gemm_ld_acc_i32_vec)
def ld_acc_i32_vector(
    src   : [i32][16] @ DRAM,
    dst   : [i32][16] @ GEMM_ACCUM,
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    pass
    for i in par(0, 16):
        dst[i] = src[i]

_do_gemm_ld_acc_i32_vec   = ("gemmini_extended_mvin( ((uint64_t) &{src_data}), ((uint32_t) {dst}.data), 16, 1 );")
@instr(_do_gemm_ld_acc_i32_vec)
def do_ld_acc_i32_vector(
    src   : [i32][16] @ DRAM,
    dst   : [i32][16] @ GEMM_ACCUM,
):
    assert stride(dst, 0) == 1
    assert stride(src, 0) == 1

    for i in par(0, 16):
        dst[i] = src[i]

ld_acc_i32_vector_v2 = ld_acc_i32_vector.rename("ld_acc_i32_vector_v2")
ld_acc_i32_vector_v2 = ld_acc_i32_vector_v2.configwrite_after('pass', ConfigLoadAcc, 'stride_set', 'True')
ld_acc_i32_vector_v2 = ld_acc_i32_vector_v2.replace(do_ld_acc_i32_vector, 'for i in _:_')
ld_acc_i32_vector_v2 = ld_acc_i32_vector_v2.replace(config_ld_acc_i32_vector, 'ConfigLoadAcc.stride_set = _')
ld_acc_i32_vector_v2 = ld_acc_i32_vector_v2.delete_pass().make_instr(_gemm_ld_acc_i32_vec)
ld_acc_i32_vector    = ld_acc_i32_vector.delete_pass().make_instr(_gemm_ld_acc_i32_vec)



_gemm_st_i8   = ("gemmini_extended_config_st({dst}.strides[0]*1, 0, 1.0f);\n"+
                 "gemmini_extended_mvout( "+
                      "((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_i8)
def st_i8(
    n     : size,
    m     : size,
    src   : [i8][n, 16] @ GEMM_SCRATCH,
    dst   : [i8][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i, j]


@proc
def clamp(src : f32, dst : i8):
    l : f32
    h : f32
    l = -128.0
    h = 127.0
    dst = select(h, src, h, src)
    dst = select(src, l, l, dst)


def new_config_st():
    @config
    class ConfigStore:
        scale : f32
        dst_stride : stride
        act : bool

    return ConfigStore

ConfigStore = new_config_st()

_gemm_st_acc_i8   = ("gemmini_extended_config_st({dst}.strides[0]*1, {act}, {scale}[0]);\n"+
                     "gemmini_extended_mvout( ((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_acc_i8)
def st_acc_i8(
    n     : size,
    m     : size,
    scale : f32,
    act   : bool,
    src   : [i32][n, 16] @ GEMM_ACCUM,
    dst   : [i8][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            src_tmp : i32
            src_tmp = src[i,j]
            tmp : f32
            acc_scale(src_tmp, tmp, scale)
            tmp2 : i8
            clamp(tmp, tmp2)
            if act == True:
                tmp2 = relu(tmp2)
            dst[i, j] = tmp2

_gemm_config_st_acc_i8   = ("gemmini_extended_config_st({dst_stride}, {act}, {scale}[0]);\n")
@instr(_gemm_config_st_acc_i8)
def config_st_acc_i8(
    scale : f32,
    dst_stride : stride,
    act : bool
):
    ConfigStore.scale = scale
    ConfigStore.dst_stride = dst_stride
    ConfigStore.act = act

_gemm_st_acc_i8   = ("gemmini_extended_mvout( ((uint64_t) {dst}.data), (uint32_t) {src}.data, {m}, {n} );")
@instr(_gemm_st_acc_i8)
def do_st_acc_i8(
    n     : size,
    m     : size,
    src   : [i32][n, 16] @ GEMM_ACCUM,
    dst   : [i8][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            src_tmp : i32
            src_tmp = src[i,j]
            tmp : f32
            acc_scale(src_tmp, tmp, ConfigStore.scale)
            tmp2 : i8
            clamp(tmp, tmp2)
            if ConfigStore.act == True:
                tmp2 = relu(tmp2)
            dst[i, j] = tmp2


st_acc_i8_v2 = st_acc_i8.rename("st_acc_i8_v2")
st_acc_i8_v2 = st_acc_i8_v2.bind_config('scale', ConfigStore, 'scale')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('tmp : _', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('src_tmp = _', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('src_tmp : _', 'ConfigStore.scale = _')
st_acc_i8_v2 = st_acc_i8_v2.fission_after('ConfigStore.scale = _', n_lifts=2)
st_acc_i8_v2 = st_acc_i8_v2.configwrite_after('ConfigStore.scale = _', ConfigStore, 'dst_stride', 'stride(dst, 0)')
st_acc_i8_v2 = st_acc_i8_v2.bind_config('act', ConfigStore, 'act')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('clamp(_)', 'ConfigStore.act = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('tmp2 : _', 'ConfigStore.act = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('acc_scale(_)', 'ConfigStore.act = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('tmp : _', 'ConfigStore.act = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('src_tmp = _', 'ConfigStore.act = _')
st_acc_i8_v2 = st_acc_i8_v2.reorder_stmts('src_tmp : _', 'ConfigStore.act = _')
st_acc_i8_v2 = st_acc_i8_v2.fission_after('ConfigStore.act = _', n_lifts=2)
st_acc_i8_v2 = st_acc_i8_v2.replace(do_st_acc_i8, 'for i in _:_')
st_acc_i8_v2 = st_acc_i8_v2.replace(config_st_acc_i8, 'ConfigStore.scale = scale')


_gemm_st_acc_i32 = ("gemmini_extended_config_st({dst}.strides[0]*4, 0, 1.0f);\n"+
                    "gemmini_extended_mvout( ((uint64_t) {dst}.data), "+
                    "((uint32_t) {src}.data | 0x20000000), {m}, {n} );")
@instr(_gemm_st_acc_i32)
def st_acc_i32(
    n     : size,
    m     : size,
    src   : [i32][n, 16] @ GEMM_ACCUM,
    dst   : [i32][n, m]  @ DRAM
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 1) == 1
    assert stride(src, 0) == 16
    assert stride(src, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i, j] = src[i, j]





_gemm_config_zero   = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n")
@instr(_gemm_config_zero)
def config_zero():
    ConfigLoad.src_stride = 0

_gemm_do_zero = ("gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                       "{m}, {n} );")
@instr(_gemm_do_zero)
def do_zero_i8(
    n   : size,
    m   : size,
    dst : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = 0.0

_gemm_zero = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"+
                 "gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                       "{m}, {n} );")
@instr(_gemm_zero)
def zero_i8(
    n   : size,
    m   : size,
    dst : [i8][n, 16] @ GEMM_SCRATCH,
):
    assert n <= 16
    assert m <= 16
    assert stride(dst, 0) == 16
    assert stride(dst, 1) == 1

    pass

    for i in par(0, n):
        for j in par(0, m):
            dst[i,j] = 0.0

zero_i8_v2 = zero_i8.rename("zero_i8_v2")
zero_i8_v2 = zero_i8_v2.configwrite_after('pass', ConfigLoad, 'src_stride', '0')
zero_i8_v2 = zero_i8_v2.replace(do_zero_i8, 'for i in _:_')
zero_i8_v2 = zero_i8_v2.replace(config_zero, 'ConfigLoad.src_stride = _')

do_zero_acc_i32 = (do_zero_i8.rename('do_zero_acc_i32')
                             .set_precision('dst', 'i32')
                             .set_memory('dst', GEMM_ACCUM)
                             .make_instr(_gemm_do_zero))
zero_acc_i32 = (zero_i8.rename('zero_acc_i32')
                          .set_precision('dst', 'i32')
                          .set_memory('dst', GEMM_ACCUM)
                          .make_instr(_gemm_zero))
zero_acc_i32_v2 = zero_acc_i32.rename("zero_acc_i32_v2")
zero_acc_i32_v2 = zero_acc_i32_v2.configwrite_after('pass', ConfigLoad, 'src_stride', '0')
zero_acc_i32_v2 = zero_acc_i32_v2.replace(do_zero_acc_i32, 'for i in _:_')
zero_acc_i32_v2 = zero_acc_i32_v2.replace(config_zero, 'ConfigLoad.src_stride = _')

zero_i8 = zero_i8.delete_pass().make_instr(_gemm_zero)
zero_i8_v2 = zero_i8_v2.delete_pass().make_instr(_gemm_zero)
zero_acc_i32    = zero_acc_i32.delete_pass().make_instr(_gemm_zero)
zero_acc_i32_v2 = zero_acc_i32_v2.delete_pass().make_instr(_gemm_zero)


_gemm_zero_vec = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"+
                 "gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                         "16, 1 );")
@instr(_gemm_zero_vec)
def zero_i8_vector(
    dst : [i8][16] @ GEMM_SCRATCH,
):
    assert stride(dst, 0) == 16

    for i in par(0, 16):
        dst[i] = 0.0







def new_config_matmul():
    @config
    class ConfigMatmul:
        done : bool

    return ConfigMatmul

ConfigMatmul = new_config_matmul()

_gemm_config_matmul = "gemmini_extended_config_ex(WS, 0, 0, 0, 1, 0, 0);\n"
@instr(_gemm_config_matmul)
def config_matmul():
    ConfigMatmul.done = True

_gemm_matmul = (
       "gemmini_extended_preload("+
            "(uint32_t)({B}.data), (uint32_t)({C}.data), "+
            "{M}, {K}, "+
            "{M}, {N}"+
       ");\n"+
       "gemmini_extended_compute_preloaded("+
            "(uint32_t)({A}.data), ~((uint32_t)0), "+
            "{K}, {N}, "+
            "16, 16"+
       ");")

@instr(_gemm_config_matmul + _gemm_matmul)
def matmul_i8(
    N : size,
    M : size,
    K : size,
    A : [i8][N, 16] @ GEMM_SCRATCH,
    B : [i8][K, 16] @ GEMM_SCRATCH,
    C : [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    pass
    for i in par(0,N):
        for j in par(0,M):
            C[i,j] = 0.0
            for k in par(0,K):
                a : i32
                b : i32

                a = A[i,k]
                b = B[k,j]

                C[i, j] += a * b

@instr(_gemm_matmul)
def do_matmul_i8(
    N : size,
    M : size,
    K : size,
    A : [i8][N, 16] @ GEMM_SCRATCH,
    B : [i8][K, 16] @ GEMM_SCRATCH,
    C : [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in par(0,N):
        for j in par(0,M):
            C[i,j] = 0.0
            for k in par(0,K):
                a : i32
                b : i32

                a = A[i,k]
                b = B[k,j]

                C[i, j] += a * b

matmul_i8_v2 = matmul_i8.rename("matmul_i8_v2")
matmul_i8_v2 = matmul_i8_v2.configwrite_after('pass', ConfigMatmul, 'done', 'True')
matmul_i8_v2 = matmul_i8_v2.replace(do_matmul_i8, 'for i in _:_')
matmul_i8_v2 = matmul_i8_v2.replace(config_matmul, 'ConfigMatmul.done = True')
matmul_i8_v2 = matmul_i8_v2.delete_pass().make_instr(_gemm_matmul)
matmul_i8    = matmul_i8.delete_pass().make_instr(_gemm_config_matmul + _gemm_matmul)




_gemm_matmul_acc = (
       "gemmini_extended_preload("+
            "(uint32_t)({B}.data), (uint32_t)({C}.data) | 0x40000000, "+
            "{M}, {K}, "+
            "{M}, {N}"+
       ");\n"+
       "gemmini_extended_compute_preloaded("+
            "(uint32_t)({A}.data), ~((uint32_t)0), "+
            "{K}, {N}, "+
            "16, 16"+
       ");")

@instr(_gemm_matmul_acc)
def matmul_acc_i8(
    N : size,
    M : size,
    K : size,
    A : [i8][N, 16] @ GEMM_SCRATCH,
    B : [i8][K, 16] @ GEMM_SCRATCH,
    C : [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    pass
    for i in par(0,N):
        for j in par(0,M):
            for k in par(0,K):
                a : i32
                b : i32

                a = A[i,k]
                b = B[k,j]

                C[i, j] += a * b

@instr(_gemm_matmul_acc)
def do_matmul_acc_i8(
    N : size,
    M : size,
    K : size,
    A : [i8][N, 16] @ GEMM_SCRATCH,
    B : [i8][K, 16] @ GEMM_SCRATCH,
    C : [i32][N, 16] @ GEMM_ACCUM,
):
    assert N <= 16
    assert M <= 16
    assert K <= 16

    for i in par(0,N):
        for j in par(0,M):
            for k in par(0,K):
                a : i32
                b : i32

                a = A[i,k]
                b = B[k,j]

                C[i, j] += a * b
matmul_acc_i8_v2 = matmul_acc_i8.rename("matmul_acc_i8_v2")
matmul_acc_i8_v2 = matmul_acc_i8_v2.configwrite_after('pass', ConfigMatmul, 'done', 'True')
matmul_acc_i8_v2 = matmul_acc_i8_v2.replace(do_matmul_acc_i8, 'for i in _:_')
matmul_acc_i8_v2 = matmul_acc_i8_v2.replace(config_matmul, 'ConfigMatmul.done = True')
matmul_acc_i8_v2 = matmul_acc_i8_v2.delete_pass().make_instr(_gemm_matmul_acc)
matmul_acc_i8    = matmul_acc_i8.delete_pass().make_instr(_gemm_config_matmul + _gemm_matmul_acc)

# --------------------------------------------------------------------------- #
#
# --------------------------------------------------------------------------- #
