from abc import ABC, abstractmethod
from enum import Enum
from collections import ChainMap
from itertools import chain
from typing import Mapping, Any
from asdl_adt import ADT, validators

from ..core.extern import Extern
from ..core.configs import Config
from ..core.memory import Memory
from ..core.prelude import Sym, SrcInfo, extclass
from ..core.LoopIR import (
    LoopIR,
    Alpha_Rename,
    SubstArgs,
    LoopIR_Do,
    Operator,
    T,
    Identifier,
    LoopIR_Rewrite,
    get_readconfigs,
    get_writeconfigs,
    get_readconfigs_expr,
)
from .internal_analysis import *


# --------------------------------------------------------------------------- #
# DataflowIR definition
# --------------------------------------------------------------------------- #


def validateAbsEnv(obj):
    if not isinstance(obj, dict):
        raise ValidationError(D, type(obj))
    for key in obj:
        if not isinstance(key, Sym):
            raise ValidationError(Sym, key)
    return obj


DataflowIR = ADT(
    """
module DataflowIR {
    proc = ( name    name,
             fnarg*  args,
             expr*   preds,
             block   body,
             srcinfo srcinfo )

    fnarg  = ( sym     name,
               expr*   hi,
               type    type,
               srcinfo srcinfo )

    block = ( stmt* stmts, absenv ctxt )

    stmt = Assign( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | Reduce( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | LoopStart( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | LoopExit( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | IfJoin( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | Alloc( sym name, expr* hi, type type )
         | Pass()
         | If( expr cond, block body, block orelse )
         | For( sym iter, expr lo, expr hi, block body )
         attributes( srcinfo srcinfo )

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | Extern( extern f, expr* args )
         | StrideExpr( sym name, int dim )
         attributes( type type, srcinfo srcinfo )

    type = Num()
         | F16()
         | F32()
         | F64()
         | INT8()
         | UINT8()
         | UINT16()
         | INT32()
         | Bool()
         | Int()
         | Index()
         | Size()
         | Stride()
}""",
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "extern": Extern,
        "binop": validators.instance_of(Operator, convert=True),
        "absenv": validateAbsEnv,
        "srcinfo": SrcInfo,
    },
    memoize={
        "Num",
        "F16",
        "F32",
        "F64",
        "INT8",
        "UINT8",
        "UINT16",
        "INT32",
        "Bool",
        "Int",
        "Index",
        "Size",
        "Stride",
    },
)


# --------------------------------------------------------------------------- #
# Helper functions for Dataflow IR
# --------------------------------------------------------------------------- #


def mk_const(val):
    return DataflowIR.Const(val, DataflowIR.Int(), null_srcinfo())


# There should be no function call or windowing at this point
class LoopIR_to_DataflowIR:
    def __init__(self, proc, stmts):
        self.loopir_proc = proc
        self.stmts = []
        self.env = ChainMap()
        for s in stmts:
            self.stmts.append([s])

        # Initialize configurations in this proc
        configs = list(
            set(
                get_writeconfigs(proc.body)
                + get_readconfigs(proc.body)
                + get_readconfigs_expr(proc.preds)
            )
        )
        for c in configs:
            orig_sym = c[0]._INTERNAL_sym(c[1])
            new_sym = orig_sym.copy()
            typ = self.map_t(c[0].lookup_type(c[1]))
            # (the most recent sym, (iter dims, dim dims), basetype)
            self.env[orig_sym] = (new_sym, ([], []), typ)

        self.dataflow_proc = self.map_proc(self.loopir_proc)

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def result(self):
        res = []
        for l in self.stmts:
            res.extend(l[1:])
        return self.dataflow_proc, res

    def init_block(self, body):
        return DataflowIR.block(body, dict())

    def map_proc(self, p):
        df_args = self._map_list(self.map_fnarg, p.args)
        df_preds = self.map_exprs(p.preds)
        df_body = self.map_stmts(p.body)
        block = self.init_block(df_body)

        return DataflowIR.proc(p.name, df_args, df_preds, block, p.srcinfo)

    def tensor_to_his(self, tensor):
        # | Tensor( expr* hi, bool is_window, type type )
        assert isinstance(tensor, T.Tensor)
        # assert tensor.is_window == False
        assert not tensor.type.is_tensor_or_window()

        return [self.map_e(hi) for hi in tensor.hi], self.map_t(tensor.type)

    def map_fnarg(self, a):
        if a.type.is_indexable() or a.type.is_bool():
            return DataflowIR.fnarg(a.name, [], self.map_t(a.type), a.srcinfo)
        else:
            # data values so renaming is necessary
            name = a.name.copy()
            his = []
            dsyms = []
            if isinstance(a.type, T.Tensor):
                his, _ = self.tensor_to_his(a.type)
                dsyms = [Sym("d" + str(d)) for d in range(len(a.type.hi))]
            typ = self.map_t(a.type.basetype())
            self.env[a.name] = (name, ([], dsyms), typ)
            return DataflowIR.fnarg(name, his, typ, a.srcinfo)

    def map_stmts(self, stmts):
        return self._map_list(self.map_s, stmts)

    def map_exprs(self, exprs):
        return self._map_list(self.map_e, exprs)

    def sym_to_read(self, syms, typ):
        return [DataflowIR.Read(s, [], typ, null_srcinfo()) for s in syms]

    def to_read(self, name, dims, typ):
        return DataflowIR.Read(
            name, self.sym_to_read(dims[0] + dims[1], typ), typ, null_srcinfo()
        )

    def map_s(self, s):
        if isinstance(s, LoopIR.Call):
            assert False, "Call statement should have been got rid of at this point"

        elif isinstance(s, LoopIR.WindowStmt):
            assert False, "Window statement should have been got rid of at this point"

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            new_name = s.name.copy()
            old_name = self.env[s.name][0]
            iters, dims = self.env[s.name][1]
            old_typ = self.env[s.name][2]
            conds = []
            for d, e in zip(reversed(dims), s.idx):
                typ = self.map_t(e.type)
                conds.append(
                    DataflowIR.BinOp(
                        "==",
                        self.map_e(e),
                        DataflowIR.Read(d, [], typ, null_srcinfo()),
                        DataflowIR.Bool(),
                        null_srcinfo(),
                    )
                )

            cond = (
                conds[0]
                if len(conds) > 0
                else DataflowIR.Const(True, DataflowIR.Bool(), null_srcinfo())
            )
            for c in conds[1:]:
                cond = DataflowIR.BinOp(
                    "and", c, cond, DataflowIR.Bool(), null_srcinfo()
                )

            body = self.map_e(s.rhs)
            orelse = self.to_read(old_name, (iters, dims), old_typ)
            self.env[s.name] = (new_name, (iters, dims), old_typ)

            if isinstance(s, LoopIR.Assign):
                return [
                    DataflowIR.Assign(
                        new_name, iters, dims, cond, body, orelse, s.srcinfo
                    )
                ]
            else:
                return [
                    DataflowIR.Reduce(
                        new_name, iters, dims, cond, body, orelse, s.srcinfo
                    )
                ]

        elif isinstance(s, LoopIR.WriteConfig):
            ir_config = s.config._INTERNAL_sym(s.field)
            assert ir_config in self.env
            df_config = ir_config.copy()
            df_rhs = self.map_e(s.rhs)
            iters, _ = self.env[ir_config][1]
            self.env[ir_config] = (df_config, (iters, []), df_rhs.type)

            return [
                DataflowIR.Assign(
                    df_config,
                    iters,
                    [],
                    DataflowIR.Const(True, DataflowIR.Bool(), null_srcinfo()),
                    df_rhs,
                    df_rhs,
                    s.srcinfo,
                )
            ]

        elif isinstance(s, LoopIR.If):
            cond = self.map_e(s.cond)

            self.push()
            body = self.map_stmts(s.body)
            bvals = {}
            for key, vals in self.env.items():
                bvals[key] = vals
            self.pop()

            self.push()
            orelse = self.map_stmts(s.orelse)
            ovals = {}
            for key, vals in self.env.items():
                ovals[key] = vals
            self.pop()

            # Create a merge node
            post_if = []
            for key, vals in self.env.items():
                bval = bvals[key]
                oval = ovals[key]

                post_name = vals[0].copy()
                bbody = self.to_read(bval[0], bval[1], bval[2])
                obody = self.to_read(oval[0], oval[1], oval[2])
                stmt = DataflowIR.IfJoin(
                    post_name,
                    vals[1][0],
                    vals[1][1],
                    cond,
                    bbody,
                    obody,
                    null_srcinfo(),
                )
                post_if.append(stmt)

                # update env
                self.env[key] = (post_name, vals[1], vals[2])

            return [
                DataflowIR.If(
                    cond, self.init_block(body), self.init_block(orelse), s.srcinfo
                )
            ] + post_if

        elif isinstance(s, LoopIR.For):
            # We first need the merge node before the body, but the problem is that we don't know what the last node is until doing the body
            # We can be safe and merge every values in the environment

            # Update self.env and prev
            prev = {}
            pre_sym = {}
            for key, vals in self.env.items():
                self.env[key] = (
                    vals[0].copy(),
                    ([s.iter] + vals[1][0], vals[1][1]),
                    vals[2],
                )
                pre_sym[key] = self.env[key]
                prev[key] = vals

            # Body
            body = self.map_stmts(s.body)

            # Construct merge nodes for both pre and post loop merges
            pre_loop = []
            post_loop = []
            iter_read = DataflowIR.Read(s.iter, [], DataflowIR.Index(), null_srcinfo())
            for key, lbefore in prev.items():
                ltop = pre_sym[key]
                llast = self.env[key]
                other_idxs = self.sym_to_read((llast[1][0] + llast[1][1])[1:], llast[2])

                # Pre loop
                pre_cond = DataflowIR.BinOp(
                    "==", iter_read, mk_const(0), DataflowIR.Bool(), null_srcinfo()
                )
                pre_body = self.to_read(lbefore[0], lbefore[1], lbefore[2])
                last_idx = [
                    DataflowIR.BinOp(
                        "-", iter_read, mk_const(1), DataflowIR.Index(), null_srcinfo()
                    )
                ]
                pre_orelse = DataflowIR.Read(
                    llast[0],
                    last_idx + other_idxs,
                    llast[2],
                    null_srcinfo(),
                )
                pre_stmt = DataflowIR.LoopStart(
                    ltop[0],
                    ltop[1][0],
                    ltop[1][1],
                    pre_cond,
                    pre_body,
                    pre_orelse,
                    null_srcinfo(),
                )
                pre_loop.append(pre_stmt)

                # Post loop
                post_name = llast[0].copy()
                post_cond = DataflowIR.BinOp(
                    ">",
                    self.map_e(s.hi),
                    self.map_e(s.lo),
                    DataflowIR.Bool(),
                    null_srcinfo(),
                )
                post_idxs = [
                    DataflowIR.BinOp(
                        "-",
                        self.map_e(s.hi),
                        mk_const(1),
                        DataflowIR.Index(),
                        null_srcinfo(),
                    )
                ]
                post_body = DataflowIR.Read(
                    llast[0],
                    post_idxs + other_idxs,
                    llast[2],
                    null_srcinfo(),
                )
                post_orelse = self.to_read(lbefore[0], lbefore[1], lbefore[2])
                post_stmt = DataflowIR.LoopExit(
                    post_name,
                    llast[1][0][1:],
                    llast[1][1],
                    post_cond,
                    post_body,
                    post_orelse,
                    null_srcinfo(),
                )
                post_loop.append(post_stmt)

                # update env
                self.env[key] = (post_name, (llast[1][0][1:], llast[1][1]), llast[2])

            return [
                DataflowIR.For(
                    s.iter,
                    self.map_e(s.lo),
                    self.map_e(s.hi),
                    self.init_block(pre_loop + body),
                    s.srcinfo,
                )
            ] + post_loop

        elif isinstance(s, LoopIR.Alloc):
            if isinstance(s.type, T.Tensor):
                assert s.type.is_window == False
                assert s.type.type.is_real_scalar()

                name = s.name.copy()
                his, _ = self.tensor_to_his(s.type)
                dsyms = [Sym("d" + str(d)) for d in range(len(s.type.hi))]
                typ = self.map_t(s.type.basetype())
                self.env[s.name] = (name, ([], dsyms), typ)

                return DataflowIR.Alloc(name, his, typ, s.srcinfo)
            else:
                assert s.type.is_real_scalar()
                name = s.name.copy()
                typ = self.map_t(s.type)
                self.env[s.name] = (name, ([], []), typ)

                return DataflowIR.Alloc(name, [], typ, s.srcinfo)

        elif isinstance(s, LoopIR.Pass):
            return [DataflowIR.Pass(s.srcinfo)]

        else:
            assert False, f"bad case {type(s)}"

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            if e.type.is_indexable() or e.type.is_bool():
                return DataflowIR.Read(e.name, [], self.map_t(e.type), e.srcinfo)
            else:
                assert e.name in self.env
                ee = self.env[e.name]
                assert len(ee[1][1]) == len(e.idx)  # dim idx
                df_idx = self.sym_to_read(ee[1][0], ee[2]) + self.map_exprs(e.idx)
                return DataflowIR.Read(ee[0], df_idx, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.ReadConfig):
            df_config = e.config._INTERNAL_sym(e.field)
            assert df_config in self.env
            ee = self.env[df_config]
            assert len(ee[1][1]) == 0
            idx = self.sym_to_read(ee[1][0], ee[2])
            return DataflowIR.Read(ee[0], idx, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.BinOp):
            if e.op == "/":
                raise NotImplementedError("division is not supported yet")

            df_lhs = self.map_e(e.lhs)
            df_rhs = self.map_e(e.rhs)
            return DataflowIR.BinOp(e.op, df_lhs, df_rhs, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.Extern):
            raise NotImplementedError("Extern is not supported yet")

            df_args = self.map_exprs(e.args)
            return DataflowIR.Extern(e.f, df_args, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.USub):
            df_arg = self.map_e(e.arg)
            return DataflowIR.USub(df_arg, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.Const):
            return DataflowIR.Const(e.val, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr):
            raise NotImplementedError("stride expression is not supported yet")

            assert e.name in self.env
            return DataflowIR.StrideExpr(
                self.env[e.name][0], e.dim, self.map_t(e.type), e.srcinfo
            )

        elif isinstance(e, LoopIR.WindowExpr):
            assert (
                False
            ), "Shouldn't be here! WindowExpr should be handled in the Call or WindowStmt cases"

        else:
            assert False, f"LoopIR type {type(e)} does not exist. WTF?"

    def _map_list(self, fn, nodes):
        res = []
        for n in nodes:
            d_ir = fn(n)
            for s in self.stmts:
                if n == s[0]:
                    s.append(d_ir)

            if isinstance(d_ir, list):
                res.extend(d_ir)
            else:
                res.append(d_ir)

        return res

    def map_t(self, t):
        if isinstance(t, T.Tensor):
            assert False
        elif isinstance(t, T.Window):
            assert False
        elif isinstance(t, T.F32):
            return DataflowIR.F32()
        elif isinstance(t, T.Num):
            return DataflowIR.Num()
        elif isinstance(t, T.F16):
            return DataflowIR.F16()
        elif isinstance(t, T.F64):
            return DataflowIR.F64()
        elif isinstance(t, T.INT8):
            return DataflowIR.INT8()
        elif isinstance(t, T.UINT8):
            return DataflowIR.UINT8()
        elif isinstance(t, T.UINT16):
            return DataflowIR.UINT16()
        elif isinstance(t, T.INT32):
            return DataflowIR.INT32()
        elif isinstance(t, T.Bool):
            return DataflowIR.Bool()
        elif isinstance(t, T.Int):
            return DataflowIR.Int()
        elif isinstance(t, T.Index):
            return DataflowIR.Index()
        elif isinstance(t, T.Size):
            return DataflowIR.Size()
        elif isinstance(t, T.Stride):
            return DataflowIR.Stride()
        else:
            assert False, f"no such type {type(t)}"


class LoopIR_Replace(LoopIR_Rewrite):
    def __init__(self, proc: LoopIR.proc, old: LoopIR.stmt, new: list):
        self.old = old
        self.new = new
        self.proc = super().apply_proc(proc)

    def result(self):
        return self.proc

    def map_s(self, s):
        if s == self.old:
            return self.new
        else:
            return super().map_s(s)


class FindStmt(LoopIR_Do):
    def __init__(self, proc, fun):
        self.stmt = None
        self.fun = fun
        super().__init__(proc)

    def result(self):
        return self.stmt

    def do_s(self, s):
        if self.stmt != None:
            return  # short circit

        if self.fun(s):
            self.stmt = s

        super().do_s(s)


def inline_calls(proc):
    while True:
        call_s = FindStmt(proc, lambda s: isinstance(s, LoopIR.Call)).result()
        if call_s == None:
            break

        win_binds = []

        def map_bind(nm, a):
            if isinstance(a, LoopIR.WindowExpr):
                stmt = LoopIR.WindowStmt(nm, a, a.srcinfo)
                win_binds.append(stmt)
                return LoopIR.Read(nm, [], a.type, a.srcinfo)
            return a

        call_bind = {
            xd.name: map_bind(xd.name, a) for xd, a in zip(call_s.f.args, call_s.args)
        }
        body = SubstArgs(call_s.f.body, call_bind).result()
        new_body = Alpha_Rename(win_binds + body).result()

        proc = LoopIR_Replace(proc, call_s, new_body).result()

    return proc


def inline_windows(proc):
    while True:
        window_s = FindStmt(proc, lambda s: isinstance(s, LoopIR.WindowStmt)).result()
        if window_s == None:
            break

        proc = DoInlineWindow(proc, window_s).result()

    return proc


# This is basically a duplication of DoInlineWindow in LoopIR_scheduling.py... but we need to change the interface to Cursor if want to merge them...
class DoInlineWindow(LoopIR_Rewrite):
    def __init__(self, proc, window):
        self.win_stmt = window
        assert isinstance(self.win_stmt, LoopIR.WindowStmt)
        self.proc = super().apply_proc(proc)

    def result(self):
        return self.proc

    def calc_idx(self, idxs):
        assert len(
            [w for w in self.win_stmt.rhs.idx if isinstance(w, LoopIR.Interval)]
        ) == len(idxs)

        new_idxs = []
        win_idx = self.win_stmt.rhs.idx
        idxs = idxs.copy()  # make function non-destructive to input
        assert len(idxs) == sum([isinstance(w, LoopIR.Interval) for w in win_idx])

        def add(x, y):
            return LoopIR.BinOp("+", x, y, T.index, x.srcinfo)

        if len(idxs) > 0 and isinstance(idxs[0], LoopIR.w_access):

            def map_w(w):
                if isinstance(w, LoopIR.Point):
                    return w
                # i is from the windowing expression we're substituting into
                i = idxs.pop(0)
                if isinstance(i, LoopIR.Point):
                    return LoopIR.Point(add(i.pt, w.lo), i.srcinfo)
                else:
                    return LoopIR.Interval(add(i.lo, w.lo), add(i.hi, w.lo), i.srcinfo)

        else:

            def map_w(w):
                return w.pt if isinstance(w, LoopIR.Point) else add(idxs.pop(0), w.lo)

        return [map_w(w) for w in win_idx]

    # used to offset the stride in order to account for
    # dimensions hidden due to window-point accesses
    def calc_dim(self, dim):
        assert dim < len(
            [w for w in self.win_stmt.rhs.idx if isinstance(w, LoopIR.Interval)]
        )

        # Because our goal here is to offset `dim` in the original
        # call argument to the point indexing to the windowing expression,
        # new_dim should essencially be:
        # `dim` + "number of LoopIR.Points in the windowing expression before the `dim` number of LoopIR.Interval"
        new_dim = 0
        for w in self.win_stmt.rhs.idx:
            if isinstance(w, LoopIR.Interval):
                dim -= 1
            if dim == -1:
                return new_dim
            new_dim += 1

    def map_s(self, s):
        # remove the windowing statement
        if s is self.win_stmt:
            return []

        # substitute the indexing at assignment and reduction statements
        if (
            isinstance(s, (LoopIR.Assign, LoopIR.Reduce))
            and self.win_stmt.name == s.name
        ):
            idxs = self.calc_idx(s.idx)
            return [type(s)(self.win_stmt.rhs.name, s.type, idxs, s.rhs, s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        # etyp    = type(e)
        win_name = self.win_stmt.name
        buf_name = self.win_stmt.rhs.name
        win_idx = self.win_stmt.rhs.idx

        if isinstance(e, LoopIR.WindowExpr) and win_name == e.name:
            new_idxs = self.calc_idx(e.idx)

            # repair window type..
            old_typ = self.win_stmt.rhs.type
            new_type = LoopIR.WindowType(
                old_typ.src_type, old_typ.as_tensor, buf_name, new_idxs
            )

            return LoopIR.WindowExpr(
                self.win_stmt.rhs.name, new_idxs, new_type, e.srcinfo
            )

        elif isinstance(e, LoopIR.Read) and win_name == e.name:
            new_idxs = self.calc_idx(e.idx)
            return LoopIR.Read(buf_name, new_idxs, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr) and win_name == e.name:
            dim = self.calc_dim(e.dim)
            return LoopIR.StrideExpr(buf_name, dim, e.type, e.srcinfo)

        return super().map_e(e)


def dataflow_analysis(proc: LoopIR.proc, loopir_stmts: list) -> DataflowIR.proc:
    proc = inline_calls(proc)
    proc = inline_windows(proc)

    # step 1 - convert LoopIR to DataflowIR with empty contexts (i.e. AbsEnvs)
    datair, stmts = LoopIR_to_DataflowIR(proc, loopir_stmts).result()

    # step 2 - run abstract interpretation algorithm to populate contexts with abs values
    ScalarPropagation(datair)

    return datair, stmts


# --------------------------------------------------------------------------- #
# Abstract Domain definitions
# --------------------------------------------------------------------------- #

ValueDomain = ADT(
    """
module ValueDomain {
    vabs  = ValConst(object val) -- real number
          | Top()
          | Bot()
  }
"""
)
V = ValueDomain


def validateValueDom(obj):
    if not isinstance(obj, V.vabs):
        raise ValidationError(A, type(obj))
    return obj


ArrayDomain = ADT(
    """
module ArrayDomain {
    abs = (sym* iterators, node tree)

    node   = Leaf(val v)
           | AffineSplit(aexpr ae,
               node  ltz, -- ae < 0 case
               node  eqz, -- ae == 0
               node  gtz  -- ae > 0
             )
           | ModSplit(aexpr ae, int m
               node neqz, -- pred % m != 0
               node  eqz, -- pred % m == 0 case
             )

    val   = SubVal(vabs av)
          | ArrayTmp(sym name, aexpr* idx)
          | ArrayVar(sym name, aexpr* idx)

    aexpr = Const(int val)
          | Var(sym name)
          | Add(aexpr lhs, aexpr rhs)
          | Mult(int coeff, aexpr ae)  
}
""",
    ext_types={
        "vabs": validateValueDom,
        "sym": Sym,
    },
    memoize={},
)
D = ArrayDomain


from . import dataflow_pprint


# --------------------------------------------------------------------------- #
# Lifting the abstract domain tree to SMT formula (AExpr)
# --------------------------------------------------------------------------- #


def mk_aexpr(op, pred):
    return A.BinOp(
        op, pred, A.Const(0, T.Int(), null_srcinfo()), T.Bool(), null_srcinfo()
    )


# Corresponds to \mathcal{L}^\#_{vabs} in the paper
def lift_to_smt_vabs(aname: A.Var, v: V.vabs):
    if isinstance(v, V.ValConst):
        c = A.Const(v.val, T.R, null_srcinfo())
        return AEq(aname, c)
    elif isinstance(v, V.Top):
        return A.Const(True, T.bool, null_srcinfo())
    elif isinstance(v, V.Bot):
        return A.Const(False, T.bool, null_srcinfo())


# Dictionary of (arrayvar, SMT variable). Necessary to have a constant symbol.
cvt_dict = dict()

# Corresponds to \mathcal{L}^\#_{\mathit{val}} in the paper
def lift_to_smt_val(aname: A.Var, e: D.val):
    if isinstance(e, D.SubVal):
        return lift_to_smt_vabs(aname, e.av)
    elif isinstance(e, D.ArrayVar):
        key = tuple([e.name] + e.idx)
        if key in cvt_dict:
            vname = cvt_dict[key]
        else:
            istr = [
                str(i)
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
                .replace("+", "p")
                .replace("*", "m")
                for i in e.idx
            ]
            sym = Sym("array_" + str(e.name) + "_".join(istr))
            vname = A.Var(sym, T.R, null_srcinfo())
            cvt_dict[key] = vname

        return AEq(aname, vname)
    else:
        assert (
            False
        ), f"{type(e)} should be substituted out by reaching lift_to_smt_val. Something is wrong about the analysis implementation."


# Lift D.aexpr to the AExpr
def lift_to_smt_a(e: D.aexpr) -> A:
    assert isinstance(e, D.aexpr)
    typ = T.Int()
    if isinstance(e, D.Var):
        return A.Var(e.name, typ, null_srcinfo())
    elif isinstance(e, D.Const):
        return A.Const(e.val, typ, null_srcinfo())
    elif isinstance(e, D.Add):
        return A.BinOp(
            "+", lift_to_smt_a(e.lhs), lift_to_smt_a(e.rhs), typ, null_srcinfo()
        )
    elif isinstance(e, D.Mult):
        return A.BinOp(
            "*",
            A.Const(e.coeff, typ, null_srcinfo()),
            lift_to_smt_a(e.ae),
            typ,
            null_srcinfo(),
        )
    else:
        assert False, "bad case!"


# Corresponds to \mathcal{L}^\#_{node} in the paper
def lift_to_smt_n(name: Sym, src: D.node):
    aname = A.Var(name, T.R, null_srcinfo())

    def map_tree(tree):
        if isinstance(tree, D.Leaf):
            return lift_to_smt_val(aname, tree.v)
        elif isinstance(tree, D.AffineSplit):
            # | Select( expr cond, expr tcase, expr fcase )
            pred = lift_to_smt_a(tree.ae)
            ltz_eq = mk_aexpr("<", pred)
            eqz_eq = mk_aexpr("==", pred)
            return A.Select(
                ltz_eq,
                map_tree(tree.ltz),
                A.Select(
                    eqz_eq,
                    map_tree(tree.eqz),
                    map_tree(tree.gtz),
                    T.bool,
                    null_srcinfo(),
                ),
                T.bool,
                null_srcinfo(),
            )
        elif isinstance(tree, D.ModSplit):
            # | ModSplit(aexpr ae, int m
            #   node neqz, -- pred % m != 0
            #   node  eqz, -- pred % m == 0 case

            pred = A.BinOp(
                "%",
                lift_to_smt_a(tree.ae),
                A.Var(tree.m, T.int, null_srcinfo()),
                DataflowIR.Int(),
                null_srcinfo(),
            )
            mod_eq = mk_aexpr("==", pred)
            return A.Select(
                mod_eq, map_tree(tree.eqz), map_tree(tree.neqz), T.bool, null_srcinfo()
            )

    return map_tree(src)


# --------------------------------------------------------------------------- #
# Transfer and lifting functions from Dataflow IR to abstract domain
# --------------------------------------------------------------------------- #

# Lifting function for dataflow ir expression to D.aexpr
# Corresponds to $A^\# : Expr \to aexpr$ in the paper
def lift_to_abs_a(e: DataflowIR.expr) -> D.aexpr:

    if isinstance(e, DataflowIR.Read):
        assert len(e.idx) == 0
        return D.Var(e.name)

    elif isinstance(e, DataflowIR.Const):
        assert isinstance(e.val, (bool, int))
        return D.Const(e.val)

    elif isinstance(e, DataflowIR.USub):
        if arg := lift_to_abs_a(e.arg):
            return D.Mult(-1, arg)

    elif isinstance(e, DataflowIR.BinOp):
        lhs = lift_to_abs_a(e.lhs)
        rhs = lift_to_abs_a(e.rhs)

        if e.op == "+":
            return D.Add(lhs, rhs)
        elif e.op == "*":
            if isinstance(lhs, D.Const):
                return D.Mult(lhs.val, rhs)
            elif isinstance(rhs, D.Const):
                return D.Mult(rhs.val, lhs)
            else:
                assert False, "shouldn't be here"
        elif e.op == "-":
            return D.Add(lhs, D.Mult(-1, rhs))
        else:
            # TODO: Support division at some point
            assert False, f"got unsupported binop {e.op}."

    assert False, f"shouldn't be here. got {e}"


# Corresponds to \delta in the paper draft
def delta(cond: DataflowIR.expr, body: D.node, orelse: D.node) -> D.node:
    assert isinstance(cond, DataflowIR.expr)
    assert isinstance(body, D.Leaf)
    assert isinstance(orelse, D.Leaf)

    # If the condition is always True or False, just return the leaf as a tree
    tree = D.Leaf(D.SubVal(V.Top()))
    if isinstance(cond, DataflowIR.Const) and (cond.val == True):
        tree = body
    elif isinstance(cond, DataflowIR.Const) and (cond.val == False):
        tree = orelse
    else:
        # operators = {+, -, *, /, mod, and, or, ==, <, <=, >, >=}
        assert isinstance(cond, DataflowIR.BinOp)

        # Handle logical operations
        if cond.op == "and":
            return delta(cond.lhs, delta(cond.rhs, body, orelse), orelse)
        elif cond.op == "or":
            return delta(cond.lhs, body, delta(cond.rhs, body, orelse))

        # FIXME: Modular inequalities are generally not supported.
        is_lhs_mod = isinstance(cond.lhs, DataflowIR.BinOp) and cond.lhs.op == "%"
        is_rhs_mod = isinstance(cond.rhs, DataflowIR.BinOp) and cond.rhs.op == "%"
        if is_lhs_mod or is_rhs_mod:
            if cond.op == "==":
                e1 = cond.lhs.lhs if is_lhs_mod else cond.rhs.lhs
                c = cond.lhs.rhs if is_lhs_mod else cond.rhs.rhs
                e2 = cond.rhs if is_lhs_mod else cond.lhs
                assert isinstance(c, DataflowIR.Const)
                tree = D.ModSplit(
                    lift_to_abs_a(
                        DataflowIR.BinOp("-", e1, e2, DataflowIR.Int(), null_srcinfo())
                    ),
                    c,
                    orelse,
                    body,
                )
            else:
                assert False, "modular inequalites are not supported yet!"

        # This is A^\#\qc{e_1 - e_2}
        eq = lift_to_abs_a(
            DataflowIR.BinOp("-", cond.lhs, cond.rhs, DataflowIR.Int(), null_srcinfo())
        )
        if cond.op == "==":
            tree = D.AffineSplit(eq, orelse, body, orelse)
        elif cond.op == "<":
            tree = D.AffineSplit(eq, body, orelse, orelse)
        elif cond.op == "<=":
            tree = D.AffineSplit(eq, body, body, orelse)
        elif cond.op == ">":
            tree = D.AffineSplit(eq, orelse, orelse, body)
        elif cond.op == ">=":
            tree = D.AffineSplit(eq, orelse, body, body)
        elif cond.op == "<":
            tree = D.AffineSplit(eq, body, orelse, orelse)
        elif cond.op == "%":
            assert False, "mod should be handled in the cases above, shouldn't be here!"
        elif cond.op == "/":
            assert False, "div is unsupported, shouldn't be here!"
        else:
            assert False, "WTF?"

    return tree


# --------------------------------------------------------------------------- #
# Simplify
# --------------------------------------------------------------------------- #


def abs_simplify(src: D.abs) -> D.abs:
    assert isinstance(src, D.abs)
    slv = SMTSolver(verbose=False)

    for itr in src.iterators:
        slv.assume(
            A.BinOp(
                ">=",
                A.Var(itr, T.Int(), null_srcinfo()),
                A.Const(0, T.Int(), null_srcinfo()),
                T.Bool(),
                null_srcinfo(),
            )
        )

    def map_tree(tree: D.node):
        if isinstance(tree, D.Leaf):
            return tree

        elif isinstance(tree, D.AffineSplit):
            # we can collapse the tree when all values are the same
            if (
                isinstance(tree.ltz, D.Leaf)
                and tree.ltz == tree.eqz == tree.gtz
                and isinstance(tree.ltz.v, D.SubVal)
            ):
                return tree.ltz

            pred = lift_to_smt_a(tree.ae)
            # check if anything is simplifiable
            ltz_eq = mk_aexpr("<", pred)
            eqz_eq = mk_aexpr("==", pred)
            gtz_eq = mk_aexpr(">", pred)
            if slv.verify(ltz_eq):
                return map_tree(tree.ltz)
            elif slv.verify(gtz_eq):
                return map_tree(tree.gtz)
            elif slv.verify(eqz_eq):
                return map_tree(tree.eqz)

            # ltz
            slv.push()
            slv.assume(ltz_eq)
            ltz = map_tree(tree.ltz)
            slv.pop()

            # eqz
            slv.push()
            slv.assume(eqz_eq)
            eqz = map_tree(tree.eqz)
            slv.pop()

            # gtz
            slv.push()
            slv.assume(gtz_eq)
            gtz = map_tree(tree.gtz)
            slv.pop()

            return D.AffineSplit(tree.ae, ltz, eqz, gtz)

        elif isinstance(tree, D.ModSplit):
            # pred: ae % m
            pred = A.BinOp(
                "%",
                lift_to_smt_a(tree.ae),
                A.Var(tree.m, T.int, null_srcinfo()),
                DataflowIR.Int(),
                null_srcinfo(),
            )
            # eqz_eq: ae % m == 0
            eqz_eq = mk_aexpr("==", pred)
            neqz_eq = A.Not(eqz_eq, T.Bool(), null_srcinfo())
            if slv.verify(eqz_eq):
                return map_tree(tree.eqz)
            elif slv.verify(neqz_eq):
                return map_tree(tree.neqz)

            # eqz
            slv.push()
            slv.assume(eqz_eq)
            eqz = map_tree(tree.eqz)
            slv.pop()

            # neqz
            slv.push()
            slv.assume(neqz_eq)
            neqz = map_tree(tree.neqz)
            slv.pop()

            return D.ModSplit(tree.ae, tree.m, neqz, eqz)
        else:
            assert False, "bad case"

    return D.abs(src.iterators, map_tree(src.tree))


# --------------------------------------------------------------------------- #
# Substitution related operations
# --------------------------------------------------------------------------- #


class Abs_Rewrite:
    def map_abs(self, a):
        iters = self.map_iters(a.iterators)
        tree = self.map_node(a.tree)
        return (iters, tree)

    def map_iters(self, itrs):
        return [self.map_iter(i) for i in itrs]

    def map_iter(self, i):
        return i

    def map_node(self, node):
        if isinstance(node, D.Leaf):
            return D.Leaf(self.map_val(node.v))
        elif isinstance(node, D.AffineSplit):
            return D.AffineSplit(
                self.map_aexpr(node.ae),
                self.map_node(node.ltz),
                self.map_node(node.eqz),
                self.map_node(node.gtz),
            )
        elif isinstance(node, D.ModSplit):
            return D.ModSplit(
                self.map_aexpr(node.ae),
                node.m,
                self.map_node(node.neqz),
                self.map_node(node.eqz),
            )
        else:
            assert False

    def map_val(self, val):
        if isinstance(val, D.SubVal):
            # TODO: May want to recurse into VABS?
            return val
        elif isinstance(val, D.ArrayTmp):
            return D.ArrayTmp(val.name, [self.map_aexpr(i) for i in val.idx])
        elif isinstance(val, D.ArrayVar):
            return D.ArrayVar(val.name, [self.map_aexpr(i) for i in val.idx])
        else:
            assert False

    def map_aexpr(self, aexpr):
        if isinstance(aexpr, D.Const):
            return aexpr
        elif isinstance(aexpr, D.Var):
            return aexpr
        elif isinstance(aexpr, D.Add):
            return D.Add(self.map_aexpr(aexpr.lhs), self.map_aexpr(aexpr.rhs))
        elif isinstance(aexpr, D.Mult):
            return D.Mult(aexpr.coeff, self.map_aexpr(aexpr.ae))


# [i0 -> e0, ..., ik -> ek]tree
class ASubs(Abs_Rewrite):
    def __init__(self, node: D.node, env: dict):
        self.dict = env
        self.node = super().map_node(node)

    def result(self):
        return self.node

    def map_aexpr(self, a):
        if isinstance(a, D.Var):
            if a.name in self.dict:
                return self.dict[a.name]

        return super().map_aexpr(a)


# vsubs : val -> \Sigma^\# -> node in the paper
def vsubs(val: D.val, env: dict) -> D.node:
    assert isinstance(val, D.val)
    assert isinstance(env, dict)

    if isinstance(val, D.ArrayVar):
        return D.Leaf(val)
    elif isinstance(val, D.SubVal):
        return D.Leaf(val)
    elif isinstance(val, D.ArrayTmp):
        if not val.name in env:
            return D.Leaf(val)

        exprs = val.idx
        itrs = env[val.name].iterators
        itr_map = dict()

        for i1, i2 in zip(itrs, exprs):
            itr_map[i1] = i2

        return ASubs(env[val.name].tree, itr_map).result()

    else:
        assert False, "bad case"


# nsubs : node -> \Sigma^\# -> node in the paper
def nsubs(node: D.node, env: dict) -> D.node:
    assert isinstance(node, D.node)
    assert isinstance(env, dict)

    if isinstance(node, D.AffineSplit):
        return D.AffineSplit(
            node.ae, nsubs(node.ltz, env), nsubs(node.eqz, env), nsubs(node.gtz, env)
        )
    elif isinstance(node, D.ModSplit):
        return D.ModSplit(node.ae, node.m, nsubs(node.neqz, env), nsubs(node.eqz, env))
    else:
        return vsubs(node.v, env)


# --------------------------------------------------------------------------- #
# Widening related operations
# --------------------------------------------------------------------------- #


def get_equations(tree: D.node, eqs: set) -> D.node:
    assert isinstance(tree, D.node)
    assert isinstance(eqs, set)
    if isinstance(tree, D.AffineSplit):
        eqs.add(tree.ae)
        get_equations(tree.ltz, eqs)
        get_equations(tree.eqz, eqs)
        get_equations(tree.gtz, eqs)
    elif isinstance(tree, D.ModSplit):
        get_equations(tree.neqz, eqs)
        get_equations(tree.eqz, eqs)
        eqs.add(tree.ae)
    return eqs


def find_intersections(dims: list, eqs: set) -> set:
    # Recursive function to generate combinations
    def get_combinations(elements, combination_length):
        if combination_length == 0:
            return [[]]
        if len(elements) < combination_length:
            return []
        else:
            # Include the first element
            with_first = get_combinations(elements[1:], combination_length - 1)
            with_first = [[elements[0]] + combo for combo in with_first]
            # Exclude the first element
            without_first = get_combinations(elements[1:], combination_length)
            # Combine both
            return with_first + without_first

    const_rep = Sym("C")

    def cvt_eq(eq: D.aexpr) -> dict:
        if isinstance(eq, D.Const):
            return {const_rep: eq.val}
        elif isinstance(eq, D.Var):
            return {eq.name: 1}
        elif isinstance(eq, D.Add):
            lhs = cvt_eq(eq.lhs)
            rhs = cvt_eq(eq.rhs)
            common = {key: (lhs[key] + rhs[key]) for key in lhs if key in rhs}
            return lhs | rhs | common
        elif isinstance(eq, D.Mult):
            arg = cvt_eq(eq.ae)
            return {key: arg[key] * eq.coeff for key in arg}

    def cvt_back(dic: dict) -> D.aexpr:
        varr = []
        for key, val in dic.items():
            if val == 0:
                continue
            if key == const_rep:
                varr.append(D.Const(val))
                continue

            var = D.Var(key)
            if val != 1:
                var = D.Mult(val, var)
            varr.append(var)

        if len(varr) > 1:
            ae = D.Add(varr[0], varr[1])
            for var in varr[2:]:
                ae = D.Add(ae, var)
        else:
            ae = varr[0]

        return ae

    target = dims[0]
    intersections = []
    cvted_eqs = []
    for eq in list(eqs):
        new_eq = cvt_eq(eq)
        if target not in new_eq:
            intersections.append(new_eq)
        else:
            cvted_eqs.append(cvt_eq(eq))

    # Generate all possible combinations of the specified length
    all_combinations = get_combinations(cvted_eqs, 2)

    new_dims = dims[1:] + [const_rep]
    for feq, seq in all_combinations:
        cur = dict()
        a_n = feq[target]
        b_n = seq[target]
        for d in new_dims:
            a_i = feq[d] if d in feq else 0
            b_i = seq[d] if d in seq else 0
            cur[d] = a_n * b_i - b_n * a_i
        intersections.append(cur)

    return [cvt_back(eq) for eq in intersections]


def merge_tree(name: Sym, tree: D.node, intersections: list) -> D.node:
    slv = SMTSolver(verbose=False)

    def map_tree(tree: D.node, flag=True):
        if isinstance(tree, D.Leaf):
            if isinstance(tree.v, D.ArrayConst):
                # TODO: probably should check index too
                if tree.v.name == name:
                    for p in intersections:
                        pred = lift_to_smt_a(p)
                        q = mk_aexpr("==", pred)
                        if slv.satisfy(q):
                            tree = map_tree(
                                D.AffineSplit(p, tree, tree, tree), flag=False
                            )
                            break

            return tree

        elif isinstance(tree, D.AffineSplit):
            pred = lift_to_smt_a(tree.ae)
            # check if anything is simplifiable
            ltz_eq = mk_aexpr("<", pred)
            eqz_eq = mk_aexpr("==", pred)
            gtz_eq = mk_aexpr(">", pred)

            # ltz
            slv.push()
            slv.assume(ltz_eq)
            ltz = map_tree(tree.ltz)
            slv.pop()

            # eqz
            eqz = tree.eqz
            if flag:
                slv.push()
                slv.assume(eqz_eq)
                eqz = map_tree(tree.eqz)
                slv.pop()

            # gtz
            slv.push()
            slv.assume(gtz_eq)
            gtz = map_tree(tree.gtz)
            slv.pop()

            return D.AffineSplit(tree.ae, ltz, eqz, gtz)

        elif isinstance(tree, D.ModSplit):
            # FIXME: this is incorrect, not using m
            pred = lift_to_smt_a(tree.ae)
            eqz_eq = mk_aexpr("==", pred)
            neqz_eq = A.Not(eqz_eq, T.Bool(), null_srcinfo())

            # eqz
            slv.push()
            slv.assume(eqz_eq)
            eqz = map_tree(tree.eqz)
            slv.pop()

            # neqz
            slv.push()
            slv.assume(neqz_eq)
            neqz = map_tree(tree.neqz)
            slv.pop()

            return D.ModSplit(tree.ae, tree.m, neqz, eqz)
        else:
            assert False, "bad case"

    return map_tree(tree)


def partition(name: Sym, src: D.abs) -> D.abs:
    assert isinstance(name, Sym)
    assert isinstance(src, D.abs)
    eqs = get_equations(src.tree, set())
    itss = find_intersections(src.iterators, eqs)
    tree = merge_tree(name, src.tree, itss)

    # FIXME: is this correct?
    # TODO: make it more generic? Like mark bottom when predicates are never satisfiable
    if isinstance(tree, D.AffineSplit):
        tree = D.AffineSplit(tree.ae, D.Leaf(D.SubVal(V.Bot())), tree.eqz, tree.gtz)

    return abs_simplify(D.abs(src.iterators, tree))


def widening(name: Sym, src: D.abs) -> D.abs:
    assert isinstance(src, D.abs)

    def map_tree(tree: D.node):
        if isinstance(tree, D.AffineSplit):
            ltz = tree.ltz
            eqz = tree.eqz
            gtz = tree.gtz

            if isinstance(tree.ltz, D.Leaf) and isinstance(tree.ltz.v, D.ArrayConst):
                if tree.ltz.v.name == name:
                    ltz = eqz

            if isinstance(tree.gtz, D.Leaf) and isinstance(tree.gtz.v, D.ArrayConst):
                if tree.gtz.v.name == name:
                    gtz = eqz

            return D.AffineSplit(
                tree.ae,
                map_tree(ltz),
                map_tree(eqz),
                map_tree(gtz),
            )
        elif isinstance(tree, D.ModSplit):
            return D.ModSplit(tree.ae, tree.m, map_tree(tree.neqz), map_tree(tree.eqz))
        else:
            return tree

    return D.abs(src.iterators, map_tree(src.tree))


# --------------------------------------------------------------------------- #
# Abstract Interpretation on DataflowIR
# --------------------------------------------------------------------------- #


class AbstractInterpretation(ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc
        # set of "original" arrays of proc arguments. We need this to distinguish ArrayVars from ArrayTmp.
        self.avars = set()
        self.fix_proc(self.proc)

    def fix_proc(self, proc: DataflowIR.proc):
        assert isinstance(proc, DataflowIR.proc)

        for a in proc.args:
            self.avars.add(a.name)

        # TODO: FIXME: Do we need to use precondition assertions?

        self.fix_block(proc.body)

    def fix_block(self, body: DataflowIR.block):
        for i in range(len(body.stmts)):
            self.fix_stmt(body.stmts[i], body.ctxt)

        # simplify
        for key, val in body.ctxt.items():
            val = self.abs_simplify(val)
            val = self.abs_simplify(val)
            body.ctxt[key] = val

    def fix_stmt(self, stmt: DataflowIR.stmt, env):
        if isinstance(
            stmt,
            (
                DataflowIR.Assign,
                DataflowIR.Reduce,
                DataflowIR.LoopStart,
                DataflowIR.LoopExit,
                DataflowIR.IfJoin,
            ),
        ):
            new_dict = {k: v for k, v in env.items() if k != stmt.lhs}
            a = self.abs_assign(stmt)
            env[stmt.lhs] = D.abs(a.iterators, nsubs(a.tree, new_dict))

        elif isinstance(stmt, DataflowIR.Pass):
            pass  # pass pass, lol

        elif isinstance(stmt, DataflowIR.Alloc):
            pass  # no need to do anything

        elif isinstance(stmt, DataflowIR.If):
            pre_body = stmt.body.ctxt
            pre_orelse = stmt.orelse.ctxt
            for nm, val in env.items():
                pre_body[nm] = val
                pre_orelse[nm] = val

            self.fix_block(stmt.body)
            self.fix_block(stmt.orelse)

            for nm, post_val in stmt.body.ctxt.items():
                env[nm] = post_val

            for nm, post_val in stmt.orelse.ctxt.items():
                env[nm] = post_val

        elif isinstance(stmt, DataflowIR.For):
            raise NotImplementedError("not worrying about fixpoing for now")

            # traverse the body in the reverse order
            for i in range(len(stmt.body.stmts) - 1, -1, -1):
                self.fix_stmt(stmt.body.stmts[i], stmt.body.ctxt)

            # Get the phi node
            assert isinstance(stmt.body.stmts[0], DataflowIR.LoopStart)
            phi = stmt.body.stmts[0].lhs

            # We need to substitute the hell out of stmt.body.ctxt[phi] at this point. Substitute until there is no
            new_dict = {k: v for k, v in stmt.body.ctxt.items() if k != phi}
            # FIXME: TODO: Change to call to nsubs
            before_before = substitute_all(stmt.body.ctxt[phi], new_dict)
            stmt.body.ctxt[phi] = before_before

            # "before" and "after" running the fixpoint
            before = self.abs_partition(phi, stmt.body.ctxt[phi])
            stmt.body.ctxt[phi] = before
            after = self.abs_simplify(self.abs_join(before, stmt.body.ctxt))
            stmt.body.ctxt[phi] = after

            # widening by value propagation
            fixed = self.abs_widening(phi, after)

            # substitute other statements that depend on phi no
            stmt.body.ctxt[phi] = self.abs_simplify(self.abs_simplify(fixed))

            for i in range(1, len(stmt.body.stmts)):
                if isinstance(
                    stmt.body.stmts[i],
                    (
                        DataflowIR.Assign,
                        DataflowIR.Reduce,
                        DataflowIR.LoopStart,
                        DataflowIR.LoopExit,
                        DataflowIR.IfJoin,
                    ),
                ):

                    nm = stmt.body.stmts[i].lhs
                    val = self.abs_join(stmt.body.ctxt[nm], stmt.body.ctxt)
                    val = self.abs_simplify(val)
                    val = self.abs_simplify(val)
                    stmt.body.ctxt[nm] = val

            for nm, post_val in stmt.body.ctxt.items():
                env[nm] = post_val

        else:
            assert False, f"bad case: {type(stmt)}"

    # Corresponds to $E^\# : \Expr \to val$ in the paper
    def fix_expr(self, e: DataflowIR.expr) -> D.val:
        if isinstance(e, DataflowIR.Read):
            return self.abs_read(e)
        elif isinstance(e, DataflowIR.Const):
            return self.abs_const(e)
        elif isinstance(e, DataflowIR.USub):
            return self.abs_usub(e)
        elif isinstance(e, DataflowIR.BinOp):
            return self.abs_binop(e)
        elif isinstance(e, DataflowIR.Extern):
            return self.abs_extern(e)
        elif isinstance(e, DataflowIR.StrideExpr):
            return self.abs_stride_expr(e)
        else:
            assert False, f"bad case {type(expr)}"

    @abstractmethod
    def abs_partition(self, name, adom):
        # Define Partitioning
        pass

    @abstractmethod
    def abs_widening(self, name, adom):
        # Define Widening
        pass

    @abstractmethod
    def abs_assign(self, stmt):
        # Define Assign
        pass

    @abstractmethod
    def abs_simplify(self, adom):
        # Define Simplify
        pass

    @abstractmethod
    def abs_join(self, adom, env):
        # Define Fixpoint on Abstract domain
        pass

    @abstractmethod
    def abs_read(self, e):
        # Define Read
        pass

    @abstractmethod
    def abs_stride_expr(self, e):
        # Define abstraction of a specific stride expression
        pass

    @abstractmethod
    def abs_const(self, e):
        # Define abstraction of a specific constant value
        pass

    @abstractmethod
    def abs_binop(self, e):
        # Implement transfer function abstraction for binary operations
        pass

    @abstractmethod
    def abs_usub(self, e):
        # Implement transfer function abstraction for unary subtraction
        pass

    @abstractmethod
    def abs_extern(self, e):
        # Implement transfer function abstraction for built-ins
        pass


class ScalarPropagation(AbstractInterpretation):
    def abs_partition(self, name: Sym, src: D.abs) -> D.abs:
        assert isinstance(name, Sym)
        assert isinstance(src, D.abs)
        return partition(name, src)

    def abs_widening(self, name: Sym, src: D.abs) -> D.abs:
        assert isinstance(name, Sym)
        assert isinstance(src, D.abs)
        return widening(name, src)

    def abs_assign(self, stmt):
        body = stmt.body
        # if reducing, then expand to x = x + rhs now we can handle both cases uniformly
        if isinstance(stmt, DataflowIR.Reduce):
            body = DataflowIR.BinOp("+", body, stmt.orelse, body.type, stmt.srcinfo)

        return D.abs(
            stmt.iters + stmt.dims,
            delta(
                stmt.cond,
                D.Leaf(self.fix_expr(body)),
                D.Leaf(self.fix_expr(stmt.orelse)),
            ),
        )

    def abs_join(self, src: D.abs, env: dict) -> D.abs:
        raise NotImplementedError("not worrying about fixpoing for now")
        assert isinstance(src, D.abs)
        assert isinstance(env, dict)

        # FIXME: TODO: Change to call to nsubs
        return substitute_all(src, env)

    def abs_simplify(self, src: D.abs) -> D.abs:
        return abs_simplify(src)

    def abs_read(self, e):
        idxs = [lift_to_abs_a(i) for i in e.idx]
        if e.name in self.avars:
            return D.ArrayVar(e.name, idxs)
        else:
            return D.ArrayTmp(e.name, idxs)

    def abs_const(self, e) -> D.val:
        return D.SubVal(V.ValConst(e.val))

    def abs_stride_expr(self, e):
        raise NotImplementedError("stride is not supported yet. Shouldn't be here!")
        return D.SubVal(V.Top())

    def abs_binop(self, e) -> D:
        return D.SubVal(V.Top())

    def abs_usub(self, e) -> D:
        return D.SubVal(V.Top())

    def abs_extern(self, e):
        raise NotImplementedError("extern is not supported yet. Shouldn't be here!")
        return D.SubVal(V.Top())
