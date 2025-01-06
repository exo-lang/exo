from abc import ABC, abstractmethod
from enum import Enum
from collections import ChainMap
from itertools import chain
from typing import Mapping, Any
from asdl_adt import ADT, validators

from .builtins import BuiltIn
from .configs import Config
from .memory import Memory
from .prelude import Sym, SrcInfo, extclass
from .LoopIR import (
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
        raise ValidationError(AbsEnv, type(obj))
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

    block = ( stmt* stmts, absenv* ctxts ) -- len(stmts) + 1 == len(ctxts)

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
         | BuiltIn( builtin f, expr* args )
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
        "builtin": BuiltIn,
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

from . import dataflow_pprint


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
        dic = []
        for i in range(len(body) + 1):
            dic.append(dict())
        return DataflowIR.block(body, dic)

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
            raise NotImplementedError(f"bad case {type(s)}")

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
            df_lhs = self.map_e(e.lhs)
            df_rhs = self.map_e(e.rhs)
            return DataflowIR.BinOp(e.op, df_lhs, df_rhs, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.BuiltIn):
            df_args = self.map_exprs(e.args)
            return DataflowIR.BuiltIn(e.f, df_args, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.USub):
            df_arg = self.map_e(e.arg)
            return DataflowIR.USub(df_arg, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.Const):
            return DataflowIR.Const(e.val, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr):
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
# Helper functions
# --------------------------------------------------------------------------- #


# class DataflowIR_Do:
#    def __init__(self, proc, *args, **kwargs):
#        self.proc = proc
#        self.do_proc(self.proc)
#
#    def do_proc(self, p):
#        [self.do_fnarg(a) for a in p.args]
#        [self.do_e(p) for p in p.preds]
#        self.do_block(p.body)
#
#    def do_fnarg(self, a):
#        [self.do_e(e) for e in a.hi]
#
#    def do_block(self, block):
#        self.do_stmts(block.stmts)
#        self.do_ctxts(block.ctxts)
#
#    def do_ctxts(self, ctxts):
#        for c in ctxts:
#            self.do_c(c)
#
#    def do_c(self, c):
#        pass
#
#    def do_stmts(self, stmts):
#        for s in stmts:
#            self.do_s(s)
#
#    def do_s(self, s):
#        if isinstance(s, (DataflowIR.Assign, DataflowIR.Reduce)):
#            self.do_e(e.cond)
#            self.do_e(e.body)
#            self.do_e(e.orelse)
#        elif isinstance(s, DataflowIR.If):
#            self.do_e(s.cond)
#            self.do_block(s.body)
#            self.do_block(s.orelse)
#        elif isinstance(s, DataflowIR.For):
#            self.do_e(s.lo)
#            self.do_e(s.hi)
#            self.do_block(s.body)
#        elif isinstance(s, DataflowIR.Alloc):
#            [self.do_e(e) for e in s.hi]
#        else:
#            assert isinstance(DataflowIR.Pass)
#
#    def do_e(self, e):
#        if isinstance(e, DataflowIR.Read):
#            [self.do_e(idx) for idx in e.idx]
#        elif isinstance(e, DataflowIR.BinOp):
#            self.do_e(e.lhs)
#            self.do_e(e.rhs)
#        elif isinstance(e, DataflowIR.BuiltIn):
#            [self.do_e(a) for a in e.args]
#        elif isinstance(e, DataflowIR.USub):
#            self.do_e(e.arg)
#        else:
#            assert isinstance(e, (DataflowIR.Const, DataflowIR.StrideExpr))
#
#
# class GetReadConfigs(DataflowIR_Do):
#    def __init__(self):
#        self.readconfigs = []
#
#    def do_e(self, e):
#        if isinstance(e, DataflowIR.ReadConfig):
#            self.readconfigs.append((e.config_field, e.type))
#        super().do_e(e)
#
#
# def _get_readconfigs(stmts):
#    gr = GetReadConfigs()
#    for stmt in stmts:
#        gr.do_s(stmt)
#    return gr.readconfigs
#
#
# class GetWriteConfigs(DataflowIR_Do):
#    def __init__(self):
#        self.writeconfigs = []
#
#    def do_s(self, s):
#        if isinstance(s, DataflowIR.WriteConfig):
#            # FIXME!!! Propagate a proper type after adding type to writeconfig
#            self.writeconfigs.append((s.config_field, T.int))
#
#        super().do_s(s)
#
#    # early exit
#    def do_e(self, e):
#        return
#
#
# def _get_writeconfigs(stmts):
#    gw = GetWriteConfigs()
#    gw.do_stmts(stmts)
#    return gw.writeconfigs


class GetValues:
    def __init__(self, proc, stmts):
        self.proc = proc
        self.stmts = stmts
        self.before = None
        self.after = None
        self.do_block(self.proc.body)

    def result(self):
        return self.before, self.after

    def do_block(self, block):
        for i, s in enumerate(block.stmts):
            if s == self.stmts[0]:
                self.before = block.ctxts[i]
            if s == self.stmts[-1]:
                self.after = block.ctxts[i + 1]
            self.do_s(s)

    def do_s(self, s):
        if isinstance(s, DataflowIR.If):
            return self.do_block(s.body) or self.do_block(s.orelse)
        elif isinstance(s, DataflowIR.For):
            return self.do_block(s.body)
        elif isinstance(s, DataflowIR.Call):
            return self.do_block(s.f.body)
        else:
            pass


# --------------------------------------------------------------------------- #
# Abstract Domain definition
# --------------------------------------------------------------------------- #

AbstractDomains = ADT(
    """
module AbstractDomains {
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

    val   = ValConst(object val) -- real number
          | Top()
          | Bot()
          | ArrayConst(sym name, aexpr* idx)

    aexpr = Const(int val)
          | Var(sym name)
          | Add(aexpr lhs, aexpr rhs)
          | Minus(aexpr lhs, aexpr rhs)
          | Mult(int coeff, aexpr ae)  
}
""",
    ext_types={
        "type": DataflowIR.type,
        "sym": Sym,
    },
    memoize={},
)
D = AbstractDomains


@extclass(AbstractDomains.abs)
def __str__(self):
    iter_strs = ". ".join(["\\" + str(i) for i in self.iterators])
    return iter_strs + str(self.tree)


@extclass(AbstractDomains.node)
def __str__(self, indent=""):
    if isinstance(self, D.Leaf):
        return f"{indent}- {str(self.v)}"
    elif isinstance(self, D.AffineSplit):
        nstr = str(self.ae)
        newdent = indent + " " * (len(nstr) + 1)
        indent = indent + "- "
        return f"""
{indent}{nstr}
{self.ltz.__str__(newdent)}
{self.eqz.__str__(newdent)}
{self.gtz.__str__(newdent)}
"""
    elif isinstance(self, D.ModSplit):
        nstr = str(self.ae) + f"%{self.m}"
        newdent = indent + " " * (len(nstr) + 1)
        indent = indent + "- "
        return f"""
{indent}{nstr}
{self.neqz.__str__(newdent)}
{self.eqz.__str__(newdent)}
"""
    else:
        assert False, "bad case"


@extclass(AbstractDomains.val)
def __str__(self):
    if isinstance(self, D.Top):
        return "⊤"
    elif isinstance(self, D.Bot):
        return "⊥"
    elif isinstance(self, D.ValConst):
        return str(self.val)
    elif isinstance(self, D.ArrayConst):
        idxs = (
            "[" + ",".join([str(i) for i in self.idx]) + "]"
            if len(self.idx) > 0
            else ""
        )
        return f"{repr(self.name)}{idxs}"
    assert False, "bad case"


@extclass(AbstractDomains.aexpr)
def __str__(self):
    if isinstance(self, D.Const):
        return str(self.val)
    elif isinstance(self, D.Var):
        return repr(self.name)
    elif isinstance(self, D.Add):
        return f"({str(self.lhs)}+{str(self.rhs)})"
    elif isinstance(self, D.Minus):
        return f"({str(self.lhs)}-{str(self.rhs)})"
    elif isinstance(self, D.Mult):
        return f"{str(self.coeff)}*{str(self.ae)}"
    assert False, "bad case"


del __str__


def _lift_aexpr(e):
    typ = T.Int()
    if isinstance(e, D.Var):
        return A.Var(e.name, typ, null_srcinfo())
    elif isinstance(e, D.Const):
        return A.Const(e.val, typ, null_srcinfo())
    elif isinstance(e, D.Add):
        return A.BinOp("+", _lift_aexpr(e.lhs), _lift_aexpr(e.rhs), typ, null_srcinfo())
    elif isinstance(e, D.Minus):
        return A.BinOp("-", _lift_aexpr(e.lhs), _lift_aexpr(e.rhs), typ, null_srcinfo())
    elif isinstance(e, D.Mult):
        return A.BinOp(
            "*",
            A.Const(e.coeff, typ, null_srcinfo()),
            _lift_aexpr(e.ae),
            typ,
            null_srcinfo(),
        )
    else:
        assert False, "bad case!"


# src[var -> term] with simplification
def substitute(var: D.ArrayConst, term: D.node, src: D.abs):
    def a_comp(x: D.aexpr, y: D.aexpr):
        if type(x) != type(y):
            return False
        if isinstance(x, D.Const):
            return x.val == y.val
        elif isinstance(x, D.Var):
            return x.name == y.name
        elif isinstance(x, (D.Add, D.Minus)):
            return a_comp(x.lhs, y.lhs) and a_comp(x.rhs, y.rhs)
        elif isinstance(x, D.Mult):
            return x.coeff == y.coeff and a_comp(x.ae, y.ae)
        else:
            assert False, "bad case"

    def arr_comp(x: D.ArrayConst, y: D.ArrayConst):
        if x.name == y.name and len(x.idx) == len(y.idx):
            if all([a_comp(xi, yi) for xi, yi in zip(x.idx, y.idx)]):
                return True
        return False

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

    def mk_aexpr(op, pred):
        return A.BinOp(
            op, pred, A.Const(0, T.Int(), null_srcinfo()), T.Bool(), null_srcinfo()
        )

    def map_tree(tree: D.node):
        if isinstance(tree, D.Leaf):
            if isinstance(tree.v, D.ArrayConst) and arr_comp(tree.v, var):
                if isinstance(term, D.AffineSplit):
                    pred = _lift_aexpr(term.ae)
                    ltz_eq = mk_aexpr("<", pred)
                    gtz_eq = mk_aexpr(">", pred)
                    eqz_eq = mk_aexpr("==", pred)
                    if slv.verify(ltz_eq):
                        return term.ltz
                    elif slv.verify(gtz_eq):
                        return term.gtz
                    elif slv.verify(eqz_eq):
                        return term.eqz
                    return term
                elif isinstance(term, D.ModSplit):
                    pred = _lift_aexpr(term.ae)
                    eqz_eq = mk_aexpr("==", pred)
                    if slv.verify(eqz_eq):
                        return term.eqz
                    elif slv.verify(A.Not(eqz_eq, T.Bool(), null_srcinfo())):
                        return term.neqz
                    return term
                else:
                    return term
            return tree

        elif isinstance(tree, D.AffineSplit):
            pred = _lift_aexpr(tree.ae)
            # ltz
            slv.push()
            ltz_eq = mk_aexpr("<", pred)
            slv.assume(ltz_eq)
            ltz = map_tree(tree.ltz)
            slv.pop()

            # eqz
            slv.push()
            eqz_eq = mk_aexpr("==", pred)
            slv.assume(eqz_eq)
            eqz = map_tree(tree.eqz)
            slv.pop()

            # gtz
            slv.push()
            gtz_eq = mk_aexpr(">", pred)
            slv.assume(gtz_eq)
            gtz = map_tree(tree.gtz)
            slv.pop()

            return D.AffineSplit(tree.ae, ltz, eqz, gtz)

        elif isinstance(tree, D.ModSplit):
            pred = _lift_aexpr(tree.ae)
            # eqz
            slv.push()
            eqz_eq = mk_aexpr("==", pred)
            slv.assume(eqz_eq)
            eqz = map_tree(tree.eqz)
            slv.pop()

            # neqz
            slv.push()
            slv.assume(A.Not(eqz_eq, T.Bool(), null_srcinfo()))
            neqz = map_tree(tree.neqz)
            slv.pop()

            return D.ModSplit(tree.ae, tree.m, neqz, eqz)
        else:
            assert False, "bad case"

    return D.abs(src.iterators, map_tree(src.tree))


# e.g. i -> i-1 in src
def sub_aexpr(var: D.aexpr, term: D.aexpr, src: D.abs):
    def a_comp(x: D.aexpr, y: D.aexpr):
        if type(x) != type(y):
            return False
        if isinstance(x, D.Const):
            return x.val == y.val
        elif isinstance(x, D.Var):
            return x.name == y.name
        elif isinstance(x, (D.Add, D.Minus)):
            return a_comp(x.lhs, y.lhs) and a_comp(x.rhs, y.rhs)
        elif isinstance(x, D.Mult):
            return x.coeff == y.coeff and a_comp(x.ae, y.ae)
        else:
            assert False, "bad case"

    def map_aexpr(ae: D.aexpr):
        if a_comp(ae, var):
            return term
        elif isinstance(ae, (D.Add, D.Minus)):
            return type(ae)(map_aexpr(ae.lhs), map_aexpr(ae.rhs))
        elif isinstance(ae, D.Mult):
            return D.Mult(ae.coeff, map_aexpr(ae.ae))
        return ae

    def map_tree(tree: D.node):
        if isinstance(tree, D.AffineSplit):
            return D.AffineSplit(
                map_aexpr(tree.ae),
                map_tree(tree.ltz),
                map_tree(tree.eqz),
                map_tree(tree.gtz),
            )
        elif isinstance(tree, D.ModSplit):
            return D.ModSplit(
                map_aexpr(tree.ae), tree.m, map_tree(tree.neqz), map_tree(tree.eqz)
            )
        else:
            if isinstance(tree.v, D.ArrayConst):
                return D.Leaf(
                    D.ArrayConst(tree.v.name, [map_aexpr(i) for i in tree.v.idx])
                )
            return tree

    return D.abs(src.iterators, map_tree(src.tree))


def wid_helper(name: Sym, tree: D.node, lastVisitedValue: D.val) -> (D.node, D.val):
    if isinstance(tree, D.Leaf):
        if isinstance(tree.v, D.ArrayConst) and tree.v.name == name:
            tree = D.Leaf(lastVisitedValue)

        return tree, tree.v
    elif isinstance(tree, D.ModSplit):
        neqz, lastVisitedValue = wid_helper(name, tree.neqz, lastVisitedValue)
        eqz, lastVisitedValue = wid_helper(name, tree.eqz, lastVisitedValue)
        return D.ModSplit(tree.ae, tree.m, neqz, eqz), lastVisitedValue
    else:
        ltz, lastVisitedValue = wid_helper(name, tree.ltz, lastVisitedValue)
        eqz, lastVisitedValue = wid_helper(name, tree.eqz, lastVisitedValue)
        gtz, lastVisitedValue = wid_helper(name, tree.gtz, lastVisitedValue)
        return D.AffineSplit(tree.ae, ltz, eqz, gtz), lastVisitedValue


def widening(name: Sym, src: D.abs) -> D.abs:
    # TODO: somehow check that src doesn't have crossings
    tree, _ = wid_helper(name, src.tree, D.Bot())
    return D.abs(src.iterators, tree)


# --------------------------------------------------------------------------- #
# Abstract Interpretation on DataflowIR
# --------------------------------------------------------------------------- #


class ScalarPropagation:
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc
        self.fix_proc(self.proc)

    def fix_proc(self, proc: DataflowIR.proc):
        assert isinstance(proc, DataflowIR.proc)

        # TODO: FIXME: Do we need to use precondition assertions?

        self.fix_block(proc.body)

    def fix_block(self, body: DataflowIR.block):
        # Assumes any inputs have already been set in body.ctxts[0]
        assert len(body.stmts) + 1 == len(body.ctxts)

        for i in range(len(body.stmts)):
            self.fix_stmt(body.ctxts[i], body.stmts[i], body.ctxts[i + 1])

    def fix_stmt(self, pre_env, stmt: DataflowIR.stmt, post_env):
        # Always propagate values bc SSA
        for nm in pre_env:
            post_env[nm] = pre_env[nm]

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
            body = stmt.body
            # if reducing, then expand to x = x + rhs now we can handle both cases uniformly
            if isinstance(stmt, DataflowIR.Reduce):
                body = DataflowIR.BinOp("+", body, stmt.orelse, body.type, stmt.srcinfo)

            def get_eq(cstmt, body, orelse):
                assert isinstance(cstmt, DataflowIR.BinOp)
                if isinstance(cstmt.rhs, DataflowIR.Const) and (cstmt.rhs.val == 0):
                    eq = cstmt.lhs
                elif isinstance(cstmt.lhs, DataflowIR.Const) and (cstmt.lhs.val == 0):
                    eq = cstmt.rhs
                else:
                    eq = DataflowIR.BinOp(
                        "-", cstmt.lhs, cstmt.rhs, DataflowIR.Int(), null_srcinfo()
                    )

                if isinstance(eq, DataflowIR.BinOp) and eq.op == "%":
                    assert isinstance(eq.rhs, DataflowIR.Const) and isinstance(
                        eq.rhs.val, int
                    )
                    modeq = self.dir_to_aexpr(eq.lhs)
                    if modeq:  # If it was a valid aexpr
                        tree = D.ModSplit(modeq, eq.rhs.val, orelse, body)

                eq = self.dir_to_aexpr(eq)

                if eq:  # If it was a valid aexpr
                    if cstmt.op == "==":
                        tree = D.AffineSplit(eq, orelse, body, orelse)
                    elif cstmt.op == ">":
                        tree = D.AffineSplit(eq, orelse, orelse, body)
                    elif cstmt.op == "<":
                        tree = D.AffineSplit(eq, body, orelse, orelse)
                    else:
                        pass

                return tree

            # If the condition is always True or False, just return the leaf as a tree
            orig_body = D.Leaf(self.fix_val(stmt.body))
            orig_orelse = D.Leaf(self.fix_val(stmt.orelse))
            body = orig_body
            orelse = orig_orelse
            tree = D.Leaf(D.Top())
            if isinstance(stmt.cond, DataflowIR.Const) and (stmt.cond.val == True):
                tree = body
            elif isinstance(stmt.cond, DataflowIR.Const) and (stmt.cond.val == False):
                tree = orelse
            else:
                assert isinstance(stmt.cond, DataflowIR.BinOp)

                # preprocess "and"
                # this might be brittle to tree structure
                cstmt = stmt.cond
                cop = cstmt.op
                while cop == "and":
                    assert isinstance(cstmt, DataflowIR.BinOp)
                    r_tree = get_eq(cstmt.rhs, body, orelse)
                    body = r_tree
                    cstmt = cstmt.lhs
                    cop = cstmt.op

                tree = get_eq(cstmt, body, orelse)

            abs_domain = D.abs(stmt.iters + stmt.dims, tree)

            # Do the substitution!
            if isinstance(orig_body.v, D.ArrayConst):
                if orig_body.v.name in pre_env:
                    # check the index of orig_body.v
                    tree = pre_env[orig_body.v.name]
                    for aidx, vidx in zip(tree.iterators, orig_body.v.idx):
                        if not isinstance(vidx, D.Var) or aidx != vidx.name:
                            tree = sub_aexpr(D.Var(aidx), vidx, tree)
                    abs_domain = substitute(orig_body.v, tree.tree, abs_domain)

            if isinstance(orig_orelse.v, D.ArrayConst):
                if orig_orelse.v.name in pre_env:
                    # check the index of orig_body.v
                    tree = pre_env[orig_orelse.v.name]
                    for aidx, vidx in zip(tree.iterators, orig_orelse.v.idx):
                        if not isinstance(vidx, D.Var) or aidx != vidx.name:
                            tree = sub_aexpr(D.Var(aidx), vidx, tree)
                    abs_domain = substitute(orig_orelse.v, tree.tree, abs_domain)

            post_env[stmt.lhs] = abs_domain

        elif isinstance(stmt, DataflowIR.Pass):
            pass  # pass pass, lol

        elif isinstance(stmt, DataflowIR.Alloc):
            pass
            # if val := self.abs_init_val(stmt.name, stmt.hi, stmt.type):
            #     post_env[stmt.name] = val

        elif isinstance(stmt, DataflowIR.If):
            # set up the loop body for fixed-point iteration
            pre_body = stmt.body.ctxts[0]
            pre_orelse = stmt.orelse.ctxts[0]
            for nm, val in pre_env.items():
                pre_body[nm] = val
                pre_orelse[nm] = val

            self.fix_block(stmt.body)
            self.fix_block(stmt.orelse)

            for nm, post_val in stmt.body.ctxts[-1].items():
                post_env[nm] = post_val

            for nm, post_val in stmt.orelse.ctxts[-1].items():
                post_env[nm] = post_val

        elif isinstance(stmt, DataflowIR.For):
            # set up the loop body for fixed-point iteration
            pre_body = stmt.body.ctxts[0]
            for nm, val in pre_env.items():
                pre_body[nm] = val

            self.fix_block(stmt.body)

            assert isinstance(
                stmt.body.stmts[-1],
                (
                    DataflowIR.Assign,
                    DataflowIR.Reduce,
                    DataflowIR.LoopStart,
                    DataflowIR.LoopExit,
                    DataflowIR.IfJoin,
                ),
            )
            lname = stmt.body.stmts[-1].lhs

            new_abs = widening(lname, stmt.body.ctxts[-1][lname])
            pre_body[lname] = new_abs
            self.fix_block(stmt.body)

            #            # run this loop until we reach a fixed-point
            #            at_fixed_point = False
            #            while not at_fixed_point:
            #            # propagate in the loop
            #                self.fix_block(stmt.body)
            #                at_fixed_point = True
            #                # copy the post-values for the loop back around to
            #                # the pre-values, by joining them together
            ##                for nm, prev_val in pre_body.items():
            #                    next_val = stmt.body.ctxts[-1][nm]
            # SANITY-CHECK: Is this correct?
            #                    at_fixed_point = at_fixed_point and greater_than(prev_val, next_val)
            #                    pre_body[nm] = self.abs_join(prev_val, next_val)

            for nm, post_val in stmt.body.ctxts[-1].items():
                post_env[nm] = post_val

        else:
            assert False, f"bad case: {type(stmt)}"

    def dir_to_aexpr(self, e):
        if isinstance(e, DataflowIR.Read):
            if len(e.idx) == 0:
                return D.Var(e.name)
        elif isinstance(e, DataflowIR.Const):
            assert isinstance(e.val, int)
            return D.Const(e.val)
        elif isinstance(e, DataflowIR.USub):
            if arg := self.dir_to_aexpr(e.arg):
                return D.Minus(D.Const(0), arg)
        elif isinstance(e, DataflowIR.BinOp):
            lhs = self.dir_to_aexpr(e.lhs)
            rhs = self.dir_to_aexpr(e.rhs)
            if not lhs or not rhs:
                return None

            if e.op == "+":
                return D.Add(lhs, rhs)
            elif e.op == "*":
                if isinstance(lhs, D.Const):
                    return D.Mult(lhs.val, rhs)
                elif isinstance(rhs, D.Const):
                    return D.Mult(rhs.val, lhs)
                else:
                    assert False, "bad!"
            elif e.op == "-":
                return D.Minus(lhs, rhs)
            else:
                pass

        return None

    def fix_val(self, e: DataflowIR.expr) -> D.val:
        if isinstance(e, DataflowIR.Read):
            idxs = [self.dir_to_aexpr(i) for i in e.idx]
            return D.ArrayConst(e.name, idxs)
        elif isinstance(e, DataflowIR.Const):
            return D.ValConst(e.val)
        elif isinstance(e, DataflowIR.USub):
            return D.Top()
        elif isinstance(e, DataflowIR.BinOp):
            return D.Top()
        elif isinstance(e, DataflowIR.BuiltIn):
            return D.Top()
        elif isinstance(e, DataflowIR.StrideExpr):
            return D.Top()
        else:
            assert False, f"bad case {type(expr)}"


"""
class AbstractInterpretation(ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc
        self.fix_proc(self.proc)

    def fix_proc(self, proc: DataflowIR.proc):
        assert isinstance(proc, DataflowIR.proc)

        # fix args
        for a in proc.args:
            if val := self.abs_init_val(a.name, a.hi, a.type):
                proc.body.ctxts[0][a.name] = val

        # TODO: FIXME: Do we need to use precondition assertions?
        # I guess it's possible to update the values based on the assertions..
        # Do we want to assign this to something? What does assertion mean? Maybe we should have just initialized buffer values with predicates when converting it to SSA form?

        self.fix_block(proc.body)

    def fix_block(self, body: DataflowIR.block):
        # Assumes any inputs have already been set in body.ctxts[0]
        assert len(body.stmts) + 1 == len(body.ctxts)

        for i in range(len(body.stmts)):
            self.fix_stmt(body.ctxts[i], body.stmts[i], body.ctxts[i + 1])

    def fix_stmt(self, pre_env, stmt: DataflowIR.stmt, post_env):
        # Always propagate values bc SSA
        for nm in pre_env:
            post_env[nm] = pre_env[nm]

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
            body = stmt.body
            # if reducing, then expand to x = x + rhs now we can handle both cases uniformly
            if isinstance(stmt, DataflowIR.Reduce):
                body = DataflowIR.BinOp("+", body, stmt.orelse, body.type, stmt.srcinfo)

            post_env[stmt.lhs] = self.abs_phi(
                stmt.iters,
                stmt.dims,
                self.fix_expr(pre_env, stmt.cond),
                self.fix_expr(pre_env, stmt.body),
                self.fix_expr(pre_env, stmt.orelse)
            )

        elif isinstance(stmt, DataflowIR.Pass):
            pass  # pass pass, lol

        elif isinstance(stmt, DataflowIR.Alloc):
            if val := self.abs_init_val(stmt.name, stmt.hi, stmt.type):
                post_env[stmt.name] = val

        elif isinstance(stmt, DataflowIR.If):
            # No need to join. Cool
            self.fix_block(stmt.body)
            self.fix_block(stmt.orelse)

        elif isinstance(stmt, DataflowIR.For):
            raise NotImplementedError
            self.fix_block(stmt.body)

        #            # set up the loop body for fixed-point iteration
        #            pre_body = stmt.body.ctxts[0]
        #            for nm, val in pre_env.items():
        #                pre_body[nm] = val
        #
        #            # initialize the loop iteration variable
        #            lo = self.fix_expr(pre_env, stmt.lo)
        #            hi = self.fix_expr(pre_env, stmt.hi)
        #            pre_body[stmt.iter] = self.abs_iter_val(stmt.iter, lo, hi)
        #
        #            # run this loop until we reach a fixed-point
        #            at_fixed_point = False
        #            while not at_fixed_point:
        #                # propagate in the loop
        #                self.fix_block(stmt.body)
        #                at_fixed_point = True
        #                # copy the post-values for the loop back around to
        #                # the pre-values, by joining them together
        #                for nm, prev_val in pre_body.items():
        #                    next_val = stmt.body.ctxts[-1][nm]
        #                    # SANITY-CHECK: Is this correct?
        #                    at_fixed_point = at_fixed_point and greater_than(prev_val, next_val)
        #                    pre_body[nm] = self.abs_join(prev_val, next_val)
        #
        #            # determine the post-env as join of pre-env and loop results
        #            for nm, pre_val in pre_env.items():
        #                loop_val = stmt.body.ctxts[-1][nm]
        #                post_env[nm] = self.abs_join(pre_val, loop_val)

        else:
            assert False, f"bad case: {type(stmt)}"

    def fix_expr(self, pre_env: D, expr: DataflowIR.expr) -> D:
        if isinstance(expr, DataflowIR.Read):
            # For now, might want to put pre_env in the future
            return self.abs_read(expr)
        elif isinstance(expr, DataflowIR.Const):
            return self.abs_const(expr.val, expr.type)
        elif isinstance(expr, DataflowIR.USub):
            arg = self.fix_expr(pre_env, expr.arg)
            return self.abs_usub(arg)
        elif isinstance(expr, DataflowIR.BinOp):
            lhs = self.fix_expr(pre_env, expr.lhs)
            rhs = self.fix_expr(pre_env, expr.rhs)
            return self.abs_binop(expr.op, lhs, rhs)
        elif isinstance(expr, DataflowIR.BuiltIn):
            args = [self.fix_expr(pre_env, a) for a in expr.args]
            return self.abs_builtin(expr.f, args)
        elif isinstance(expr, DataflowIR.StrideExpr):
            return self.abs_stride_expr(expr.name, expr.dim)
        else:
            assert False, f"bad case {type(expr)}"

    @abstractmethod
    def abs_read(self, expr):
        # Define Read

    @abstractmethod
    def abs_phi(self, dims, cond, body, orelse):
        # Define phi node/values

    @abstractmethod
    def abs_init_val(self, name, his, typ):
        # Define initial argument values

    @abstractmethod
    def abs_stride_expr(self, name, dim):
        # Define abstraction of a specific stride expression

    @abstractmethod
    def abs_const(self, val, typ):
        # Define abstraction of a specific constant value

    @abstractmethod
    def abs_binop(self, op, lval, rval):
        # Implement transfer function abstraction for binary operations

    @abstractmethod
    def abs_usub(self, arg):
        # Implement transfer function abstraction for unary subtraction

    @abstractmethod
    def abs_builtin(self, builtin, args):
        # Implement transfer function abstraction for built-ins


def greater_than(bexpr, val):
    if bexpr == val:
        return True

    if isinstance(val, D.Top):
        return True

    if not isinstance(bexpr, D.Or):
        return False

    exists = False
    if isinstance(bexpr.rhs, D.Or):
        exists |= greater_than(bexpr.rhs, val)
    if isinstance(bexpr.lhs, D.Or):
        exists |= greater_than(bexpr.lhs, val)
    if bexpr.rhs == val or bexpr.lhs == val:
        return True

    return exists


class ScalarPropagation(AbstractInterpretation):
    def abs_read(self, expr):
        pass

    def abs_phi(self, iters, dims, cond, body, orelse):
        assert isinstance(iters, list)
        assert isinstance(dims, list)
        assert isinstance(cond, D.dexpr)
        assert isinstance(body, D.dexpr)
        assert isinstance(orelse, D.dexpr)

        if len(dims) > 0:
            # Not gonna worry about arrays for now!
            return None

        return D.Select(dims, cond, body, orelse)

    def abs_init_val(self, name, his, typ):
        assert isinstance(name, Sym)
        assert isinstance(his, list)
        assert isinstance(typ, DataflowIR.type)

        if len(his) > 0:
            # Not gonna worry about arrays for now
            return None

        return D.Top()

    def abs_stride_expr(self, name, dim):
        assert isinstance(name, Sym)
        assert isinstance(dim, int)

        return D.Var(Sym(name.name() + str(dim)), DataflowIR.stride)

    def abs_const(self, val, typ) -> D:
        return D.ValConst(val)

    def abs_binop(self, op, lval: D, rval: D) -> D:

        if isinstance(lval, D.Top) or isinstance(rval, D.Top):
            return D.Top()

        # front_ops = {"+", "-", "*", "/", "%",
        #              "<", ">", "<=", ">=", "==", "and", "or"}
        if isinstance(lval, D.Const) and isinstance(rval, D.Const):
            typ = lval.type
            lval = lval.val
            rval = rval.val
            if op == "+":
                val = lval + rval
            elif op == "-":
                val = lval - rval
            elif op == "*":
                val = lval * rval
            elif op == "/":
                val = lval / rval  # THIS IS WRONG
            elif op == "%":
                val = lval % rval
            else:
                typ = T.bool  # What would be bool here?
                if op == "<":
                    val = lval < rval
                elif op == ">":
                    val = lval > rval
                elif op == "<=":
                    val = lval <= rval
                elif op == ">=":
                    val = lval >= rval
                elif op == "==":
                    val = lval == rval
                elif op == "and":
                    val = lval and rval
                elif op == "or":
                    val = lval or rval
                else:
                    assert False, f"Bad Case Operator: {op}"

            return D.Const(val, typ)

        # TODO: and, or short circuiting here

        if op == "/":
            # NOTE: THIS doesn't work right for integer division...
            # c1 / c2
            # 0 / x == 0
            if isinstance(lval, D.Const) and lval.val == 0:
                return lval

        if op == "%":
            if isinstance(rval, D.Const) and rval.val == 1:
                return D.Const(0, lval.type)

        if op == "*":
            # x * 0 == 0
            if isinstance(lval, D.Const) and lval.val == 0:
                return lval
            elif isinstance(rval, D.Const) and rval.val == 0:
                return rval

        return D.BinOp(op, lval, rval, lval.type)  # FIXME: realliy?

    def abs_usub(self, arg: D) -> D:
        if isinstance(arg, D.Top):
            return D.Top()
        return D.USub(arg, arg.type)

    def abs_builtin(self, builtin, args):
        # TODO: We should not attempt to do precise analysis on builtins
        return D.Top()
"""
