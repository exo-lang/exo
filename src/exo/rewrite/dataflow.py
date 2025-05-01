from abc import ABC, abstractmethod
from enum import Enum
from collections import ChainMap, defaultdict
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


# TODO: separatae absenv from DataflowIR
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
    def __init__(self, proc, stmts, sym):
        self.loopir_proc = proc
        self.stmts = []
        # self.sym and self.syms are a bit hacky.. ways of getting new syms
        self.sym = sym
        self.syms = []
        self.env = ChainMap()
        for s in stmts:
            self.stmts.append([s])

        self.dataflow_proc = self.map_proc(self.loopir_proc)

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def update_env(self, name, args):
        if self.sym == name:
            self.syms.append(args[0])
        self.env[name] = args

    def result(self):
        res = []
        for l in self.stmts:
            res.extend(l[1:])
        return self.dataflow_proc, res, self.syms

    def init_block(self, body):
        return DataflowIR.block(body, dict())

    def map_proc(self, p):
        df_args = self._map_list(self.map_fnarg, p.args)

        # Initialize configurations in this proc
        configs = list(
            set(
                get_writeconfigs(self.loopir_proc.body)
                + get_readconfigs(self.loopir_proc.body)
                + get_readconfigs_expr(self.loopir_proc.preds)
            )
        )
        for c in configs:
            orig_sym = c[0]._INTERNAL_sym(c[1])
            new_sym = orig_sym.copy()
            typ = self.map_t(c[0].lookup_type(c[1]))
            # (the most recent sym, (iter dims, dim dims), basetype)
            self.update_env(orig_sym, (new_sym, ([], []), typ))
            # Treat the initial configuration as function arguments!
            df_args.append(DataflowIR.fnarg(new_sym, [], typ, null_srcinfo()))

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
            self.update_env(a.name, (name, ([], dsyms), typ))
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
            self.update_env(s.name, (new_name, (iters, dims), old_typ))

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
            self.update_env(ir_config, (df_config, (iters, []), df_rhs.type))

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
                self.update_env(key, (post_name, vals[1], vals[2]))

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
                self.update_env(
                    key, (vals[0].copy(), ([s.iter] + vals[1][0], vals[1][1]), vals[2])
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
                    "==", iter_read, self.map_e(s.lo), DataflowIR.Bool(), null_srcinfo()
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
                self.update_env(
                    key, (post_name, (llast[1][0][1:], llast[1][1]), llast[2])
                )

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
                self.update_env(s.name, (name, ([], dsyms), typ))

                return DataflowIR.Alloc(name, his, typ, s.srcinfo)
            else:
                assert s.type.is_real_scalar()
                name = s.name.copy()
                typ = self.map_t(s.type)
                self.update_env(s.name, (name, ([], []), typ))

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
            df_lhs = self.map_e(e.lhs)
            df_rhs = self.map_e(e.rhs)
            return DataflowIR.BinOp(e.op, df_lhs, df_rhs, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.Extern):
            df_args = self.map_exprs(e.args)
            return DataflowIR.Extern(e.f, df_args, self.map_t(e.type), e.srcinfo)

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
          | ArrayVar(sym name, aexpr* idx)

    aexpr = Const(int val)
          | Var(sym name)
          | Add(aexpr lhs, aexpr rhs)
          | Mult(int coeff, aexpr ae)  
          | Div(aexpr ae, int divisor)
          | Mod(aexpr ae, int m)
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
# Join for D.val, including SubVal and ArrayVar
# --------------------------------------------------------------------------- #


def subval_join(v1, v2):
    assert isinstance(v1, D.val)
    assert isinstance(v2, D.val)

    if type(v1) != type(v2):
        return D.SubVal(V.Top())

    if isinstance(v1, D.SubVal):
        if isinstance(v1.av, V.Bot):
            return v2
        if isinstance(v2.av, V.Bot):
            return v1
        if (
            isinstance(v1.av, V.ValConst)
            and isinstance(v2.av, V.ValConst)
            and v1.av.val == v2.av.val
        ):
            return v1

        return D.SubVal(V.Top())

    elif isinstance(v2, D.ArrayVar):
        if v1.name == v2.name and all(
            [aexpr_eq(a1, a2) for a1, a2 in zip(v1.idx, v2.idx)]
        ):
            return v1
        else:
            return D.SubVal(V.Top())
    else:
        assert False, "bad case"


# --------------------------------------------------------------------------- #
# Equivalence for D.val, used with overlay
# --------------------------------------------------------------------------- #


def subval_eq(v1, v2):
    assert isinstance(v1, D.val)
    assert isinstance(v2, D.val)

    if type(v1) != type(v2):
        return D.SubVal(V.Bot())

    if isinstance(v1, D.SubVal):
        if v1 == v2:
            return D.SubVal(V.Top())
    if isinstance(v1, D.ArrayVar):
        if v1.name == v2.name and all(
            [aexpr_eq(a1, a2) for a1, a2 in zip(v1.idx, v2.idx)]
        ):
            return D.SubVal(V.Top())
    return D.SubVal(V.Bot())


def is_all_top(a):
    if isinstance(a, D.abs):
        return is_all_top(a.tree)
    if isinstance(a, D.Leaf):
        if isinstance(a.v, D.SubVal) and isinstance(a.v.av, V.Top):
            return True
    if isinstance(a, D.AffineSplit):
        return is_all_top(a.ltz) and is_all_top(a.eqz) and is_all_top(a.gtz)
    if isinstance(a, D.ModSplit):
        return is_all_top(a.neqz) and is_all_top(a.eqz)

    return False


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
    elif isinstance(e, D.Div):
        return A.BinOp(
            "/",
            lift_to_smt_a(e.ae),
            A.Const(e.divisor, typ, null_srcinfo()),
            typ,
            null_srcinfo(),
        )
    elif isinstance(e, D.Mod):
        return A.BinOp(
            "%",
            lift_to_smt_a(e.ae),
            A.Const(e.m, typ, null_srcinfo()),
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


def has_array_access(e: DataflowIR.expr) -> bool:
    if isinstance(e, DataflowIR.Read):
        if len(e.idx) > 0:
            return True
    elif isinstance(e, DataflowIR.BinOp):
        return has_array_access(e.lhs) or has_array_access(e.rhs)
    return False


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
        elif e.op == "/":
            assert isinstance(rhs, D.Const)
            return D.Div(lhs, rhs.val)
        elif e.op == "%":
            assert isinstance(rhs, D.Const)
            return D.Mod(lhs, rhs.val)
        else:
            # TODO: Support division at some point
            assert False, f"got unsupported binop {e.op} in {e}."

    assert False, f"shouldn't be here. got {e}"


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
            # When assumptions are unsatisfiable, this is a cell without integer points. We Bottom such cases.
            if not slv.satisfy((A.Const(True, T.bool, null_srcinfo()))):
                return D.Leaf(D.SubVal(V.Bot()))

            # we can collapse the tree when all values are the same
            if (
                isinstance(tree.ltz, D.Leaf)
                and isinstance(tree.eqz, D.Leaf)
                and isinstance(tree.gtz, D.Leaf)
                and (type(tree.ltz) == type(tree.eqz) == type(tree.gtz))
                and (tree.ltz.v == tree.eqz.v == tree.gtz.v)
            ):
                return tree.ltz

            pred = lift_to_smt_a(tree.ae)

            # If ltz branch is unsatisfiable and the values for eqz and gtz branches are equivalent, we can collapse this node.
            if (
                isinstance(tree.eqz, D.Leaf)
                and isinstance(tree.gtz, D.Leaf)
                and (type(tree.eqz) == type(tree.gtz))
                and (tree.eqz.v == tree.gtz.v)
            ):
                if not slv.satisfy(mk_aexpr("<", pred)):
                    return tree.eqz

            # If gtz branch is unsatisfiable and the values for eqz and ltz branches are equivalent, we can collapse this node.
            if (
                isinstance(tree.eqz, D.Leaf)
                and isinstance(tree.ltz, D.Leaf)
                and (type(tree.eqz) == type(tree.ltz))
                and (tree.eqz.v == tree.ltz.v)
            ):
                if not slv.satisfy(mk_aexpr(">", pred)):
                    return tree.eqz

            # check if anything is simplifiable
            ltz_eq = mk_aexpr("<", pred)
            eqz_eq = mk_aexpr("==", pred)
            gtz_eq = mk_aexpr(">", pred)

            if slv.verify(eqz_eq):
                return map_tree(tree.eqz)
            elif slv.verify(gtz_eq):
                return map_tree(tree.gtz)
            elif slv.verify(ltz_eq):
                return map_tree(tree.ltz)

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
        elif isinstance(aexpr, D.Div):
            return D.Div(self.map_aexpr(aexpr.ae), aexpr.divisor)
        elif isinstance(aexpr, D.Mod):
            return D.Mod(self.map_aexpr(aexpr.ae), aexpr.m)
        else:
            assert False


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


# --------------------------------------------------------------------------- #
# Substitution for aexpr
# --------------------------------------------------------------------------- #

# Helper function to negate an expression.
def negate(ae):
    if isinstance(ae, D.Const):
        return D.Const(-ae.val)
    elif isinstance(ae, D.Var):
        return D.Mult(-1, ae)
    elif isinstance(ae, D.Add):
        return D.Add(negate(ae.lhs), negate(ae.rhs))
    elif isinstance(ae, D.Mult):
        return D.Mult(-ae.coeff, ae.ae)
    else:
        raise TypeError("Unknown aexpr type")


# Function to "divide" an expression by an integer.
# It assumes that every constant multiplier divides evenly.
def divide_expr(ae, divisor):
    if divisor == 0:
        raise ValueError("Division by zero")
    if isinstance(ae, D.Const):
        if ae.val % divisor != 0:
            raise ValueError("Non-integer division result")
        return D.Const(ae.val // divisor)
    elif isinstance(ae, D.Var):
        # Treat a variable as 1*Var.
        if 1 % divisor != 0:
            raise ValueError("Non-integer division result")
        # Only works if divisor is 1 or -1.
        return D.Mult(1 // divisor, ae)
    elif isinstance(ae, D.Add):
        lhs = divide_expr(ae.lhs, divisor)
        rhs = divide_expr(ae.rhs, divisor)
        if isinstance(rhs, D.Const) and rhs.val == 0:
            return lhs
        if isinstance(lhs, D.Const) and lhs.val == 0:
            return rhs
        return D.Add(lhs, rhs)
    elif isinstance(ae, D.Mult):
        if ae.coeff % divisor != 0:
            raise ValueError(
                "Non-integer division result in multiplication, todo implement fractions"
            )
        if isinstance(ae.ae, D.Const) and ae.ae.val == 0:
            return D.Const(0)
        r = ae.coeff // divisor
        if r == 1:
            return ae.ae
        elif r == 0:
            return D.Const(0)
        return D.Mult(r, ae.ae)
    else:
        raise TypeError("Unknown aexpr type")


# Function to "collect" the target variable.
# It returns a pair (coeff, rest) such that: ae == coeff*target + rest.
def collect(ae, target):
    if isinstance(ae, D.Const):
        return (0, ae)
    elif isinstance(ae, D.Var):
        if ae.name == target:
            return (1, D.Const(0))
        else:
            return (0, ae)
    elif isinstance(ae, D.Add):
        coeff_lhs, rest_lhs = collect(ae.lhs, target)
        coeff_rhs, rest_rhs = collect(ae.rhs, target)
        return (coeff_lhs + coeff_rhs, D.Add(rest_lhs, rest_rhs))
    elif isinstance(ae, D.Mult):
        coeff_inside, rest_inside = collect(ae.ae, target)
        return (ae.coeff * coeff_inside, D.Mult(ae.coeff, rest_inside))
    else:
        raise TypeError("Unknown aexpr type")


# The main function: solve the equation ae = 0 for the given target variable.
def solve_for(ae, target):
    coeff, rest = collect(ae, target)
    if coeff == 0:
        return None
    # Equation is: coeff*target + rest = 0  ->  target = -(rest) / coeff.
    return divide_expr(negate(rest), coeff)


# FIXME: TODO:
# It's not clear what to do in cases like "2*i == d", where i will be 1/2*d,
# but we only allow integer multiplier!
# I think this will be a real issue..
# We won't be able to handle statements like x[2*i] :eyes:
# Use fractions!!

# is it possible to have multiple eqs conflicting?? I think the first one should be prioritized in that case.
def eliminate_target_dim(val, tgt):
    # can we just ASubs on the subtree?? from eq branch?
    # that actually should be correct. We got rid of tgt from that branch, so we can just proceed.
    assert isinstance(val, D.node)

    if isinstance(val, D.Leaf):
        return val
    elif isinstance(val, D.AffineSplit):
        r = solve_for(val.ae, tgt)
        if r == None:
            eqz = eliminate_target_dim(val.eqz, tgt)
        else:
            eqz = ASubs(val.eqz, {tgt: r}).result()

        return D.AffineSplit(
            val.ae,
            eliminate_target_dim(val.ltz, tgt),
            eqz,
            eliminate_target_dim(val.gtz, tgt),
        )
    elif isinstance(val, D.ModSplit):
        return D.ModSplit(
            val.ae,
            val.m,
            eliminate_target_dim(val.neqz, tgt),
            eliminate_target_dim(val.eqz, tgt),
        )
    else:
        assert False, "bad else"


# --------------------------------------------------------------------------- #
# Overlay cell decomposition
# --------------------------------------------------------------------------- #


def overlay(a1: D.abs or D.node, a2: D.abs or D.node, fun) -> D.node:
    """
    return {(d1 \intersects d2, fun(v1, v2) | d1 \in a1 and d2 \in a2}
    """
    if isinstance(a1, D.abs):
        assert isinstance(a2, D.abs)

        if len(a1.iterators) != len(a2.iterators) or not all(
            [i1 == i2 for i1, i2 in zip(a1.iterators, a2.iterators)]
        ):
            raise ValueError(f"Iterators of a1 and a2 should match exactly")

        slv = SMTSolver(verbose=False)

        for itr in a1.iterators:
            slv.assume(
                A.BinOp(
                    ">=",
                    A.Var(itr, T.Int(), null_srcinfo()),
                    A.Const(0, T.Int(), null_srcinfo()),
                    T.Bool(),
                    null_srcinfo(),
                )
            )

        return overlay_with_smt(slv, a1.tree, a2.tree, fun)
    else:
        assert isinstance(a1, D.node)
        assert isinstance(a2, D.node)

        slv = SMTSolver(verbose=False)

        return overlay_with_smt(slv, a1, a2, fun)


def mk_mod_expr(ae, m):
    return A.BinOp(
        "%", lift_to_smt_a(ae), A.Const(m, T.int, null_srcinfo()), T.int, null_srcinfo()
    )


def aexpr_eq(e1, e2):
    assert isinstance(e1, D.aexpr) and isinstance(e2, D.aexpr)

    # First check that they're the same type
    if type(e1) != type(e2):
        return False

    if isinstance(e1, D.Const):
        return e1.val == e2.val
    elif isinstance(e1, D.Var):
        return e1.name == e2.name
    elif isinstance(e1, D.Add):
        return aexpr_eq(e1.lhs, e2.lhs) and aexpr_eq(e1.rhs, e2.rhs)
    elif isinstance(e1, D.Mult):
        return e1.coeff == e2.coeff and aexpr_eq(e1.ae, e2.ae)
    elif isinstance(e1, D.Div):
        return e1.divisor == e2.divisor and aexpr_eq(e1.ae, e2.ae)
    elif isinstance(e1, D.Mod):
        return e1.m == e2.m and aexpr_eq(e1.ae, e2.ae)
    else:
        assert False, "bad case"


def affine_overlay_helper(slv, t1, t2, fun):
    """
    t1 is a affine split, t2 can be anything
    """
    pred_expr = lift_to_smt_a(t1.ae)

    # check if anything is simplifiable
    ltz_eq = mk_aexpr("<", pred_expr)
    eqz_eq = mk_aexpr("==", pred_expr)
    gtz_eq = mk_aexpr(">", pred_expr)

    if slv.verify(ltz_eq):
        return overlay_with_smt(slv, t1.ltz, t2, fun)
    if slv.verify(eqz_eq):
        return overlay_with_smt(slv, t1.eqz, t2, fun)
    if slv.verify(gtz_eq):
        return overlay_with_smt(slv, t1.gtz, t2, fun)

    slv.push()
    slv.assume(ltz_eq)
    new_ltz = overlay_with_smt(slv, t1.ltz, t2, fun)
    slv.pop()

    slv.push()
    slv.assume(eqz_eq)
    new_eqz = overlay_with_smt(slv, t1.eqz, t2, fun)
    slv.pop()

    slv.push()
    slv.assume(gtz_eq)
    new_gtz = overlay_with_smt(slv, t1.gtz, t2, fun)
    slv.pop()

    return D.AffineSplit(t1.ae, new_ltz, new_eqz, new_gtz)


def mod_overlay_helper(slv, t1, t2, fun):
    """
    t1 is a mod-split, t2 can be anything
    """
    pred_expr = mk_mod_expr(t1.ae, t1.m)
    eq_expr = mk_aexpr("==", pred_expr)
    neq_expr = A.Not(eq_expr, T.bool, null_srcinfo())

    if slv.verify(eq_expr):
        return overlay_with_smt(slv, t1.eqz, t2, fun)
    if slv.verify(neq_expr):
        return overlay_with_smt(slv, t1.neqz, t2, fun)

    slv.push()
    slv.assume(eq_expr)
    new_eqz = overlay_with_smt(slv, t1.eqz, t2, fun)
    slv.pop()

    slv.push()
    slv.assume(neq_expr)
    new_neqz = overlay_with_smt(slv, t1.neqz, t2, fun)
    slv.pop()

    return D.ModSplit(t1.ae, t1.m, new_neqz, new_eqz)


def overlay_with_smt(slv, t1, t2, fun):
    """
    Overlay trees t1 and t2, pruning any branch whose intersection is unsatisfiable.
    Returns the overlayed D.node, or a 'bottom' leaf if unsatisfiable.
    """
    # TODO: This will Bottom out the infeasible branch, but I'm commenting this out since it's a bit confusing
    # First check if current path constraints are already unsatisfiable
    # if not slv.satisfy(A.Const(True, T.bool, null_srcinfo())):
    #    return D.Leaf(D.SubVal(V.Bot()))

    # -- Case 1: both are leaves
    if isinstance(t1, D.Leaf) and isinstance(t2, D.Leaf):
        # Now we combine the leaf values with fun
        return D.Leaf(fun(t1.v, t2.v))

    # -- Case 2: t1 is a leaf but t2 is a split
    if isinstance(t1, D.Leaf):
        if isinstance(t2, D.AffineSplit):
            return affine_overlay_helper(slv, t2, t1, fun)

        if isinstance(t2, D.ModSplit):
            return mod_overlay_helper(slv, t2, t1, fun)

    # -- Case 3: t2 is a leaf but t1 is a split (symmetric to case 2)
    if isinstance(t2, D.Leaf):
        if isinstance(t1, D.AffineSplit):
            return affine_overlay_helper(slv, t1, t2, fun)

        elif isinstance(t1, D.ModSplit):
            return mod_overlay_helper(slv, t1, t2, fun)

    # -- Case 4: both are AffineSplit
    if isinstance(t1, D.AffineSplit) and isinstance(t2, D.AffineSplit):
        # If they split on the same expression, unify sub-branches directly
        if aexpr_eq(t1.ae, t2.ae):  # just a syntactic equivalence for now
            pred_expr = lift_to_smt_a(t1.ae)

            slv.push()
            slv.assume(mk_aexpr("<", pred_expr))  # from t1
            merged_ltz = overlay_with_smt(slv, t1.ltz, t2.ltz, fun)
            slv.pop()

            slv.push()
            slv.assume(mk_aexpr("==", pred_expr))
            merged_eqz = overlay_with_smt(slv, t1.eqz, t2.eqz, fun)
            slv.pop()

            slv.push()
            slv.assume(mk_aexpr(">", pred_expr))
            merged_gtz = overlay_with_smt(slv, t1.gtz, t2.gtz, fun)
            slv.pop()

            return D.AffineSplit(t1.ae, merged_ltz, merged_eqz, merged_gtz)

        else:
            return affine_overlay_helper(slv, t1, t2, fun)

    # -- Case 5: both are ModSplit
    if isinstance(t1, D.ModSplit) and isinstance(t2, D.ModSplit):
        # If they split on the *same* (aexpr, modulus), unify sub-branches
        if aexpr_eq(t1.ae, t2.ae) and t1.m == t2.m:
            # Both do `ae % m == 0` vs `!=0`; unify eqz with eqz, neqz with neqz
            pred_expr = mk_mod_expr(t1.ae, t1.m)

            slv.push()
            slv.assume(mk_aexpr("==", pred_expr))
            merged_eqz = overlay_with_smt(slv, t1.eqz, t2.eqz, fun)
            slv.pop()

            slv.push()
            slv.assume(A.Not(mk_aexpr("==", pred_expr), T.bool, null_srcinfo()))
            merged_neqz = overlay_with_smt(slv, t1.neqz, t2.neqz, fun)
            slv.pop()

            return D.ModSplit(t1.ae, t1.m, merged_neqz, merged_eqz)
        else:
            return mod_overlay_helper(slv, t1, t2, fun)

    # -- Case 6: one is AffineSplit, the other is ModSplit
    if isinstance(t1, D.AffineSplit) and isinstance(t2, D.ModSplit):
        return affine_overlay_helper(slv, t1, t2, fun)

    if isinstance(t1, D.ModSplit) and isinstance(t2, D.AffineSplit):
        return mod_overlay_helper(slv, t1, t2, fun)

    assert False, "shouldn't reach here"


# --------------------------------------------------------------------------- #
# Abstract Interpretation on DataflowIR
# --------------------------------------------------------------------------- #


class AbstractInterpretation(ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc
        # set of "original" arrays of proc arguments. We need this to distinguish ArrayVars from Top
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
            body = stmt.body
            # if reducing, then expand to x = x + rhs now we can handle both cases uniformly
            if isinstance(stmt, DataflowIR.Reduce):
                body = DataflowIR.BinOp("+", body, stmt.orelse, body.type, stmt.srcinfo)

            env[stmt.lhs] = D.abs(
                stmt.iters + stmt.dims,
                self.abs_ternary(
                    stmt.cond,
                    self.fix_expr(body, env),
                    self.fix_expr(stmt.orelse, env),
                ),
            )

        elif isinstance(stmt, DataflowIR.Pass):
            pass  # pass pass, lol

        elif isinstance(stmt, DataflowIR.Alloc):
            pass  # no need to do anything

        elif isinstance(stmt, DataflowIR.If):
            # Basically just pass-through

            # Initialize
            for nm, val in env.items():
                stmt.body.ctxt[nm] = val
                stmt.orelse.ctxt[nm] = val

            self.fix_block(stmt.body)
            self.fix_block(stmt.orelse)

            # Write back
            for nm, post_val in stmt.body.ctxt.items():
                env[nm] = post_val
            for nm, post_val in stmt.orelse.ctxt.items():
                env[nm] = post_val

        elif isinstance(stmt, DataflowIR.For):

            for nm, val in env.items():
                stmt.body.ctxt[nm] = val

            pre_env = dict()
            self.fix_block(stmt.body)
            for nm, val in stmt.body.ctxt.items():
                pre_env[nm] = val

            count = 0
            while True:

                # fixpoint iteration
                self.fix_block(stmt.body)

                all_eq = True
                for nm, val in stmt.body.ctxt.items():
                    # Don't widen if it does not depend on this loop
                    if stmt.iter not in val.iterators:
                        continue

                    # Use overlay to compare equality of pre_env and val.
                    # subval_eq will top if the values are equivalent, so will need to check that all leaves are top by is_all_top
                    if is_all_top(overlay(pre_env[nm], val, subval_eq)):
                        continue

                    # Eliminate target dim, pre-processing to the widening (kinda). just approximation.
                    val = D.abs(
                        val.iterators, eliminate_target_dim(val.tree, stmt.iter)
                    )

                    # Widening
                    w_res = self.abs_widening(pre_env[nm], val, count)

                    # if the result of the widening is all Top, that means we gave up so exit the loop.
                    if is_all_top(w_res):
                        continue

                    all_eq = False
                    stmt.body.ctxt[nm] = w_res
                    pre_env[nm] = w_res

                if all_eq:
                    break

                count += 1

            for nm, val in pre_env.items():
                env[nm] = val

        else:
            assert False, f"bad case: {type(stmt)}"

    # Corresponds to E^\# : \Expr \to \Sigma^\# \to val in the paper
    def fix_expr(self, e: DataflowIR.expr, env) -> D.val:
        if isinstance(e, DataflowIR.Read):
            return self.abs_read(e.name, e.idx, env)

        elif isinstance(e, DataflowIR.Const):
            return self.abs_const(e.val)

        elif isinstance(e, DataflowIR.USub):
            return self.abs_usub(e.arg)

        elif isinstance(e, DataflowIR.BinOp):
            return self.abs_binop(e.op, e.lhs, e.rhs)

        elif isinstance(e, DataflowIR.Extern):
            return self.abs_extern(e.f, e.args)

        elif isinstance(e, DataflowIR.StrideExpr):
            return self.abs_stride(e.name, e.dim)

        else:
            assert False, f"bad case {type(expr)}"

    @abstractmethod
    def abs_ternary(
        self, cond: DataflowIR.expr, body: D.node, orelse: D.node
    ) -> D.node:
        """Approximate the ternary phi node"""

    @abstractmethod
    def abs_read(self, name, idx):
        """Approximate the DataflorIR Reads"""

    @abstractmethod
    def abs_const(self, val):
        """Approximate the constant"""

    @abstractmethod
    def abs_usub(self, arg):
        """Approximate the usub"""

    @abstractmethod
    def abs_binop(self, op, lhs, rhs):
        """Approximate the binop"""

    @abstractmethod
    def abs_extern(self, func: Extern, args: list) -> D.node:
        """Approximate the extern"""

    @abstractmethod
    def abs_stride(self, name, dim) -> D.node:
        """Approximate the stride"""

    @abstractmethod
    def abs_widening(self, a1, a2, count):
        """Approximate the loop!"""
