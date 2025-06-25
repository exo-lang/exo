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

# Sympy
import sympy as sm


# --------------------------------------------------------------------------- #
# DataflowIR definition
# --------------------------------------------------------------------------- #


def validateAbsEnv(obj):
    if not isinstance(obj, dict):
        raise ValidationError(D, type(obj))
    for key in obj:
        if not isinstance(key, sm.Symbol):
            raise ValidationError(sm.Symbol, key)
    return obj


# TODO: separatae absenv from DataflowIR
DataflowIR = ADT(
    """
module DataflowIR {
    proc = ( name    name,
             fnarg*  args,
             expr*   preds,
             dict    sym_table, -- table of sm.Symbol to Sym
             stmt*   body,
             absenv  ctxt,
             srcinfo srcinfo )

    fnarg  = ( sym     name,
               expr*   hi,
               type    type,
               srcinfo srcinfo )

    stmt = Assign( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | Reduce( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | LoopStart( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | LoopExit( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | IfJoin( sym lhs, sym *iters, sym* dims, expr cond, expr body, expr orelse )
         | Alloc( sym name, expr* hi, type type )
         | Pass()
         | If( expr cond, stmt* body, stmt* orelse )
         | For( sym iter, expr lo, expr hi, stmt* body )
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
        "dict": dict,
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

        return DataflowIR.proc(p.name, df_args, df_preds, {}, df_body, {}, p.srcinfo)

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

            return [DataflowIR.If(cond, body, orelse, s.srcinfo)] + post_if

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
                    pre_loop + body,
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


from sympy.core.relational import Boolean

ArrayDomain = ADT(
    """
module ArrayDomain {
    abs = (sym* iterators, expr *poly, node tree) -- iterators is generator list, poly is a list of original polynomials

    node   = Leaf(val v, dict sample) -- leaf has a value and sample points!
           | LinSplit(cell *cells)

    cell = Cell(rel eq, node tree)

    val   = SubVal(vabs av)
          | ArrayVar(sym name, expr* idx)
          | ScalarExpr(expr poly)
}
""",
    ext_types={
        "vabs": validateValueDom,
        "sym": sm.Symbol,
        "expr": sm.Expr,
        "rel": Boolean,
        "dict": dict,
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
def lift_to_smt_a(e) -> A:
    # def lift_to_smt_a(e: D.aexpr) -> A:
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


# Lifting function for dataflow ir expression to Sympy's Expr
def lift_to_sympy(e: DataflowIR.expr, table: dict) -> sm.Expr:

    if isinstance(e, DataflowIR.Read):
        assert len(e.idx) == 0
        sym = sm.Symbol(e.name.__repr__())
        table[sym] = e.name
        return sym

    elif isinstance(e, DataflowIR.Const):
        val = e.val
        if isinstance(val, bool):
            return sm.S.true if val else sm.S.false
        if isinstance(val, int):
            return sm.Integer(val)
        raise TypeError(f"Unsupported constant type: {type(val)}")

    elif isinstance(e, DataflowIR.USub):
        return -lift_to_sympy(e.arg, table)

    elif isinstance(e, DataflowIR.BinOp):
        lhs = lift_to_sympy(e.lhs, table)
        rhs = lift_to_sympy(e.rhs, table)

        op = e.op
        if op == "+":
            return sm.Add(lhs, rhs, evaluate=False)
        elif op == "-":
            return sm.Add(lhs, sm.Mul(-1, rhs, evaluate=False), evaluate=False)
        elif op == "*":
            return sm.Mul(lhs, rhs, evaluate=False)
        elif op == "/":
            return sm.Mul(lhs, sm.Pow(rhs, -1, evaluate=False), evaluate=False)
        elif op == "==":
            return sm.Eq(lhs, rhs, evaluate=False)
        elif op == "<":
            return sm.Lt(lhs, rhs, evaluate=False)
        elif op == ">":
            return sm.Gt(lhs, rhs, evaluate=False)
        elif op == ">=":
            return sm.Le(lhs, rhs, evaluate=False)
        elif op == "<=":
            return sm.Ge(lhs, rhs, evaluate=False)
        elif op == "%":
            return sm.Mod(lhs, rhs, evaluate=False)
        elif op == "and":
            return sm.And(lhs, rhs, evaluate=False)
        elif op == "or":
            return sm.Or(lhs, rhs, evaluate=False)

        # and, or. and other stuff should not be here!

        raise ValueError(f"Unsupported binary operator: {op}")

    raise TypeError(f"Unhandled DataflowIR node: {e!r}")


# --------------------------------------------------------------------------- #
# Substitution related operations
# --------------------------------------------------------------------------- #

# ─────────────────────────────────────────────────────────────────────────────
#  Generic, immutable mapper --------------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
class Abs_Rewrite:
    """Walk every node of an ArrayDomain tree, building a *new* tree on the way.

    To specialise, override *only* the small `map_*` methods; the scaffolding
    takes care of recursion and immutability.
    """

    # ── entry point ----------------------------------------------------------
    def map_abs(self, a: D.abs) -> D.abs:
        return D.abs(
            self.map_iters(a.iterators),
            [self.map_expr(p) for p in a.poly],
            self.map_node(a.tree),
        )

    # ── iterators ------------------------------------------------------------
    def map_iters(self, itrs):
        return [self.map_iter(i) for i in itrs]

    def map_iter(self, i):  # ← identity-by-default
        return i

    # ── expressions / relations ---------------------------------------------
    def map_expr(self, e):
        return e  # identity

    def map_rel(self, r):
        return r

    # ── NEW: samples ---------------------------------------------------------
    def map_sample(self, sample: dict) -> dict:
        """Rewrite the dictionary *value* expressions.
        The *keys* (variables) stay unchanged here.
        """
        return {k: self.map_expr(v) for k, v in sample.items()}

    # ── nodes ----------------------------------------------------------------
    def map_node(self, node: D.node):
        if isinstance(node, D.Leaf):
            return D.Leaf(
                self.map_val(node.v),
                self.map_sample(node.sample),  # *** now rewritten ***
            )

        elif isinstance(node, D.LinSplit):
            return D.LinSplit([self.map_cell(c) for c in node.cells])

        raise TypeError(f"Unknown node variant {type(node)}")

    def map_cell(self, cell: D.cell):
        return D.Cell(self.map_rel(cell.eq), self.map_node(cell.tree))

    # ── values ---------------------------------------------------------------
    def map_val(self, val: D.val):
        if isinstance(val, D.SubVal):
            return val  # nothing inside to touch

        if isinstance(val, D.ScalarExpr):
            return D.ScalarExpr(self.map_expr(val.poly))

        if isinstance(val, D.ArrayVar):
            return D.ArrayVar(val.name, [self.map_expr(i) for i in val.idx])

        raise TypeError(f"Unknown val variant {type(val)}")


# ─────────────────────────────────────────────────────────────────────────────
#  Concrete subclass:  substitution ------------------------------------------
# ─────────────────────────────────────────────────────────────────────────────
class ASubs(Abs_Rewrite):
    r"""
    Apply a *single* SymPy substitution  Σ = {sym : expr}  to a CAD.

    * Works for **numeric** maps  {x: 3, y: -7/2}                       – projection
    * Works for **symbolic** maps {x: x - 1,  y: y + z} (affine change) – pull-back
    """

    # ── construction ---------------------------------------------------------
    def __init__(self, node: D.node, env: dict[sm.Symbol, sm.Expr]):
        self._env = env
        self._out = super().map_node(node)  # build transformed tree

    def result(self) -> D.node:  # public accessor
        return self._out

    # ── expression-level rewrite --------------------------------------------
    def map_expr(self, e):
        return e.xreplace(self._env)

    def map_rel(self, r):
        return r.xreplace(self._env)

    # ── sample-point logic ---------------------------------------------------
    def map_sample(self, sample: dict) -> dict:
        new_s = {}

        for var, old_val in sample.items():

            if var not in self._env:
                expr = (
                    sm.sympify(old_val) if not hasattr(old_val, "xreplace") else old_val
                )
                new_s[var] = expr.xreplace(self._env)
                continue

            rhs = self._env[var]  # Σ(var)

            # case 1 : constant substitution  (x → 3)
            if not rhs.free_symbols & {var}:
                new_s[var] = rhs.xreplace(self._env)
                continue

            # case 2 : affine/self-referential (x → x - 1, etc.)
            sol = sm.solve(sm.Eq(rhs, old_val), var, dict=True)
            new_s[var] = sol[0][var] if sol else old_val.xreplace(self._env)

        return new_s


# --------------------------------------------------------------------------- #
# Abstract Interpretation on DataflowIR
# --------------------------------------------------------------------------- #
class AbstractInterpretation(ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc
        # set of "original" arrays of proc arguments. We need this to distinguish ArrayVars from Top
        self.avars = set()
        # set of scalar variables defined in the current scope!
        self.svars = set()
        self.sym_table = proc.sym_table
        self.fix_proc(self.proc)

    def fix_proc(self, proc: DataflowIR.proc):
        assert isinstance(proc, DataflowIR.proc)

        for a in proc.args:
            sym = sm.Symbol(a.name.__repr__())
            self.sym_table[sym] = a.name
            if len(a.hi) > 0:
                self.avars.add(sym)
            else:
                self.svars.add(sym)

        # TODO: FIXME: Do we need to use precondition assertions?

        self.fix_stmts(proc.body, proc.ctxt)

    def fix_stmts(self, stmts: list[DataflowIR.stmt], env):
        for i in range(len(stmts)):
            self.fix_stmt(stmts[i], env)

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

            gens = []
            for itr in stmt.iters + stmt.dims:
                sym = sm.Symbol(itr.__repr__())
                self.sym_table[sym] = itr
                gens.append(sym)

            lhs_name = sm.Symbol(stmt.lhs.__repr__())
            self.sym_table[lhs_name] = stmt.lhs

            a_body = self.fix_expr(body, env)
            a_orelse = self.fix_expr(stmt.orelse, env)
            env[lhs_name] = self.abs_ternary(
                gens,
                stmt.cond,
                a_body,
                a_orelse,
            )

        elif isinstance(stmt, DataflowIR.Pass):
            pass  # pass pass, lol

        elif isinstance(stmt, DataflowIR.Alloc):
            pass  # no need to do anything

        elif isinstance(stmt, DataflowIR.If):
            # Basically just pass-through

            self.fix_stmts(stmt.body, env)
            self.fix_stmts(stmt.orelse, env)

        elif isinstance(stmt, DataflowIR.For):

            itr_sym = sm.Symbol(stmt.iter.__repr__())
            self.svars.add(itr_sym)

            pre_env = dict()
            self.fix_stmts(stmt.body, env)
            for nm, val in env.items():
                pre_env[nm] = val

            count = 0
            while True:

                # fixpoint iteration
                tmp_env = pre_env.copy()
                self.fix_stmts(stmt.body, tmp_env)
                all_eq = True
                for nm, val in tmp_env.items():

                    # Don't widen if it does not depend on this loop
                    if (
                        val.iterators == [] or itr_sym != val.iterators[0]
                    ):  # should be just executed once!! recover env from the first iteration!
                        continue

                    if self.issubsetof(val, pre_env[nm]):
                        continue

                    w_res = self.abs_widening(pre_env[nm], val, count, itr_sym)

                    # if the result of the widening is None, that means we gave up so exit the loop.
                    if not w_res:
                        assert False, "widening returned None, debug"

                    all_eq = False
                    pre_env[nm] = w_res

                if all_eq:
                    break

                count += 1

            self.svars.remove(itr_sym)
            for nm, val in pre_env.items():
                env[nm] = val

        else:
            assert False, f"bad case: {type(stmt)}"

    # Corresponds to E^\# : \Expr \to \Sigma^\# \to val in the paper
    def fix_expr(self, e: DataflowIR.expr, env) -> D.abs:
        if isinstance(e, DataflowIR.Read):
            return self.abs_read(e.name, e.idx, env)

        elif isinstance(e, DataflowIR.Const):
            return self.abs_const(e.val)

        elif isinstance(e, DataflowIR.USub):
            return self.abs_usub(e.arg)

        elif isinstance(e, DataflowIR.BinOp):
            lhs = self.fix_expr(e.lhs, env)
            rhs = self.fix_expr(e.rhs, env)
            return self.abs_binop(e.op, lhs, rhs)

        elif isinstance(e, DataflowIR.Extern):
            return self.abs_extern(e, env)

        elif isinstance(e, DataflowIR.StrideExpr):
            return self.abs_stride(e.name, e.dim)

        else:
            assert False, f"bad case {type(expr)}"

    @abstractmethod
    def abs_ternary(
        self, iterators: list, cond: DataflowIR.expr, body: D.node, orelse: D.node
    ) -> D.abs:
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
    def abs_extern(self, func):
        """Approximate the extern"""

    @abstractmethod
    def abs_stride(self, name, dim):
        """Approximate the stride"""

    @abstractmethod
    def abs_widening(self, a1, a2, count):
        """Approximate the loop!"""

    @abstractmethod
    def issubsetof(self, a1, a2):
        """a1 \subsetof a2?"""
