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
)
from .internal_analysis import *

# --------------------------------------------------------------------------- #
# Abstract Domain definition
# --------------------------------------------------------------------------- #

AbstractDomains = ADT(
    """
module AbstractDomains {
    dexpr = Top()
          | Var( sym name, type type )
          | Const( object val, type type )
          | Or( dexpr lhs, dexpr rhs, type type )
          | USub( dexpr arg, type type )
}
""",
    ext_types={
        "type": LoopIR.type,
        "sym": Sym,
        "binop": validators.instance_of(Operator, convert=True),
    },
    memoize={},
)
D = AbstractDomains


@extclass(AbstractDomains.dexpr)
def __str__(self):
    if isinstance(self, D.Top):
        return "⊤"
    elif isinstance(self, D.Const):
        return str(self.val)
    elif isinstance(self, D.Var):
        return str(self.name)
    elif isinstance(self, D.Or):
        return str(self.lhs) + " ∨ " + str(self.rhs)
    elif isinstance(self, D.USub):
        return "-" + str(self.arg)

    assert False, "bad case"


del __str__


def lift_dexpr(e, key=None):
    if isinstance(e, D.Var):
        op = A.Var(e.name, e.type, null_srcinfo())
    elif isinstance(e, D.Const):
        op = A.Const(e.val, e.type, null_srcinfo())
    elif isinstance(e, D.Or):
        return A.BinOp(
            "or",
            lift_dexpr(e.lhs, key=key),
            lift_dexpr(e.rhs, key=key),
            e.type,
            null_srcinfo(),
        )
    elif isinstance(e, D.USub):
        op = A.USub(lift_dexpr(e.arg), e.type, null_srcinfo())
    else:
        assert isinstance(e, D.Top)
        op = A.Const(True, T.bool, null_srcinfo())

    if key:
        return A.BinOp("==", key, op, T.bool, null_srcinfo())
    else:
        return op


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

    stmt = Assign( sym lhs, sym* dims, expr cond, expr body, expr orelse )
         | Reduce( sym lhs, sym* dims, expr cond, expr body, expr orelse )
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
# Top Level Call to Dataflow analysis
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

    def tensor_to_dims(self, tensor):
        # | Tensor( expr* hi, bool is_window, type type )
        assert isinstance(tensor, T.Tensor)
        assert tensor.is_window == False
        assert not tensor.type.is_tensor_or_window()

        return [self.map_e(hi) for hi in tensor.hi], self.map_t(tensor.type)

    def map_fnarg(self, a):
        if a.type.is_indexable() or a.type.is_bool():
            return DataflowIR.fnarg(a.name, [], self.map_t(a.type), a.srcinfo)
        else:
            # data values so renaming is necessary
            name = a.name.copy()
            dims = []
            dsyms = []
            if isinstance(a.type, T.Tensor):
                dims, _ = self.tensor_to_dims(a.type)
                dsyms = [Sym("d" + str(d)) for d in range(len(a.type.hi))]
            typ = self.map_t(a.type.basetype())
            self.env[a.name] = (name, dsyms, typ)
            return DataflowIR.fnarg(name, dims, typ, a.srcinfo)

    def map_stmts(self, stmts):
        return self._map_list(self.map_s, stmts)

    def map_exprs(self, exprs):
        return self._map_list(self.map_e, exprs)

    def sym_to_read(self, syms, typ):
        return [DataflowIR.Read(s, [], typ, null_srcinfo()) for s in syms]

    def map_s(self, s):
        if isinstance(s, LoopIR.Call):
            assert False, "Call statement should have been got rid of at this point"

        elif isinstance(s, LoopIR.WindowStmt):
            assert False, "Window statement should have been got rid of at this point"

        elif isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            new_name = s.name.copy()
            dims = self.env[s.name][1]
            cond = DataflowIR.Const(True, DataflowIR.Bool(), null_srcinfo())
            for d, e in zip(reversed(dims), s.idx):
                typ = self.map_t(e.type)
                c = DataflowIR.BinOp(
                    "==",
                    DataflowIR.Read(d, [], typ, null_srcinfo()),
                    self.map_e(e),
                    DataflowIR.Bool(),
                    null_srcinfo(),
                )
                cond = DataflowIR.BinOp(
                    "and", c, cond, DataflowIR.Bool(), null_srcinfo()
                )

            body = self.map_e(s.rhs)
            orelse = DataflowIR.Read(
                self.env[s.name][0],
                self.sym_to_read(self.env[s.name][1], self.env[s.name][2]),
                self.env[s.name][2],
                null_srcinfo(),
            )
            self.env[s.name] = (new_name, dims, self.env[s.name][2])

            if isinstance(s, LoopIR.Assign):
                return [
                    DataflowIR.Assign(new_name, dims, cond, body, orelse, s.srcinfo)
                ]
            else:
                return [
                    DataflowIR.Reduce(new_name, dims, cond, body, orelse, s.srcinfo)
                ]

        elif isinstance(s, LoopIR.WriteConfig):
            ir_config = s.config._INTERNAL_sym(s.field)
            df_config = ir_config.copy()
            df_rhs = self.map_e(s.rhs)
            self.env[ir_config] = (df_config, [], df_rhs.type)

            return [
                DataflowIR.Assign(
                    df_config,
                    [],
                    DataflowIR.Const(True, DataflowIR.Bool(), null_srcinfo()),
                    df_rhs,
                    df_rhs,
                    s.srcinfo,
                )
            ]

        elif isinstance(s, LoopIR.If):
            # | If( expr cond, block body, block orelse )

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
                bbody = DataflowIR.Read(
                    bval[0], self.sym_to_read(bval[1], bval[2]), bval[2], null_srcinfo()
                )
                # TODO: not quite right, because orelse branch might not exist and/or value might not be updated. In that case we need to pull values from before the if.
                obody = DataflowIR.Read(
                    oval[0], self.sym_to_read(oval[1], oval[2]), oval[2], null_srcinfo()
                )
                stmt = DataflowIR.Assign(
                    post_name, vals[1], cond, bbody, obody, null_srcinfo()
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
            # loopir :  For( sym iter, expr lo, expr hi, stmt* body, loop_mode loop_mode )
            # datair : For( sym iter, expr lo, expr hi, block body )

            # We first need the merge node before the body, but the problem is that we don't know what the last node is until doing the body
            # We can be safe and merge every values in the environment

            # Update self.env and prev
            prev = {}
            pre_sym = {}
            for key, vals in self.env.items():
                self.env[key] = (vals[0].copy(), [s.iter] + vals[1], vals[2])
                pre_sym[key] = self.env[key]
                prev[key] = vals

            # Body
            body = self.map_stmts(s.body)

            # Construct merge nodes for both pre and post loop merges
            pre_loop = []
            post_loop = []
            iter_read = DataflowIR.Read(s.iter, [], DataflowIR.Index(), null_srcinfo())
            for key, llast in self.env.items():
                ltop = pre_sym[key]
                lbefore = prev[key]

                # Pre loop
                pre_cond = DataflowIR.BinOp(
                    "==", iter_read, mk_const(0), DataflowIR.Bool(), null_srcinfo()
                )
                pre_body = DataflowIR.Read(
                    lbefore[0],
                    self.sym_to_read(lbefore[1], lbefore[2]),
                    lbefore[2],
                    null_srcinfo(),
                )
                last_idx = [
                    DataflowIR.BinOp(
                        "-", iter_read, mk_const(1), DataflowIR.Bool(), null_srcinfo()
                    )
                ]
                pre_orelse = DataflowIR.Read(
                    llast[0],
                    last_idx + self.sym_to_read(llast[1][1:], llast[2]),
                    llast[2],
                    null_srcinfo(),
                )
                pre_stmt = DataflowIR.Assign(
                    ltop[0], ltop[1], pre_cond, pre_body, pre_orelse, null_srcinfo()
                )
                pre_loop.append(pre_stmt)

                # Post loop
                post_name = llast[0].copy()
                post_cond = DataflowIR.BinOp(
                    ">", iter_read, self.map_e(s.lo), DataflowIR.Bool(), null_srcinfo()
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
                    post_idxs + self.sym_to_read(llast[1][1:], llast[2]),
                    llast[2],
                    null_srcinfo(),
                )
                post_orelse = DataflowIR.Read(
                    lbefore[0],
                    self.sym_to_read(lbefore[1], lbefore[2]),
                    lbefore[2],
                    null_srcinfo(),
                )
                post_stmt = DataflowIR.Assign(
                    post_name,
                    llast[1][1:],
                    post_cond,
                    post_body,
                    post_orelse,
                    null_srcinfo(),
                )
                post_loop.append(post_stmt)

                # update env
                self.env[key] = (post_name, llast[1][1:], llast[2])

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
            assert isinstance(s.type, T.Tensor)
            assert s.type.is_window == False
            assert s.type.type.is_real_scalar()
            # loopir: Alloc( sym name, type type, mem mem )
            # datair: Alloc( sym name, expr* hi, type type )

            name = s.name.copy()
            his, _ = self.tensor_to_dims(s.type)
            dsyms = [Sym("d" + str(d)) for d in range(len(s.type.hi))]
            typ = self.map_t(s.type.basetype())
            self.env[s.name] = (name, dsyms, typ)

            return DataflowIR.Alloc(name, his, typ, s.srcinfo)

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
                diff = len(ee[1]) - len(e.idx)
                df_idx = self.sym_to_read(ee[1][:diff], ee[2]) + self.map_exprs(e.idx)
                assert len(ee[1]) == len(df_idx)

                return DataflowIR.Read(ee[0], df_idx, self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.ReadConfig):
            df_config = e.config._INTERNAL_sym(e.field)
            assert df_config in self.env
            ee = self.env[df_config]
            idx = self.sym_to_read(ee[1][: len(ee[1])], ee[2])
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
                self.env[e.name], e.dim, self.map_t(e.type), e.srcinfo
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
    # ScalarPropagation(datair)

    return datair, stmts


# --------------------------------------------------------------------------- #
# Helper functions
# --------------------------------------------------------------------------- #


class DataflowIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc = proc

        self.do_proc(self.proc)

    def do_proc(self, p):
        for a in p.args:
            self.do_t(a.type)
        for pred in p.preds:
            self.do_e(pred)

        self.do_block(p.body)

    def do_block(self, block):
        self.do_stmts(block.stmts)
        self.do_ctxts(block.ctxts)

    def do_ctxts(self, ctxts):
        for c in ctxts:
            self.do_c(c)

    def do_c(self, c):
        pass

    def do_stmts(self, stmts):
        for s in stmts:
            self.do_s(s)

    def do_s(self, s):
        styp = type(s)
        if styp is DataflowIR.Assign or styp is DataflowIR.Reduce:
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
            self.do_t(s.type)
        elif styp is DataflowIR.WriteConfig:
            self.do_e(s.rhs)
        elif styp is DataflowIR.If:
            self.do_e(s.cond)
            self.do_block(s.body)
            self.do_block(s.orelse)
        elif styp is DataflowIR.For:
            self.do_e(s.lo)
            self.do_e(s.hi)
            self.do_block(s.body)
        elif styp is DataflowIR.Call:
            self.do_proc(s.f)
            for e in s.args:
                self.do_e(e)
        elif styp is DataflowIR.WindowStmt:
            self.do_e(s.rhs)
        elif styp is DataflowIR.Alloc:
            self.do_t(s.type)
        else:
            pass

    def do_e(self, e):
        etyp = type(e)
        if etyp is DataflowIR.Read:
            for e in e.idx:
                self.do_e(e)
        elif etyp is DataflowIR.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        elif etyp is DataflowIR.BuiltIn:
            for a in e.args:
                self.do_e(a)
        elif etyp is DataflowIR.USub:
            self.do_e(e.arg)
        else:
            pass

        self.do_t(e.type)

    def do_t(self, t):
        if isinstance(t, T.Tensor):
            for i in t.hi:
                self.do_e(i)
        elif isinstance(t, T.Window):
            self.do_t(t.src_type)
            self.do_t(t.as_tensor)
        else:
            pass


class GetReadConfigs(DataflowIR_Do):
    def __init__(self):
        self.readconfigs = []

    def do_e(self, e):
        if isinstance(e, DataflowIR.ReadConfig):
            self.readconfigs.append((e.config_field, e.type))
        super().do_e(e)


def get_readconfigs(stmts):
    gr = GetReadConfigs()
    for stmt in stmts:
        gr.do_s(stmt)
    return gr.readconfigs


class GetWriteConfigs(DataflowIR_Do):
    def __init__(self):
        self.writeconfigs = []

    def do_s(self, s):
        if isinstance(s, DataflowIR.WriteConfig):
            # FIXME!!! Propagate a proper type after adding type to writeconfig
            self.writeconfigs.append((s.config_field, T.int))

        super().do_s(s)

    # early exit
    def do_e(self, e):
        return


def get_writeconfigs(stmts):
    gw = GetWriteConfigs()
    gw.do_stmts(stmts)
    return gw.writeconfigs


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
# Abstract Interpretation on DataflowIR
# --------------------------------------------------------------------------- #


class AbstractInterpretation(ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc

        self.fix_proc(self.proc)

    def fix_proc(self, proc: DataflowIR.proc):
        assert isinstance(proc, DataflowIR.proc)

        # TODO: Do we need to use precondition assertions?
        # Like assertion checks??
        # I guess that depends on how we're using assertions in the downstream new_eff.py. leave it for now.

        # setup initial values
        init_env = proc.body.ctxts[0]
        for a in proc.args:
            if not a.type.is_tensor_or_window():
                init_env[a.name] = self.abs_init_val(a.name, a.type)

        # Initialize all the configuration states used in this proc
        configs = get_readconfigs(self.proc.body.stmts) + get_writeconfigs(
            self.proc.body.stmts
        )
        for c, typ in configs:
            init_env[c] = self.abs_init_val(c, typ)

        self.fix_block(proc.body)

    def fix_block(self, body: DataflowIR.block):
        """Assumes any inputs have already been set in body.ctxts[0]"""
        assert len(body.stmts) + 1 == len(body.ctxts)

        for i in range(len(body.stmts)):
            self.fix_stmt(body.ctxts[i], body.stmts[i], body.ctxts[i + 1])

    def fix_stmt(self, pre_env, stmt: DataflowIR.stmt, post_env):
        if isinstance(stmt, (DataflowIR.Assign, DataflowIR.Reduce)):
            # Ignore buffers with len(idx) > 0
            if len(stmt.idx) == 0:
                # if reducing, then expand to x = x + rhs
                rhs_e = stmt.rhs
                if isinstance(stmt, DataflowIR.Reduce):
                    read_buf = DataflowIR.Read(
                        stmt.name, stmt.idx, stmt.type, stmt.srcinfo
                    )
                    rhs_e = DataflowIR.BinOp(
                        "+", read_buf, rhs_e, rhs_e.type, rhs_e.srcinfo
                    )
                # now we can handle both cases uniformly
                rval = self.fix_expr(pre_env, rhs_e)
                post_env[stmt.name] = rval

                # propagate un-touched variables
                for nm in pre_env:
                    if nm != stmt.name:
                        post_env[nm] = pre_env[nm]
            else:
                # propagate un-touched variables
                for nm in pre_env:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.WriteConfig):
            rval = self.fix_expr(pre_env, stmt.rhs)
            post_env[stmt.config_field] = rval

            # propagate un-touched variables
            for nm in pre_env:
                if nm != stmt.config_field:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, (DataflowIR.Pass, DataflowIR.WindowStmt)):
            # propagate un-touched variables
            for nm in pre_env:
                post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.Alloc):

            if not stmt.type.is_tensor_or_window():
                post_env[stmt.name] = self.abs_alloc_val(stmt.name, stmt.type)

            # propagate un-touched variables
            for nm in pre_env:
                post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.If):
            # TODO: Handle constraints!!
            # TODO: Add support for path-dependency in analysis
            # TODO: Add support for "I know cond is true!"
            pre_body, post_body = stmt.body.ctxts[0], stmt.body.ctxts[-1]
            pre_else, post_else = stmt.orelse.ctxts[0], stmt.orelse.ctxts[-1]

            for nm, val in pre_env.items():
                pre_body[nm] = val
                pre_else[nm] = val

            self.fix_block(stmt.body)
            self.fix_block(stmt.orelse)

            for nm in pre_env:
                bodyval = post_body[nm]
                elseval = post_else[nm]
                val = self.abs_join(bodyval, elseval)
                post_env[nm] = val

        elif isinstance(stmt, DataflowIR.For):
            # set up the loop body for fixed-point iteration
            pre_body = stmt.body.ctxts[0]
            for nm, val in pre_env.items():
                pre_body[nm] = val

            # initialize the loop iteration variable
            lo = self.fix_expr(pre_env, stmt.lo)
            hi = self.fix_expr(pre_env, stmt.hi)
            pre_body[stmt.iter] = self.abs_iter_val(stmt.iter, lo, hi)

            # run this loop until we reach a fixed-point
            at_fixed_point = False
            while not at_fixed_point:
                # propagate in the loop
                self.fix_block(stmt.body)
                at_fixed_point = True
                # copy the post-values for the loop back around to
                # the pre-values, by joining them together
                for nm, prev_val in pre_body.items():
                    next_val = stmt.body.ctxts[-1][nm]
                    # SANITY-CHECK: Is this correct?
                    at_fixed_point = at_fixed_point and greater_than(prev_val, next_val)
                    pre_body[nm] = self.abs_join(prev_val, next_val)

            # determine the post-env as join of pre-env and loop results
            for nm, pre_val in pre_env.items():
                loop_val = stmt.body.ctxts[-1][nm]
                post_env[nm] = self.abs_join(pre_val, loop_val)

        elif isinstance(stmt, DataflowIR.Call):
            subst = dict()
            for sig, arg in zip(stmt.f.args, stmt.args):
                if isinstance(arg, DataflowIR.ReadConfig):
                    subst[arg.config_field] = sig.name
                elif isinstance(arg, DataflowIR.Read):
                    subst[arg.name] = sig.name

            # Run fixpoint on this subprocedure call
            self.fix_proc(stmt.f)

            # abs_join the values in the callee _if_ the name is binded to the sub-procedure call
            for nm, pre_val in pre_env.items():
                subproc_nm = subst[nm] if nm in subst else nm
                if not subproc_nm in stmt.f.body.ctxts[-1]:
                    post_env[nm] = pre_val
                else:
                    subproc_val = stmt.f.body.ctxts[-1][subproc_nm]
                    post_env[nm] = self.abs_join(pre_val, subproc_val)

        else:
            assert False, f"bad case: {type(stmt)}"

    def fix_expr(self, pre_env: D, expr: DataflowIR.expr) -> D:
        if isinstance(expr, DataflowIR.Read):
            if len(expr.idx) > 0:
                return D.Top()

            return pre_env[expr.name]
        elif isinstance(expr, DataflowIR.Const):
            return self.abs_const(expr.val, expr.type)
        elif isinstance(expr, DataflowIR.USub):
            arg = self.fix_expr(pre_env, expr.arg)
            return self.abs_usub(arg)
        elif isinstance(expr, DataflowIR.BinOp):
            lhs = self.fix_expr(pre_env, expr.lhs)
            rhs = self.fix_expr(pre_env, expr.rhs)
            return self.abs_binop(expr.op, lhs, rhs)

        elif isinstance(expr, DataflowIR.ReadConfig):
            return pre_env[expr.config_field]

        # TODO: Fix them
        elif isinstance(expr, DataflowIR.BuiltIn):
            args = [self.fix_expr(pre_env, a) for a in expr.args]
            return self.abs_builtin(expr.f, args)
        elif isinstance(expr, DataflowIR.StrideExpr):
            return self.abs_stride_expr(expr.name, expr.dim)
        elif isinstance(expr, DataflowIR.WindowExpr):
            raise NotImplementedError("windowexpr??")
        else:
            assert False, f"bad case {type(expr)}"

    @abstractmethod
    def abs_init_val(self, name, typ):
        """Define initial argument values"""

    @abstractmethod
    def abs_alloc_val(self, name, typ):
        """Define initial value of an allocation"""

    @abstractmethod
    def abs_iter_val(self, name, lo, hi):
        """Define value of an iteration variable"""

    @abstractmethod
    def abs_stride_expr(self, name, dim):
        """Define abstraction of a specific stride expression"""

    @abstractmethod
    def abs_const(self, val, typ):
        """Define abstraction of a specific constant value"""

    @abstractmethod
    def abs_join(self, lval, rval):
        """Define join in the abstract value lattice"""

    @abstractmethod
    def abs_binop(self, op, lval, rval):
        """Implement transfer function abstraction for binary operations"""

    @abstractmethod
    def abs_usub(self, arg):
        """Implement transfer function abstraction for unary subtraction"""

    @abstractmethod
    def abs_builtin(self, builtin, args):
        """Implement transfer function abstraction for built-ins"""


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
    def abs_init_val(self, name, typ):
        return D.Var(name, typ)

    def abs_alloc_val(self, name, typ):
        return D.Var(name, typ)

    def abs_iter_val(self, name, lo, hi):
        # TODO: shouldn't we be able to range the iteration range/
        # lo_cons = A.BinOp("<=", lo, name, T.index, null_srcinfo())
        # hi_cons = A.BinOp("<", name, hi, T.index, null_srcinfo())
        # return AAnd(lo_cons, hi_cons)
        return D.Var(name, T.index)

    def abs_stride_expr(self, name, dim):
        return D.Var(Sym(name.name() + str(dim)), T.stride)

    def abs_const(self, val, typ) -> D:
        return D.Const(val, typ)

    def abs_join(self, lval: D, rval: D):

        if isinstance(lval, D.Top):
            return rval
        elif isinstance(rval, D.Top):
            return lval
        elif lval == rval:
            return lval
        elif isinstance(lval, D.Var) and isinstance(rval, D.Var):
            if lval.name == rval.name:
                return lval
        elif isinstance(lval, D.Const) and isinstance(rval, D.Const):
            if lval.val == rval.val:
                return lval
        elif isinstance(lval, D.Or):
            if greater_than(lval, rval):
                return lval
        elif isinstance(rval, D.Or):
            if greater_than(rval, lval):
                return rval

        typ = lval.type
        if lval.type != rval.type:
            if lval.type.is_real_scalar() and rval.type.is_real_scalar():
                typ = T.R
            else:
                # TODO: Decide whether or not to typecheck here
                typ = T.R
                # assert (
                #    False
                # ), f"type should be the same to join? {lval.type} and {rval.type}"

        return D.Or(lval, rval, typ)

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

        # memo
        # 0 + x == x
        # TOP + C(0) = abs({ x + y | x in conc(TOP), y in conc(C(0)) })
        #            = abs({ x + 0 | x in anything })
        #            = abs({ x | x in anything })
        #            = TOP
        return D.Top()

    def abs_usub(self, arg: D) -> D:
        if isinstance(arg, D.Top):
            return D.Top()
        return D.USub(arg, arg.type)

    def abs_builtin(self, builtin, args):
        # TODO: We should not attempt to do precise analysis on builtins
        return D.Top()
