from abc import ABC, abstractmethod
from enum import Enum
from itertools import chain
from typing import Mapping, Any
from asdl_adt import ADT, validators

from .builtins import BuiltIn
from .configs import Config
from .memory import Memory
from .prelude import Sym, SrcInfo, extclass
from .LoopIR import LoopIR, Alpha_Rename, SubstArgs, LoopIR_Do, Operator, T, Identifier
from .new_dataflow_core import *

# --------------------------------------------------------------------------- #
# Abstract Domain definition
# --------------------------------------------------------------------------- #

# { A : ( [dim_0, dim_1, dim_2] , [(constraints, tgt), (constraints, tgt)]) }

AbstractDomains = ADT(
    """
module AbstractDomains {
    mexpr = Unk()
          | Var( sym name )
          | Const( object val, type type )
          | BinOp( binop op, mexpr lhs, mexpr rhs )
          | Array( sym name, avar *dims )
    path = ( aexpr nc, aexpr sc, mexpr tgt ) -- perform weak update for now
    env   = ( avar *dims, path* paths ) -- This can handle index access uniformly!
}
""",
    ext_types={
        "type": LoopIR.type,
        "sym": Sym,
        "aexpr": A.expr,
        "avar": A.Var,
        "binop": validators.instance_of(Operator, convert=True),
    },
    memoize={},
)
D = AbstractDomains

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
               type    type,
               srcinfo srcinfo )

    block = ( stmt* stmts, absenv* ctxts ) -- len(stmts) + 1 == len(ctxts)

    stmt = Assign( sym name, type type, expr* idx, expr rhs )
         | Reduce( sym name, type type, expr* idx, expr rhs )
         | WriteConfig( sym config_field, expr rhs )
         | Pass()
         | If( expr cond, block body, block orelse )
         | For( sym iter, expr lo, expr hi, block body )
         | Alloc( sym name, type type )
         | InlinedCall( proc f, block body ) -- f is only there for comments
         attributes( srcinfo srcinfo )

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | BuiltIn( builtin f, expr* args )
         | StrideExpr( sym name, int dim )
         | ReadConfig( sym config_field )
         attributes( type type, srcinfo srcinfo )

}""",
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "builtin": BuiltIn,
        "config": Config,
        "binop": validators.instance_of(Operator, convert=True),
        "type": LoopIR.type,
        "absenv": validateAbsEnv,
        "srcinfo": SrcInfo,
    },
    memoize={},
)

from . import dataflow_pprint


# --------------------------------------------------------------------------- #
# Top Level Call to Dataflow analysis
# --------------------------------------------------------------------------- #

aexpr_false = A.Const(False, T.bool, null_srcinfo())
aexpr_true = A.Const(True, T.bool, null_srcinfo())


class LoopIR_to_DataflowIR:
    def __init__(self, proc):
        self.loopir_proc = proc
        self.dataflow_proc = self.map_proc(self.loopir_proc)

    def result(self):
        return self.dataflow_proc

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

    def map_fnarg(self, a):
        return DataflowIR.fnarg(a.name, a.type, a.srcinfo)

    def map_stmts(self, stmts):
        return self._map_list(self.map_s, stmts)

    def map_exprs(self, exprs):
        return self._map_list(self.map_e, exprs)

    def map_s(self, s):
        if isinstance(s, (LoopIR.Call, LoopIR.WindowStmt)):
            raise NotImplementedError(
                "LoopIR.Call and LoopIR.WindowStmt should be inlined when we reach here!"
            )

        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)):
            df_idx = self.map_exprs(s.idx)
            df_rhs = self.map_e(s.rhs)
            if isinstance(s, LoopIR.Assign):
                return DataflowIR.Assign(s.name, s.type, df_idx, df_rhs, s.srcinfo)
            else:
                return DataflowIR.Reduce(s.name, s.type, df_idx, df_rhs, s.srcinfo)

        elif isinstance(s, LoopIR.WriteConfig):
            # TODO: Confirm with Gilbert!
            df_config_sym = Sym(f"{config.name}_{field}")
            df_rhs = self.map_e(s.rhs)

            return DataflowIR_WriteConfig(df_config_sym, df_rhs, s.srcinfo)

        elif isinstance(s, LoopIR.If):
            df_cond = self.map_e(s.cond)
            df_body = self.map_stmts(s.body)
            df_orelse = self.map_stmts(s.orelse)

            return DataflowIR.If(
                df_cond, self.init_block(df_body), self.init_block(df_orelse), s.srcinfo
            )

        elif isinstance(s, LoopIR.For):
            df_lo = self.map_e(s.lo)
            df_hi = self.map_e(s.hi)
            df_body = self.map_stmts(s.body)

            return DataflowIR.For(
                s.iter, df_lo, df_hi, self.init_block(df_body), s.srcinfo
            )

        elif isinstance(s, LoopIR.Alloc):
            return DataflowIR.Alloc(s.name, s.type, s.srcinfo)

        elif isinstance(s, LoopIR.Pass):
            return DataflowIR.Pass(s.srcinfo)

        else:
            raise NotImplementedError(f"bad case {type(s)}")

    def map_e(self, e):
        if isinstance(e, LoopIR.Read):
            df_idx = self.map_exprs(e.idx)
            return DataflowIR.Read(e.name, df_idx, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.BinOp):
            df_lhs = self.map_e(e.lhs)
            df_rhs = self.map_e(e.rhs)
            return DataflowIR.BinOp(e.op, df_lhs, df_rhs, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.BuiltIn):
            df_args = self.map_exprs(e.args)
            return DataflowIR.BuiltIn(e.f, df_args, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.USub):
            df_arg = self.map_e(e.arg)
            return DataflowIR.USub(df_arg, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.WindowExpr):
            raise NotImplementedError("WindowExpr should not appear here!")

        elif isinstance(e, LoopIR.ReadConfig):
            # TODO: This needs to coodinate with Writeconfig
            raise NotImplementedError("Implement ReadConfig")

        elif isinstance(e, LoopIR.Const):
            return DataflowIR.Const(e.val, e.type, e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr):
            return DataflowIR.StrideExpr(e.name, e.dim, e.type, e.srcinfo)

        else:
            raise NotImplementedError(f"bad case {type(e)}")

    @staticmethod
    def _map_list(fn, nodes):
        return [fn(n) for n in nodes]


def dataflow_analysis(proc: LoopIR.proc) -> DataflowIR.proc:
    # step 1 - convert LoopIR to DataflowIR with empty contexts (i.e. AbsEnvs)
    # TODO: inline functioncall -> inline windowstmt -> lowering
    # FIXME: new_proc = inline_func(proc)
    # FIXME: new_proc = inline_windowstmt(proc)
    datair = LoopIR_to_DataflowIR(proc).result()

    # step 2 - run abstract interpretation algorithm
    #           to populate contexts with sound values
    # TODO: call constant propagation
    ConstantPropagation(datair)

    return datair


# --------------------------------------------------------------------------- #
# Abstract Interpretation on DataflowIR
# --------------------------------------------------------------------------- #


@extclass(AbstractDomains.mexpr)
def __str__(self):
    if self == D.Unk():
        return "Unknown"
    if isinstance(self, D.Const):
        return str(self.val)
    if isinstance(self, D.BinOp):
        return str(self.lhs) + str(e.op) + str(self.rhs)
    if isinstance(self, D.Array):
        dim_str = "["
        for d in self.dims:
            dim_str += str(d)
        dim_str += "]"
        return str(self.name) + dim_str

    assert False, "bad case"


@extclass(AbstractDomains.path)
def __str__(self):
    return "(" + str(self.nc) + ", " + str(self.sc) + ") : " + str(self.tgt)


@extclass(AbstractDomains.env)
def __str__(self):
    return (
        "{ ("
        + ", ".join([str(d) for d in self.dims])
        + ") , ["
        + ", ".join([str(p) for p in self.paths])
        + "]}"
    )


del __str__


# {x : ((x_0), [(0 <= x_0 < 2, 2.0), (1 <= x_0 < 3, 4.0)])}


def update(env: D.env, rval: list[D.path]):
    pre_paths = [p for p in env.paths]
    rval_paths = [p for p in rval]

    merge_paths = []
    for pre_path in env.paths:
        for rval_path in rval:
            pre_cons = pre_path.nc.simplify()
            rval_cons = rval_path.nc.simplify()

            if isinstance(pre_path.tgt, D.Unk):
                pre_paths.remove(pre_path)
            elif pre_cons == rval_cons:
                # TODO: Handle strong update
                merge_paths.append(D.path(rval_cons, rval_path.sc, rval_path.tgt))
                pre_paths.remove(pre_path)
                rval_paths.remove(rval_path)

    return D.env(env.dims, pre_paths + merge_paths + rval_paths)


def bind_cons(cons: A.expr, rval: list[D.path]):
    new_paths = []

    for path in rval:
        new_nc = A.BinOp("and", path.nc, cons, T.bool, null_srcinfo())
        new_path = D.path(new_nc.simplify(), path.sc, path.tgt)
        new_paths.append(new_path)

    return new_paths


def propagate_cons(cons: A.expr, env: D.env):
    return D.env(env.dims, bind_cons(cons, env.paths))


def ir_to_aexpr(e: DataflowIR.expr):
    if isinstance(e, DataflowIR.Const):
        ae = A.Const(e.val, e.type, null_srcinfo())
    elif isinstance(e, DataflowIR.Read):
        ae = A.Var(e.name, e.type, null_srcinfo())
    elif isinstance(e, Sym):
        ae = A.Var(e, T.index, null_srcinfo())
    else:
        assert False, f"got {e} of type {type(e)}"

    return ae


class AbstractInterpretation(ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc

        # setup initial values
        init_env = self.proc.body.ctxts[0]
        for a in proc.args:
            if a.type.is_numeric():
                init_env[a.name] = self.abs_init_val(a.name, a.type)

        # We probably ought to somehow use precondition assertions
        # TODO: leave it for now
        # { n == 16; }
        # for p in proc.preds:
        #    self.assume_pred(p, init_env)

        self.fix_block(self.proc.body)

    def fix_block(self, body: DataflowIR.block):
        """Assumes any inputs have already been set in body.ctxts[0]"""
        assert len(body.stmts) + 1 == len(body.ctxts)

        for i in range(len(body.stmts)):
            self.fix_stmt(body.ctxts[i], body.stmts[i], body.ctxts[i + 1])

    def fix_stmt(self, pre_env, stmt: DataflowIR.stmt, post_env):

        if isinstance(stmt, (DataflowIR.Assign, DataflowIR.Reduce)):
            # if reducing, then expand to x = x + rhs
            rhs_e = stmt.rhs
            if isinstance(stmt, DataflowIR.Reduce):
                read_buf = DataflowIR.Read(
                    stmt.name, stmt.idx, rhs_e.type, stmt.srcinfo
                )
                rhs_e = DataflowIR.BinOp("+", read_buf, rhs_e, rhs_e.type, stmt.srcinfo)

            # now we can handle both cases uniformly
            rval = self.fix_expr(pre_env, rhs_e)

            # Handle constraints
            cons = A.Const(True, T.bool, null_srcinfo())
            for b, e in zip(pre_env[stmt.name].dims, stmt.idx):
                eq = A.BinOp("==", b, ir_to_aexpr(e), T.bool, null_srcinfo())
                cons = A.BinOp("and", cons, eq, T.bool, null_srcinfo())

            rval = bind_cons(cons, rval)

            post_env[stmt.name] = update(pre_env[stmt.name], rval)

            # propagate un-touched variables
            for nm in pre_env:
                if nm != stmt.name:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.WriteConfig):
            rval = self.fix_expr(pre_env, stmt.rhs)

            post_env[stmt.config_field] = update(pre_env[stmt.config_field], rval)

            # propagate un-touched variables
            for nm in pre_env:
                if nm != stmt.config_field:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.Pass):
            # propagate un-touched variables
            for nm in pre_env:
                post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.Alloc):

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
            # TODO: Add support for loop-condition analysis in some way?

            pre_body = stmt.body.ctxts[0]
            iter_cons = self.abs_iter_val(
                ir_to_aexpr(stmt.iter), ir_to_aexpr(stmt.lo), ir_to_aexpr(stmt.hi)
            )
            for nm, val in pre_env.items():
                pre_body[nm] = propagate_cons(iter_cons, val)

            # Commenting out the following. We don't need to run a fixed-point

            # set up the loop body for fixed-point iteration
            # run this loop until we reach a fixed-point
            # at_fixed_point = False
            # while not at_fixed_point:
            # propagate in the loop
            #     self.fix_block(stmt.body)
            #     at_fixed_point = True
            # copy the post-values for the loop back around to
            # the pre-values, by joining them together
            #     for nm, prev_val in pre_body.items():
            #         next_val = stmt.body.ctxts[-1][nm]
            #         val = self.abs_join(prev_val, next_val)
            #         at_fixed_point = at_fixed_point and prev_val == val
            #         pre_body[nm] = val

            self.fix_block(stmt.body)

            # determine the post-env as join of pre-env and loop results
            for nm, pre_val in pre_env.items():
                loop_val = stmt.body.ctxts[-1][nm]
                post_env[nm] = self.abs_join(pre_val, loop_val)

        elif isinstance(stmt, DataflowIR.InlinedCall):
            # TODO: Decide how Inlined Calls work
            pre_body, post_body = stmt.body.ctxts[0], stmt.body.ctxts[-1]
            pre_else, post_else = stmt.orelse.ctxts[0], stmt.orelse.ctxts[-1]

            for nm, val in pre_env.items():
                stmt.body.ctxts[0][nm] = val

            self.fix_block(stmt.body)

            # Left Off: Oh No, do we preserve variable names when inlining?
        else:
            assert False, f"bad case: {type(stmt)}"

    def fix_expr(self, pre_env: D.env, expr: DataflowIR.expr) -> list[D.path]:
        if isinstance(expr, DataflowIR.Read):
            return pre_env[expr.name].paths
        elif isinstance(expr, DataflowIR.Const):
            return self.abs_const(expr.val, expr.type)
        elif isinstance(expr, DataflowIR.USub):
            arg = self.fix_expr(pre_env, expr.arg)
            return self.abs_usub(arg)
        elif isinstance(expr, DataflowIR.BinOp):
            lhs = self.fix_expr(pre_env, expr.lhs)
            rhs = self.fix_expr(pre_env, expr.rhs)
            return self.abs_binop(expr.op, lhs, rhs)

        # TODO: Fix them
        elif isinstance(expr, DataflowIR.BuiltIn):
            args = [self.fix_expr(pre_env, a) for a in expr.args]
            return self.abs_builtin(expr.f, args)
        elif isinstance(expr, DataflowIR.StrideExpr):
            return self.abs_stride_expr(expr.name, expr.dim)
        elif isinstance(expr, DataflowIR.ReadConfig):
            return pre_env[expr.name]
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


def make_empty_path(me: D.mexpr) -> D.path:
    return [D.path(aexpr_true, aexpr_false, me)]


def make_unk() -> D.path:
    return [D.path(aexpr_true, aexpr_false, D.Unk())]


def make_unk_array(buf_name: Sym, dims: list) -> D.path:
    return [D.path(aexpr_true, aexpr_false, D.Array(buf_name, dims))]


class ConstantPropagation(AbstractInterpretation):
    def abs_init_val(self, name, typ):
        if isinstance(typ, T.Tensor):
            dims = []
            for i in range(len(typ.hi)):
                dims.append(
                    A.Var(Sym(name.name() + "_" + str(i)), T.index, null_srcinfo())
                )
            return D.env(dims, make_unk_array(name, dims))
        else:
            return D.env([], make_unk())

    def abs_alloc_val(self, name, typ):
        if isinstance(typ, T.Tensor):
            dims = []
            for i in range(len(typ.hi)):
                dims.append(
                    A.Var(Sym(name.name() + "_" + str(i)), T.index, null_srcinfo())
                )
            return D.env(dims, make_unk_array(name, dims))
        else:
            return D.env([], make_unk())

    def abs_iter_val(self, name, lo, hi):
        lo_cons = A.BinOp("<=", lo, name, T.index, null_srcinfo())
        hi_cons = A.BinOp("<", name, hi, T.index, null_srcinfo())
        return AAnd(lo_cons, hi_cons)

    def abs_stride_expr(self, name, dim):
        assert False, "unimplemented"

    def abs_const(self, val, typ) -> D.path:
        # TODO: Ignore constraints for now
        return make_empty_path(D.Const(val, typ))

    def abs_join(self, lval: D.env, rval: D.env):
        for d1, d2 in zip(lval.dims, rval.dims):
            assert d1 == d2

        new_env = update(lval, rval.paths)

        return new_env

    def abs_binop(self, op, lval: list[D.path], rval: list[D.path]) -> list[D.path]:
        # TODO: Support constraints
        # TODO: FIX!!
        lval = lval[0].tgt
        rval = rval[0].tgt

        if isinstance(lval, D.Unk) or isinstance(rval, D.Unk):
            return make_empty_path(D.Unk())

        # front_ops = {"+", "-", "*", "/", "%",
        #              "<", ">", "<=", ">=", "==", "and", "or"}
        if isinstance(lval, D.Const) and isinstance(rval, D.Const):
            typ = lval.type
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

            return make_empty_path(D.Const(val, typ))

        # TODO: and, or short circuiting here

        if op == "/":
            # NOTE: THIS doesn't work right for integer division...
            # c1 / c2
            # 0 / x == 0
            if isinstance(lval, D.Const) and lval.val == 0:
                return make_empty_path(lval)

        if op == "%":
            if isinstance(rval, D.Const) and rval.val == 1:
                return make_empty_path(D.Const(0, lval.type))

        if op == "*":
            # x * 0 == 0
            if isinstance(lval, D.Const) and lval.val == 0:
                return make_empty_path(lval)
            elif isinstance(rval, D.Const) and rval.val == 0:
                return make_empty_path(rval)

        # memo
        # 0 + x == x
        # TOP + C(0) = abs({ x + y | x in conc(TOP), y in conc(C(0)) })
        #            = abs({ x + 0 | x in anything })
        #            = abs({ x | x in anything })
        #            = TOP
        return make_empty_path(D.Unk())

    def abs_usub(self, arg: D.path) -> D.path:
        arg = arg.tgt

        if isinstance(arg, D.Const):
            return make_empty_path(D.Const(-arg.val, arg.typ))
        return make_empty_path(arg)

    def abs_builtin(self, builtin, args):
        # TODO: Fix
        if any([not isinstance(a, D.Const) for a in args]):
            return D.Unk()
        vargs = [a.val for a in args]

        # TODO: write a short circuit for select builtin
        return D.Const(builtin.interpret(vargs), args[0].typ)
