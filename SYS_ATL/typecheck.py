from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR import UAST, LoopIR, front_ops, bin_ops
from .LoopIR import T

from .memory import *

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# This pass converts from UAST to LoopIR
#   In the process, different kinds of expressions are identified and
#   checked for proper usage.
#
#   Non-Goals:
#     - The UAST already has symbol conversion sorted out, so we
#       don't need to do that or to capture additional Python values
#       due to metaprogramming etc.
#     - We're not going to check bounds correctness here
#     - We're not going to check for parallelism-induced race-conditions here
#

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Helper functions

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# The typechecker


class TypeChecker:
    def __init__(self, proc):
        self.uast_proc = proc
        self.env = Environment()
        self.errors = []

        args = []
        for a in proc.args:
            typ = self.check_t(a.type, a.srcinfo)
            self.env[a.name] = typ
            mem = a.mem
            if mem is None:
                mem = DRAM
            args.append(LoopIR.fnarg(a.name, typ, mem, a.srcinfo))

        preds = []
        for p in proc.preds:
            pred = self.check_e(p)
            if pred.type != T.err and pred.type != T.bool:
                self.err(pred, f"expected a bool expression")
            preds.append(pred)

        body = self.check_stmts(proc.body)

        if not proc.name:
            self.err(proc, "expected all procedures to be named")

        self.loopir_proc = LoopIR.proc(name =proc.name or "anon",
                                       args =args,
                                       preds=preds,
                                       body =body,
                                       instr=proc.instr,
                                       eff  =None,
                                       srcinfo=proc.srcinfo)

        # do error checking here
        if len(self.errors) > 0:
            raise TypeError("Errors occurred during typechecking:\n" +
                            "\n".join(self.errors))

    def get_loopir(self):
        return self.loopir_proc

    def err(self, node, msg):
        self.errors.append(f"{node.srcinfo}: {msg}")

    def check_stmts(self, body):
        assert len(body) > 0
        stmts = []
        for s in body:
            stmts.append( self.check_single_stmt(s) )
        return stmts

    def check_access(self, node, nm, idx, lvalue=False):
        # check indexing
        idx = [ self.check_e(i) for i in idx ]
        for i in idx:
            if i.type != T.err and not i.type.is_indexable():
                self.err(i, f"cannot index with expression of type '{i.type}'")

        # check compatibility with buffer type
        typ     = self.env[nm]
        if typ is T.err:
            pass
        elif typ.is_numeric():
            if len(typ.shape()) < len(idx):
                self.err(node, f"expected access of variable " +
                               f"'{nm}' of type '{typ}' to have " +
                               f"no more than {len(typ.shape())} " +
                               f"indices, but {len(idx)} were found.")
                typ = T.err
            elif lvalue and len(typ.shape()) != len(idx):
                self.err(node, f"expected lvalue access of variable " +
                               f"'{nm}' of type '{typ}' to have " +
                               f"exactly {len(typ.shape())} " +
                               f"indices, but {len(idx)} were found.")
                typ = T.err
            else:
                for _ in idx:
                    typ = typ.type
        elif lvalue:
            self.err(node, f"cannot assign/reduce to '{nm}', " +
                           f"a non-numeric variable of type '{typ}'")
            typ = T.err

        return idx, typ

    def check_single_stmt(self, stmt):
        if type(stmt) is UAST.Assign or type(stmt) is UAST.Reduce:
            idx, typ = self.check_access(stmt, stmt.name, stmt.idx,
                                         lvalue=True)
            assert (typ.is_real_scalar() or typ is T.err)
            rhs     = self.check_e(stmt.rhs)
            if rhs.type != T.err and not rhs.type.is_real_scalar():
                self.err(rhs, f"cannot assign/reduce a "+
                              f"'{rhs.type}' type value")

            IRnode  = (LoopIR.Assign if type(stmt) is UAST.Assign else
                       LoopIR.Reduce)
            return IRnode(stmt.name, typ, None, idx, rhs, None, stmt.srcinfo)

        elif type(stmt) is UAST.Pass:
            return LoopIR.Pass(None, stmt.srcinfo)

        elif type(stmt) is UAST.If:
            cond    = self.check_e(stmt.cond)
            if cond.type != T.err and cond.type != T.bool:
                self.err(cond, f"expected a bool expression")
            body    = self.check_stmts(stmt.body)
            ebody   = []
            if len(stmt.orelse) > 0:
                ebody   = self.check_stmts(stmt.orelse)
            return LoopIR.If(cond, body, ebody, None, stmt.srcinfo)

        elif type(stmt) is UAST.ForAll:
            self.env[stmt.iter] = T.index

            # handle standard ParRanges
            parerr = ("currently only supporting for-loops of the form:\n" +
                      "  for _ in par(0, affine_expression):")

            if (type(stmt.cond) is not UAST.ParRange or
                    type(stmt.cond.lo) is not UAST.Const or
                    stmt.cond.lo.val != 0):
                self.err(stmt.cond, parerr)

            hi = self.check_e(stmt.cond.hi)
            if hi.type != T.err and not hi.type.is_sizeable():
                self.err(hi, "expected loop bound of type 'int' or "
                             "type 'size'")

            body = self.check_stmts(stmt.body)
            return LoopIR.ForAll(stmt.iter, hi, body, None, stmt.srcinfo)

        elif type(stmt) is UAST.Alloc:
            typ = self.check_t(stmt.type, stmt.srcinfo)
            self.env[stmt.name] = typ
            mem = stmt.mem
            if mem is None:
                mem = DRAM
            return LoopIR.Alloc(stmt.name, typ, mem, None, stmt.srcinfo)

        elif type(stmt) is UAST.Call:
            args    = [ self.check_e(a) for a in stmt.args ]

            # because procedures have dependently-typed signatures,
            # we need to re-map size types as we type-check
            size_map = {}
            for call_a,sig_a in zip(args, stmt.f.args):
                is_err = True
                if call_a.type == T.err:
                    pass
                elif sig_a.type is T.size:
                    if call_a.type == T.size:
                        if type(call_a) is not LoopIR.Read:
                            self.err(call_a, "expected size arguments to be "+
                                             "simply variables or constants "+
                                             "for now")
                        else:
                            is_err = False
                            size_map[sig_a.name] = call_a.name
                    elif call_a.type == T.int:
                        if type(call_a) is not LoopIR.Const:
                            self.err(call_a, "expected size arguments to be "+
                                             "simply variables or constants "+
                                             "for now")
                        else:
                            is_err = False
                            size_map[sig_a.name] = call_a.val
                    else:
                        self.err(call_a, "expected argument of 'size' or "+
                                         "'int' type, but got argument of "+
                                        f"type '{call_a.type}'")

                elif sig_a.type is T.index:
                    if not call_a.type.is_indexable():
                        self.err(call_a, "expected index-type expression, "+
                                         f"but got type {call_a.type}")
                    else:
                        is_err = False

                elif sig_a.type.is_numeric():
                    if len(call_a.type.shape()) != len(sig_a.type.shape()):
                        self.err(call_a,
                                 f"expected argument of type '{sig_a.type}', "
                                 f"but got '{call_a.type}'")

                    #sig_type = sig_a.type.subst(size_map)
                    #if call_a.type != sig_type:
                    #    self.err(call_a,
                    #             f"expected argument of type '{sig_type}'")

                    # ensure scalars are simply variable names
                    elif call_a.type.is_real_scalar():
                        if (type(call_a) is not LoopIR.Read or
                            len(call_a.idx) != 0):
                            self.err(call_a, "expected scalar arguments "+
                                             "to be simply variable names "+
                                             "for now")
                        else:
                            is_err = False
                    else:
                        is_err = False

                else: assert False, "bad argument type case"

                if is_err:
                    return LoopIR.Pass(None, stmt.srcinfo)

            # if no errors were hit, then we get to here
            return LoopIR.Call(stmt.f, args, None, stmt.srcinfo)

        else:
            assert False, "not a loopir in check_stmts"

    def check_e(self, e):
        if type(e) is UAST.Read:
            idx, typ = self.check_access(e, e.name, e.idx, lvalue=False)
            return LoopIR.Read(e.name, idx, typ, e.srcinfo)

        elif type(e) is UAST.Const:
            # TODO: What should be the default const type?
            if type(e.val) is float:
                return LoopIR.Const(e.val, T.R, e.srcinfo)
            elif type(e.val) is int:
                return LoopIR.Const(e.val, T.int, e.srcinfo)
            else:
            # We currently don't allow constant bool type
                self.err(e, f"literal of unexpected type '{type(e.val)}' "
                            f"and value: {e.val}")
                return LoopIR.Const(0, T.err, e.srcinfo)

        elif type(e) is UAST.USub:
            arg = self.check_e(e.arg)
            if arg.type.is_real_scalar() or arg.type.is_indexable():
                neg1 = -1.0 if arg.type.is_real_scalar() else -1
                return LoopIR.BinOp("*",
                                    LoopIR.Const(neg1, arg.type, e.srcinfo),
                                    arg,
                                    arg.type, e.srcinfo)
            elif arg.type != T.err:
                self.err(e, f"cannot negate expression of type '{arg.type}'")
            return LoopIR.Const(0, T.err, e.srcinfo)

        elif type(e) is UAST.BinOp:
            lhs = self.check_e(e.lhs)
            rhs = self.check_e(e.rhs)
            typ = T.err
            if lhs.type == T.err or rhs.type == T.err:
                typ = T.err
            elif e.op == "and" or e.op == "or":
                if lhs.type is not T.bool:
                    self.err(lhs, "expected 'bool' argument to logical op")
                if rhs.type is not T.bool:
                    self.err(rhs, "expected 'bool' argument to logical op")
                typ = T.bool
            elif e.op == "==" and lhs.type == T.bool and rhs.type == T.bool:
                self.err(e, "using \"==\" for boolean not supported. Use "+
                            "\"and\" instead")
                typ = T.err
            elif (e.op == "<" or e.op == "<=" or e.op == "==" or
                  e.op == ">" or e.op == ">="):
                if not lhs.type.is_indexable():
                    self.err(lhs, f"expected 'index' or 'size' argument to "+
                                  f"comparison op: {e.op}")
                if not rhs.type.is_indexable():
                    self.err(rhs, f"expected 'index' or 'size' argument to "+
                                  f"comparison op: {e.op}")
                typ = T.bool
            elif (e.op == "+" or e.op == "-" or e.op == "*" or
                  e.op == "/" or e.op == "%"):
                if lhs.type.is_real_scalar():
                    if not rhs.type.is_real_scalar():
                        self.err(rhs, "expected scalar type")
                        typ = T.err
                    elif e.op == "%":
                        self.err(e, "cannot compute modulus of 'R' values")
                        typ = T.err
                    else:
                        if lhs.type == T.R:
                            typ = T.R
                        elif lhs.type == T.f32:
                            typ = T.f32
                        elif lhs.type == T.f64:
                            typ = T.f64
                        elif lhs.type == T.int8:
                            typ = T.int8
                elif rhs.type.is_real_scalar():
                    self.err(lhs, "expected scalar type")
                elif lhs.type == T.bool or rhs.type == T.bool:
                    self.err(lhs, "cannot perform arithmetic on 'bool' values")
                    typ = T.err
                elif type(lhs.type) is T.Tensor or type(rhs.type) is T.Tensor:
                    self.err(lhs, "cannot perform arithmetic on tensors")
                    typ = T.err
                else:
                    assert lhs.type.is_indexable()
                    assert rhs.type.is_indexable()
                    if e.op == "/" or e.op == "%":
                        if rhs.type != T.int or type(rhs) is not LoopIR.Const:
                            self.err(rhs, "cannot divide or modulo by a "+
                                          "non-constant value")
                            typ = T.err
                        typ = lhs.type
                    elif e.op == "*":
                        if lhs.type == T.int:
                            typ = rhs.type
                        elif rhs.type == T.int:
                            typ = lhs.type
                        else:
                            self.err(e, "cannot multiply two non-constant "+
                                        "indexing/sizing expressions, since "+
                                        "the result would be non-affine")
                            typ = T.err
                    else: # + or -
                        if lhs.type == T.index or rhs.type == T.index:
                            typ = T.index
                        elif lhs.type == T.size or rhs.type == T.size:
                            typ = T.size
                        else:
                            typ = T.int

            else: assert False, f"bad op: '{e.op}'"

            return LoopIR.BinOp(e.op, lhs, rhs, typ, e.srcinfo)

        elif type(e) is UAST.ParRange:
            assert False, ("parser should not place ParRange anywhere "+
                           "outside of a for-loop condition")
        else:
            assert False, "not a LoopIR in check_e"

    def check_t(self, typ, srcinfo):
        if type(typ) is UAST.Num:
            return T.R
        elif type(typ) is UAST.F32:
            return T.f32
        elif type(typ) is UAST.F64:
            return T.f64
        elif type(typ) is UAST.INT8:
            return T.int8
        elif type(typ) is UAST.Bool:
            return T.bool
        elif type(typ) is UAST.Int:
            return T.int
        elif type(typ) is UAST.Size:
            return T.size
        elif type(typ) is UAST.Index:
            return T.index
        elif type(typ) is UAST.Tensor:
            if is_pos_int(typ.hi):
                hi = LoopIR.Const(typ.hi, T.int, srcinfo)
            else:
                hi = LoopIR.Read(typ.hi, [], T.size, srcinfo)
            return T.Tensor(hi, self.check_t(typ.type, srcinfo))
        else:
            assert False, "bad case"
