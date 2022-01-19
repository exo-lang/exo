from collections import ChainMap

from .LoopIR_effects import Effects as E
from .asts import UAST, PAST, LoopIR

front_ops = {'+', '-', '*', '/', '%', '<', '>', '<=', '>=', '==', 'and', 'or'}

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Untyped AST

UAST = UAST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Pattern AST

PAST = PAST

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Loop IR

LoopIR = LoopIR


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Types

class T:
    Num = LoopIR.Num
    F32 = LoopIR.F32
    F64 = LoopIR.F64
    INT8 = LoopIR.INT8
    INT32 = LoopIR.INT32
    Bool = LoopIR.Bool
    Int = LoopIR.Int
    Index = LoopIR.Index
    Size = LoopIR.Size
    Stride = LoopIR.Stride
    Error = LoopIR.Error
    Tensor = LoopIR.Tensor
    Window = LoopIR.WindowType
    R = Num()
    f32 = F32()
    int8 = INT8()
    i8 = INT8()
    int32 = INT32()
    i32 = INT32()
    f64 = F64()
    bool = Bool()  # note: accessed as T.bool outside this module
    int = Int()
    index = Index()
    size = Size()
    stride = Stride()
    err = Error()

    def is_type(self):
        return isinstance(self, LoopIR.type)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #

# convert from LoopIR.expr to E.expr
def lift_to_eff_expr(e):
    if isinstance(e, LoopIR.Read):
        assert len(e.idx) == 0
        return E.Var(e.name, e.type, e.srcinfo)
    elif isinstance(e, LoopIR.Const):
        return E.Const(e.val, e.type, e.srcinfo)
    elif isinstance(e, LoopIR.BinOp):
        return E.BinOp(e.op,
                       lift_to_eff_expr(e.lhs),
                       lift_to_eff_expr(e.rhs),
                       e.type, e.srcinfo)
    elif isinstance(e, LoopIR.USub):
        return E.BinOp('-',
                       E.Const(0, e.type, e.srcinfo),
                       lift_to_eff_expr(e.arg),
                       e.type, e.srcinfo)
    elif isinstance(e, LoopIR.StrideExpr):
        return E.Stride(e.name, e.dim, e.type, e.srcinfo)
    elif isinstance(e, LoopIR.ReadConfig):
        return E.ConfigField(e.config, e.field,
                             e.config.lookup(e.field)[1], e.srcinfo)

    else:
        assert False, "bad case, e is " + str(type(e))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Standard Pass Templates for Loop IR


class LoopIR_Rewrite:
    def __init__(self, proc, instr=None, *args, **kwargs):
        self.orig_proc = proc

        args = [self.map_fnarg(a) for a in self.orig_proc.args]
        preds = [self.map_e(p) for p in self.orig_proc.preds]
        preds = [p for p in preds
                 if not (isinstance(p, LoopIR.Const) and p.val)]
        body = self.map_stmts(self.orig_proc.body)

        eff = self.map_eff(self.orig_proc.eff)

        self.proc = LoopIR.proc(name=self.orig_proc.name,
                                args=args,
                                preds=preds,
                                body=body,
                                instr=instr,
                                eff=eff,
                                srcinfo=self.orig_proc.srcinfo)

    def result(self):
        return self.proc

    def map_fnarg(self, a):
        return LoopIR.fnarg(a.name, self.map_t(a.type), a.mem, a.srcinfo)

    def map_stmts(self, stmts):
        return [s for b in stmts
                for s in self.map_s(b)]

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            return [styp(s.name, self.map_t(s.type), s.cast,
                         [self.map_e(a) for a in s.idx],
                         self.map_e(s.rhs), self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.WriteConfig:
            return [LoopIR.WriteConfig(s.config, s.field, self.map_e(s.rhs),
                                       self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.WindowStmt:
            return [LoopIR.WindowStmt(s.lhs, self.map_e(s.rhs),
                                      self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.If:
            return [LoopIR.If(self.map_e(s.cond), self.map_stmts(s.body),
                              self.map_stmts(s.orelse),
                              self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.ForAll:
            return [LoopIR.ForAll(s.iter, self.map_e(s.hi),
                                  self.map_stmts(s.body),
                                  self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.Seq:
            return [LoopIR.Seq(s.iter, self.map_e(s.hi),
                               self.map_stmts(s.body),
                               self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.Call:
            return [LoopIR.Call(s.f, [self.map_e(a) for a in s.args],
                                self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.Alloc:
            return [LoopIR.Alloc(s.name, self.map_t(s.type), s.mem,
                                 self.map_eff(s.eff), s.srcinfo)]
        else:
            return [s]

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            return LoopIR.Read(e.name, [self.map_e(a) for a in e.idx],
                               self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.BinOp:
            return LoopIR.BinOp(e.op, self.map_e(e.lhs), self.map_e(e.rhs),
                                self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.BuiltIn:
            return LoopIR.BuiltIn(e.f, [self.map_e(a) for a in e.args],
                                  self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.USub:
            return LoopIR.USub(self.map_e(e.arg), self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.WindowExpr:
            return LoopIR.WindowExpr(e.name,
                                     [self.map_w_access(w) for w in e.idx],
                                     self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.ReadConfig:
            return LoopIR.ReadConfig(e.config, e.field,
                                     self.map_t(e.type), e.srcinfo)
        else:
            # constant case cannot have variable-size tensor type
            # stride expr case has stride type
            return e

    def map_w_access(self, w):
        if isinstance(w, LoopIR.Interval):
            return LoopIR.Interval(self.map_e(w.lo),
                                   self.map_e(w.hi), w.srcinfo)
        else:
            return LoopIR.Point(self.map_e(w.pt), w.srcinfo)

    def map_t(self, t):
        ttyp = type(t)
        if ttyp == T.Tensor:
            return T.Tensor([self.map_e(r) for r in t.hi],
                            t.is_window, self.map_t(t.type))
        elif ttyp == T.Window:
            return T.Window(self.map_t(t.src_type), self.map_t(t.as_tensor),
                            t.src_buf,
                            [self.map_w_access(w) for w in t.idx])
        else:
            return t

    def map_eff(self, eff):
        if eff is None:
            return eff
        return E.effect([self.map_eff_es(es) for es in eff.reads],
                        [self.map_eff_es(es) for es in eff.writes],
                        [self.map_eff_es(es) for es in eff.reduces],
                        [self.map_eff_ce(ce) for ce in eff.config_reads],
                        [self.map_eff_ce(ce) for ce in eff.config_writes],
                        eff.srcinfo)

    def map_eff_es(self, es):
        return E.effset(es.buffer,
                        [self.map_eff_e(i) for i in es.loc],
                        es.names,
                        self.map_eff_e(es.pred) if es.pred else None,
                        es.srcinfo)

    def map_eff_ce(self, ce):
        return E.config_eff(ce.config,
                            ce.field,
                            self.map_eff_e(ce.value) if ce.value else None,
                            self.map_eff_e(ce.pred) if ce.pred else None,
                            ce.srcinfo)

    def map_eff_e(self, e):
        if isinstance(e, E.BinOp):
            return E.BinOp(e.op, self.map_eff_e(e.lhs),
                           self.map_eff_e(e.rhs), e.type, e.srcinfo)
        else:
            return e


class LoopIR_Do:
    def __init__(self, proc, *args, **kwargs):
        self.proc = proc

        for a in self.proc.args:
            self.do_t(a.type)
        for p in self.proc.preds:
            self.do_e(p)

        self.do_stmts(self.proc.body)

    def do_stmts(self, stmts):
        for s in stmts:
            self.do_s(s)

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            for e in s.idx:
                self.do_e(e)
            self.do_e(s.rhs)
            self.do_t(s.type)
        elif styp is LoopIR.WriteConfig:
            self.do_e(s.rhs)
        elif styp is LoopIR.WindowStmt:
            self.do_e(s.rhs)
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            self.do_e(s.hi)
            self.do_stmts(s.body)
        elif styp is LoopIR.Call:
            for e in s.args:
                self.do_e(e)
        elif styp is LoopIR.Alloc:
            self.do_t(s.type)
        else:
            pass

        self.do_eff(s.eff)

    def do_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            for e in e.idx:
                self.do_e(e)
        elif etyp is LoopIR.BinOp:
            self.do_e(e.lhs)
            self.do_e(e.rhs)
        elif etyp is LoopIR.BuiltIn:
            for a in e.args:
                self.do_e(a)
        elif etyp is LoopIR.USub:
            self.do_e(e.arg)
        elif etyp is LoopIR.WindowExpr:
            for w in e.idx:
                self.do_w_access(w)
        else:
            pass

        self.do_t(e.type)

    def do_w_access(self, w):
        if isinstance(w, LoopIR.Interval):
            self.do_e(w.lo)
            self.do_e(w.hi)
        elif isinstance(w, LoopIR.Point):
            self.do_e(w.pt)
        else:
            assert False, "bad case"

    def do_t(self, t):
        if isinstance(t, T.Tensor):
            for i in t.hi:
                self.do_e(i)
        elif isinstance(t, T.Window):
            self.do_t(t.src_type)
            self.do_t(t.as_tensor)
            for w in t.idx:
                self.do_w_access(w)
        else:
            pass

    def do_eff(self, eff):
        if eff is None:
            return
        for es in eff.reads:
            self.do_eff_es(es)
        for es in eff.writes:
            self.do_eff_es(es)
        for es in eff.reduces:
            self.do_eff_es(es)

    def do_eff_es(self, es):
        for i in es.loc:
            self.do_eff_e(i)
        if es.pred:
            self.do_eff_e(es.pred)

    def do_eff_e(self, e):
        if isinstance(e, E.BinOp):
            self.do_eff_e(e.lhs)
            self.do_eff_e(e.rhs)


class FreeVars(LoopIR_Do):
    def __init__(self, node):
        assert isinstance(node, list)
        self.env = ChainMap()
        self.fv = set()

        for n in node:
            if isinstance(n, LoopIR.stmt):
                self.do_s(n)
            elif isinstance(n, LoopIR.expr):
                self.do_e(n)
            elif isinstance(n, E.effect):
                self.do_eff(n)
            else:
                assert False, "expected stmt, expr, or effect"

    def result(self):
        return self.fv

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def do_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name not in self.env:
                self.fv.add(s.name)
        elif styp is LoopIR.WindowStmt:
            self.env[s.lhs] = True
        elif styp is LoopIR.If:
            self.do_e(s.cond)
            self.push()
            self.do_stmts(s.body)
            self.do_stmts(s.orelse)
            self.pop()
            self.do_eff(s.eff)
            return
        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            self.do_e(s.hi)
            self.push()
            self.env[s.iter] = True
            self.do_stmts(s.body)
            self.pop()
            self.do_eff(s.eff)
            return
        elif styp is LoopIR.Alloc:
            self.env[s.name] = True

        super().do_s(s)

    def do_e(self, e):
        etyp = type(e)
        if (etyp is LoopIR.Read or
                etyp is LoopIR.WindowExpr or
                etyp is LoopIR.StrideExpr):
            if e.name not in self.env:
                self.fv.add(e.name)

        super().do_e(e)

    def do_t(self, t):
        if isinstance(t, T.Window):
            if t.src_buf not in self.env:
                self.fv.add(t.src_buf)

        super().do_t(t)

    def do_eff_es(self, es):
        if es.buffer not in self.env:
            self.fv.add(es.buffer)

        self.push()
        for x in es.names:
            self.env[x] = True

        super().do_eff_es(es)
        self.pop()

    def do_eff_e(self, e):
        if isinstance(e, E.Var) and e.name not in self.env:
            self.fv.add(e.name)

        super().do_eff_e(e)


class Alpha_Rename(LoopIR_Rewrite):
    def __init__(self, node):
        self.env = ChainMap()
        self.node = []

        if isinstance(node, LoopIR.proc):
            self.node = self.map_proc(node)
        else:
            assert isinstance(node, list)
            for n in node:
                if isinstance(n, LoopIR.stmt):
                    self.node += self.map_s(n)
                elif isinstance(n, LoopIR.expr):
                    self.node += [self.map_e(n)]
                elif isinstance(n, E.effect):
                    self.node += [self.map_eff(n)]
                else:
                    assert False, "expected stmt or expr or effect"

    def result(self):
        return self.node

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def map_proc(self, proc):
        args = [self.map_fnarg(fa) for fa in proc.args]
        preds = [self.map_e(e) for e in proc.preds]
        body = self.map_stmts(proc.body)
        eff = self.map_eff(proc.eff)

        return LoopIR.proc(proc.name, args, preds, body,
                           proc.instr, eff, proc.srcinfo)

    def map_fnarg(self, fa):
        nm = fa.name.copy()
        self.env[fa.name] = nm
        typ = self.map_t(fa.type)
        return LoopIR.fnarg(nm, typ, fa.mem, fa.srcinfo)

    def map_s(self, s):
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            nm = self.env[s.name] if s.name in self.env else s.name
            return [styp(nm, self.map_t(s.type), s.cast,
                         [self.map_e(a) for a in s.idx],
                         self.map_e(s.rhs), self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.WindowStmt:
            rhs = self.map_e(s.rhs)
            lhs = s.lhs.copy()
            self.env[s.lhs] = lhs
            return [LoopIR.WindowStmt(lhs, rhs,
                                      self.map_eff(s.eff), s.srcinfo)]
        elif styp is LoopIR.If:
            self.push()
            stmts = super().map_s(s)
            self.pop()
            return stmts

        elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
            hi = self.map_e(s.hi)
            eff = self.map_eff(s.eff)
            self.push()
            itr = s.iter.copy()
            self.env[s.iter] = itr
            stmts = [styp(itr, hi, self.map_stmts(s.body),
                          eff, s.srcinfo)]
            self.pop()
            return stmts

        elif styp is LoopIR.Alloc:
            nm = s.name.copy()
            self.env[s.name] = nm
            return [LoopIR.Alloc(nm, self.map_t(s.type), s.mem,
                                 self.map_eff(s.eff), s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        etyp = type(e)
        if etyp is LoopIR.Read:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.Read(nm, [self.map_e(a) for a in e.idx],
                               self.map_t(e.type), e.srcinfo)
        elif etyp is LoopIR.WindowExpr:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.WindowExpr(nm,
                                     [self.map_w_access(a) for a in e.idx],
                                     self.map_t(e.type), e.srcinfo)

        elif etyp is LoopIR.StrideExpr:
            nm = self.env[e.name] if e.name in self.env else e.name
            return LoopIR.StrideExpr(nm, e.dim, e.type, e.srcinfo)

        return super().map_e(e)

    def map_eff_es(self, es):
        self.push()
        names = [nm.copy() for nm in es.names]
        for orig, new in zip(es.names, names):
            self.env[orig] = new

        buf = self.env[es.buffer] if es.buffer in self.env else es.buffer
        eset = E.effset(buf,
                        [self.map_eff_e(i) for i in es.loc],
                        names,
                        self.map_eff_e(es.pred) if es.pred else None,
                        es.srcinfo)
        self.pop()
        return eset

    def map_eff_e(self, e):
        if isinstance(e, E.Var):
            nm = self.env[e.name] if e.name in self.env else e.name
            return E.Var(nm, e.type, e.srcinfo)

        return super().map_eff_e(e)

    def map_t(self, t):
        ttyp = type(t)
        if ttyp == T.Window:
            src_buf = t.src_buf
            if t.src_buf in self.env:
                src_buf = self.env[t.src_buf]

            return T.Window(self.map_t(t.src_type), self.map_t(t.as_tensor),
                            src_buf,
                            [self.map_w_access(w) for w in t.idx])

        return super().map_t(t)


class SubstArgs(LoopIR_Rewrite):
    def __init__(self, nodes, binding):
        assert isinstance(nodes, list)
        assert isinstance(binding, dict)
        assert all(isinstance(v, LoopIR.expr) for v in binding.values())
        assert not any(
            isinstance(v, LoopIR.WindowExpr) for v in binding.values())
        self.env = binding
        self.nodes = []
        for n in nodes:
            if isinstance(n, LoopIR.stmt):
                self.nodes += self.map_s(n)
            elif isinstance(n, LoopIR.expr):
                self.nodes += [self.map_e(n)]
            else:
                assert False, "expected stmt or expr"

    def result(self):
        return self.nodes

    def map_s(self, s):
        # this substitution could refer to a read or a window expression
        styp = type(s)
        if styp is LoopIR.Assign or styp is LoopIR.Reduce:
            if s.name in self.env:
                e = self.env[s.name]
                assert isinstance(e, LoopIR.Read) and len(e.idx) == 0
                return [styp(e.name, self.map_t(s.type), s.cast,
                             [self.map_e(a) for a in s.idx],
                             self.map_e(s.rhs), s.eff, s.srcinfo)]

        return super().map_s(s)

    def map_e(self, e):
        # this substitution could refer to a read or a window expression
        if isinstance(e, LoopIR.Read):
            if e.name in self.env:
                if len(e.idx) == 0:
                    return self.env[e.name]
                else:
                    sub_e = self.env[e.name]
                    assert (isinstance(sub_e, LoopIR.Read) and
                            len(sub_e.idx) == 0)
                    return LoopIR.Read(sub_e.name,
                                       [self.map_e(a) for a in e.idx],
                                       e.type, e.srcinfo)
        elif isinstance(e, LoopIR.WindowExpr):
            if e.name in self.env:
                if len(e.idx) == 0:
                    return self.env[e.name]
                else:
                    sub_e = self.env[e.name]
                    assert (isinstance(sub_e, LoopIR.Read) and len(
                        sub_e.idx) == 0)
                    return LoopIR.WindowExpr(sub_e.name,
                                             [self.map_w_access(a) for a in
                                              e.idx],
                                             self.map_t(e.type), e.srcinfo)

        elif isinstance(e, LoopIR.StrideExpr):
            if e.name in self.env:
                sub_e = self.env[e.name]
                return LoopIR.StrideExpr(sub_e.name, e.dim, e.type, e.srcinfo)

        return super().map_e(e)

    def map_eff_es(self, es):
        # this substitution could refer to a read or a window expression
        new_es = super().map_eff_es(es)
        if es.buffer in self.env:
            sub_e = self.env[es.buffer]
            assert isinstance(sub_e, LoopIR.Read) and len(sub_e.idx) == 0
            new_es.buffer = sub_e.name
        return new_es

    def map_eff_e(self, e):
        if isinstance(e, E.Var):
            if e.name in self.env:
                if e.type.is_indexable():
                    sub_e = self.env[e.name]
                    assert sub_e.type.is_indexable()
                    return lift_to_eff_expr(sub_e)
                else:  # Could be config value (e.g. f32)
                    sub_e = self.env[e.name]
                    return lift_to_eff_expr(sub_e)

        return super().map_eff_e(e)

    def map_t(self, t):
        ttyp = type(t)
        if ttyp == T.Window:
            src_buf = t.src_buf
            if t.src_buf in self.env:
                src_buf = self.env[t.src_buf].name

            return T.Window(self.map_t(t.src_type), self.map_t(t.as_tensor),
                            src_buf,
                            [self.map_w_access(w) for w in t.idx])

        return super().map_t(t)
