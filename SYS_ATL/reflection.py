from . import query_asts as QAST
from .LoopIR import LoopIR, T
from .prelude import *




@extclass(QAST.Call)
def __str__(self):
    return f"{self.proc}(_)"
del __str__

@extclass(QAST.Alloc)
def __str__(self):
    return f"{self.name} : _"
del __str__

@extclass(QAST.WriteConfig)
def __str__(self):
    return f"{self.config.name()}.{self.field} = {str(self.rhs)}"
del __str__

@extclass(QAST.Assign)
@extclass(QAST.Reduce)
def __str__(self):
    if len(self.idx) > 0:
        return f"{self.name}[_] = {str(self.rhs)}"
    else:
        return f"{self.name} = {str(self.rhs)}"
del __str__

@extclass(QAST.For)
def __str__(self):
    return f"for {self.name} in par(0, {str(self.hi)}):_"
del __str__

@extclass(QAST.If)
def __str__(self):
    cond = str(self.cond)
    return f"if {cond}:_"
del __str__

@extclass(QAST.Read)
def __str__(self):
    if len(self.idx) > 0:
        return f"{self.name}[_]"
    else:
        return f"{self.name}"
del __str__

@extclass(QAST.Const)
def __str__(self):
    return f"{self.val}"
del __str__

@extclass(QAST.USub)
def __str__(self):
    return f"-{str(self.arg)}"
del __str__

@extclass(QAST.BinOp)
def __str__(self):
    lhs = str(self.lhs)
    rhs = str(self.rhs)
    return f"{lhs} {self.op} {rhs}"
del __str__



# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Conversion from LoopIR AST to QAST



class LoopIR_to_QAST:
  def __init__(self, loopir_node):
    self.loopir_node  = loopir_node

    self.names        = dict()

    if isinstance(loopir_node, LoopIR.proc):
      self.qast       = self.map_proc(loopir_node)
    elif isinstance(loopir_node, list):
      if len(loopir_node) == 0:
        self.qast     = []
      elif isinstance(loopir_node[0], LoopIR.stmt):
        self.qast     = self.map_stmts(loopir_node)
      else:
        assert False, f"cannot process list of {type(loopir_node[0])}"
    elif isinstance(loopir_node, LoopIR.fnarg):
      self.qast       = self.map_fnarg(loopir_node)
    elif isinstance(loopir_node, LoopIR.stmt):
      self.qast       = self.map_stmt(loopir_node)
    elif isinstance(loopir_node, LoopIR.expr):
      self.qast       = self.map_expr(loopir_node)
    elif isinstance(loopir_node, LoopIR.type):
      self.qast       = self.map_type(loopir_node)

  def result(self):
    return self.qast

  def getname(self, sym):
    return str(sym)

  def bindname(self, sym):
    return str(sym)

  def map_proc(self, proc):
    name = proc.name
    return QAST.Proc(name, [ self.map_fnarg(fa) for fa in proc.args ],
                           [ self.map_expr(p)   for p  in proc.preds ],
                           self.map_stmts(proc.body),
                           proc.instr)

  def map_fnarg(self, fa):
    name = self.bindname(fa.name)
    return QAST.FnArg(name, self.map_type(fa.type), fa.mem)

  def map_stmts(self, body):
    return [ self.map_stmt(s) for s in body ]

  def map_stmt(self, s):
    styp = type(s)
    if styp is LoopIR.Assign or styp is LoopIR.Reduce:
      qtyp = QAST.Assign if styp is LoopIR.Assign else QAST.Reduce
      name = self.getname(s.name)
      return qtyp(name, self.map_type(s.type),
                  [ self.map_expr(i) for i in s.idx ],
                  self.map_expr(s.rhs))
    elif styp is LoopIR.WriteConfig:
      return QAST.WriteConfig(s.config, s.field, self.map_expr(s.rhs))
    elif styp is LoopIR.Pass:
      return QAST.Pass()
    elif styp is LoopIR.If:
      return QAST.If(self.map_expr(s.cond),
                     self.map_stmts(s.body),
                     self.map_stmts(s.orelse))
    elif styp is LoopIR.ForAll or styp is LoopIR.Seq:
      name    = self.bindname(s.iter)
      is_par  = styp is LoopIR.ForAll
      return QAST.For(name, QAST.Const(0,QAST.int()), self.map_expr(s.hi),
                            self.map_stmts(s.body), is_par)
    elif styp is LoopIR.Alloc:
      name = self.bindname(s.name)
      return QAST.Alloc(name, self.map_type(s.type), s.mem)
    elif styp is LoopIR.Call:
      return QAST.Call(s.f.name, [ self.map_expr(a) for a in s.args ])
    elif styp is LoopIR.WindowStmt:
      name = self.bindname(s.name)
      return QAST.WindowStmt(name, self.map_expr(s.rhs))
    else:
      assert False, f"bad case: {styp}"

  def map_expr(self, e):
    etyp = type(e)
    if etyp is LoopIR.Read:
      return QAST.Read(self.getname(e.name),
                       [ self.map_expr(i) for i in e.idx ],
                       self.map_type(e.type))
    elif etyp is LoopIR.Const:
      return QAST.Const(e.val, self.map_type(e.type))
    elif etyp is LoopIR.USub:
      return QAST.USub(self.map_expr(e.arg), self.map_type(e.type))
    elif etyp is LoopIR.BinOp:
      return QAST.BinOp(e.op, self.map_expr(e.lhs),
                              self.map_expr(e.rhs), self.map_type(e.type))
    elif etyp is LoopIR.BuiltIn:
      return QAST.BuiltIn(e.f.name(),
                          [ self.map_expr(a) for a in e.args ],
                          self.map_type(e.type))
    elif etyp is LoopIR.WindowExpr:
      name = self.getname(e.name)
      def map_w(w):
        if isinstance(w, LoopIR.Interval):
          return QAST.Interval(self.map_expr(w.lo), self.map_expr(w.hi))
        else:
          return QAST.Point(self.map_expr(w.pt))
      return QAST.WindowExpr(name,
                             [ map_w(w) for w in e.idx ],
                             self.map_type(e.type))
    elif etyp is LoopIR.StrideExpr:
      name = self.getname(e.name)
      return QAST.StrideExpr(name, e.dim,
                             self.map_type(e.type))
    elif etyp is LoopIR.ReadConfig:
      return QAST.ReadConfig(e.config, e.field,
                             self.map_type(e.type))
    else:
      assert False, f"bad case: {etyp}"

  def map_type(self, typ):
    if typ == T.R:
      return QAST.R()
    elif typ == T.f32:
      return QAST.f32()
    elif typ == T.f64:
      return QAST.f64()
    elif typ == T.i8:
      return QAST.i8()
    elif typ == T.i32:
      return QAST.i32()
    elif typ == T.bool:
      return QAST.bool()
    elif typ == T.int:
      return QAST.int()
    elif typ == T.index:
      return QAST.index()
    elif typ == T.size:
      return QAST.size()
    elif typ == T.stride:
      return QAST.stride()
    elif typ.is_tensor_or_window():
      as_tensor = typ.as_tensor if isinstance(typ, T.Window) else typ
      return QAST.tensor([ self.map_expr(e) for e in as_tensor.hi ],
                         as_tensor.is_window, self.map_type(as_tensor.type))
    else:
      assert False, f"bad case: {type(typ)}"
