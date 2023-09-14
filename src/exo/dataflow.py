# import re
from collections import OrderedDict, ChainMap
from enum import Enum
from itertools import chain
from typing import Mapping, Any

# from typing import Type

from asdl_adt import ADT, validators

from .builtins import BuiltIn
from .configs import Config
from .memory import Memory
from .prelude import Sym, SrcInfo, extclass

from .LoopIR import Alpha_Rename, SubstArgs, LoopIR_Do, Operator

# --------------------------------------------------------------------------- #
# Top Level Call
# --------------------------------------------------------------------------- #

# Probably actually make this a class (pass) so it can inherit from LoopIR_Do
def dataflow_analysis(proc: LoopIR.proc) -> DataflowIR.proc:
    # step 1 - convert LoopIR to DataflowIR
    #           with empty contexts (i.e. AbsEnvs)
    # step 2 - run abstract interpretation algorithm
    #           to populate contexts with sound values
    pass


# Big Question: How do we represent the result of dataflow analysis?

# Option 1: Take LoopIR and transform into a similar but different IR
#           which includes dataflow annotations and "desugars"/eliminates
#           constructs we don't care about for analysis purposes
#           (e.g. just inline all windowing)
#
#
#  pyAST  -->  UAST
#         --[type checking, inference, bounds checking]-->  [LoopIR]
#         --[backend checks]-->  C-code as strings
#
#  [LoopIR]  --[Primitive Transformation 42]-->  [LoopIR]
#
#   As part of Primitive Transformation:
#       [LoopIR]  --[Dataflow analysis]-->  "AST annotated w/dataflow results"
#                 --[modified new_eff_analysis]-->  AExpr/Query
#                                                   (Discharged to SMT)
#
#

"""
A Lattice is a set X equipped with
    - a partial order < relation and
    - two binary operators, join (^) and meet (v)
    - two constants, top and bottom
i.e. a Lattice is a quadruple (X, <, ^, v)

satisfying the following axioms:
    * join and meet are both commutative and associative
    * absorption: a v (a ^ b) = a,  a ^ (a v b) = a

Consequences
    * idempotency: a v a = a,   a ^ a = a
    * meet is the greatest lower bound of the two elements (aka, min)
    * join is the least upper bound of the two elements (aka, max)
    * note meet and join are not well defined in arbitrary partial orders

Examples:
    * powerset of a set forms a lattice
    * logical formulas form a lattice (meet is and, join is or)

Def. Lattice Homomorphism:
    A function f : X -> Y from one lattice to another is a homomorphism if
    *   f(a v b) = f(a) v f(b)
    *   f(a ^ b) = f(a) ^ f(b)
    *   Every lattice homomorphism is monotonic:
            i.e. if a <= b, then f(a) <= f(b)
"""


"""
P(V_1 x V_2 x ...)
I have a concrete domain of values V.
Define the "Concrete" lattice for V to be Powerset(V)
(note every powerset is a lattice, with subset as partial order,
    union as join, and intersection as meet)

An abstraction of V (eqv. Powerset(V)) is a lattice A s.t. we have
two lattice homomorphisms:
    abstraction (abs : P(V) -> A)
    concretization (conc : A -> P(V))
satisfying a property: namely that they are "adjoint" in the following sense
        abs(conc(x)) = x
        conc(abs(x)) <= x
"""

"""
This is an attempt to think something through, not an implementation plan:

A product lattice of lattices A and B is given as
    - set of values A x B
    - top = (top,top)
    - bot = (bot,bot)
    - (a0,b0) <= (a1,b1) iff. a0 <= a1 and b0 <= b1
    - (a0,b0) ^ (a1,b1) = (a0^a1, b0^b1)
    - (a0,b0) v (a1,b1) = (a0 v a1, b0 v b1)

Can we form a product abstraction?
Well, we have some V which A abstracts and some W which B abstracts?
No, we're actually interested in the case where A and B both abstract V.
So, what we are actually interested in isn't a totally arbitrary product...

Product Abstraction (on a common concrete domain V)
Given abstractions (A,absA,concA) and (B,absB,concB) both of V,
Construct a product abstraction:
    - product lattice A x B
    - absAB(X)    = ???
    - concAB(a,b) = concA(a) ^ concB(b)

conclusion 1: any (bot,_) or (_,bot) must be bot in A x B!
              i.e. we cannot simply use a product lattice
conjecture: this occurs for any (singleton) set, not just bottom

Major Conclusion:
    - Trying to form product abstractions is a bad idea!
"""

"""
Btw, suppose I have P(X) the power set of X.
Using "type theory" notation, I can also talk about the set of
all functions X -> Bool, or all predicates on X
(e.g. in Coq one would write X -> Prop)
All of these are (ignoring stupid logical foundation issues)
essentially the same.  Why?  Well, consider some S in P(X)
S is a subset of X; define f_S(x) = "x in S".
Similarly, using set comprehension and given f : X -> Bool,
define S_f = { x | f(x) }

(one more thing for fun)
X -> P(Y)  ~==  X -> Y -> Bool  ~==  X x Y -> Bool  ~==  P(X x Y)
                                                    ~==  Rel X Y
"""

"""
A program is a control-flow-graph, and btw, let's fission each basic block
into invidividual SSA statements (including unique variable names,
and phi nodes)

A program point is basically an edge between statements in this CFG
In some sense, program points are the real "states" and the statements
are transitions.

More precisely the state of my machine is
(PC (i.e. program point), stack (i.e. variable environment mapping))

Abstract States
(PC, stack but values from A instead of from V)

How do I abstract a statement (i.e. function) y = f(x)
    Well, first x is now a abstract value, so concretize it to a set of values
    Then, we know how to map each individual value with f
    This produces a set of values for y
    Then re-abstract this set
In other words...
    _y = _f(_x) = abs({ f(x) | x in conc(_x) })
That's a definition; we need to work it out for any given
    choice of language (i.e. statements/functions f_i) and
    choice of abstract lattice (i.e. A, abs, conc)

How do we abstract multiple incoming edges to some program point?
(i.e. how do we abstract phi nodes)
Answer: phi is join. done.
"""

"""
The Abstract Interpretation Algorithm:
    Propagate abstract values through statements in any order.
    This will compute a fixed-point assignment of variables to
    abstract values at every program point

    This algorithm will terminate if the abstract lattice has finite
    height.
"""

"""
Abs     = V U {top, bot}   -- only decision we've made
Conc    = P(V)
abs  : Conc -> Abs
conc : Abs -> Conc
s.t.
abs(conc(x)) = x
conc(abs(X)) >= X

What is the specific definition of `conc`?
conc(v) = {v} for v in V
conc(bot) = {}
conc(top) = V


"""


AbsEnv: TypeAlias = Mapping[Sym, Any]


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
             block   body )

    fnarg  = ( sym     name,
               type    type )

    block = ( stmt* stmts, absenv* ctxts ) -- len(stmts) + 1 == len(ctxts)

    stmt = Assign( sym name, type type, string? cast, expr* idx, expr rhs )
         | Reduce( sym name, type type, string? cast, expr* idx, expr rhs )
         | WriteConfig( sym config_field, expr rhs )
         | Pass()
         | If( expr cond, block body, block orelse )
         | Seq( sym iter, expr lo, expr hi, block body )
         | Alloc( sym name, type type )
         | InlinedCall( proc f, block body ) -- f is only there for comments

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | BuiltIn( builtin f, expr* args )
         | StrideExpr( sym name, int dim )
         | ReadConfig( sym config_field )
         attributes( type type )

}""",
    ext_types={
        "name": validators.instance_of(Identifier, convert=True),
        "sym": Sym,
        "builtin": BuiltIn,
        "config": Config,
        "binop": validators.instance_of(Operator, convert=True),
        "type": LoopIR.type,
        "absenv": validateAbsEnv,
    },
    memoize={},
)

# Option 2: Leave the input LoopIR as is, and create auxiliary datastructures
#           which allow us to "lookup" dataflow results for different variables
#           at different points in the code.
#
#           For instance, use Python dictionaries to hold the annotations


class AbstractInterpretation(collections.ABC):
    def __init__(self, proc: DataflowIR.proc):
        self.proc = proc

        # setup initial values
        init_env = self.proc.body.ctxts[0]
        for a in proc.args:
            init_env[a.name] = self.abs_init_val(a.name, a.type)

        # We probably ought to somehow use precondition assertions
        # TODO

        self.fix_block(self.proc.body)

    def fix_block(self, body: DataflowIR.block):
        """Assumes any inputs have already been set in body.ctxts[0]"""
        assert len(body.stmts) + 1 == len(body.ctxts)

        for s, pre, post in zip(body.stmts, body.ctxts[:-1], body.ctxts[1:]):
            self.fix_stmt(pre, s, post)

    def fix_stmt(self, pre_env, stmt: DataflowIR.stmt, post_env):
        if isinstance(stmt, (DataflowIR.Assign, DataflowIR.Reduce)):
            # TODO: Design approach for parameterization over idx

            # if reducing, then expand to x = x + rhs
            rhs_e = stmt.rhs
            if isinstance(stmt, DataflowIR.Reduce):
                read_buf = DataflowIR.Read(stmt.name, stmt.idx)
                rhs_e = DataflowIR.BinOp("+", read_buf, rhs_e)
            # now we can handle both cases uniformly
            rval = self.fix_expr(pre_env, rhs_e)
            # need to be careful for buffers (no overwrite guarantee)
            if len(stmt.idx) > 0:
                rval = self.abs_join(pre_env[stmt.name], rval)
            post_env[stmt.name] = rval

            # propagate un-touched variables
            for nm in pre_env:
                if nm != stmt.name:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.WriteConfig):
            rval = self.fix_expr(pre_env, stmt.rhs)
            post_env[stmt.config_field] = rval

            # propagate un-touched variables
            for nm in pre_env:
                if nm != stmt.config_field:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.Pass):
            # propagate un-touched variables
            for nm in pre_env:
                post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.Alloc):
            # TODO: Add support for parameterization over idx?

            post_env[stmt.name] = self.abs_alloc_val(stmt.name, stmt.type)

            # propagate un-touched variables
            for nm in pre_env:
                post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.If):
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

        elif isinstance(stmt, DataflowIR.Seq):
            # TODO: Add support for loop-condition analysis in some way?

            # set up the loop body for fixed-point iteration
            pre_body = stmt.body.ctxts[0]
            for nm, val in pre_env.items():
                pre_body[nm] = val
            # initialize the loop iteration variable
            lo = self.fix_expr(pre_env, stmt.lo)
            hi = self.fix_expr(pre_env, stmt.hi)
            pre_body[stmt.iter] = self.abs_iter_val(lo, hi)

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
                    val = self.abs_join(prev_val, next_val)
                    at_fixed_point = at_fixed_point and prev_val == val
                    pre_body[nm] = val

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

    def fix_expr(self, pre_env, expr: DataflowIR.expr):
        if isinstance(expr, DataflowIR.Read):
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
        elif isinstance(expr, DataflowIR.BuiltIn):
            args = [self.fix_expr(pre_env, a) for a in expr.args]
            return self.abs_builtin(expr.f, args)
        elif isinstance(expr, DataflowIR.StrideExpr):
            return self.abs_stride_expr(expr.name, expr.dim)
        elif isinstance(expr, DataflowIR.ReadConfig):
            return pre_env[expr.config_field]
        else:
            assert False, f"bad case {type(expr)}"

    """
    stmt = Assign( sym name, type type, string? cast, expr* idx, expr rhs )
         | Reduce( sym name, type type, string? cast, expr* idx, expr rhs )
         | WriteConfig( sym config_field, expr rhs )
         | Pass()
         | If( expr cond, block body, block orelse )
         | Seq( sym iter, expr lo, expr hi, block body )
         | Alloc( sym name, type type )
         | InlinedCall( proc f, block body ) -- f is only there for comments

    expr = Read( sym name, expr* idx )
         | Const( object val )
         | USub( expr arg )  -- i.e.  -(...)
         | BinOp( binop op, expr lhs, expr rhs )
         | BuiltIn( builtin f, expr* args )
         | StrideExpr( sym name, int dim )
         | ReadConfig( sym config_field )
         attributes( type type )
    """

    @abstractmethod
    def abs_init_val(self, name, typ):
        """Define initial argument values"""

    @abstractmethod
    def abs_alloc_val(self, name, typ):
        """Define initial value of an allocation"""

    @abstractmethod
    def abs_iter_val(self, lo, hi):
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


AbstractDomains = ADT(
    """
module AbstractDomains {
    cprop = CTop() | CBot()
          | Const( object val, type type )
          | CStrideExpr( sym name, int dim )
    
    iprop = ITop() | IBot()
          | Interval( int lo, int hi ) -- use for integers
}
""",
    ext_types={
        "type": LoopIR.type,
        "sym": Sym,
    },
    memoize={"CTop", "CBot", "Const", "ITop", "IBot", "Interval", "IConst"},
)
A = AbstractDomains


class ConstantPropagation(AbstractInterpretation):
    def abs_init_val(self, name, typ):
        return A.CTop()

    def abs_alloc_val(self, name, typ):
        return A.CTop()

    def abs_iter_val(self, lo, hi):
        return A.CTop()

    def abs_stride_expr(self, name, dim):
        return A.CStrideExpr(name, dim)

    def abs_const(self, val, typ):
        return A.Const(val, typ)

    def abs_join(self, lval: A.cprop, rval: A.cprop):
        if lval == A.CBot():
            return rval
        elif rval == A.CBot():
            return lval
        elif lval == A.CTop() or rval == A.CTop():
            return A.CTop()
        else:
            assert isinstance(lval, A.Const) and isinstance(rval, A.Const)
            if lval.val == rval.val:
                return lval
            else:
                return A.CTop()

    def abs_binop(self, op, lval, rval):
        if isinstance(lval, A.CBot) or isinstance(rval, A.CBot):
            return A.CBot()

        if isinstance(lval, A.Const) and isinstance(rval, A.Const):
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
                assert False, f"Bad Case Operator: {op}"
            return A.Const(val, lval.type)

        #        if op == "*":
        #            if (one val is 0):
        #                return zero_val

        if op == "/":
            # NOTE: THIS doesn't work right for integer division...
            # c1 / c2
            # 0 / x == 0
            if isinstance(lval, A.Const) and lval.val == 0:
                return lval

        return A.CTop()

        if op == "+" or op == "-":
            return A.CTop()
            # 0 + x == x
            # TOP + C(0) = abs({ x + y | x in conc(TOP), y in conc(C(0)) })
            #            = abs({ x + 0 | x in anything })
            #            = abs({ x | x in anything })
            #            = TOP
        elif op == "*":
            # x * 0 == 0
            if isinstance(lval, A.Const) and lval.val == 0:
                return lval
            elif isinstance(rval, A.Const) and rval.val == 0:
                return rval
            else:
                return A.CTop()

        else:
            return A.CTop()

    # front_ops = {"+", "-", "*", "/", "%",
    #              "<", ">", "<=", ">=", "==", "and", "or"}

    def abs_usub(self, arg):
        if isinstance(arg, A.Const):
            return A.Const(-arg.val, arg.typ)
        return arg

    def abs_builtin(self, builtin, args):
        return CTop()


class IntervalAnalysis(AbstractInterpretation):
    def abs_init_val(self, name, typ):
        return A.ITop()

    def abs_alloc_val(self, name, typ):
        return A.ITop()

    def abs_iter_val(self, lo, hi):
        if isinstance(lo, A.IBot) or isinstance(hi, A.IBot):
            return A.IBot()
        else:
            return self.abs_join(lo, hi)

    def abs_join(self, lval: A.iprop, rval: A.iprop):
        if isinstance(lval, A.ITop) or isinstance(rval, A.ITop):
            return A.ITop()
        elif isinstance(lval, A.IBot):
            return rval
        elif isinstance(rval, A.IBot):
            return lval
        else:
            assert isinstance(lval, A.Interval) and isinstance(rval, A.Interval)
            return A.Interval(min(lval.lo, rval.lo), max(lval.hi, rval.hi))
