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

from .LoopIR import Alpha_Rename, SubstArgs, LoopIR_Do

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

Changed: TypeAlias = bool


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

    def join_env(self, cur_env, updates) -> bool:
        """Mutate cur_env to incorporate updates
        Return True if something changed, False if all the same"""
        found_change = False
        for k, v in updates.items():
            if k in cur_env:
                old_v = cur_env[k]
                new_v = self.abs_join(old_v, v)
                found_change = found_change or (old_v == new_v)
                cur_env[k] = new_v
            else:
                found_change = True
                cur_env[k] = v

        return found_change

    def fix_block(self, body: DataflowIR.block) -> Changed:
        """Assumes any inputs have already been set in body.ctxts[0]"""
        assert len(body.stmts) + 1 == len(body.ctxts)

        found_change = False
        for s, pre, post in zip(body.stmts, body.ctxts[:-1], body.ctxts[1:]):
            found_change = found_change or self.fix_stmt(pre, s, post)

        return found_change

    def fix_stmt(self, pre_env, stmt: DataflowIR.stmt, post_env) -> Changed:
        found_change = False
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
            found_change = post_env.get(stmt.name, None) != rval
            post_env[stmt.name] = rval

            # propagate un-touched variables
            for nm in pre_env:
                if nm != stmt.name:
                    post_env[nm] = pre_env[nm]

        elif isinstance(stmt, DataflowIR.WriteConfig):
            rval = self.fix_expr(pre_env, stmt.rhs)
            found_change = post_env.get(stmt.config_field, None) != rval
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

            found_change = found_change or self.fix_block(stmt.body)
            found_change = found_change or self.fix_block(stmt.orelse)

            for nm in pre_env:
                bodyval = post_body[nm]
                elseval = post_else[nm]
                val = self.abs_join(bodyval, elseval)
                found_change = found_change or (post_env.get(nm, None) != val)
                post_env[nm] = val

        elif isinstance(stmt, DataflowIR.Seq):
            # Left Off: We didn't fix this part up
            env = pre_env.copy()
            env[stmt.iter] = self.abs_top(T.index)

            self.join_env(stmt.body.ctxts[0], env)

            loop_change = True
            while loop_change:
                loop_change = False
                loop_change = self.fix_block(stmt.body)
                loop_change = loop_change or self.join_env(
                    stmt.body.ctxts[0], self.body.ctxts[-1]
                )

            found_change = self.join_env(post_env, stmts.body.ctxts[-1])
        elif isinstance(stmt, DataflowIR.InlinedCall):
            pre_body, post_body = stmt.body.ctxts[0], stmt.body.ctxts[-1]
            pre_else, post_else = stmt.orelse.ctxts[0], stmt.orelse.ctxts[-1]

            for nm, val in pre_env.items():
                stmt.body.ctxts[0][nm] = val

            found_change = self.fix_block(stmt.body)

            # Left Off: Oh No, do we preserve variable names when inlining?
        else:
            assert False, f"bad case: {type(stmt)}"

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
         | ReadConfig( config config, string field )
         attributes( type type )
    """

    @abstractmethod
    def abs_init_val(self, name, typ):
        """Define initial argument values"""

    @abstractmethod
    def abs_alloc_val(self, name, typ):
        """Define initial value of an allocation"""

    @abstractmethod
    def abs_top(self, typ):
        """Return encoding of Lattice top value"""

    @abstractmethod
    def abs_join(self, lval, rval):
        """Define join in the abstract value lattice"""

    @abstractmethod
    def abs_binop(self, op, lval, rval):
        """Implement transfer function abstraction for binary operations"""

    @abstractmethod
    def abs_unsub(self, arg):
        """Implement transfer function abstraction for unary subtraction"""


AbstractDomains = ADT(
    """
module AbstractDomains {
    cprop = CTop() | CBot()
          | Const( object val, type type )
}
""",
    ext_types={
        "type": LoopIR.type,
    },
    memoize={"CTop", "CBot", "Const"},
)


class ConstantPropagation(AbstractInterpretation):
    def abs_init_val(self, name, typ):
        return Top()

    def abs_join(self, lval: Abs, rval: Abs):
        if lval == Bot():
            return rval
        elif rval == Bot():
            return lval
        elif lval == Top() or rval == Top():
            return top
        else:
            assert isinstance(lval, Const) and isinstance(rval, Const)
            if lval.val == rval.val:
                return lval
            else:
                return top

    def abs_binop(self, op, lval, rval):
        if isinstance(lval, Const) and isinstance(rval, Const):
            return Const(binop_meaning(op)(lval.val, rval.val))
        elif lval == Bot() or rval == Bot():
            return Bot()
        else:
            return Top()

        pass
