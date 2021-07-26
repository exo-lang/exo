from .asdl.adt import ADT
from .asdl.adt import memo as ADTmemo

from .prelude import *
from .LoopIR import LoopIR, T, LoopIR_Rewrite

from collections import ChainMap
from functools import reduce

import pysmt
from pysmt import shortcuts as SMT

def _get_smt_solver():
    factory = pysmt.factory.Factory(pysmt.shortcuts.get_env())
    slvs    = factory.all_solvers()
    if len(slvs) == 0: raise OSError("Could not find any SMT solvers")
    return pysmt.shortcuts.Solver(name=next(iter(slvs)))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #



class DoReplace(LoopIR_Rewrite):
    def __init__(self, proc, subproc, stmt_block):
        # ensure that subproc and stmt_block match in # of statements
        n_stmts = len(subproc.body)
        if len(stmt_block) < n_stmts:
            raise SchedulingError("Not enough statements to match")
        stmt_block = stmt_block[:n_stmts]

        self.subproc        = subproc
        self.target_block   = stmt_block

        super().__init__(proc)

    def map_stmts(self, stmts):
        # see if we can find the target block in this block
        n_stmts = len(self.target_block)
        match_i = None
        for i,s in enumerate(stmts):
            if s == self.target_block[0]:
                if stmts[i:i+n_stmts] == self.target_block:
                    match_i = i
                    break

        if match_i is None:
            return super().map_stmts(stmts)
        else: # process the match
            res = Unification(self.subproc, stmts).result()

            new_args = [self.sym_to_expr(s, stmt.srcinfo) for s in res]
            new_call = LoopIR.Call(self.subproc, new_args, None, stmt.srcinfo)

            return stmts[ : match_i] + [new_call] + stmts[match_i+n_stmts : ]

    # make this more efficient by not rewriting
    # most of the sub-trees
    def map_e(self,e):
        return e
    def map_t(self,t):
        return t
    def map_eff(self,eff):
        return eff


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unification "System of Equations" grammar

# the problem designates disjoint subsets of the variables used
# as "holes" (variables to solve for) and "knowns" (variables to express
# a solution as an affine combination of).  Any variable not in either of
# those lists is unknown but not permissible in a solution expression.
UEq = ADT("""
module UEq {
    problem = ( sym*    holes,  -- symbols the solution is requested for
                sym*    knowns, -- symbols allowed in solution expressions
                pred*   preds   -- conj of equations
              )

    pred    = Conj( pred* preds )
            | Disj( pred* preds )
            | Cases( sym case_var, case* cases )
            | Eq( expr lhs, expr rhs )

    case    = ( string label, pred pred )
    
    -- affine expressions
    expr  =  Const(int val)
          |  Var( sym name )
          |  Add( expr lhs, expr rhs )
          |  Scale( int coeff, expr e )

} """, {
    'sym':          lambda x: type(x) is Sym,
})


# -------------------------------------- #
# Conversion to Strings for Debug

def _str_uexpr(e, prec=0):
    etyp = type(e)
    if etyp is UEq.Const:
        return str(e.val)
    elif etyp is UEq.Var:
        return str(e.name)
    elif etyp is UEq.Add:
        s = f"{_str_uexpr(e.lhs,0)} + {_str_uexpr(e.rhs,1)}"
        if prec > 0:
            s = f"({s})"
        return s
    elif etyp is UEq.Scale:
        return f"{e.coeff}*{_str_uexpr(e.e,10)}"
    else: assert False, "bad case"

@extclass(UEq.Const)
@extclass(UEq.Var)
@extclass(UEq.Add)
@extclass(UEq.Scale)
def __str__(self):
    return _str_uexpr(self)
del __str__

@extclass(UEq.case)
def __str__(self):
    return f"{self.label}: {self.pred}"
del __str__

def _str_upred(p, prec=0):
    ptyp = type(p)
    if ptyp is UEq.Eq:
        return f"{p.lhs} == {p.rhs}"
    elif ptyp is UEq.Conj or ptyp is UEq.Disj:
        op  = ' and ' if UEq.Conj else ' or '
        s   = op.join([ _str_upred(pp,1) for pp in p.preds ])
        if prec > 0:
            s = f"({s})"
        return s
    elif ptyp is UEq.Cases:
        return (f'cases({p.case_var}) '+
                ' | '.join([ f"({case})" for case in p.cases ]))
    else: assert False, "bad case"

@extclass(UEq.Conj)
@extclass(UEq.Disj)
@extclass(UEq.Eq)
def __str__(self):
    return _str_upred(self)
del __str__

@extclass(UEq.problem)
def __str__(prob):
    lines = [ "Holes:   "+', '.join([ str(x) for x in prob.holes ]) ,
              "Knowns:  "+', '.join([ str(x) for x in prob.knowns ]) ]
    lines += [ str(p) for p in prob.preds ]
    return '\n'.join(lines)
del __str__


# -------------------------------------- #
# How to solve this system of equations

@extclass(UEq.problem)
def solve(prob):
    solver      = _get_smt_solver()

    known_list  = prob.knowns
    known_idx   = { k : i for i,k in enumerate(known_list) }
    Nk          = len(known_list)

    var_set     = dict()
    case_set    = dict()
    def get_var(x):
        if x in var_set:
            return var_set[x]
        else:
            vec = ([ SMT.Symbol(f"{repr(x)}_{repr(k)}", SMT.INT)
                     for k in known_list ] +
                   [ SMT.Symbol(f"{repr(x)}_const", SMT.INT) ])
            var_set[x] = vec
            return vec
    def get_case(x):
        if x not in case_set:
            case_set[x] = SMT.Symbol(f"{repr(x)}", SMT.INT)
        return case_set[x]
    # initialize all hole variables, ensuring they are defined
    for x in prob.holes:
        get_var(x)

    def lower_e(e):
        if type(e) is UEq.Const:
            return ([SMT.Int(0)] * Nk) + [SMT.Int(e.val)]
        elif type(e) is UEq.Var:
            if e.name in known_idx:
                one_hot = [SMT.Int(0)] * (Nk+1)
                one_hot[known_idx[e.name]] = SMT.Int(1)
                return one_hot
            else:
                return get_var(e.name)
        elif type(e) is UEq.Add:
            lhs = lower_e(e.lhs)
            rhs = lower_e(e.rhs)
            return [ SMT.Plus(x,y) for x,y in zip(lhs,rhs) ]
        elif type(e) is UEq.Scale:
            arg = lower_e(e.e)
            return [ SMT.Times(e.coeff, a) for a in arg ]
        else: assert False, "bad case"

    def lower_p(p):
        if type(p) is UEq.Eq:
            lhs = lower_e(p.lhs)
            rhs = lower_e(p.rhs)
            return SMT.And(*[ SMT.Equals(x,y) for x,y in zip(lhs,rhs) ])
        elif type(p) is UEq.Conj:
            return SMT.And(*[ lower_p(pp) for pp in p.preds ])
        elif type(p) is UEq.Disj:
            return SMT.Or(*[ lower_p(pp) for pp in p.preds ])
        elif type(p) is UEq.Cases:
            case_var = get_case(p.case_var)
            def per_case(i,c):
                pp = lower_p(c.pred)
                is_case = SMT.Equals(case_var, SMT.Int(i))
                return SMT.Implies(is_case, pp)
            return SMT.Or(*[ per_case(i,c) for i,c in enumerate(p.cases) ])
        else: assert False, "bad case"

    prob_pred   = SMT.And(*[ lower_p(p) for p in prob.preds ])
    if not solver.is_sat(prob_pred):
        return None
    else:
        solutions = dict()
        for x in prob.holes:
            x_syms  = get_var(x)
            x_vals  = solver.get_py_values(x_syms)
            expr    = None
            for x,v in zip(known_list, x_vals):
                v = int(v)
                if v == 0:
                    continue
                elif v == 1:
                    term = UEq.Var(x)
                else:
                    term = UEq.Scale(v, UEq.Var(x))

                expr = term if expr is None else UEq.Add(expr, term)

            # constant offset
            off     = UEq.Const(int(x_vals[-1]))
            expr    = off if expr is None else UEq.Add(expr, off)

            solutions[x] = expr

        # report on case decisions
        for x in case_set:
            val = solver.get_py_value(case_set[x])
            solutions[x] = int(val)

        return solutions



#   def foo(n : size, x : R[n]):
#       for i in par(0,n):
#           x[i] = 0.0
#
#
#   for j in par(0,m):
#       y[j,3] = 0.0
#
#   ===
#
#       foo(?n, ?x)
#
#   foo(m,y[j,:])
#
#
# Description of Naively normalized system (in disjunctive normal form)
#
#       dnf_sys     = ( dnf_conj* conjuncts ) -- i.e.   c0 \/ c1 \/ ...
#       dnf_conj    = ( dnf_eq* eqs )         -- i.e.  eq0 /\ eq1 /\ ...
#       dnf_eq      = ( expr lhs, expr rhs )
#
# How to solve the DNF system
#   for each `dnf_conj`:
#       try to solve the conjunction of linear equations.
#       if success, then success overall
#       otherwise, continue and try the next `dnf_conj`
#
# How to solve a conjunction of linear equations
#   translate into a matrix thing Ax = b
#       specificially the matrix S = [A,b]
#   "solve using an integer-only version of gaussian elimination"
#       i.e. compute row-echelon-form of S; call it S'
#   yielding A'x = b'  where S' = [A',b']
#
#   Suppose x = [ ?x ; x_sym ]
#   and A' has the form after reduction
#       [ ?A , A_sym ]*[ ?x ; x_sym ] = b'
#   Suppose we were able to normalize so that ?A = Id (maybe more 0s?)
#                                                     (maybe no entry for
#                                                      some of the ?x holes)
#   If there are extra rows entirely 0 within ?A, but not within A_sym,
#   then we're saying that there must necessarily hold certain equations
#   between the x_sym values.  Does this mean that there are no solutions
#   i.e. that we are overconstrained, because for instance, we are saying
#   that we want to unify i == j  where i and j are both free variables
#   in the original code. (hence not necessarily true) Or more egregiously
#   we could have an equation row like 0 === 1, which is obviously not
#   satisfiable/true in any sense.
#
#   Now suppose something more like an ideal ?A = Id situation.
#   Then in this case, each row can be read out as the sought-after
#   substitution defining some hole in ?x.
#
#   In this case, we still have the problem that that solution could use
#   some bound variable that is not in scope at the proposed call-site.
#   In this case, we don't really have a solution.  So somehow, we need
#   to set up the system of equations s.t. we never get such solutions
#   or s.t. if there exists a solution without such out-of-scope variables
#   (i.e. a good solution) then we will find that "good solution" first.
#   Put another way, if we find a "bad solution" hopefully we've constructed
#   the problem so that finding that bad solution implies that there are no
#   good solutions.


# Gilbert's sympy solving an integer linear system notes
#
# from sympy import matrices as spmat
# from sympy.core.numbers import ilcm
#
# # how to declare a matrix
# A = spmat.Matrix(...)
#
# # how to perform row-reduction on a matrix
# # rref = Row Reduced Echelon Form
# R, pivots   = A.rref()

# Make separate equation editing paths
# 1. Initial construction
#      Address aliases (Alloc, windowStmt)
#        Substitute the rest of body by this new symbol?
# 2. Address Read and Assign/Reduce, and WindowExpr
#    lhs.name == rhs.name
#    Indexing expressions
#    i. subproc arg is a Tensor
#       body's index === args's index
#       call arg is just arg
#    ii.type is window :
#    x[0, 1, j+1] === ?src[j]
#    x[0, 1, 1:]
#       1. if the index is bounded
#           ---> Interval
# 3. Linear system solver
#
#       Generate placeholder points/Intervals (forall arg's index,
#                           exact matching index should exist in body's index)
#       The fact that arg's index exists => body's index is Interval, not point
#       Body's index didn't match with any arg's index => Point
#       body's index === placeholder
#    windowexpr is very similar to Read, except that index is w_access!
#       indices should be exact same if arg is Tensor
#       arg is window : generate placeholder and body's index === placeholder


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Unification compiler pass



class Unification:
    def __init__(self, subproc, stmt_block):
        self.equations = []

        self.unknowns  = []
        self.new_syms  = []
        
        # TODO: Asserts
        # We don't have inequality in EQs IR

        # TODO: Size
        # Inequality...??

        self.init_stmts(subproc.body, stmt_block)

    def err(self):
        raise TypeError("subproc and pattern don't match")

    def result(self):
        return Equations.system(self.equations)


    def init_stmts(self, sub_body, stmt_block):
        if len(body) != len(stmt_block):
            self.err()

        for sub, block in zip(sub_body, stmt_block):
            init_stmt(sub, block)
        

    def init_stmt(self, sub_stmt, stmt):

        # If stmt type are different, emit error
        if type(stmt) is not type(sub_stmt):
            self.err()

        if type(stmt) is LoopIR.ForAll:
            iter_eq = Equations.SimpleEq( Equations.Var(sub_stmt.iter), Equations.Var(stmt.iter) )

            if sub_stmt.hi in self.unknowns:
                lhs_eq = Equations.Hole(sub_stmt.hi)
            elif type(sub_stmt.hi) is LoopIR.Const:
                lhs_eq = Equations.Const(sub_stmt.hi.val)
            else:
                lhs_eq = Equations.Var(sub_stmt.hi)

            if type(stmt.hi) is LoopIR.Const:
                rhs_eq = Equations.Const(stmt.hi.val)
            else:
                rhs_eq = Equations.Symbol(stmt.hi)

            hi_eq   = Equations.SimpleEq( lhs_eq, rhs_eq )

            self.equations.append( iter_eq )
            self.equations.append( hi_eq )

            self.init_stmts(sub_stmt.body, stmt.body)

        elif type(stmt) is LoopIR.If:
            self.init_expr(sub_stmt.cond, stmt.cond)

            self.init_stmts(sub_stmt.body, stmt.body)
            self.init_stmts(sub_stmt.orelse, stmt.orelse)

        elif type(stmt) is LoopIR.Call:
            if sub_stmt.f != stmt.f:
                self.err()

            for e1, e2 in zip(sub_stmt.args, stmt.args):
                if e1.type.is_tensor_or_window():
                        # This is uninterpreted "holes" at this p
                    self.init_expr(e1, e2)

#            | Alloc  ( sym name, type type, mem? mem )
        elif type(stmt) is LoopIR.Alloc:
            # TODO: This should be syntactic check
            if sub_stmt.type.is_real_scalar() and stmt.type.is_real_scalar():
                pass # Good
            elif sub_stmt.type.is_tensor() and stmt.type.is_tensor():
#            | Tensor     ( expr* hi, bool is_window, type type )
                for sub_h, h in zip(sub_stmt.type.hi, stmt.type.hi):
                    self.init_expr(sub_h, h)
            else:
                self.err()

            # Substitute sub_stmt body's name to stmt.name
            # TODO: ! stmt_rest
            sub_body = subst(sub_stmt.body, sub_stmt.name, stmt.name)

            self.init_stmts(sub_body, stmt.body)

#            | WindowStmt( sym lhs, expr rhs )
        elif type(stmt) is LoopIR.WindowStmt:

            assert sub_stmt.rhs is LoopIR.WindowExpr and stmt.rhs is LoopIR.WindowExpr

# x[:], res[4:]
# x[4],  res[0]
             # TODO: This should be a syntactic check!?
#            | WindowExpr( sym name, w_access* idx )
#            | WindowType ( type src, type as_tensor, expr window )
            for sub_idx, idx in zip(sub_stmt.rhs.idx, stmt.rhs.idx):
                self.init_w_access(sub_idx, idx)

            # TODO: Check sub_stmt.rhs.type == stmt.rhs.type

            # Rename
            sub_body = subst(sub_stmt.body, sub_stmt.lhs, stmt.lhs)
            init_stmts(...,...)
            #self.equations.append( (sub_stmt, stmt) )

        # This is uninterpreted "holes" at this point!
        elif type(stmt) is LoopIR.Assign or type(stmt) is LoopIR.Reduce:
            self.equations.append( (sub_stmt, stmt) )

        elif type(stmt) is LoopIR.Pass:
            pass

        else:
            assert False, "bad case!"


    def init_expr(self, e1, e2):
        if type(e1) is not type(e2):
            self.err()

        # numeric type should match syntactically
        if type(e1) is LoopIR.Read:
            if e1.type.is_tensor_or_window():
            # This is uninterpreted "holes" at this point!
                self.equations.append( (e1, e2) )
            else:
                assert len(e1.idx) == 0
                self.equations.append( (e1.name, e2.name) )

        elif type(e1) is LoopIR.Const:
            if e1.val != e2.name:
                self.err()

        elif type(e1) is LoopIR.USub:
            self.init_expr(e1.arg, e2.arg)

        elif type(e1) is LoopIR.BinOp:
            if e1.op != e2.op:
                self.err()

            self.init_expr(e1.lhs, e2.lhs)
            self.init_expr(e1.rhs, e2.rhs)
# 1+0 , 1
        elif type(e1) is LoopIR.WindowExpr:
            # Uninterpreted "holes" at this point!
            self.equations.append( (e1, e2) )

        elif type(e1) is LoopIR.StrideAssert:
            pass

        else:
            assert False, "bad case"
