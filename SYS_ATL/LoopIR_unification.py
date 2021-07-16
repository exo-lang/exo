from .prelude import *
from .LoopIR import LoopIR, T, LoopIR_Rewrite

from collections import ChainMap
import sympy as SYMPY


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
            res = Unification(self.subproc, stmts)

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


def subst(stmts, old_sym, new_sym):
    # Replace all old_sym in stmts with new_sym
    pass

# Description of a "System of Equations"
#
#       system      = (basic_eq* eqs)
#                   -- meaning:     eqs[0] /\ eqs[1] /\ eqs[2] /\ ...
#       basic_eq    = SimpleEq( expr lhs, expr rhs )
#                   -- meaning:     lhs == rhs
#                   | DisjOfEq( expr* lhs, expr* rhs )
#                   -- meaning:     lhs[0] == rhs[0] \/ lhs[1] == rhs[1] \/ ...
#
#     expr  =  Const(int val)
#           |  Hole( sym name )     -- values we want to solve for
#           |  Var( sym name )      -- unknowns that cannot be used in solution, e.g. j
#           |  Symbol( sym name )   -- symbols that can be in solution
#           |  Add( expr lhs, expr rhs )
#           |  Scale( int coeff, expr e )
#
#
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
#
#
#
#

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
class Unification:
    def __init__(self, subproc, stmt_block):
        self.equations = []
        self.unknowns = []

        # Initialize asserts
        for a in subproc.preds:
            self.equations.append(a)

        # Initialize size
        for a in subproc.args:
            if type(a.type) is T.size:
                self.equation.append(a > 0)
            self.unknowns.append(a)

        self.init_stmts(subproc.body, stmt_block)

        self.expand_eqs()

    def expand_eqs():
        for (e1, e2) in self.equations:
            if (type(e1) is LoopIR.Read or
                type(e1) is LoopIR.Assign or
                type(e1) is LoopIR.Reduce or
                type(e1) is LoopIR.WindowExpr):

                self.expand_eq(e1, e2)

    def expand_eq(e1, e2):
        if type(e1) is LoopIR.Read:
            self.equations.append( (e1.name, e2.name) )

            if e1.type.is_tensor() and e2.type.is_tensor():
#       body's index === args's index
#       call arg is just arg
                for e1_idx, e2_idx in zip(e1.idx, e2.idx):
                    self.equations.append( (e1_idx, e2_idx) )

            elif e1.type.is_window() and e2.type.is_window():
                # The caller window expression should have a form of
                # res[0, 1, i, j, 3] === x[?a, ?b, ?c, ?d, ?e] (x[i, j, l] in body)
                # The first thing we can resolve is bounded read, which are i and j
                #  ---> ?c and ?d are Intervals
                # All other indices are Points
                # So just construct a equation

                pass
            else:
                assert False, "type error"


    def err(self):
        raise TypeError("subproc and pattern don't match")

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents


#    stmt    = Assign ( sym name, type type, string? cast, expr* idx, expr rhs )
#            | Reduce ( sym name, type type, string? cast, expr* idx, expr rhs )

    def init_stmts(self, body, stmt_block):

        if len(body) == 0 and len(stmt_block) == 0:
            return
        if len(body) != len(stmt_block):
            self.err()

        stmt = stmt_block[0]
        sub_stmt = body[0]
        stmt_rest = stmt_block[1:]
        sub_stmt_rest = body[1:]

        # If stmt type are different, emit error
        if type(stmt) is not type(sub_stmt):
            self.err()

        if type(stmt) is LoopIR.ForAll:
            # Substitute all occurance of subproc_stmt.iter in
            # subproc body to stmt.iter
            sub_body = subst(sub_stmt.body, sub_stmt.iter, stmt.iter)

            # hi == hi
            self.equations.append( (sub_stmt.hi, stmt.hi ) )

            self.init_stmts(sub_body, stmt.body)

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
