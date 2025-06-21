from .dataflow import *
from .cad.cad import hongproj, simplify_alg_sub, get_nice_roots, get_sample_point
import sympy as sm
from sympy.logic.boolalg import BooleanFunction
from sympy.core.relational import Relational


def dataflow_analysis(
    proc: LoopIR.proc, loopir_stmts: list, syms=None
) -> DataflowIR.proc:
    proc = inline_calls(proc)
    proc = inline_windows(proc)

    # step 1 - convert LoopIR to DataflowIR with empty contexts (i.e. AbsEnvs)
    datair, stmts, d_syms = LoopIR_to_DataflowIR(proc, loopir_stmts, syms).result()

    # step 2 - run abstract interpretation algorithm to populate contexts with abs values
    Strategy1(datair)

    return datair, stmts, d_syms


def nice_root(poly, var):
    """
    Return the exact root of a linear polynomial a*var + b.
    Works even when a or b contain other symbols.
    For non-linear polynomials fall back to 'root(poly)'.
    """
    P = sm.Poly(poly, var)
    if P.degree() == 1:
        a, b = P.all_coeffs()  # a*var + b
        return sm.simplify(-b / a)  # symbolic division, then simplify
    assert False, "hmm"


# ---------------------------------------------------------------------------
#  Cylindrical Algebraic Decomposition that produces an ArrayDomain instance
# ---------------------------------------------------------------------------
def cylindrical_algebraic_decomposition(F, gens):
    """Return the CAD of *F* with respect to *gens* encoded as a single
    ``ArrayDomain`` value.

    Parameters
    ----------
    F : Sequence[Expr]
        Input polynomial system.
    gens : Sequence[Symbol]
        Ordered tuple of variables (highest index last).

    Notes
    -----
    - The function re‑uses the existing helper routines *hongproj*,
      *simplify_alg_sub*, *get_nice_roots* and *get_sample_point* that are
      assumed to be in scope (exactly as in the original implementation).
    - All inner nodes are built with ``D.LinSplit`` and each child is a
      ``D.Cell`` holding a SymPy *Relational* object that describes the
      corresponding section of the space.
    - A 0‑dimensional cell obtains the empty dictionary ``{}`` as its sample
      point, which is the unique point of \(\mathbb{R}^0\).
    """

    # ---------------------------------------------------------------------
    # 2.  Projection phase (Hong’s projection)
    # ---------------------------------------------------------------------
    proj_sets = [F]
    for i in range(len(gens) - 1):
        proj_sets.append(list(hongproj(proj_sets[-1], gens[i])))

    # ---------------------------------------------------------------------
    # 3.  Helper: build the tree recursively (lifting phase)
    # ---------------------------------------------------------------------
    def lift(level: int, partial_sample: dict):
        """Return a ``D.node`` describing the stack over *partial_sample*."""

        # ---------- leaf --------------------------------------------------
        if level < 0:
            # Initial, placeholder value. To be colored by propagate_values
            val = D.SubVal(V.Top())
            # In R^0 the sample point is the empty tuple ⇒ ``{}``
            return D.Leaf(val, dict(partial_sample))

        var = gens[level]
        projs = proj_sets[level]

        # ---------- collect roots for the current partial sample ---------
        roots_to_poly = defaultdict(list)
        for p in projs:
            p_sub = simplify_alg_sub(p, partial_sample)
            for r in get_nice_roots(p_sub):
                roots_to_poly[r].append(p)
        ordered = sorted(roots_to_poly)

        # ---------- generate 1‑dimensional cells --------------------------
        cells = []

        if not ordered:  # no roots ⇒ only the unbounded interval
            smpl = get_sample_point(sm.S.NegativeInfinity, sm.S.Infinity)
            child = lift(level - 1, {**partial_sample, var: smpl})
            cells.append(D.Cell(sm.S.true, child))

        else:
            # left unbounded interval (−∞, r₀)
            r0 = ordered[0]
            smpl = get_sample_point(sm.S.NegativeInfinity, r0)
            rel = var < nice_root(roots_to_poly[r0][0].as_expr(), var)
            child = lift(level - 1, {**partial_sample, var: smpl})
            cells.append(D.Cell(rel, child))

            # root points and the open intervals between successive roots
            for r_lo, r_hi in zip(ordered, ordered[1:]):
                p_eq = roots_to_poly[r_lo][0]
                rel_eq = sm.Eq(p_eq.as_expr(), 0)
                child = lift(level - 1, {**partial_sample, var: r_lo})
                cells.append(D.Cell(rel_eq, child))

                # ---------- open strip (r_lo , r_hi) --------------------
                p_lo = roots_to_poly[r_lo][0].as_expr()
                p_hi = roots_to_poly[r_hi][0].as_expr()

                smpl = get_sample_point(r_lo, r_hi)  # sample inside
                child = lift(level - 1, {**partial_sample, var: smpl})

                # choose the correct inequality directions
                s_lo = sm.sign(p_lo.subs({**partial_sample, var: smpl}))
                s_hi = sm.sign(p_hi.subs({**partial_sample, var: smpl}))

                ineq_lo = (p_lo > 0) if s_lo > 0 else (p_lo < 0)
                ineq_hi = (p_hi > 0) if s_hi > 0 else (p_hi < 0)

                guard = sm.And(ineq_lo, ineq_hi)
                cells.append(D.Cell(guard, child))

            # last root and right unbounded interval (rₙ, ∞)
            r_last = ordered[-1]
            p_eq = roots_to_poly[r_last][0]
            rel_eq = sm.Eq(p_eq.as_expr(), 0)
            child = lift(level - 1, {**partial_sample, var: r_last})
            cells.append(D.Cell(rel_eq, child))

            smpl = get_sample_point(r_last, sm.S.Infinity)
            child = lift(level - 1, {**partial_sample, var: smpl})
            rel_iv = var > nice_root(roots_to_poly[r_last][0].as_expr(), var)
            cells.append(D.Cell(rel_iv, child))

        # ---------- bundle the stack for this variable -------------------
        return D.LinSplit(cells)

    # ---------------------------------------------------------------------
    # 4.  Assemble the root ``ArrayDomain.abs`` value
    # ---------------------------------------------------------------------
    tree = lift(len(gens) - 1, {})
    return D.abs(gens, F, tree)


def _sample_satisfies(expr, sample) -> bool:
    value = expr.xreplace(sample)
    return bool(value.simplify())


def _lookup_value(node: D.node, sample):
    """
    Recursively follow `sample` through `node` (a subtree of a2).
    Returns the leaf's .v if the point ends up in exactly one cell,
    otherwise None.
    """
    if isinstance(node, D.Leaf):
        return node.v

    # LinSplit: pick the first cell whose guard is satisfied.
    for cell in node.cells:
        if _sample_satisfies(cell.eq, sample):
            return _lookup_value(cell.tree, sample)

    # Point lies outside the CAD partition (should not happen in a true CAD):
    return None


# ────────────────────────────────────────────
# new immutable copier
# ────────────────────────────────────────────
def propagate_values(dst: D.abs, src1: D.abs, src2: D.abs, cond):
    """
    Return a *new* ArrayDomain.abs:
      • Traverses `dst`'s CAD without mutating it.
      • For each leaf sample:
          - if  cond(sample)  holds ⟶ take value from `src1`;
          - else                       take value from `src2`.
      • If the sample does not belong to any cell of the chosen source
        CAD, the original value in `dst` is kept.
    """

    def rebuild(node: D.node):
        # ── Leaf ───────────────────────────────────────────────────────────
        if isinstance(node, D.Leaf):
            choose_src = (
                src1.tree if _sample_satisfies(cond, node.sample) else src2.tree
            )
            tgt_val = _lookup_value(choose_src, node.sample)
            assert tgt_val is not None
            # create a *new* Leaf with the (possibly) updated value
            return D.Leaf(tgt_val, node.sample)

        # ── LinSplit ───────────────────────────────────────────────────────
        new_cells = [
            D.Cell(cell.eq, rebuild(cell.tree))  # new subtree per child
            for cell in node.cells
        ]

        return D.LinSplit(new_cells)

    # build an entirely new abs element (iterators & polynomials unchanged)
    return D.abs(dst.iterators, dst.poly, rebuild(dst.tree))


def extract_poly(expr):
    """
    Return a list of SymPy *Expr* objects, one for every polynomial that
    appears in a relational sub-expression of `expr`.

    A “polynomial” here is taken as  (lhs - rhs)  of each relation, so

        x - y > 0                      →   x - y
        x + y == 0  &  2*x - z < 0     →   [x + y, 2*x - z]

    The order follows a left-to-right traversal of the Boolean tree.
    """
    polys = []

    def visit(node):
        # And / Or / Implies / Equivalent … – all BooleanFunction subclasses
        if isinstance(node, BooleanFunction):
            # NOT has exactly one argument
            if isinstance(node, sm.Not):
                visit(node.args[0])
            else:
                for arg in node.args:
                    visit(arg)

        # Any comparison:  Eq, Ne, Lt, Le, Gt, Ge, StrictGreaterThan, …
        elif isinstance(node, Relational):
            polys.append(sm.expand(node.lhs - node.rhs))

        # other node types (Symbol, Number, ...) are ignored

    visit(expr)
    return polys


from sympy.solvers.simplex import lpmin, InfeasibleLPError
from sympy.core.relational import StrictGreaterThan, StrictLessThan, Ge, Le


def has_integer_solution(inequalities):
    # 1 .  Replace strict signs by non-strict ones, shifting the RHS by ±1
    normalised = []
    for rel in inequalities:
        if isinstance(rel, StrictGreaterThan):  #  expr > rhs  ⇒  expr ≥ rhs+1
            normalised.append(Ge(rel.lhs, rel.rhs + 1))
        elif isinstance(rel, StrictLessThan):  #  expr < rhs  ⇒  expr ≤ rhs-1
            normalised.append(Le(rel.lhs, rel.rhs - 1))
        else:  #  ≤, ≥ or = already OK
            normalised.append(rel)

    # 2 .  Ask the simplex solver to “minimise 0” (pure feasibility test)
    try:
        lpmin(0, tuple(normalised))
        return True  # feasible  ⇒  there is an (integer) solution
    except InfeasibleLPError:
        return False  # contradictory system


# ───────────────────────── helpers ──────────────────────────────────────────
def _iter_leaves(node, eqs: tuple):
    """Depth-first generator over every Leaf in a CAD tree."""
    if isinstance(node, D.Leaf):
        yield node
    else:  # LinSplit
        for cell in node.cells:
            new_eqs = eqs
            if isinstance(cell.eq, sm.Eq):
                new_eqs = eqs + (cell.eq,)
            else:
                # skip if cell.eq is unsatisfiable for integer
                polys = sm.And.make_args(cell.eq) | set(eqs)
                if not has_integer_solution(polys):
                    continue

            yield from _iter_leaves(cell.tree, new_eqs)


def _vabs_subsetof(v1, v2):
    """
    Return True iff v1 ⊑ v2 w.r.t. the ValueDomain lattice.

    Bot   ⊑ everything
    c     ⊑ Top                    (for any constant c)
    c₁    ⊑ c₂     ⇔ c₁ == c₂      (same constant)
    Top   ⊑ Top only
    """
    # ⊥ is below everything
    if isinstance(v1, V.Bot):
        return True

    # ⊤ is only below itself
    if isinstance(v1, V.Top):
        return isinstance(v2, V.Top)

    # v1 is a concrete constant
    if isinstance(v1, V.ValConst):
        # Constant is below ⊤
        if isinstance(v2, V.Top):
            return True
        # Two constants: only if they are the *same* constant
        if isinstance(v2, V.ValConst):
            return v1.val == v2.val
        # Otherwise (v2 is Bot or some unsupported kind) -> False
        return False

    # Fallback: if both have exactly the same representation we treat them equal
    return type(v1) is type(v2)


def _val_subset(v1: D.val, v2: D.val) -> bool:
    """Return True iff v1 subset of v2"""
    if isinstance(v1, D.SubVal) and isinstance(v1.av, V.Bot):
        return True

    if isinstance(v1, D.SubVal) and isinstance(v2, D.SubVal):
        return _vabs_subsetof(v1.av, v2.av)

    if isinstance(v1, D.ArrayVar) and isinstance(v2, D.ArrayVar):
        if v1.name != v2.name or len(v1.idx) != len(v2.idx):
            return False
        # compare each index modulo algebraic equivalence
        return all(sm.simplify(e1 - e2) == 0 for e1, e2 in zip(v1.idx, v2.idx))

    return False  # mismatched variants


def evaluate_poly(poly, values):
    # Accept either a Poly or a raw SymPy expression
    expr = poly.as_expr() if isinstance(poly, sm.Poly) else sm.sympify(poly)

    # Perform *simultaneous* substitution to avoid unintended cascades
    # (subs(..., simultaneous=True) is new in SymPy 1.11 – fall back
    #  to ordinary .subs if you are on an older version)
    try:
        out = expr.subs(values, simultaneous=True)
    except TypeError:
        out = expr.subs(values)

    return sm.simplify(out)


class Strategy1(AbstractInterpretation):
    def abs_stride(self, name, dim):
        return D.abs([], [], D.Leaf(D.SubVal(V.Top()), {}))

    def abs_extern(self, func, args):
        return D.abs([], [], D.Leaf(D.SubVal(V.Top()), {}))

    def abs_binop(self, op, lhs, rhs):
        return D.abs([], [], D.Leaf(D.SubVal(V.Top()), {}))

    def abs_usub(self, arg):
        return D.abs([], [], D.Leaf(D.SubVal(V.Top()), {}))

    def abs_const(self, val):
        return D.abs([], [], D.Leaf(D.SubVal(V.ValConst(val)), {}))

    def abs_read(self, ename, idx, env):
        # return True if arrays have indirect accesses
        if any([has_array_access(i) for i in idx]):
            assert False, "implement"
            return D.Leaf(D.SubVal(V.Top()))

        name = sm.Symbol(ename.__repr__())
        self.sym_table[name] = ename
        idxs = [lift_to_sympy(i, self.sym_table).simplify() for i in idx]
        if name in self.avars:
            return D.abs([], [], D.Leaf(D.ArrayVar(name, idxs), {}))
        else:
            # bot if not found in env, substitute the array access if it does
            if name in env:
                rabs = env[name]
                itr_map = dict()
                iters = []
                for i1, i2 in zip(rabs.iterators, idxs):
                    if i2.free_symbols:
                        iters.append(i1)
                    itr_map[i1] = i2

                new_poly = list({evaluate_poly(p, itr_map) for p in rabs.poly})

                return D.abs(iters, new_poly, ASubs(rabs.tree, itr_map).result())
            else:
                return D.abs([], [], D.Leaf(D.SubVal(V.Bot()), {}))

    # Corresponds to \delta in the paper draft
    def abs_ternary(
        self,
        iterators: list[sm.Symbol],
        cond: DataflowIR.expr,
        body: D.abs,
        orelse: D.abs,
    ) -> D.abs:
        assert isinstance(cond, DataflowIR.expr)
        assert isinstance(body, D.abs)
        assert isinstance(orelse, D.abs)

        # If the condition is always True or False, just return the leaf as a tree
        if isinstance(cond, DataflowIR.Const) and (cond.val == True):
            return body
        elif isinstance(cond, DataflowIR.Const) and (cond.val == False):
            return orelse
        elif isinstance(cond, DataflowIR.BinOp):
            # operators = {+, -, *, /, mod, and, or, ==, <, <=, >, >=}

            if has_array_access(cond.lhs) or has_array_access(cond.rhs):
                assert False, "unimplemented"
                # TODO: Implement this
                # if the condition is value dependent, we have to join body and orelse values.
                # call cad with (p1 + p2, iterators)
                # and join values in the propagate_values call!

            # Get polynomials from cond, body, and orelse, and call cad
            p1 = body.poly
            p2 = orelse.poly
            sym_cond = lift_to_sympy(cond, self.sym_table)
            p3 = extract_poly(sym_cond)

            # TODO: This is a dirty hack for handling constants like "n"
            fsyms = []
            for p in p1 + p2 + p3:
                for fsym in p.free_symbols:
                    if fsym not in iterators and fsym not in fsyms:
                        fsyms.append(fsym)

            tree = cylindrical_algebraic_decomposition(
                p1 + p2 + p3, iterators + fsyms
            )  # initialize sample points for each leaf

            # Propagate the values to the resulting cad tree
            # by evaluating sample points on the cond, body, and orelse branches
            # because the sample points should be unique, it should represent the cells

            new_tree = propagate_values(tree, body, orelse, sym_cond)

            return new_tree

        assert False, "something is wrong!"

    def issubsetof(self, a1: D.abs, a2: D.abs) -> bool:
        """
        is a1 subset of a2?
        for all leaves in a1, get the corresponding value in a2
        """
        assert a1.iterators == a2.iterators
        for v_leaf in _iter_leaves(a1.tree, tuple()):
            v_a2 = _lookup_value(a2.tree, v_leaf.sample)
            if v_a2 is None or not _val_subset(v_leaf.v, v_a2):
                return False
        return True

    # =============================================================================
    # The Widening Operator
    # =============================================================================
    def abs_widening(self, a1: D.abs, a2: D.abs, count: int) -> D.abs:
        """
        Widen "a2" for loop approxmation
        """

        assert len(a2.iterators) == len(a1.iterators)

        if count >= 1:
            return None

        def visit(node: D.node) -> D.node:
            if isinstance(node, D.Leaf):
                return node  # unchanged

            new_cells = []
            prev_val = None
            for cell in node.cells:
                new_tree = visit(cell.tree)

                if isinstance(new_tree, D.Leaf):
                    val = new_tree.v
                    if not prev_val or (
                        isinstance(cell.eq, sm.Equality)
                        and not (
                            isinstance(val, D.SubVal) and isinstance(val.av, V.Bot)
                        )
                    ):
                        if val != prev_val:
                            new_cells.append(D.Cell(cell.eq, new_tree))
                            prev_val = val
                            continue

                    # Case 1: If inequality, merge with the previous cell
                    # Case 2: If the cell is equality but the value is Bottom, merge with the previous cell
                    # Case 3: If the equality has the same value as the previous cells!
                    new_eq = sm.Or(new_cells[-1].eq, cell.eq)
                    new_cells[-1] = D.Cell(new_eq, new_cells[-1].tree)
                else:
                    new_cells.append(D.Cell(cell.eq, new_tree))

            return D.LinSplit(new_cells)

        return D.abs(a2.iterators, a2.poly, visit(a2.tree))
