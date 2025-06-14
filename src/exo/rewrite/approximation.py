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
            rel = sm.S.true  # whole line: no defining relation
            cells.append(D.Cell(rel, child))

        else:
            # left unbounded interval (−∞, r₀)
            r0 = ordered[0]
            smpl = get_sample_point(sm.S.NegativeInfinity, r0)
            rel = var < r0
            child = lift(level - 1, {**partial_sample, var: smpl})
            cells.append(D.Cell(rel, child))

            # root points and the open intervals between successive roots
            for r_lo, r_hi in zip(ordered, ordered[1:]):
                p_eq = roots_to_poly[r_lo][0]
                rel_eq = sm.Eq(p_eq.as_expr(), 0)
                child = lift(level - 1, {**partial_sample, var: r_lo})
                cells.append(D.Cell(rel_eq, child))

                smpl = get_sample_point(r_lo, r_hi)
                rel_iv = sm.And(var > r_lo, var < r_hi)
                child = lift(level - 1, {**partial_sample, var: smpl})
                cells.append(D.Cell(rel_iv, child))

            # last root and right unbounded interval (rₙ, ∞)
            r_last = ordered[-1]
            p_eq = roots_to_poly[r_last][0]
            rel_eq = sm.Eq(p_eq.as_expr(), 0)
            child = lift(level - 1, {**partial_sample, var: r_last})
            cells.append(D.Cell(rel_eq, child))

            smpl = get_sample_point(r_last, sm.S.Infinity)
            rel_iv = var > r_last
            child = lift(level - 1, {**partial_sample, var: smpl})
            cells.append(D.Cell(rel_iv, child))

        # ---------- bundle the stack for this variable -------------------
        return D.LinSplit(cells)

    # ---------------------------------------------------------------------
    # 4.  Assemble the root ``ArrayDomain.abs`` value
    # ---------------------------------------------------------------------
    tree = lift(len(gens) - 1, {})
    return D.abs(gens, F, tree)


def _sample_satisfies(expr, sample) -> bool:
    subs = {sym: v for sym, v in sample.items()}
    value = expr.xreplace(subs)
    return bool(value)


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
            tgt_val = _lookup_value(choose_src, node.sample) or node.v
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
            if isinstance(node, Not):
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
            return D.Leaf(D.SubVal(V.Top()))

        name = sm.Symbol(ename.__repr__())
        self.sym_table[name] = ename
        idxs = [lift_to_sympy(i, self.sym_table) for i in idx]
        if name in self.avars:
            return D.abs([], [], D.Leaf(D.ArrayVar(name, idxs), {}))
        else:
            # bot if not found in env, substitute the array access if it does
            if name in env:
                rabs = env[name]
                itr_map = dict()
                for i1, i2 in zip(rabs.iterators, idxs):
                    itr_map[i1] = i2
                return D.abs(
                    rabs.iterators, rabs.poly, ASubs(rabs.tree, itr_map).result()
                )
            else:
                return D.abs([], [], D.Leaf(D.SubVal(V.Top()), {}))

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

            tree = cylindrical_algebraic_decomposition(
                p1 + p2 + p3, iterators
            )  # initialize sample points for each leaf

            # Propagate the values to the resulting cad tree
            # by evaluating sample points on the cond, body, and orelse branches
            # because the sample points should be unique, it should represent the cells

            new_tree = propagate_values(tree, body, orelse, sym_cond)

            return new_tree

        assert False, "something is wrong!"

    # =============================================================================
    # The Widening Operator
    # =============================================================================

    # TODO: We should probably use ternary decision "diagrams" to compress the leaf duplications (?)

    # maybe we should pass a count to widening so that we can debug and terminate when necessary
    # widening has information of iteration count
    def abs_widening(self, a1: D.abs, a2: D.abs, count: int) -> D.abs:
        """
        Widening for loop approxmation
        """

        assert len(a2.iterators) == len(a1.iterators)

        if count >= 0:
            tree = D.Leaf(D.SubVal(V.Top()))
        else:
            tree = overlay(a1, a2, subval_join)

        # and we can just run whatever on a3
        # to satisfy the x \widen y >= x \join y
        return D.abs(a2.iterators, tree)

        if isinstance(a2.tree, D.Leaf):
            return a2

        variables = a2.iterators

        # Sort half spaces into dimension
        regions = {}
        for reg in extract_regions(a2.tree, a2.iterators):
            d = len(variables) - sum(e == "eqz" for (_, e) in reg["path"])
            assert d >= 0
            regions[d] = [] if d not in regions else regions[d]
            regions[d].append(reg)

        # Get intersections!
        eqs = get_eqs_from_tree(a2.tree)
        print()
        print(variables)
        print([str(s) for s in eqs])
        intersections = find_intersections(variables, eqs)

        for d in regions.keys():
            print("d : ", d)
            for reg in regions[d]:
                print([(str(p), str(e)) for p, e in reg["path"]])
            print()
        print("Intersections:", [(str(dim), str(eq)) for dim, eq in intersections])
        refined_regions = regions.get(0, [])
        scanned_variables = variables.copy()

        for dim in range(1, len(variables) + 1):
            if dim not in regions:
                continue

            print("here dim: ", dim)
            intersection_pairs = []
            ivar = None
            for idim, intersection in intersections:
                if idim != dim - 1:
                    continue

                coeff, const = linearize_aexpr(intersection, variables)
                for k in coeff.keys():
                    if k in scanned_variables and abs(coeff.get(k, 0)) > 1e-9:
                        if ivar is not None and k != ivar:
                            assert False
                        ivar = k
                        scanned_variables.remove(k)
                        break
                val = -const / coeff[ivar]
                intersection_pairs.append((val, intersection))
            # Sort the intersection pairs by value.
            intersection_pairs.sort(key=lambda pair: pair[0])

            tmp_regions = []
            for reg in regions[dim]:
                tmp_regions.extend(
                    refine_region(
                        reg, a2.iterators, refined_regions, intersection_pairs
                    )
                )
            refined_regions.extend(tmp_regions)

        # FIXME: dictionary reconstruction might be buggy
        dict_tree = build_dict_tree(refined_regions)

        reconstructed_tree = dict_tree_to_node(dict_tree)

        # print("\nReconstructed Abstract Domain Tree:")
        # print(reconstructed_tree)
        a = abs_simplify(
            abs_simplify(abs_simplify(D.abs(a2.iterators, reconstructed_tree)))
        )
        #    print("\nPrevious Abstract Domain Tree:")
        #    print(a1)

        return a
