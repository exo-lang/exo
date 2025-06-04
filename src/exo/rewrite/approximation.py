from .dataflow import *


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


# --------------------------------------------------------------------------- #
# Widening related operations
# --------------------------------------------------------------------------- #

import numpy as np
from scipy.optimize import linprog

# =============================================================================
# Linearization and halfspace conversion
# =============================================================================


def linearize_aexpr(aexpr, variables):
    """
    Convert an aexpr into (coeff, const) where:
      - coeff is a dict mapping variable names to coefficients,
      - const is the constant.
    aexpr is one of: Const(val), Var(name), Add(lhs, rhs), Mult(coeff, ae)
    """
    if isinstance(aexpr, D.Const):
        return ({var: 0 for var in variables}, aexpr.val)
    elif isinstance(aexpr, D.Var):
        coeff = {var: 0 for var in variables}
        coeff[aexpr.name] = 1
        return (coeff, 0)
    elif isinstance(aexpr, D.Add):
        coeff1, const1 = linearize_aexpr(aexpr.lhs, variables)
        coeff2, const2 = linearize_aexpr(aexpr.rhs, variables)
        coeff = {var: coeff1.get(var, 0) + coeff2.get(var, 0) for var in variables}
        return (coeff, const1 + const2)
    elif isinstance(aexpr, D.Mult):
        coeff_inner, const_inner = linearize_aexpr(aexpr.ae, variables)
        coeff = {var: aexpr.coeff * coeff_inner.get(var, 0) for var in variables}
        return (coeff, aexpr.coeff * const_inner)
    else:
        raise ValueError(f"Unknown aexpr: {aexpr}")


def coeffs_to_array(coeff, variables):
    """Return a numpy array of coefficients in the order given by variables."""
    return np.array([coeff.get(var, 0) for var in variables], dtype=float)


def get_halfspaces_for_aexpr(aexpr, branch, variables, eps=1e-6):
    """
    Given an aexpr and a branch type ("ltz", "leq", "eqz", or "gtz"),
    return a list of halfspaces (each is an array [a0, a1, ..., a_{n-1}, b])
    representing a0*x0 + ... + a_{n-1}*x_{n-1} + b <= 0.
    For eqz we return two inequalities.
    """
    coeff, const = linearize_aexpr(aexpr, variables)
    A = coeffs_to_array(coeff, variables)
    if branch == "ltz":
        return [np.append(A, const + eps)]
    elif branch == "gtz":
        return [np.append(-A, -const + eps)]
    elif branch == "leq":
        return [np.append(A, const)]
    elif branch == "eqz":
        hs1 = np.append(A, const)
        hs2 = np.append(-A, -const)
        return [hs1, hs2]
    else:
        raise ValueError(f"Unknown branch type: {branch}")


# =============================================================================
# Region Extraction
# =============================================================================


def extract_regions(node, iterators, halfspaces=None, path=None):
    """
    Recursively traverse the abstract domain tree to collect regions (cells).
    For each leaf, record:
      - "halfspaces": list of halfspace constraints defining the cell,
      - "path": list of (aexpr, branch) tuples representing decisions,
      - "leaf_value": the cell's marking.
    """
    if halfspaces is None:
        halfspaces = []
    if path is None:
        path = []

    regions = []
    if isinstance(node, D.Leaf):
        regions.append(
            {
                "halfspaces": list(halfspaces),
                "path": list(path),
                "value": node,
            }
        )
    elif isinstance(node, D.LinSplit):
        if isinstance(node.cond, D.Lt):
            hs_t = get_halfspaces_for_aexpr(node.cond.lhs, "ltz", iterators)
            hs_f = get_halfspaces_for_aexpr(node.cond.lhs, "gtz", iterators)
        elif isinstance(node.cond, D.Le):
            hs_t = get_halfspaces_for_aexpr(node.cond.lhs, "leq", iterators)
            hs_f = get_halfspaces_for_aexpr(node.cond.lhs, "gtz", iterators)
        elif isinstance(node.cond, D.Eq):
            hs_t = get_halfspaces_for_aexpr(node.cond.lhs, "eqz", iterators)
            hs_f = get_halfspaces_for_aexpr(
                node.cond.lhs, "ltz", iterators
            ) + get_halfspaces_for_aexpr(node.cond.lhs, "gtz", iterators)
        else:
            hs_t = hs_f = []

        for hs in hs_t:
            halfspaces.append(hs)
        path.append((node.cond.lhs, "t"))
        regions.extend(extract_regions(node.t_branch, iterators, halfspaces, path))
        path.pop()
        for _ in hs_t:
            halfspaces.pop()

        for hs in hs_f:
            halfspaces.append(hs)
        path.append((node.cond.lhs, "f"))
        regions.extend(extract_regions(node.f_branch, iterators, halfspaces, path))
        path.pop()
        for _ in hs_f:
            halfspaces.pop()
    else:
        raise ValueError("Unknown node type encountered during extraction.")
    return regions


# =============================================================================
# "Find Intersection" without sympy
# =============================================================================

# We use a constant symbol represented by the string "C".
const_rep = Sym("C")


def get_combinations(elements, combination_length):
    """Return all combinations (as lists) of the given length from elements."""
    if combination_length == 0:
        return [[]]
    if len(elements) < combination_length:
        return []
    else:
        with_first = get_combinations(elements[1:], combination_length - 1)
        with_first = [[elements[0]] + combo for combo in with_first]
        without_first = get_combinations(elements[1:], combination_length)
        return with_first + without_first


def cvt_eq(eq: D.aexpr) -> dict:
    if isinstance(eq, D.Const):
        return {const_rep: eq.val}
    elif isinstance(eq, D.Var):
        return {eq.name: 1}
    elif isinstance(eq, D.Add):
        lhs = cvt_eq(eq.lhs)
        rhs = cvt_eq(eq.rhs)
        common = {key: (lhs[key] + rhs[key]) for key in lhs if key in rhs}
        return lhs | rhs | common
    elif isinstance(eq, D.Mult):
        arg = cvt_eq(eq.ae)
        return {key: arg[key] * eq.coeff for key in arg}


def cvt_back(dic: dict) -> D.aexpr:
    varr = []
    for key, val in dic.items():
        if val == 0:
            continue
        if key == const_rep:
            varr.append(D.Const(val))
            continue

        var = D.Var(key)
        if val != 1:
            var = D.Mult(val, var)
        varr.append(var)

    if len(varr) > 1:
        ae = D.Add(varr[0], varr[1])
        for var in varr[2:]:
            ae = D.Add(ae, var)
    else:
        ae = varr[0]

    return ae


def find_intersections(dims, eqs):
    """
    Given a list of dimension names (e.g. ["i", "d0"]) and a set of equations (D.aexpr),
    produce a list of candidate intersection equations that do not depend on dims[0].
    The algorithm:
      1. For each eq in eqs, convert it to a dictionary.
      2. If the target (dims[0]) is not in the dictionary, add it directly.
      3. Otherwise, collect equations that contain dims[0] and then, for every
         pair, eliminate dims[0] by forming the combination:
             a_n * b_i - b_n * a_i
         for each remaining dimension.
      4. Convert each resulting dictionary back to a D.aexpr.
    """
    intersections = []
    cvted_eqs = []
    for eq in list(eqs):
        dic = cvt_eq(eq)
        if dims[0] not in dic:
            new_dim = sum(dic.get(d, 0) != 0 for d in dims[1:])
            intersections.append((new_dim, dic))
        else:
            cvted_eqs.append(dic)

    # For two equations, eliminate dims[0] using all combinations of 2.
    for feq, seq in get_combinations(cvted_eqs, 2):
        a_n = feq.get(dims[0], 0)
        b_n = seq.get(dims[0], 0)
        cur = {}
        for d in dims[1:] + [const_rep]:
            a_i = feq.get(d, 0)
            b_i = seq.get(d, 0)
            cur[d] = a_n * b_i - b_n * a_i

        # Skip if two equations were parallel
        new_dim = sum(cur.get(d, 0) != 0 for d in dims[1:])
        if new_dim == 0:
            continue

        # Make the coefficient for dims[1] positive.
        if cur.get(dims[1], 0) < 0:
            for k in cur:
                cur[k] = -cur[k]
        intersections.append((new_dim, cur))

    print(intersections)
    # Convert all dictionary representations back to D.aexpr.
    return [(dim, cvt_back(dic)) for dim, dic in intersections]


def get_eqs_from_tree(t):
    """ """
    if isinstance(t, D.Leaf):
        return set()
    elif isinstance(t, D.LinSplit):
        if isinstance(t.cond, D.Eq):
            return (
                {t.cond.lhs}
                | get_eqs_from_tree(t.t_branch)
                | get_eqs_from_tree(t.f_branch)
            )
        return get_eqs_from_tree(t.t_branch) | get_eqs_from_tree(t.f_branch)
    else:
        assert False


# =============================================================================
# Revised Region Refinement (using find_intersections)
# =============================================================================


def refine_region(region, variables, candidates, intersection_pairs):
    """ """
    #    print("  Original candidates:")
    #    for r in candidates:
    #        print("     path:", [(str(a), br) for a, br in r["path"]], " val: ", r["value"])

    # FIXME: This might not be general for multi-dimension
    # Removing the paths from candidates if it exists in the region. This is projecting kind of projecting the world to this region.
    new_candidates = []
    for reg in candidates:
        new_reg = dict()
        new_reg["halfspaces"] = reg["halfspaces"]
        new_reg["value"] = reg["value"]
        tmp_path = []
        for p in reg["path"]:
            if p not in region["path"]:
                tmp_path.append(p)
        new_reg["path"] = tmp_path
        new_candidates.append(new_reg)

    candidates = new_candidates

    #    print("  Removed candidates:")
    #    for r in candidates:
    #        print("     path:", [(str(a), br) for a, br in r["path"]], " val: ", r["value"])

    print()
    print(f"Original Region:")
    print("  Path:", [(str(a), br) for a, br in region["path"]])
    print("  Leaf value:", region["value"])
    print("  Halfspaces:")
    for hs in region["halfspaces"]:
        print("   ", hs)

    half_spaces = []

    def append_hspaces(pre_h, pre_p, p, e):
        half_spaces.append(
            (pre_h + get_halfspaces_for_aexpr(p, e, variables), pre_p + [(p, e)])
        )

    orig_h = list(region["halfspaces"])
    orig_p = list(region["path"])

    if intersection_pairs:
        append_hspaces(orig_h, orig_p, intersection_pairs[0][1], "ltz")
        append_hspaces(orig_h, orig_p, intersection_pairs[0][1], "eqz")
        tmp_spaces = []
        tmp_paths = []
        for i in range(1, len(intersection_pairs)):
            tmp_spaces.extend(
                get_halfspaces_for_aexpr(intersection_pairs[i - 1][1], "gtz", variables)
            )
            tmp_paths.append((intersection_pairs[i - 1][1], "gtz"))
            append_hspaces(
                orig_h + tmp_spaces, orig_p + tmp_paths, intersection_pairs[i][1], "ltz"
            )
            append_hspaces(
                orig_h + tmp_spaces, orig_p + tmp_paths, intersection_pairs[i][1], "eqz"
            )
        append_hspaces(
            orig_h + tmp_spaces, orig_p + tmp_paths, intersection_pairs[-1][1], "gtz"
        )
    else:
        half_spaces = [(orig_h, orig_p)]

    print("\nRegions after Refinement:")
    res = []
    for i, hs in enumerate(half_spaces):
        region_i = {
            "halfspaces": hs[0],
            "path": hs[1],
            "value": region["value"],
        }
        rep = find_feasible_point(region_i["halfspaces"], len(variables))
        if rep is None:
            continue

        print(f"Region {i}:")
        print("  Path:", [(str(a), br) for a, br in region_i["path"]])
        print("  Leaf value:", region_i["value"])
        print("  Halfspaces:")
        for hs in region_i["halfspaces"]:
            print("   ", hs)
        print("  Representative point:", rep)
        color = compute_candidate_color(rep, candidates, variables)
        print("  Candidate color:", color)
        if (
            isinstance(region_i["value"], D.Leaf)
            and isinstance(region_i["value"].v, D.SubVal)
            and isinstance(region_i["value"].v.av, V.Bot)
            and color is not None
        ):
            region_i["value"] = color
            print("  colored")
        else:
            print("  orig_value: ", region_i["value"])
        print()

        res.append(region_i)

    return res


def vertical_line_intersect(rep_point, paths, variables, tol=1e-9):
    lower_bound = float("-inf")
    upper_bound = float("inf")

    for aexpr, rel in paths:
        # linearize_aexpr returns (coeff_dict, constant_term)
        coeff, const = linearize_aexpr(aexpr, variables)
        # Incorporate the constant from linearization.
        constant = const + sum(
            coeff.get(var, 0) * rep_point[j]
            for j, var in enumerate(variables[1:], start=1)
        )
        a0 = coeff.get(variables[0], 0)

        # Adjust the constraint based on its relation.
        if rel == "gtz":
            # f(x) > 0 becomes -f(x) <= 0.
            a0 = -a0
            constant = -constant
        elif rel == "eqz":
            # Equality: f(x) = 0 forces a unique value.
            if abs(a0) < tol:
                if abs(constant) > tol:
                    return None
                continue
            eq_bound = -constant / a0
            lower_bound = max(lower_bound, eq_bound)
            upper_bound = min(upper_bound, eq_bound)
            continue
        # For "ltz" (or similar), we have f(x) <= 0.
        if abs(a0) < tol:
            if constant > tol:
                return None
            continue

        # Solve: a0 * x0 + constant <= 0  =>
        #   if a0 > 0: x0 <= -constant/a0, else x0 >= -constant/a0.
        bound = -constant / a0
        if a0 > 0:
            upper_bound = min(upper_bound, bound)
        else:
            lower_bound = max(lower_bound, bound)

    # We’re considering the vertical ray: x0 <= rep_point[0].
    feasible_upper = min(upper_bound, rep_point[0])
    if lower_bound > feasible_upper + tol:
        return None

    return feasible_upper


# =============================================================================
# Candidate Color Computation (as before)
# =============================================================================


def compute_candidate_color(rep_point, candidates, variables):
    """
    Given a representative point rep_point and a list of candidate hyperplanes
    (each as (aexpr, color)), select the candidate that intersects the vertical line
    (in the target direction) at the highest coordinate below rep_point.
    """
    best_i = float("-inf")
    best_color = None
    for candidate in candidates:
        candidate_i = vertical_line_intersect(rep_point, candidate["path"], variables)
        # print(
        #    "  Path:",
        #    [(str(a), br) for a, br in candidate["path"]],
        #    " color: ",
        #    candidate["value"],
        # )
        # print("    i=", candidate_i)
        if candidate_i != None and candidate_i < rep_point[0] and candidate_i > best_i:
            best_i = candidate_i
            best_color = candidate["value"]
    return best_color


# =============================================================================
# Reconstruction of the Abstract Tree
# =============================================================================


def insert_region_path(dict_tree, path, leaf_value):
    """
    Insert a region (represented by its path and leaf_value) into dict_tree.
    Instead of using a plain string, we wrap the aexpr in an AexprKey so that the
    original aexpr is preserved.
    The key is of the form (AexprKey(aexpr), branch).
    """
    current = dict_tree
    for aexpr, branch in path:
        key = (aexpr, branch)
        if key not in current:
            current[key] = {}
        current = current[key]
    current["leaf"] = leaf_value


def build_dict_tree(regions):
    dict_tree = {}
    #    print("build_dict_tree")
    for reg in regions:
        #        print( "  Path:", [(str(a), br) for a, br in reg["path"]], ", value:", reg["leaf_value"],)
        insert_region_path(dict_tree, reg["path"], reg["value"])
    #    print()
    return dict_tree


def dict_tree_to_node(dict_tree):
    """
    Reconstruct an abstract domain tree from dict_tree.
    At each node, the keys (other than "leaf") are of the form (AexprKey(aexpr), branch).
    We group by the common splitting expression.
    Missing branches are filled with a default bottom node.
    """
    if "leaf" in dict_tree and len(dict_tree) == 1:
        return dict_tree["leaf"]

    # Group keys by the AexprKey (splitting expression)
    grouping = {}
    for key in dict_tree.keys():
        if key == "leaf":
            continue
        aexpr, branch = key
        grouping.setdefault(aexpr, {})[branch] = dict_tree[key]

    # For this node, assume there is only one splitting expression.
    # (If there are more, you'll need to decide how to merge them.)
    aexpr = next(iter(grouping.keys()))
    subtrees = grouping[aexpr]
    # Retrieve subtrees for the three branches; if missing, use a default bottom.
    ltz_subtree = subtrees.get("ltz", {"leaf": D.Leaf(D.SubVal(V.Top()))})
    eqz_subtree = subtrees.get("eqz", {"leaf": D.Leaf(D.SubVal(V.Top()))})
    gtz_subtree = subtrees.get("gtz", {"leaf": D.Leaf(D.SubVal(V.Top()))})

    node_ltz = dict_tree_to_node(ltz_subtree)
    node_eqz = dict_tree_to_node(eqz_subtree)
    node_gtz = dict_tree_to_node(gtz_subtree)

    # Use the original aexpr from the key
    inner = D.LinSplit(D.Eq(aexpr), node_eqz, node_gtz)
    return D.LinSplit(D.Lt(aexpr), node_ltz, inner)


# =============================================================================
# Feasible Point Helpers (using scipy)
# =============================================================================


def find_feasible_point(halfspaces, dim):
    """
    Given a list of halfspaces (each as [a0,...,a_{dim}, b] representing
    a0*x0+...+a_{dim-1}*x_{dim-1}+b <= 0), use linprog to find a feasible point.
    """
    if halfspaces == []:
        return None

    A_ub = []
    b_ub = []
    for hs in halfspaces:
        A_ub.append(hs[:dim])
        b_ub.append(-hs[dim])
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    c = np.zeros(dim)
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, 1000)] * dim, method="highs")
    if res.success:
        return res.x
    else:
        return None


class Strategy1(AbstractInterpretation):
    def abs_stride(self, name, dim):
        return D.Leaf(D.SubVal(V.Top()))

    def abs_extern(self, func, args):
        return D.Leaf(D.SubVal(V.Top()))

    def abs_binop(self, op, lhs, rhs):
        return D.Leaf(D.SubVal(V.Top()))

    def abs_usub(self, arg):
        return D.Leaf(D.SubVal(V.Top()))

    def abs_const(self, val):
        return D.Leaf(D.SubVal(V.ValConst(val)))

    def abs_read(self, name, idx, env):
        # return True if arrays have indirect accesses
        if any([has_array_access(i) for i in idx]):
            return D.Leaf(D.SubVal(V.Top()))

        idxs = [lift_to_abs_a(i) for i in idx]
        if name in self.avars:
            return D.Leaf(D.ArrayVar(name, idxs))
        else:
            # bot if not found in env, substitute the array access if it does
            if name in env:
                itr_map = dict()
                for i1, i2 in zip(env[name].iterators, idxs):
                    itr_map[i1] = i2
                return ASubs(env[name].tree, itr_map).result()
            else:
                return D.Leaf(D.SubVal(V.Top()))

    # Corresponds to \delta in the paper draft
    def abs_ternary(
        self, cond: DataflowIR.expr, body: D.node, orelse: D.node
    ) -> D.node:
        assert isinstance(cond, DataflowIR.expr)
        assert isinstance(body, D.node)
        assert isinstance(orelse, D.node)

        # If the condition is always True or False, just return the leaf as a tree
        tree = D.Leaf(D.SubVal(V.Top()))
        if isinstance(cond, DataflowIR.Const) and (cond.val == True):
            tree = body
        elif isinstance(cond, DataflowIR.Const) and (cond.val == False):
            tree = orelse
        elif isinstance(cond, DataflowIR.Read):
            # boolean case
            assert len(cond.idx) == 0
            assert isinstance(cond.type, DataflowIR.Bool)
            # TODO: Try to handle this more precisely, not just joining bodies?
            return overlay(body, orelse, subval_join)
        else:
            # operators = {+, -, *, /, mod, and, or, ==, <, <=, >, >=}
            assert isinstance(cond, DataflowIR.BinOp)

            # Handle logical operations
            if cond.op == "and":
                return self.abs_ternary(
                    cond.lhs, self.abs_ternary(cond.rhs, body, orelse), orelse
                )
            elif cond.op == "or":
                return self.abs_ternary(
                    cond.lhs, body, self.abs_ternary(cond.rhs, body, orelse)
                )

            if has_array_access(cond.lhs) or has_array_access(cond.rhs):
                # TODO: This is a loose approximation for mutable control state,
                # because we're just treating the condition as Top, always.
                return overlay(body, orelse, subval_join)

            # FIXME: Support modular inequalities for constant cases.
            is_lhs_mod = isinstance(cond.lhs, DataflowIR.BinOp) and cond.lhs.op == "%"
            is_rhs_mod = isinstance(cond.rhs, DataflowIR.BinOp) and cond.rhs.op == "%"
            if is_lhs_mod or is_rhs_mod:
                if cond.op == "==":
                    e1 = cond.lhs.lhs if is_lhs_mod else cond.rhs.lhs
                    c = cond.lhs.rhs if is_lhs_mod else cond.rhs.rhs
                    e2 = cond.rhs if is_lhs_mod else cond.lhs
                    assert isinstance(c, DataflowIR.Const)
                    return D.LinSplit(
                        D.Eq(
                            D.Mod(
                                lift_to_abs_a(
                                    DataflowIR.BinOp(
                                        "-", e1, e2, DataflowIR.Int(), null_srcinfo()
                                    )
                                ),
                                c.val,
                            )
                        ),
                        body,
                        orelse,
                    )
                else:
                    assert False, "modular inequalites are not supported yet!"

            # This is A^\#\qc{e_1 - e_2}
            eq = lift_to_abs_a(
                DataflowIR.BinOp(
                    "-", cond.lhs, cond.rhs, DataflowIR.Int(), null_srcinfo()
                )
            )
            if cond.op == "==":
                tree = D.LinSplit(D.Eq(eq), body, orelse)
            elif cond.op == "<":
                tree = D.LinSplit(D.Lt(eq), body, orelse)
            elif cond.op == "<=":
                tree = D.LinSplit(D.Le(eq), body, orelse)
            elif cond.op == ">":
                tree = D.LinSplit(D.Lt(D.Mult(-1, eq)), body, orelse)
            elif cond.op == ">=":
                tree = D.LinSplit(D.Le(D.Mult(-1, eq)), body, orelse)
            elif cond.op == "%":
                assert (
                    False
                ), "mod should be handled in the cases above, shouldn't be here!"
            elif cond.op == "/":
                assert False, "div is unsupported, shouldn't be here!"
            else:
                assert False, "WTF?"

        return tree

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
