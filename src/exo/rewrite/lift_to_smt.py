import sympy as sm
from exo.rewrite.internal_analysis import *
from exo.rewrite.dataflow import D, V
from sympy.core.relational import Relational

# --------------------------------------------------------------------------- #
# Lifting the abstract domain tree to SMT formula (AExpr)
# --------------------------------------------------------------------------- #
def mk_aexpr(op, pred):
    return A.BinOp(
        op, pred, A.Const(0, T.Int(), null_srcinfo()), T.Bool(), null_srcinfo()
    )


def lift_to_smt_a(e, env) -> A.expr:
    """
    Convert a SymPy expression `e` into an AExpr.
    - takes a dictionary of sm.Symbol to Sym
    """

    # ---------------------------- literals ---------------------------------
    if isinstance(e, bool):
        return A.Const(e, T.bool, null_srcinfo())

    elif isinstance(e, (int, sm.Integer)):
        return A.Const(int(e), T.int, null_srcinfo())

    elif isinstance(e, (float, sm.Float, sm.Rational)):
        return A.Const(float(e), T.R, null_srcinfo())

    elif isinstance(e, sm.Symbol):
        return A.Var(env[e], T.R, null_srcinfo())  # need to convert to Sym

    # --------------------------- arithmetic --------------------------------
    elif isinstance(e, sm.Add):
        args = [lift_to_smt_a(a, env) for a in e.args]
        acc = args[0]
        for a in args[1:]:
            acc = A.BinOp("+", acc, a, T.R, null_srcinfo())
        return acc

    elif isinstance(e, sm.Mul):
        args = [lift_to_smt_a(a, env) for a in e.args]
        acc = args[0]
        for a in args[1:]:
            acc = A.BinOp("*", acc, a, T.R, null_srcinfo())
        return acc

    # --------------------------- relational --------------------------------
    elif isinstance(e, Relational):
        lhs = lift_to_smt_a(e.lhs, env)
        rhs = lift_to_smt_a(e.rhs, env)
        op_map = {
            sm.Eq: "==",
            sm.Lt: "<",
            sm.Le: "<=",
            sm.Gt: ">",
            sm.Ge: ">=",
        }
        for cls, op in op_map.items():
            if isinstance(e, cls):
                return A.BinOp(op, lhs, rhs, T.bool, null_srcinfo())

    # ----------------------------- logical ---------------------------------
    elif isinstance(e, sm.And):
        return AAnd(*(lift_to_smt_a(a, env) for a in e.args))

    elif isinstance(e, sm.Or):
        return AOr(*(lift_to_smt_a(a, env) for a in e.args))

    elif isinstance(e, sm.Not):
        return ANot(lift_to_smt_a(e.args[0], env))

    # --------------------------- fallback ----------------------------------
    raise AssertionError(f"Unsupported SymPy construct: {type(e)} – {e}")


# ---------------------------------------------------------------------------
# ArrayDomain.val  ➜  AExpr
# ---------------------------------------------------------------------------

# Corresponds to \mathcal{L}^\#_{vabs} in the paper
def lift_to_smt_vabs(aname: A.Var, v: V.vabs):
    if isinstance(v, V.ValConst):
        c = A.Const(v.val, T.R, null_srcinfo())
        return AEq(aname, c)
    elif isinstance(v, V.Top):
        return A.Const(True, T.bool, null_srcinfo())
    elif isinstance(v, V.Bot):
        return A.Const(False, T.bool, null_srcinfo())


# cache for ArrayVar  → fresh SMT symbols
cvt_dict: dict[tuple, A.Var] = {}


def lift_to_smt_val(aname: A.Var, e: D.val, env: dict) -> A.expr:
    """
    Constrain the SMT variable `aname` (A.Var) to equal the ArrayDomain value `e`
    by producing an AExpr<boolean>.

    Handles:
      • SubVal(vabs)      – abstract numeric value
      • ArrayVar(name,idx)– read of concrete array element
      • ScalarExpr(poly)  – scalar polynomial expression (new!)
    """
    # ------------------------------------------------------------------ SubVal
    if isinstance(e, D.SubVal):
        return lift_to_smt_vabs(aname, e.av)

    # ---------------------------------------------------------------- ArrayVar
    elif isinstance(e, D.ArrayVar):
        # Memoise a *single* SMT variable per array+indices combination
        key = (e.name, *e.idx)
        vname = cvt_dict.get(key)
        if vname is None:
            # build a readable symbol such as  array_A_0_2pI  …
            idx_str = "_".join(
                str(i)
                .replace("(", "")
                .replace(")", "")
                .replace("-", "_")
                .replace("+", "p")
                .replace("*", "m")
                for i in e.idx
            )
            sym = Sym(f"array_{e.name}_{idx_str}")
            vname = A.Var(sym, T.R, null_srcinfo())
            cvt_dict[key] = vname

        return AEq(aname, vname)

    # ------------------------------------------------------------- ScalarExpr
    elif isinstance(e, D.ScalarExpr):
        # Convert the affine/polynomial expression to an AExpr and equate
        rhs = lift_to_smt_a(e.poly, env)  # assumes helper already exists
        return AEq(aname, rhs)

    # ----------------------------------------------------------- Fallback / ✘
    raise AssertionError(f"Unhandled ArrayDomain.val constructor: {type(e)}")


# ---------------------------------------------------------------------------
# ArrayDomain  ➜  AExpr  (updated for LinSplit / Cell)
# ---------------------------------------------------------------------------


def lift_to_smt_n(name: Sym, src: D.node, env: dict):
    """
    Convert an ArrayDomain node `src` into an AExpr constraining the SMT
    variable `name`.

    Leaf        : same as before (value constraint).
    LinSplit    : guarded union of Cells, compiled to nested A.Selects.

    Parameters
    ----------
    name : Sym          – the SMT variable being constrained
    src  : D.node       – ArrayDomain node

    Returns
    -------
    A.expr (always of type bool)
    """
    aname = A.Var(name, T.R, null_srcinfo())  # the SMT variable

    # ---------- helpers ----------------------------------------------------
    def _lift_rel(rel):
        """
        Turn Cell.rel (the guard) into an AExpr<bool>.
        - Already an A.expr        ➜ keep
        - Python bool              ➜ wrap in ABool
        - Anything else            ➜ assume affine expr, feed to lift_to_smt_a
        """
        if isinstance(rel, A.expr):
            return rel
        if isinstance(rel, bool):
            return ABool(rel)
        # Fallback: treat it as the affine predicate used in the old code
        return lift_to_smt_a(rel, env)

    # ---------- recursive walk ---------------------------------------------
    def walk(node: D.node):
        # Leaf  -------------------------------------------------------------
        if isinstance(node, D.Leaf):
            return lift_to_smt_val(aname, node.v, env)

        # LinSplit  ---------------------------------------------------------
        if isinstance(node, D.LinSplit):
            # Build nested selects, right-associative (last cell is the "else")
            else_branch = ABool(False)  # ⊥ for “no cell hit”
            for cell in reversed(node.cells):
                guard = _lift_rel(cell.eq)
                then = walk(cell.tree)
                else_branch = A.Select(guard, then, else_branch, T.bool, null_srcinfo())
            return else_branch

        # Unknown node kind -------------------------------------------------
        raise AssertionError(f"Unhandled ArrayDomain node: {type(node)}")

    # -----------------------------------------------------------------------
    return walk(src)
