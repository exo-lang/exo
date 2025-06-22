import sympy as sm
from typing import Sequence
from .dataflow import D, DataflowIR
from sympy.core.relational import Relational


def _canon_expr(e: sm.Expr) -> sm.Expr:
    """Return a deterministic polynomial representation."""
    e = sm.expand(e)  # put in polynomial form
    e = sm.simplify(e)  # remove obvious cancellations, gcd-like things
    return e


# ---------------------------------------------------------------------
# canonicaliser
# ---------------------------------------------------------------------
from sympy.logic.boolalg import BooleanFunction


def _canon_rel(r: Relational) -> Relational:
    """
    Recursively canonicalise a (possibly nested) SymPy Boolean/Relational
    expression.  Every Relational leaf is replaced by its ``canonical``
    form while the Boolean structure (And/Or/Not/…) is preserved.
    """
    # Handle logical combinations such as And/Or/Not/…
    if isinstance(r, BooleanFunction):
        new_args = tuple(_canon_rel(arg) for arg in r.args)
        return r.func(*new_args)

    # At this point `r` should be a bare relational (Eq, Lt, Ge, …) or boolean!
    return r.canonical


# ──────────────────────────────────────────────────────────────────────────────
# helpers ─ canonicalise leaf value & sample dict
# ──────────────────────────────────────────────────────────────────────────────
def _canon_val(v: D.val) -> D.val:
    """Canonicalise the indices of an ArrayVar (if present)."""
    if isinstance(v, D.ArrayVar):
        new_idx = list(_canon_expr(e) for e in v.idx)
        return D.ArrayVar(v.name, new_idx)
    # SubVal / other custom value types are assumed to be immutable objects
    return v


def _canon_sample(sample: dict) -> dict:
    return {k: _canon_expr(expr) for k, expr in sample.items()}


# ──────────────────────────────────────────────────────────────────────────────
# public API
# ──────────────────────────────────────────────────────────────────────────────
def _canon_abs(abs_dom: D.abs) -> D.abs:
    """
    Return a *new* `ArrayDomain.abs` whose
      * all `poly` entries,
      * all cell relations (`eq`),
      * all index expressions and sample points
    are written in a stable, canonical form.
    """

    # ── canonicalise the polynomial list ─────────────────────────────────
    new_poly: Sequence[sm.Expr] = list(_canon_expr(p) for p in abs_dom.poly)

    # ── recurse over the CAD ─────────────────────────────────────────────
    def rebuild(node: D.node) -> D.node:
        # ---------- Leaf -------------------------------------------------
        if isinstance(node, D.Leaf):
            new_val = _canon_val(node.v)
            new_sample = _canon_sample(node.sample)
            return D.Leaf(new_val, new_sample)

        # ---------- LinSplit --------------------------------------------
        new_cells = [
            D.Cell(_canon_rel(cell.eq), rebuild(cell.tree)) for cell in node.cells
        ]
        return D.LinSplit(list(new_cells))  # keep ADT fields immutable

    # build the brand new `abs`
    return D.abs(abs_dom.iterators, new_poly, rebuild(abs_dom.tree))


# ──────────────────────────────────────────────────────────────────────────────
# Canonicalise every absenv inside a DataflowIR.proc
# ──────────────────────────────────────────────────────────────────────────────
DIR = DataflowIR  # shorter alias


def _canon_dir(proc: DIR.proc) -> DIR.proc:
    new_ctxt = {k: _canon_abs(v) for k, v in proc.ctxt.items()}

    return DIR.proc(
        proc.name,
        proc.args,
        proc.preds,
        proc.sym_table,
        proc.body,
        new_ctxt,
        proc.srcinfo,
    )
