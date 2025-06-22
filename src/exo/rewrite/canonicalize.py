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
    """
    Walk the whole `DataflowIR` tree, call `_canon_abs` on every entry of a
    block's `absenv`, and return an *updated* `proc`.

    The traversal is shallow for statements that do **not** embed a `block`
    (Assign, Reduce, …) and recursive for the two constructs that *do* contain
    blocks (`If`, `For`).
    """

    # ---- local helpers -------------------------------------------------------
    def rebuild_block(blk: DIR.block) -> DIR.block:
        """Return a new block with
        • canonicalised ctxt      (absenv)
        • rebuilt stmts (recursing where needed)
        """
        # 1. canonicalise the abstract-environment dictionary
        new_ctxt = {k: _canon_abs(v) for k, v in blk.ctxt.items()}

        # 2. rebuild the statement list
        new_stmts: List[DIR.stmt] = []
        for s in blk.stmts:
            # ——— statements that hold *nested* blocks ————————————————
            if isinstance(s, DIR.If):
                new_body = rebuild_block(s.body)
                new_orelse = rebuild_block(s.orelse)
                new_stmts.append(
                    DIR.If(s.cond, new_body, new_orelse, srcinfo=s.srcinfo)
                )

            elif isinstance(s, DIR.For):
                new_body = rebuild_block(s.body)
                new_stmts.append(
                    DIR.For(s.iter, s.lo, s.hi, new_body, srcinfo=s.srcinfo)
                )

            # ——— all other statements are block-free; keep as-is ————————
            else:
                new_stmts.append(s)

        return DIR.block(list(new_stmts), new_ctxt)

    # ---- rebuild the whole proc ---------------------------------------------
    new_body = rebuild_block(proc.body)

    return DIR.proc(
        proc.name, proc.args, proc.preds, proc.sym_table, new_body, proc.srcinfo
    )
