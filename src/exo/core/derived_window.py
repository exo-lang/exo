"""Derived windows (WindowStmt) as codegen-only aliases.

PROTOTYPE (branch akeley98/derived_window); see endgame_plan/plan_derived_window.md.

Invariant after desugar_derived_windows: no expression or statement indexes
the lhs of a WindowStmt. Every use of a derived window `w = x[widx]` is
rewritten to index the underlying tensor `x` directly. A WindowExpr that
was (or should be, for SpecialWindow) taken from `w` records this in its
`derived` attribute, which rewrites pass through un-inspected.

unify_derived_windows runs just before codegen and converts each
`x[idx] (derived=w)` back to `w[idx']`, i.e. what the WindowExpr would have
been prior to desugaring.
"""

from __future__ import annotations

from collections import ChainMap
from typing import Optional

from .LoopIR import LoopIR, LoopIR_Rewrite, T, chain_window_idx, create_window_type
from .prelude import Sym


class _ScopedRewrite(LoopIR_Rewrite):
    """LoopIR_Rewrite with a lexically-scoped env of WindowStmt rhs."""

    def __init__(self):
        # WindowStmt lhs Sym -> WindowStmt
        self.env = ChainMap()

    def push(self):
        self.env = self.env.new_child()

    def pop(self):
        self.env = self.env.parents

    def map_s(self, s):
        if isinstance(s, (LoopIR.For, LoopIR.If)):
            self.push()
            stmts = super().map_s(s)
            self.pop()
            return stmts
        return super().map_s(s)

    def map_stmts(self, stmts):
        # Body of For/If is mapped inside the push/pop of map_s, but
        # each statement list (body/orelse) is still its own scope.
        self.push()
        result = super().map_stmts(stmts)
        self.pop()
        return result


def _root(window_stmt):
    """(root tensor Sym, root-relative idx) of a desugared WindowStmt

    Uses rhs.name/rhs.idx, not rhs.type: after desugaring the rhs always
    indexes the root tensor, while some rewrites leave T.Window.src_buf stale.
    """
    return window_stmt.rhs.name, window_stmt.rhs.idx


def _pts(idx):
    return [w.pt if isinstance(w, LoopIR.Point) else w for w in idx]


class DesugarDerivedWindows(_ScopedRewrite):
    def __init__(self, proc):
        super().__init__()
        self.result = self.apply_proc(proc)

    def map_s(self, s):
        if isinstance(s, LoopIR.WindowStmt):
            new_s = (super().map_s(s) or [s])[0]
            self.env[s.name] = new_s
            return [new_s]
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)) and s.name in self.env:
            new_s = (super().map_s(s) or [s])[0]
            root, widx = _root(self.env[s.name])
            idx = _pts(chain_window_idx(widx, new_s.idx))
            return [new_s.update(name=root, idx=idx, derived=s.name)]
        return super().map_s(s)

    def map_e(self, e):
        new_e = super().map_e(e) or e
        if not hasattr(e, "name") or e.name not in self.env:
            return new_e if new_e is not e else None
        wstmt = self.env[e.name]
        root, widx = _root(wstmt)
        if isinstance(e, LoopIR.Read):
            if new_e.idx:
                idx = _pts(chain_window_idx(widx, new_e.idx))
                return new_e.update(name=root, idx=idx, derived=e.name)
            # Bare window passed as call argument.
            return wstmt.rhs.update(derived=e.name, srcinfo=e.srcinfo)
        elif isinstance(e, LoopIR.WindowExpr):
            assert e.derived is None
            # Don't trust new_e.type: may be relative to an inlined window param.
            idx = chain_window_idx(widx, new_e.idx)
            # Keep the use's as_tensor (its shape doesn't depend on the root).
            typ = T.Window(wstmt.rhs.type.src_type, new_e.type.as_tensor, root, idx)
            return new_e.update(name=root, idx=idx, type=typ, derived=e.name)
        elif isinstance(e, LoopIR.StrideExpr):
            dims = [d for d, w in enumerate(widx) if isinstance(w, LoopIR.Interval)]
            return new_e.update(name=root, dim=dims[e.dim])
        else:
            assert False, f"unexpected {type(e)} of derived window"


def desugar_derived_windows(proc):
    return DesugarDerivedWindows(proc).result


def desugar_derived_windows_stmts(stmts):
    d = DesugarDerivedWindows.__new__(DesugarDerivedWindows)
    _ScopedRewrite.__init__(d)
    return d.apply_stmts(stmts)


class DerivedWindowError(TypeError):
    pass


def _sub(a, lo):
    if isinstance(lo, LoopIR.Const) and lo.val == 0:
        return a
    diff = _affine(a)
    _affine(lo, -1, diff)
    if all(v == 0 for k, v in diff.items() if k != ""):
        return LoopIR.Const(diff.get("", 0), T.int, a.srcinfo)
    # Try cancelling lo against top-level summands of a, e.g. (i + 4*j) - 4*j
    summands = []

    def flatten(e):
        if isinstance(e, LoopIR.BinOp) and e.op == "+":
            flatten(e.lhs)
            flatten(e.rhs)
        else:
            summands.append(e)

    flatten(a)
    target = {k: v for k, v in _affine(lo).items() if v != 0}
    kept = []
    for t in summands:
        t_aff = {k: v for k, v in _affine(t).items() if v != 0}
        if t_aff and all(target.get(k) == v for k, v in t_aff.items()):
            for k in t_aff:
                del target[k]
        else:
            kept.append(t)
    if not target and kept:
        result = kept[0]
        for t in kept[1:]:
            result = LoopIR.BinOp("+", result, t, T.index, a.srcinfo)
        return result
    if isinstance(a, LoopIR.BinOp) and a.op == "+" and str(a.lhs) == str(lo):
        return a.rhs
    return LoopIR.BinOp("-", a, lo, T.index, a.srcinfo)


def _affine(e, scale=1, acc=None):
    """Affine normal form {atom: coeff} with "" as the constant's key.
    Non-affine subterms are opaque atoms keyed by their printed form."""
    acc = {} if acc is None else acc

    def add(k, v):
        acc[k] = acc.get(k, 0) + v

    if isinstance(e, LoopIR.Const):
        add("", scale * e.val)
    elif isinstance(e, LoopIR.Read) and not e.idx:
        add(e.name, scale)
    elif isinstance(e, LoopIR.USub):
        _affine(e.arg, -scale, acc)
    elif isinstance(e, LoopIR.BinOp) and e.op in ("+", "-"):
        _affine(e.lhs, scale, acc)
        _affine(e.rhs, scale if e.op == "+" else -scale, acc)
    elif (
        isinstance(e, LoopIR.BinOp) and e.op == "*" and isinstance(e.lhs, LoopIR.Const)
    ):
        _affine(e.rhs, scale * e.lhs.val, acc)
    elif (
        isinstance(e, LoopIR.BinOp) and e.op == "*" and isinstance(e.rhs, LoopIR.Const)
    ):
        _affine(e.lhs, scale * e.rhs.val, acc)
    else:
        add(("opaque", str(e)), scale)
    return acc


def _same(a, b):
    # Affine normalization; TODO consider SMT with context (if conditions).
    diff = _affine(a)
    _affine(b, -1, diff)
    return all(v == 0 for v in diff.values())


class UnifyDerivedWindows(_ScopedRewrite):
    def __init__(self, proc):
        super().__init__()
        self.result = self.apply_proc(proc)

    def unify_idx(self, node, idx):
        """Return (window Sym, window stmt, idx relative to the window)

        Point coordinates of the WindowStmt must equal the access's (affine
        normalization). Intervals are NOT checked for containment in the
        window: TMA tail tiles routinely extend past the Sm90_tensorMap
        window, relying on the UNSOUND allow_out_of_bounds hack
        (see AccessInfo.allow_out_of_bounds). For plain windows an access
        outside the window but inside the tensor is the same address anyway.
        """
        w = node.derived
        wstmt = self.env.get(w)
        if wstmt is None:
            raise DerivedWindowError(
                f"{node.srcinfo}: {node} requests derived window {w}, "
                f"which is not in scope (statements re-ordered or deleted?)"
            )
        root, widx = _root(wstmt)
        if root != node.name:
            raise DerivedWindowError(
                f"{node.srcinfo}: {node} requests derived window {w} of {root}"
            )
        new_idx = []
        for wd, ed in zip(widx, idx):
            is_pt = not isinstance(ed, LoopIR.Interval)
            pt = ed.pt if isinstance(ed, LoopIR.Point) else ed
            if isinstance(wd, LoopIR.Point):
                if not is_pt or not _same(wd.pt, pt):
                    raise DerivedWindowError(
                        f"{node.srcinfo}: {node} cannot be expressed through "
                        f"{w} = {wstmt.rhs}: point coordinate {wd.pt} not matched"
                    )
            elif isinstance(ed, LoopIR.Point):
                new_idx.append(ed.update(pt=_sub(ed.pt, wd.lo)))
            elif isinstance(ed, LoopIR.Interval):
                new_idx.append(ed.update(lo=_sub(ed.lo, wd.lo), hi=_sub(ed.hi, wd.lo)))
            else:
                new_idx.append(_sub(ed, wd.lo))
        return w, wstmt, new_idx

    def map_s(self, s):
        if isinstance(s, LoopIR.WindowStmt):
            new_s = (super().map_s(s) or [s])[0]
            self.env[s.name] = s  # root form, pre-unify
            return [new_s]
        if isinstance(s, (LoopIR.Assign, LoopIR.Reduce)) and s.derived is not None:
            new_s = (super().map_s(s) or [s])[0]
            w, _, idx = self.unify_idx(new_s, new_s.idx)
            return [new_s.update(name=w, idx=idx, derived=None)]
        return super().map_s(s)

    def map_e(self, e):
        new_e = super().map_e(e)
        if not isinstance(e, (LoopIR.WindowExpr, LoopIR.Read)) or e.derived is None:
            return new_e
        e = new_e or e
        w, wstmt, idx = self.unify_idx(e, e.idx)
        if isinstance(e, LoopIR.Read):
            return e.update(name=w, idx=idx, derived=None)
        typ = create_window_type(w, wstmt.rhs.type, idx)
        return LoopIR.WindowExpr(w, idx, None, typ, e.srcinfo)


def unify_derived_windows(proc):
    return UnifyDerivedWindows(proc).result


def check_no_config_reads(window_expr, err):
    """Window statement indices must not read configs: desugaring moves the
    index computation to each use, which is wrong if the config changes."""
    found = []

    class _Find(LoopIR_Rewrite):
        def map_e(self, e):
            if isinstance(e, LoopIR.ReadConfig):
                found.append(e)
            return super().map_e(e)

    _Find().apply_e(window_expr)
    if found:
        err(f"window statement index cannot read config {found[0]}")
