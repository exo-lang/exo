from .inspection import *

exo_exceptions_ = {
    ValueError,
    TypeError,
    SchedulingError,
    InvalidCursorError,
}


def attempt(op, errs=exo_exceptions_):
    errs = tuple(errs)

    def rewrite(p, *args, rs=False, **kwargs):
        try:
            res = op(p, *args, **kwargs), True
        except errs:
            res = p, False
        if not rs:
            res = res[0]
        return res

    return rewrite


def apply(op):
    def rewrite(proc, cursors, *args, **kwargs):
        for c in cursors:
            proc = op(proc, c, *args, **kwargs)
        return proc

    return rewrite


def predicate(op, pred):
    def rewrite(proc, *args, **kwargs):
        if pred(proc, *args, **kwargs):
            proc = op(proc, *args, **kwargs)
        return proc

    return rewrite


def make_pass(op, trav_start=nlr_stmts):
    def rewrite(proc, block=InvalidCursor(), *args, **kwargs):
        stmts = trav_start(proc, block)
        return apply(op)(proc, stmts, *args, **kwargs)

    return rewrite


def lift_rc(op, attr):
    def rewrite(*args, **kwargs):
        proc, cursors = op(*args, **kwargs, rc=True)
        c = getattr(cursors, attr)
        return proc, c

    return rewrite


def repeate(op):
    op_attempt = attempt(op)

    def rewrite(p, *args, **kwargs):
        success = True
        while success:
            p, success = op_attempt(p, *args, **kwargs, rs=True)
        return p

    return rewrite


__all__ = ["apply", "attempt", "make_pass", "lift_rc", "repeate", "predicate"]
