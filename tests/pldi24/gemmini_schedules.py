from __future__ import annotations

import exo.API_cursors as pc
from exo.libs.memories import GEMM_SCRATCH, GEMM_ACCUM
from exo import proc, instr, DRAM, config, ExoType
from exo.stdlib.scheduling import *
from exo.stdlib.stdlib import *
from exo.stdlib.inspection import *


def repeat(p, func, stmt, fwd=lambda p, c: p.forward(c)):
    while True:
        try:
            p = func(p, fwd(p, stmt))
        except:
            break
    return p


def lift_config(p, config):
    def lift_config_helper(p, config):
        p = repeat(p, reorder_stmts, config, lambda p, c: p.forward(c).expand(1, 0))
        p = fission(p, config.after(), unsafe_disable_checks=True)
        p = remove_loop(p, config.parent())
        p = repeat(p, reorder_stmts, config, lambda p, c: p.forward(c).expand(1, 0))
        return p

    return repeat(p, lift_config_helper, config)


def replace_and_inline(p, tup):
    """
    args:
        - tup: a tuple of an equivalent @instr procedures (instrucion 1, instruction 2)

    given:
        @instr(...)
        def instr1(...):
            s1;
        &&
        @instr(...)
        def instr2(...):
            some_config_statement
            s1;

    rewrite is:

    for i in seq(0, hi):
        s1;
        s2

    --->

    some_config_statement
    for i in seq(0, hi):
        s1
        s2
    """
    p = replace_once(p, tup[0])
    p = call_eqv(p, tup[0], tup[1])
    c = p.find(str(tup[1].name()) + "(_)").parent()
    p = inline(p, tup[1])

    for s in p.forward(c).body():
        if isinstance(s, pc.WindowStmtCursor):
            p = inline_window(p, s)
        if isinstance(s, pc.CallCursor):
            if str(s.subproc().name()).startswith("config"):
                p = lift_config(p, s)

    return p


def fission_as_much_as_possible(p, cursor):
    """
    for i in ...:
        for j in ...:
            s1
            s2        <- cursor
            s3
    --->
    for i in ...:
        for j in ...:
            s2

    for i in ...:
        for j in ...:
            s1
            s3
    """
    cursor = p.forward(cursor)
    p = reorder_top(p, cursor)
    gap_c = cursor.after()
    while True:
        try:
            p = fission(p, gap_c)
            gap_c = p.forward(gap_c).parent().after()
        except:
            break

    return p


def sink_if(p, if_cursor):
    """
    if ...:           <- if_cursor
        for i in ...:
            for j in ...:
                s1
    --->
    for i in ...:
        for j in ...:
            if ...:   <- if_cursor
                s1
    """
    while True:
        if not isinstance(if_cursor.body()[0], pc.ForCursor):
            break
        else:
            p = lift_scope(p, if_cursor.body()[0])
            if_cursor = if_cursor.body()[0]
    return p


def add_guard(p, c):
    """
    for i in ...:
        for j in ...:
            s1[j]      <- c
    --->
    for i in ...:
        for j in ...:
            if i == 0:
                s1[j]  <- c
    """
    c = p.forward(c)
    while True:
        c = c.parent()
        if not isinstance(c, pc.ForCursor):
            break
        try:
            hi = c.hi().value()
            name = c.name()
            child = p.forward(c).body()[0]
            p = remove_loop(p, c)
            p = add_loop(p, child, name, hi, guard=True)
            if_cursor = p.forward(child).parent().body()[0]
            p = sink_if(p, if_cursor)
        except:
            continue
    return p


def remove_redundant_loops(p, c, num=0):
    """
    for i in ...:
        for j in ...:
            s1[j]      <- c
    --->
    for j in ...:
        s1[j]          <- c
    """
    c = p.forward(c)
    cur_depth = 0
    while True:
        c = c.parent()
        if not isinstance(c, pc.ForCursor):
            break
        try:
            if cur_depth >= num:
                break
            hi = c.hi().value()
            name = c.name()
            child = p.forward(c).body()[0]
            p = remove_loop(p, c)
            cur_depth += 1
        except:
            continue
    return p


def find_child_loop(loop_c, name):
    """
    args:
         - loop_c: cursor to a loop
         - name:   loop iteration name to find

    input:
        for i in ...:         <- loop_c
            for j in ...:
                for k in ...: <- name

    return:
        a target loop cursor (k), depth of the target loop (2)
    """
    count = 0
    while True:
        try:
            if len(loop_c.body()) == 1:
                count += 1
                child_loop = loop_c.body()[0]
                if isinstance(child_loop, pc.ForCursor) and child_loop.name() == name:
                    return child_loop, count
                loop_c = loop_c.body()[0]
            else:
                break
        except:
            break

    return None, None


def fuse_two_loops(p, c):
    """
    for i in ...:         <- c
        for j in ...:
            s1
    for k in ...:         <- c.next()
        for i in ...:
            s2
    ---->
    for i in ...:         <- c
        for j in ...:
            s1
        for k in ...:
            s2
    """
    try:
        next_c = c.next()
    except:
        return p, False

    if isinstance(c, pc.ForCursor) and isinstance(next_c, pc.ForCursor):
        if c.name() == next_c.name() and c.hi().value() == next_c.hi().value():
            p = fuse(p, c, next_c, unsafe_disable_check=True)
            return p, True
        else:
            tgt_c, count = find_child_loop(next_c, c.name())
            if tgt_c:
                p = lift_scope_n(p, tgt_c, n_lifts=count)
                p = fuse(p, c, tgt_c, unsafe_disable_check=True)
                return p, True

    return p, False


def fuse_all_loops(p, cursor):
    """
    recursively calls fuse_two_loops to all the loops
    """
    while True:
        if isinstance(cursor, pc.ForCursor):
            p = fuse_all_loops(p, cursor.body()[0])

        # Fuse in current scope
        p, b = fuse_two_loops(p, cursor)

        if b:
            cursor = p.forward(cursor)
        else:
            try:
                cursor = p.forward(cursor).next()
            except:
                break

    return p


def autolift_alloc(p, alloc_c, dep_set=None, max_size=0, lift=True):
    """
    for i in seq(0, 10):
        for j in seq(0, 20):
            a : R          <- alloc_c, dep_set = {'i'}
            a[i] = ...
    ---->
    a : R[10]              <- if size is less than max_size
    for i in seq(0, n):
        for j in seq(0, m):
            a[i] = ...
    """
    alloc_c = p.forward(alloc_c)
    loop_c = get_enclosing_loop(p, alloc_c)
    accum_size = 1
    while True:
        try:
            if not isinstance(loop_c, pc.ForCursor):
                break
            if dep_set == None or loop_c.name() in dep_set:
                if accum_size * loop_c.hi().value() <= max_size:
                    p = expand_dim(p, alloc_c, loop_c.hi().value(), loop_c.name())
                    accum_size = accum_size * loop_c.hi().value()
            if lift:
                p = lift_alloc(p, alloc_c)
            loop_c = loop_c.parent()
        except:
            break
    return p


def bind_and_lift(p, expr_c, max_size=0, lift=True):
    """
    for i in seq(0, 10):
        for j in seq(0, 20):
            C[...] = A[i] <- expr_c
    ---->
    a : R[10]             <- if size is less than max_size
    for i in seq(0, n):
        for j in seq(0, m):
            a[i] = A[i]
            C[...] = a[i]
    """
    dep_set = set(d for d in get_symbols(p, expr_c))
    p = bind_expr(p, [expr_c], expr_c.name() + "_tmp")
    assign_c = expr_c.parent()
    load_c = p.forward(assign_c).prev()
    alloc_c = p.forward(assign_c).prev().prev()
    # TODO: set_precision should handle ExoType
    if assign_c.rhs().type() == ExoType.I8:
        p = set_precision(p, alloc_c, "i8")

    return (
        autolift_alloc(p, alloc_c, dep_set, max_size=max_size, lift=lift),
        load_c,
        alloc_c,
    )


def reorder_top(p, c):
    """
    for i in seq(0, 10):
        s1
        s2
        s3  <- c
    ---->
    for i in seq(0, 10):
        s3  <- c
        s1
        s2
    """
    c = p.forward(c)
    while True:
        try:
            p = reorder_stmts(p, c.expand(1, 0))
            c = p.forward(c)
        except:
            break
    return p


def lift_scope_n(p, c, n_lifts=1):
    """
    for i in seq(0, 10):
        for j in seq(0, 10):
            for k in seq(0, 10):
                if ...:  <- c
                    s1
    ----> if n_lifts == 2:
    for i in seq(0, 10):
        if ...:  <- c
            for j in seq(0, 10):
                for k in seq(0, 10):
                    s1
    """
    for i in range(0, n_lifts):
        p = lift_scope(p, c)
    return p


def recurse_add(idx):
    if isinstance(idx, pc.ReadCursor):
        return [idx.name()]
    elif isinstance(idx, pc.BinaryOpCursor):
        return recurse_add(idx.lhs()) + recurse_add(idx.rhs())
    else:
        return []


def reorder_loops_from_idx(proc, cursor):
    """
    for k in seq(0, 10):
        for j in seq(0, 10):
            for i in seq(0, 10):
                a[i, j, k] = ... <--- cursor
    ---->:
    for i in seq(0, 10):
        for j in seq(0, 10):
            for k in seq(0, 10):
                a[i, j, k] = ...
    """
    cursor = proc.forward(cursor)
    idx_list = []
    for idx in cursor.idx():
        idx_list.extend(recurse_add(idx))

    top_par = get_enclosing_loop(proc, cursor)
    while True:
        if (
            isinstance(top_par.parent(), pc.InvalidCursor)
            or len(top_par.parent().body()) != 1
        ):
            break
        top_par = top_par.parent()

    while True:
        if len(top_par.body()) != 1 or not idx_list:
            break

        if top_par.name() == idx_list[0]:
            idx_list.pop(0)
        elif top_par.name() in idx_list:
            new_loop, _ = find_child_loop(top_par, idx_list[0])
            if new_loop != None:
                proc = lift_scope(proc, new_loop)
                top_par = proc.forward(top_par.parent()).body()[0]
            else:
                idx_list.pop(0)
        else:
            top_par = top_par.body()[0]

    return proc


def unroll_all(p, lis):
    for l in lis:
        p = unroll_loop(p, l)
    return p
