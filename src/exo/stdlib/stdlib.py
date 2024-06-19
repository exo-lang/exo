from __future__ import annotations
from dataclasses import dataclass

from exo import *
from exo.syntax import *
from exo.API_cursors import *

from .scheduling import *
from .inspection import *
from .higher_order import *
from .rc_wrappers import *


@dataclass
class bind_and_set_expr_cursors:
    alloc: AllocCursor
    bound_expr: ExprCursor
    expr_reads: ExprCursor


def bind_and_set_expr(proc, exprs, precision, memory, new_name=None, rc=False):
    if new_name is None:
        new_name = next(get_unique_names(proc))

    expr = exprs if isinstance(exprs, ExprCursor) else exprs[0]
    stmt = get_enclosing_stmt(proc, expr)
    proc = bind_expr(proc, exprs, new_name)
    proc = set_precision(proc, new_name, precision)
    proc = set_memory(proc, new_name, memory)

    alloc = get_declaration(proc, stmt, new_name)
    bound_expr = alloc.next().rhs()

    if not rc:
        return proc
    # Disabled since forwarding after replace is not supported now
    # exprs = [proc.forward(e) for e in exprs]
    return proc, bind_and_set_expr_cursors(alloc, bound_expr, exprs)


def parallelize_and_lift_alloc(proc, alloc_cursor, n_lifts=1):
    """
    for i in seq(0, hi):
        B1;
        name: type[shape];
        B2;

    ----->

    name: type[hi][shape]
    for i in seq(0, hi):
        B1;
        B2;
    """
    alloc_cursor = proc.forward(alloc_cursor)
    for i in range(n_lifts):
        alloc_cursor = proc.forward(alloc_cursor)
        enclosing_scope = alloc_cursor.parent()
        if isinstance(enclosing_scope, ForCursor):
            proc = expand_dim(
                proc,
                alloc_cursor,
                expr_to_string(enclosing_scope.hi()),
                enclosing_scope.name(),
            )
        proc = lift_alloc(proc, alloc_cursor)
    return proc


def parallelize_allocs(proc, cursor):
    if not isinstance(cursor, (ForCursor, IfCursor)):
        raise SchedulingError(
            f"Got type {type(cursor)}, expected {ForCursor} or {IfCursor}"
        )

    allocs = filter(lambda s: isinstance(s, AllocCursor), nlr_stmts(proc, cursor))
    func = lambda proc, alloc: parallelize_and_lift_alloc(
        proc, alloc, get_distance(proc, alloc, cursor)
    )
    return apply(func)(proc, allocs)


def interleave_loop(proc, loop, factor=None, par_reduce=False, memory=DRAM, tail="cut"):
    """
    for i in seq(0, c):
        S1
        S2
        S3

    ----->
    s1 x c
    s2 x c
    s3 x c
    """
    if factor == 1:
        return proc

    loop = proc.forward(loop)

    def rewrite(proc, loop, factor=None, par_reduce=False, memory=DRAM, tail="cut"):
        loop = proc.forward(loop)
        if factor is not None:
            proc, (outer, loop, _) = divide_loop_(
                proc, loop, factor, tail=tail, rc=True
            )
        else:
            outer = loop.parent()
        if par_reduce:
            proc = parallelize_all_reductions(proc, outer, memory=memory, unroll=True)
            loop = proc.forward(outer).body()[0]
        allocs = filter(lambda s: isinstance(s, AllocCursor), loop.body())
        proc = apply(parallelize_and_lift_alloc)(proc, allocs)

        stmts = list(proc.forward(loop).body())
        proc = apply(fission)(proc, [s.after() for s in stmts[:-1]])
        proc = apply(unroll_loop)(proc, [proc.forward(s).parent() for s in stmts])
        return proc

    if tail in {"cut", "cut_and_guard"}:
        proc = rewrite(proc, loop, factor, par_reduce, memory, tail)
    elif tail == "recursive":
        if factor is None:
            raise SchedulingError(
                "Cannot specify recursive tail strategy and factor=None"
            )
        proc, (_, inners, _) = divide_loop_recursive(
            proc, loop, factor, tail="cut", rc=True
        )
        proc = apply(rewrite)(proc, inners, par_reduce=par_reduce, memory=memory)
    elif tail == "specialize":
        if factor is None:
            raise SchedulingError(
                "Cannot specify recursive tail strategy and factor=None"
            )
        proc = rewrite(proc, loop, factor, par_reduce, memory, tail="cut")
        tail_loop = proc.forward(loop).next()
        proc, (stmts,) = binary_specialize(
            proc, tail_loop, tail_loop.hi(), [i for i in range(factor)], rc=True
        )
        proc = apply(rewrite)(proc, stmts, par_reduce=par_reduce, memory=memory)
    else:
        raise SchedulingError(f"Unknown tail strategy: {tail}")
    return proc


@dataclass
class hoist_stmt_cursors:
    allocs: list[AllocCursor]
    stmt: StmtCursor
    loop: ForCursor

    def __iter__(self):
        yield self.allocs
        yield self.stmt
        yield self.loop


def hoist_stmt(proc, stmt, rc=False):
    """
    for i in seq(0, hi):
        B1;
        s;
        B2;

    --->

    s;
    for i in seq(0, hi):
        B1;
        B2;
    """
    stmt = proc.forward(stmt)

    # Type Checking
    if not isinstance(stmt, StmtCursor):
        raise SchedulingError("Cannot hoist cursor that are not statements")

    loop = stmt.parent()

    # Pre-condition 1: a scope exists
    if not isinstance(loop, ForCursor):
        raise SchedulingError("Statement is not within a loop")

    # Pre-condition 2: fail-fast, no dependency on a loop
    deps = list(get_symbols(proc, stmt))
    if isinstance(loop, ForCursor) and loop.name() in deps:
        raise SchedulingError(
            "Cannot hoist cursor to a statement that depends on enclosing loop"
        )

    # Alloc is a special case
    if isinstance(stmt, AllocCursor):
        proc = lift_alloc(proc, stmt)
        if not rc:
            return proc
        loop = proc.forward(loop)
        stmt = proc.forward(stmt)
        return proc, hoist_stmt_cursors([stmt], stmt, loop)

    allocs = []

    # Reorder the statement to the top of the loop
    while not isinstance(stmt.prev(), InvalidCursor):
        prev_stmt = stmt.prev()
        if isinstance(prev_stmt, AllocCursor) and prev_stmt.name() in deps:
            proc = lift_alloc(proc, prev_stmt)
            allocs.append(prev_stmt)
        else:
            proc = reorder_stmts(proc, stmt.expand(1, 0))
        stmt = proc.forward(stmt)

    # Pull the statement on its own outside the loop
    if len(loop.body()) > 1:
        proc, (_, loop) = fission_(proc, stmt.after(), rc=True)
        stmt = proc.forward(stmt)
    proc = remove_loop(proc, stmt.parent())

    if not rc:
        return proc

    allocs = [proc.forward(a) for a in allocs]
    stmt = proc.forward(stmt)
    loop = proc.forward(loop)
    return proc, hoist_stmt_cursors(allocs, stmt, loop)


@dataclass
class hoist_from_loop_cursors:
    hoisted: list
    loop: ForCursor

    def __iter__(self):
        yield self.hoisted
        yield self.loop


def hoist_from_loop(proc, loop, rc=False):
    loop = proc.forward(loop)

    if not is_loop(proc, loop):
        raise SchedulingError(f"loop must of type {ForCursor} not {type(loop)}")

    rcs = []

    def hoist_non_alloc(proc, stmt):
        stmt = proc.forward(stmt)
        if isinstance(stmt, AllocCursor):
            return proc
        proc, cursors = hoist_stmt(proc, stmt, rc=True)
        rcs.append(cursors)
        return proc

    proc = apply(attempt(hoist_non_alloc))(proc, loop.body())

    if not rc:
        return proc

    if not rcs:
        return proc, hoist_from_loop_cursors([], loop)

    loop = rcs[-1].loop
    stmt_allocs = []
    for cursors in rcs:
        stmt = proc.forward(cursors.stmt)
        allocs = tuple(proc.forward(alloc) for alloc in cursors.allocs)
        stmt_allocs.append((stmt, allocs))
    return proc, hoist_from_loop_cursors(stmt_allocs, loop)


@dataclass
class jam_stmt_cursors:
    loop: ForCursor
    stmt: StmtCursor

    def __iter__(self):
        yield self.loop
        yield self.stmt


def jam_stmt(proc, stmt, unsafe_disable_check=False, rc=False):
    stmt = proc.forward(stmt)
    loop = stmt.next()
    if not is_loop(proc, loop):
        raise SchedulingError("Next statement must be a loop.")

    proc = add_loop(
        proc,
        stmt,
        loop.name(),
        FormattedExprStr("_ - _", loop.hi(), loop.lo()),
        unsafe_disable_check=unsafe_disable_check,
    )
    stmt = proc.forward(stmt)
    stmt_loop = proc.forward(stmt).parent()
    proc = shift_loop(proc, stmt_loop, FormattedExprStr("_", loop.lo()))
    proc = simplify(proc)
    proc = fuse(proc, stmt_loop, loop, unsafe_disable_check=unsafe_disable_check)
    proc = repeate(reorder_stmt_forward)(proc, stmt)

    if not rc:
        return proc
    stmt = proc.forward(stmt)
    stmt_loop = proc.forward(stmt_loop)
    return proc, jam_stmt_cursors(stmt_loop, stmt)


def parallelize_reduction(
    proc, reduce_stmt, factor=None, memory=DRAM, nth_loop=1, unroll=False
):
    # Auto-coersion
    if isinstance(unroll, bool):
        unroll = (unroll, unroll)

    reduce_stmt = proc.forward(reduce_stmt)

    if not is_reduce(proc, reduce_stmt):
        raise TypeError(f"reduce_stmt must of type {ReduceCursor}")

    reduce_loop = get_enclosing_loop(proc, reduce_stmt, nth_loop)
    control_vars = get_symbols(proc, reduce_stmt.idx())
    if reduce_loop.name() in control_vars:
        raise SchedulingError("Statement isn't a reduction over the specified loop.")

    if factor is not None:
        proc, (reduce_loop, _, _) = divide_loop_(
            proc, reduce_loop, factor, tail="guard", rc=True
        )
        proc = simplify(proc)
        nth_loop += 1

    # Stage reduction around the loop we are reducing over
    proc = reorder_loops(proc, reduce_loop)
    proc = auto_stage_mem(
        proc,
        reduce_loop,
        reduce_stmt.name(),
        accum=True,
    )
    proc = simplify(proc)

    # Set the memory of the newly created buffer
    reduce_loop = proc.forward(reduce_loop)
    alloc = reduce_loop.prev().prev()
    proc = set_memory(proc, alloc, memory)

    # Parallelize the reduction
    reduce_loop = proc.forward(reduce_loop)
    proc = parallelize_and_lift_alloc(proc, reduce_loop.prev().prev())

    # Fission the zero and store-back stages
    proc = fission(proc, reduce_loop.before())
    proc = fission(proc, reduce_loop.after())

    # Reorder the loop nest back
    reduce_loop = proc.forward(reduce_loop)
    proc = reorder_loops(proc, reduce_loop.parent())

    # Unroll any loops
    reduce_loop = proc.forward(reduce_loop)
    if unroll[0]:
        proc = unroll_loop(proc, reduce_loop.prev())
    if unroll[1]:
        proc = unroll_loop(proc, reduce_loop.next())

    if factor is not None:
        proc = undo_divide_and_guard_loop(proc, reduce_loop)
    return proc


def parallelize_all_reductions(proc, loop, factor=None, memory=DRAM, unroll=False):
    loop = proc.forward(loop)

    def rewrite(proc, s):
        s = proc.forward(s)
        reduc_loop = proc.forward(loop)
        nth_loop = 0
        for parent in get_parents(proc, s):
            if is_loop(proc, parent):
                nth_loop += 1
            if parent == reduc_loop:
                break
        return parallelize_reduction(proc, s, factor, memory, nth_loop, unroll)

    return make_pass(attempt(rewrite))(proc, loop.body())


def unroll_and_jam(proc, loop, factor, unroll=(True, True, True)):
    loop = proc.forward(loop)
    inner_loops = [i for i in loop.body() if isinstance(i, ForCursor)]
    if len(inner_loops) > 1:
        raise SchedulingError("Multiple loops found, decision is ambigious")
    if len(inner_loops) == 0:
        raise SchedulingError("No loops found")

    return interleave_outer_loop_with_inner_loop(
        proc, loop, inner_loops[0], factor, unroll=unroll
    )


def unroll_and_jam_parent(proc, loop, factor, unroll=(True, True, True)):
    loop = proc.forward(loop)
    outer_loop = loop.parent()
    if not isinstance(outer_loop, ForCursor):
        raise SchedulingError("parent is not a loop")
    return interleave_outer_loop_with_inner_loop(
        proc, outer_loop, loop, factor, unroll=unroll
    )


def interleave_outer_loop_with_inner_loop(
    proc,
    outer_loop_cursor,
    inner_loop_cursor,
    interleave_factor,
    unroll=(True, True, True),
):
    # TODO: check if inner_loop is directly in the body of outer_loop
    outer_loop_cursor = proc.forward(outer_loop_cursor)
    inner_loop_cursor = proc.forward(inner_loop_cursor)

    if (
        isinstance(outer_loop_cursor.hi(), LiteralCursor)
        and outer_loop_cursor.hi().value() == interleave_factor
    ):
        middle_loop_cursor = outer_loop_cursor
    else:
        proc = divide_loop(
            proc,
            outer_loop_cursor,
            interleave_factor,
            (outer_loop_cursor.name() + "o", outer_loop_cursor.name() + "i"),
            tail="cut",
        )

        outer_loop_cursor = proc.forward(outer_loop_cursor)
        middle_loop_cursor = outer_loop_cursor.body()[0]
    middle_loop_stmts = list(middle_loop_cursor.body())

    proc = simplify(proc)
    for stmt in middle_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = parallelize_and_lift_alloc(proc, stmt)
    inner_loop_cursor = proc.forward(inner_loop_cursor)

    if not isinstance(inner_loop_cursor.prev(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.before())
        inner_loop_cursor = proc.forward(inner_loop_cursor)
        if unroll[0]:
            proc = unroll_loop(proc, inner_loop_cursor.parent().prev())

    if not isinstance(inner_loop_cursor.next(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.after())
        inner_loop_cursor = proc.forward(inner_loop_cursor)
        if unroll[2]:
            proc = unroll_loop(proc, inner_loop_cursor.parent().next())

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    proc = reorder_loops(proc, inner_loop_cursor.parent())
    if unroll[1]:
        proc = unroll_loop(proc, inner_loop_cursor.parent())

    return proc


def fission_into_singles(proc, cursor):
    if not isinstance(cursor, (ForCursor, IfCursor)):
        raise SchedulingError(
            f"Got type {type(cursor)}, expected {ForCursor} or {IfCursor}"
        )

    cursor = proc.forward(cursor)

    def dfs(proc, cursor, n_lifts=0):
        if n_lifts and not is_end_of_body(proc, cursor):
            proc = fission(proc, cursor.after(), n_lifts)
        children = get_children(proc, cursor)
        children = filter(lambda s: isinstance(s, StmtCursor), children)
        return apply(dfs)(proc, children, n_lifts + 1)

    proc = parallelize_allocs(proc, cursor)
    return dfs(proc, cursor)


@dataclass
class vectorize_cursors:
    loop: ForCursor

    def __iter__(self):
        yield self.loop


def vectorize_predicate_tail(
    proc,
    loop,
    vec_width,
    precision,
    mem_type,
    instructions=[],
    rules=[],
    tail="cut_and_predicate",
    rc=False,
):
    proc = parallelize_all_reductions(proc, loop, factor=vec_width, memory=mem_type)
    proc = stage_compute(proc, loop, precision, mem_type, rules)
    proc, (outer, inner, _) = divide_loop_(proc, loop, vec_width, rc=True)
    proc = simplify(proc)
    proc = fission_into_singles(proc, inner)

    if tail == "cut_and_predicate":
        if_c = inner.body()[0]
        cut = FormattedExprStr("_ - 1", outer.hi())
        proc = attempt(cut_loop)(proc, outer, cut)
        outer = proc.forward(outer)
        proc = dce(proc, outer.body())
    proc = replace_all_stmts(proc, instructions)

    if not rc:
        return proc
    outer = proc.forward(outer)
    return proc, vectorize_cursors(outer)


def vectorize(
    proc,
    loop,
    vec_width,
    precision,
    mem_type,
    instructions=[],
    rules=[],
    tail="cut_and_predicate",
    rc=False,
):
    if tail in {"predicate", "cut_and_predicate"}:
        return vectorize_predicate_tail(
            proc, loop, vec_width, precision, mem_type, instructions, rules, tail, rc
        )
    proc, (outer, inner, _) = divide_loop_(proc, loop, vec_width, tail=tail, rc=True)
    proc = parallelize_all_reductions(proc, outer, memory=mem_type)

    outer = proc.forward(outer)
    inner = outer.body()[0]

    proc = stage_compute(proc, inner, precision, mem_type, rules)
    proc = fission_into_singles(proc, inner)

    proc = replace_all_stmts(proc, instructions)

    if not rc:
        return proc
    outer = proc.forward(outer)
    return proc, vectorize_cursors(outer)


def tile_loops(proc, loop_tile_pairs, perfect=False):

    loop_tile_pairs = [(proc.forward(i[0]), i[1]) for i in loop_tile_pairs]

    inner_loops = []
    for i in range(len(loop_tile_pairs)):
        outer_loop = loop_tile_pairs[i][0]
        tile_size = loop_tile_pairs[i][1]
        new_names = (outer_loop.name() + "o", outer_loop.name() + "i")
        if perfect or (
            isinstance(outer_loop.hi(), LiteralCursor)
            and (outer_loop.hi().value() % tile_size == 0)
        ):
            proc = divide_loop(proc, outer_loop, tile_size, new_names, perfect=True)
        else:
            proc = divide_loop(proc, outer_loop, tile_size, new_names, tail="cut")
        inner_loop = proc.forward(outer_loop).body()[0]
        inner_loops.append(inner_loop)

    for i in range(len(loop_tile_pairs) - 2, -1, -1):
        inner_loop = inner_loops[i]
        tile_size = loop_tile_pairs[i][1]
        for j in range(i + 1, len(loop_tile_pairs)):
            loop = loop_tile_pairs[j][0]
            proc = interleave_outer_loop_with_inner_loop(
                proc, inner_loop, loop, tile_size, (False, False, False)
            )
    return proc, [proc.forward(l) for l in inner_loops]


def tile_loops_bottom_up(proc, loop, tiles):

    cur_loop = loop
    for i in tiles[:-1]:
        if not len(cur_loop.body()) == 1:
            raise SchedulingError("All loop must have a body length of 1")
        if not isinstance(cur_loop.body()[0], ForCursor):
            raise SchedulingError("Did not find a nested loop")
        cur_loop = cur_loop.body()[0]

    loops = []
    cur_loop = loop
    for i in tiles:
        loops.append((cur_loop, i))
        cur_loop = cur_loop.body()[0]

    def get_depth(loop):
        if not isinstance(loop, (ForCursor, IfCursor)):
            return 0
        return max([get_depth(i) for i in loop.body()]) + 1

    def push_loop_in(proc, loop, depth):
        loop = proc.forward(loop)
        if get_depth(loop) == depth:
            return proc
        count = len(loop.body())
        for stmt in list(loop.body())[:-1]:
            proc = fission(proc, stmt.after())
        loop = proc.forward(loop)
        loops = []
        for i in range(count):
            loops.append(loop)
            loop = loop.next()
        for loop in loops:
            if get_depth(loop) == depth:
                continue
            loop = proc.forward(loop)
            child = loop.body()[0]
            if isinstance(child, ForCursor):
                proc = reorder_loops(proc, loop)
                forwarded_loop = proc.forward(loop)
            elif isinstance(child, IfCursor):
                proc = lift_scope(proc, child)
                child = proc.forward(child)
                forwarded_loop = child.body()[0]
            else:
                assert False, "Invalid"
            proc = push_loop_in(proc, forwarded_loop, depth)
        return proc

    for depth, (loop, tile) in enumerate(loops[::-1]):
        if tile is not None:
            proc, (_, inner, tail) = divide_loop_(
                proc, loop, tile, tail="cut_and_guard", rc=True
            )
            proc = push_loop_in(proc, inner, depth + 1)
            proc = push_loop_in(proc, tail.body()[0], depth + 1)
        else:
            proc = push_loop_in(proc, loop, depth + 1)

    return proc


def auto_stage_mem(proc, block, buff, new_buff_name=None, accum=False, rc=False):
    if new_buff_name is None:
        new_buff_name = next(get_unique_names(proc))

    if not isinstance(block, BlockCursor):
        block = proc.forward(block)
        block = block.as_block()

    block_nodes = list(lrn(proc, block))
    block_loops = list(filter(lambda s: is_loop(proc, s), block_nodes))
    block_accesses = filter(
        lambda s: is_access(proc, s) and s.name() == buff, block_nodes
    )

    def eval_rng(expr, env):
        expr = proc.forward(expr)
        if is_binop(proc, expr):
            lhs_rng = eval_rng(expr.lhs(), env)
            rhs_rng = eval_rng(expr.rhs(), env)
            join = lambda l, op, r: f"({l}) {op} ({r})"
            rng_l = join(lhs_rng[0], expr.op(), rhs_rng[0])
            rng_r = join(lhs_rng[1], expr.op(), rhs_rng[1])
            if is_add(proc, expr) or is_sub(proc, expr):
                return (rng_l, rng_r)
            else:
                rng = (rng_l, rng_r)
                if is_literal(proc, expr.rhs()):
                    if expr.rhs().value() < 0:
                        rng = rng[::-1]
                elif is_literal(proc, expr.lhs()):
                    if expr.lhs().value() < 0:
                        rng = rng[::-1]
                else:
                    assert False, "Unreachable case"
                return rng
        elif is_unary_minus(proc, expr):
            rng = eval_rng(expr.arg(), env)
            return (f"-({rng[1]})", f"-({rng[0]})")
        elif is_read(proc, expr):
            if expr.name() in env:
                return env[expr.name()]
            else:
                read = f"{expr.name()}"
                return (read, read)
        elif is_literal(proc, expr):
            val = f"{expr.value()}"
            return (val, val)
        else:
            assert False, "Unreachable case"

    def get_window(cursor):
        parents = get_parents(proc, cursor)
        my_loops = list(filter(lambda s: s in block_loops, parents))
        my_loops = my_loops[::-1]
        env = {}
        for loop in my_loops:
            lo_rng = eval_rng(loop.lo(), env)
            hi_rng = eval_rng(loop.hi(), env)
            env[loop.name()] = (lo_rng[0], hi_rng[1])
        window = tuple(eval_rng(idx, env) for idx in cursor.idx())
        return window

    def window_to_str(window):
        def get_dim_win(dim):
            if dim[0] == dim[1]:
                return dim[0]
            return f"{dim[0]}:{dim[1]}"

        dims = [get_dim_win(i) for i in window]
        window = ",".join(dims)
        window = f"{buff}[{window}]" if window else buff
        return window

    def my_stage_mem(proc, window):
        window = window_to_str(window)
        return attempt(stage_mem, errs=(SchedulingError,))(
            proc, block, window, new_buff_name, accum, rs=True
        )

    declaration = get_declaration(proc, block[0], buff)
    if not declaration.is_tensor():
        return stage_mem(proc, block, buff, new_buff_name, accum)

    # Start with the window as the entire buffer
    buff_window = [["0", eval_rng(dim, {})[0]] for dim in declaration.shape()]

    # Get the window per access
    accessess_windows = [get_window(access) for access in block_accesses]

    # Prune the candidates for each dim x side
    candidates = [(set(), set()) for i in buff_window]
    for access_window in accessess_windows:
        for i, dim in enumerate(access_window):
            for side in range(2):
                candidates[i][side].add(dim[side])
                candidates[i][side].add(dim[side])

    # Tighten each dimension and side of the buffer window
    for i, dim in enumerate(buff_window):
        for side in range(0, 2):
            # Prune
            if len(candidates[i][side]) == 1:
                dim[side] = next(iter(candidates[i][side]))
                continue
            for candidate in candidates[i][side]:
                old_side = dim[side]
                dim[side] = candidate
                new_proc, success = my_stage_mem(proc, buff_window)
                if not success:
                    dim[side] = old_side
    window = window_to_str(buff_window)
    return stage_mem_(proc, block, window, new_buff_name, accum, rc=rc)


def ordered_stage_expr(proc, expr_cursors, new_buff_name, precision, n_lifts=1):
    if not isinstance(expr_cursors, list):
        expr_cursors = [expr_cursors]

    if not all([isinstance(cursor, ExprCursor) for cursor in expr_cursors]):
        raise SchedulingError("auto_stage_mem expects a read a cursor")

    expr_cursors = [proc.forward(c) for c in expr_cursors]
    original_stmt = get_enclosing_stmt(proc, expr_cursors[0])

    proc = bind_expr(proc, expr_cursors, new_buff_name)
    original_stmt = proc.forward(original_stmt)
    assign_cursor = original_stmt.prev()
    alloc_cursor = assign_cursor.prev()
    expr_cursor = assign_cursor.rhs()
    deps = list(get_symbols(proc, expr_cursor))

    assert isinstance(assign_cursor, AssignCursor)
    assert isinstance(alloc_cursor, AllocCursor)

    anchor_stmt = assign_cursor

    def hoist_as_loop(proc, stmt_cursor):
        stmt_cursor = proc.forward(stmt_cursor)
        while not isinstance(stmt_cursor.prev(), InvalidCursor):
            proc = reorder_stmts(proc, stmt_cursor.expand(1, 0))
            stmt_cursor = proc.forward(stmt_cursor)

        proc = fission(proc, stmt_cursor.after())

        return proc

    for i in range(n_lifts):
        parent = anchor_stmt.parent()

        if not isinstance(parent, ForCursor):
            raise SchedulingError("Not implemented yet")
        if parent.name() in deps:
            proc = parallelize_and_lift_alloc(proc, alloc_cursor)
        else:
            proc = lift_alloc(proc, alloc_cursor)

        proc = hoist_as_loop(proc, anchor_stmt)
        anchor_stmt = proc.forward(anchor_stmt)
        anchor_stmt = anchor_stmt.parent()

    alloc_cursor = proc.forward(alloc_cursor)
    loop_nest = alloc_cursor.next()

    def try_removing_loops(proc, loop):
        child_stmt = loop.body()[0]
        if isinstance(child_stmt, ForCursor):
            proc = try_removing_loops(proc, child_stmt)
        try:
            proc = remove_loop(proc, loop)
        except:
            pass
        return proc

    proc = try_removing_loops(proc, loop_nest)
    alloc_cursor = proc.forward(alloc_cursor)
    proc = set_precision(proc, alloc_cursor, precision)
    scopes_nest = alloc_cursor.next()

    def lift_all_ifs(proc, scope, depth=0):
        if isinstance(scope, IfCursor):
            for i in range(depth):
                proc = lift_scope(proc, scope)
        child_stmt = scope.body()[0]
        if isinstance(child_stmt, (ForCursor, IfCursor)):
            proc = lift_all_ifs(proc, child_stmt, depth + 1)
        return proc

    proc = lift_all_ifs(proc, scopes_nest)

    return proc


def _eliminate_dead_code_pruned(proc, s):
    s = proc.forward(s)
    if isinstance(s, ForCursor) and is_loop_bounds_const(s):
        return proc
    else:
        return eliminate_dead_code(proc, s)


dce = make_pass(attempt(_eliminate_dead_code_pruned))


def unroll_buffers(proc, block=InvalidCursor(), mem=None):
    def rewrite(proc, alloc):
        alloc = proc.forward(alloc)
        if not isinstance(alloc, AllocCursor):
            return proc
        if not alloc.is_tensor():
            return proc
        diff = int(alloc.mem() is mem)
        if len(alloc.shape()) - diff:
            if isinstance(alloc.shape()[0], LiteralCursor):
                return unroll_buffer(proc, alloc, 0)
        return proc

    while True:
        new_proc = make_pass(rewrite)(proc, block)
        if new_proc == proc:
            break
        proc = new_proc
    return new_proc


def unfold_reduce(proc, reduce):
    if not isinstance(reduce, ReduceCursor):
        raise SchedulingError("Expected a reduce cursor")

    proc = auto_stage_mem(proc, reduce, reduce.name())
    reduce = proc.forward(reduce)
    alloc = reduce.prev().prev()
    proc = merge_writes(proc, reduce.as_block().expand(delta_lo=1, delta_hi=0))
    assign = proc.forward(alloc).next()
    proc = inline_assign(proc, assign)
    proc = delete_buffer(proc, alloc)

    return proc


def fma_rule(proc, expr):
    expr = proc.forward(expr)

    if is_add(proc, expr):
        if is_mul(proc, expr.lhs()):
            # (a * b) + c
            return [expr.lhs().lhs(), expr.lhs().rhs(), expr.rhs()]
        elif is_mul(proc, expr.rhs()):
            # a + (b * c)
            return [expr.lhs(), expr.rhs().lhs(), expr.rhs().rhs()]

    return None


def abs_rule(proc, expr):
    expr = proc.forward(expr)
    if is_select(proc, expr):
        args = expr.args()
        if (
            is_literal(proc, args[0], 0.0)
            and is_unary_minus(proc, args[3])
            and are_exprs_equal(proc, args[1], args[2])
            and are_exprs_equal(proc, args[1], args[3].arg())
        ):
            return [[args[1], args[2], args[3].arg()]]
    return None


def stage_expr_into_memory(proc, exprs, precision, memory):
    if not isinstance(exprs, list):
        exprs = [exprs]

    expr = proc.forward(exprs[0])

    # No need to stage if expr is already assigned
    # to the target memory
    parent = expr.parent()
    if (
        isinstance(parent, AssignCursor)
        and get_declaration(proc, expr, parent.name()).mem() is memory
    ):
        return proc, expr

    # No need to stage if expr is already read
    # from the target memory
    if (
        isinstance(expr, ReadCursor)
        and get_declaration(proc, expr, expr.name()).mem() is memory
    ):
        return proc, expr
    return lift_rc(bind_and_set_expr, "bound_expr")(proc, exprs, precision, memory)


def stage_compute(
    proc,
    block=InvalidCursor(),
    precision="R",
    memory=DRAM,
    children_ops=[],
):

    if not isinstance(children_ops, list):
        raise SchedulingError("Expected children_ops to be a list")

    def get_numeric_children(proc, cursor=InvalidCursor()):
        check = lambda c: hasattr(c, "type") and c.type().is_numeric()
        yield from filter(check, get_children(proc, cursor))

    children_ops.append(get_numeric_children)

    def stage(proc, exprs):
        proc, expr = stage_expr_into_memory(proc, exprs, precision, memory)
        for children_op in children_ops:
            if children := children_op(proc, expr):
                break
        return apply(stage)(proc, children)

    allocs = filter(lambda s: isinstance(s, AllocCursor), nlr_stmts(proc, block))
    proc = apply(set_memory)(proc, allocs, memory)
    proc = make_pass(attempt(unfold_reduce))(proc, block)
    assigns = filter(lambda s: isinstance(s, AssignCursor), lrn_stmts(proc, block))
    exprs = [assign.rhs() for assign in assigns]
    proc = apply(stage)(proc, exprs)
    # TODO: uncomment once bug in Exo is fixed
    # proc = inline_copies(proc, block)
    proc = make_pass(attempt(fold_into_reduce))(proc, block)
    proc = dealiasing_pass(proc, block)
    return proc


def check_call_site(proc, call_cursor):
    if not isinstance(call_cursor, CallCursor):
        raise TypeError("call_cursor must be a CallCursor")

    ###################################################################
    # build an env of symbols this call statement observes.
    # e.g. {x: (DRAM, "f32"), y: (Neon, "f64")}
    ###################################################################
    env = {}
    obs_stmts = get_observed_stmts(call_cursor)
    allocs = filter(lambda s: isinstance(s, AllocCursor), obs_stmts)
    for s in list(proc.args()) + list(allocs):
        if s.type().is_numeric():
            mem = s.mem()
            env[s.name()] = (mem, s.type())
    ###################################################################
    # Check consistency at call site
    ###################################################################
    call_args = call_cursor.args()
    callee_parameters = call_cursor.subproc().args()
    for arg, par in zip(call_args, callee_parameters):
        par_type = par.type()
        if par_type.is_numeric():
            par_mem = par.mem()
            arg_mem, arg_type = env[arg.name()]
            if not issubclass(arg_mem, par_mem) or arg_type is not par_type:
                return False
    return True


@dataclass
class replace_cursors:
    call: CallCursor

    def __iter__(self):
        yield self.call


# TODO: change this to work for blocks
def checked_replace(proc, stmt, subproc, quiet=False):
    stmt = proc.forward(stmt)
    parent = stmt.parent()
    index = get_index_in_body(proc, stmt)

    try:
        proc = replace(proc, stmt, subproc, quiet=quiet)
    except:
        raise SchedulingError("failed to replace")

    is_else = False
    if (
        isinstance(parent, IfCursor)
        and not isinstance(parent.orelse(), InvalidCursor)
        and index < len(parent.orelse())
        and parent.orelse()[index] == stmt
    ):
        is_else = True
    if not isinstance(parent, InvalidCursor):
        parent = proc.forward(parent)
    else:
        parent = proc

    call = parent.body()[index] if not is_else else parent.orelse()[index]
    if not check_call_site(proc, call):
        raise SchedulingError("Call site inconsistency")
    return proc


def replace_all_stmts(proc, instructions):
    if not isinstance(instructions, list):
        instructions = [instructions]

    for stmt in nlr_stmts(proc):
        try:
            stmt = proc.forward(stmt)
        except InvalidCursorError:
            continue

        for instr in instructions:
            try:
                proc = checked_replace(proc, stmt, instr, quiet=True)
                break
            except SchedulingError:
                pass
    return proc


def bound_loop_by_if(proc, loop):
    loop = proc.forward(loop)
    err = "Expected loop to be of the following structure:\nfor iter in seq(lo, hi):\n\t if iter < e:"
    if len(loop.body()) != 1 or not isinstance(loop.body()[0], IfCursor):
        raise SchedulingError(err)

    if_c = loop.body()[0]
    if not isinstance(if_c.orelse(), InvalidCursor):
        raise SchedulingError(err)

    if (
        not isinstance(if_c.cond().lhs(), ReadCursor)
        or if_c.cond().lhs().name() != loop.name()
        or if_c.cond().op() != "<"
    ):
        raise SchedulingError(err)

    if_c = loop.body()[0]
    proc = cut_loop(proc, loop, FormattedExprStr("_ + _", loop.lo(), if_c.cond().rhs()))
    loop1 = proc.forward(loop)
    loop2 = loop1.next()
    proc = eliminate_dead_code(proc, loop1.body()[0])
    # This should step can be skipped in general, but there is a problem in
    # Exo where the check of the inner if would fail if the loop bounds
    # are equal or dead in general.
    proc, success = attempt(eliminate_dead_code)(proc, loop2, rs=True)
    if success:
        return proc
    proc = eliminate_dead_code(proc, loop2.body()[0])
    # proc = delete_pass(proc, loop2), but it doesn't forward it
    return proc


def undo_divide_and_guard_loop(proc, loop):
    loop = proc.forward(loop)
    proc = mult_loops(proc, loop, loop.name()[:-1])
    proc = simplify(proc)
    proc = bound_loop_by_if(proc, loop)
    return proc


def unroll_loops(proc, block=InvalidCursor(), threshold=None):
    def pred(proc, s):
        s = proc.forward(s)
        if not (is_loop(proc, s) and is_loop_bounds_const(proc, s)):
            return False
        return threshold is None or s.hi().value() - s.lo().value() <= threshold

    return make_pass(predicate(unroll_loop, pred), lrn_stmts)(proc, block)


def cleanup(proc):
    proc = simplify(proc)
    proc = unroll_loops(proc, threshold=1)
    proc = dce(proc)
    try:
        proc.find("pass")
        proc = delete_pass(proc)
    except SchedulingError:
        pass
    return proc


def reorder_stmt_forward(proc, stmt):
    stmt = proc.forward(stmt)
    block = stmt.as_block().expand(0, 1)
    return reorder_stmts(proc, block)


def reorder_stmt_backwards(proc, stmt):
    stmt = proc.forward(stmt)
    block = stmt.as_block().expand(-1, 0)
    return reorder_stmts(proc, block)


@dataclass
class divide_loop_recursive_cursors:
    outer_loops: list
    inner_loops: list
    tail_loop: ForCursor

    def __iter__(self):
        yield self.outer_loops
        yield self.inner_loops
        yield self.tail_loop


def divide_loop_recursive(proc, loop, factor, tail="cut", rc=False):
    if tail not in {"cut", "cut_and_guard"}:
        raise SchedulingError("tail strategy must be cut or cut_and_guard")
    outer_loops = []
    inner_loops = []
    tail_loop = loop
    while factor > 1:
        proc, (outer, inner, tail_loop) = divide_loop_(
            proc, tail_loop, factor, tail=tail, rc=True
        )
        outer_loops.append(outer)
        inner_loops.append(inner)
        factor = factor // 2
    if not rc:
        return proc
    outer_loops = [proc.forward(c) for c in outer_loops]
    inner_loops = [proc.forward(c) for c in inner_loops]
    return proc, divide_loop_recursive_cursors(outer_loops, inner_loops, tail_loop)


@dataclass
class binary_specialize_cursors:
    stmts: Cursor

    def __iter__(self):
        yield self.stmts


def binary_specialize(proc, stmt, expr, values, rc=False):
    stmt = proc.forward(stmt)
    if isinstance(expr, ExprCursor):
        expr = proc.forward(expr)
        expr = expr_to_string(expr)
    get_cond = lambda op, v: f"{expr} {op} {v}"

    if len(values) == 1:
        raise SchedulingError("Cannot specialize given one value!")
    values = sorted(values)
    stmt = proc.forward(stmt)

    stmts = []

    def rewrite(proc, stmt, values):
        if len(values) == 1:
            # This should be redundant if the user provided correct inputs!
            # So, it is really a check that the inputs the user provided cover the full range.
            proc, (if_stmt,) = specialize_(
                proc, stmt, get_cond("==", values[0]), rc=True
            )
            proc = simplify(proc)
            proc = eliminate_dead_code(proc, if_stmt)
            stmts.append(if_stmt.body()[0])
            stmts.append(if_stmt.orelse()[0])
            return proc
        md = len(values) // 2
        proc, (if_stmt,) = specialize_(proc, stmt, get_cond("<", values[md]), rc=True)
        proc = rewrite(proc, if_stmt.body()[0], values[:md])
        proc = rewrite(proc, if_stmt.orelse()[0], values[md:])
        return proc

    proc = rewrite(proc, stmt, values)
    if not rc:
        return proc

    filtered_stmts = []
    for s in stmts:
        try:
            stmt = proc.forward(s)
            if not isinstance(stmt, PassCursor):
                filtered_stmts.append(stmt)
        except InvalidCursorError:
            pass
    return proc, binary_specialize_cursors(filtered_stmts)


def cse(proc, block, precision):
    if not isinstance(block, BlockCursor):
        block = proc.forward(block).as_block()

    nodes = list(lrn(proc, block))

    # First do CSE on the buffer accesses
    accesses = filter(lambda c: is_access(proc, c), nodes)
    accesses = filter(lambda c: is_stmt(proc, c) or c.type().is_numeric(), accesses)

    buff_map = {}

    for access in accesses:
        idx_ac = buff_map.setdefault(access.name(), {})
        idx_str = expr_to_string(access.idx())
        idx_ac.setdefault(idx_str, []).append(access)

    for buff, idx_map in buff_map.items():
        for idx, access_list in idx_map.items():
            if len(access_list) > 1:
                if all(is_read(proc, c) for c in access_list):
                    proc = bind_and_set_expr(proc, access_list, precision, DRAM)
                else:
                    staging_block = get_bounding_block(proc, access_list)
                    proc = auto_stage_mem(proc, staging_block, buff)
    return proc


inline_copies = make_pass(predicate(attempt(inline_assign), is_copy))


def dealias(proc, stmt):
    stmt = proc.forward(stmt)

    if not is_assign(proc, stmt) and not is_reduce(proc, stmt):
        raise TypeError("stmt must be an assign or reduce")

    nodes = list(lrn(proc, stmt.rhs()))

    accesses = filter(lambda c: is_access(proc, c), nodes)
    accesses = filter(lambda c: is_stmt(proc, c) or c.type().is_numeric(), accesses)

    buff_map = {}
    for access in accesses:
        buff_map.setdefault(access.name(), []).append(access)

    for buff, buff_accesses in buff_map.items():
        for access in buff_accesses[:-1]:
            decl = get_declaration(proc, stmt, access.name())
            proc = bind_and_set_expr(proc, access, decl.type(), decl.mem())

    if stmt.name() in buff_map:
        decl = get_declaration(proc, stmt, stmt.name())
        proc = bind_and_set_expr(proc, stmt.rhs(), decl.type(), decl.mem())

    return proc


dealiasing_pass = make_pass(attempt(dealias, errs=(TypeError,)))


def round_loop(proc, loop, factor, up=True):
    tail = "guard" if up else "cut"
    proc, (loop, _, _) = divide_loop_(proc, loop, factor, tail=tail, rc=True)
    proc = mult_loops(proc, loop, loop.name()[:-1])
    return proc


@dataclass
class cut_loop_and_unroll_cursors:
    loop: ForCursor

    def __iter__(self):
        yield self.loop


def cut_loop_and_unroll(proc, loop, const, front=True, rc=False):
    loop = proc.forward(loop)
    cut = (
        FormattedExprStr("_ + 1", loop.lo())
        if front
        else FormattedExprStr("_ - 1", loop.hi())
    )
    proc, (const_loop, loop) = cut_loop_(proc, loop, cut, rc=True)
    if not front:
        const_loop, loop = loop, const_loop
    proc = shift_loop(proc, const_loop, 0)
    proc = simplify(proc)
    proc = unroll_loop(proc, const_loop)
    proc = shift_loop(proc, loop, 0)
    if not rc:
        return proc
    return proc, cut_loop_and_unroll_cursors(loop)
