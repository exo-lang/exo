from __future__ import annotations
from dataclasses import dataclass

from exo import *
from exo.libs.memories import *
from exo.platforms.x86 import *
from exo.platforms.neon import *
from exo.syntax import *
from exo.stdlib.scheduling import *
from exo.API_cursors import *
from .introspection import get_stmt_dependencies, get_declaration, get_expr_dependencies


class BLAS_SchedulingError(Exception):
    pass


def expr_to_string(expr_cursor, subst={}):
    def expr_list_to_string(expr_list, subst):
        expr_str_list = [expr_to_string(i, subst) for i in expr_list]
        if not expr_str_list:
            return ""
        return "[" + ", ".join(expr_str_list) + "]"

    if isinstance(expr_cursor, ExprListCursor):
        return expr_list_to_string(expr_cursor, subst)

    if not isinstance(expr_cursor, ExprCursor):
        raise BLAS_SchedulingError("Cursor must be an ExprCursor")
    if isinstance(expr_cursor, ReadCursor):
        name = str(expr_cursor.name())
        if name in subst:
            return f"({subst[name]})"
        idx_str = expr_list_to_string(expr_cursor.idx(), subst)
        return f"({name}{idx_str})"
    elif isinstance(expr_cursor, ReadConfigCursor):
        raise BLAS_SchedulingError("ReadConfigCursor is not supported")
    elif isinstance(expr_cursor, LiteralCursor):
        val_str = str(expr_cursor.value())
        return f"({val_str})"
    elif isinstance(expr_cursor, UnaryMinusCursor):
        arg_str = expr_to_string(expr_cursor.arg, subst)
        return f"(-{arg_str})"
    elif isinstance(expr_cursor, BinaryOpCursor):
        binop_str = expr_cursor.op()
        lhs_str = expr_to_string(expr_cursor.lhs(), subst)
        rhs_str = expr_to_string(expr_cursor.rhs(), subst)
        return f"({lhs_str}{binop_str}{rhs_str})"
    elif isinstance(expr_cursor, BuiltInCursor):
        name = expr_cursor.name()
        args_str = expr_list_to_string(expr_cursor.args(), subst)
        return f"({name}({args_str[1:-1]}))"
    elif isinstance(expr_cursor, WindowExprCursor):
        raise BLAS_SchedulingError("WindowExprCursor is not supported")
    else:
        assert False, "Undefined Type"


def get_enclosing_scope(cursor, scope_type):
    if not scope_type in (ForSeqCursor, IfCursor):
        raise BLAS_SchedulingError("scope type must be ForSeqCursor or IfCursor")

    cursor = cursor.parent()
    while not isinstance(cursor, (scope_type, InvalidCursor)):
        cursor = cursor.parent()

    if isinstance(cursor, InvalidCursor):
        raise BLAS_SchedulingError("No enclosing scope found")

    return cursor


def get_enclosing_loop(cursor):
    return get_enclosing_scope(cursor, ForSeqCursor)


def get_enclosing_if(cursor):
    return get_enclosing_scope(cursor, IfCursor)


def get_statement(cursor):

    while not isinstance(cursor, StmtCursor):
        cursor = cursor.parent()

    return cursor


def is_already_divided(loop_cursor, div_factor):
    return (
        len(loop_cursor.body()) == 1
        and isinstance(loop_cursor.body()[0], ForSeqCursor)
        and isinstance(loop_cursor.body()[0].hi(), LiteralCursor)
        and loop_cursor.body()[0].hi().value() == div_factor
    )


def stage_expr(proc, expr_cursors, new_name, memory=DRAM, n_lifts=1):
    """
    for i in seq(0, hi):
        s (e(i));

    ----->

    new_name: R[hi]
    for i in seq(0, hi):
        new_name[i] = e(i)
    for i in seq(0, hi):
        s (new_name[i]);
    """

    if not isinstance(expr_cursors, list):
        expr_cursors = [expr_cursors]

    expr_cursors = [proc.forward(c) for c in expr_cursors]
    enclosing_loop = get_enclosing_loop(expr_cursors[0])
    stmt = get_statement(expr_cursors[0])
    proc = bind_expr(proc, expr_cursors, new_name, cse=True)
    stmt = proc.forward(stmt)
    bind_stmt = stmt.prev()
    alloc_stmt = bind_stmt.prev()
    proc = set_memory(proc, alloc_stmt, memory)
    proc = expand_dim(
        proc, alloc_stmt, expr_to_string(enclosing_loop.hi()), enclosing_loop.name()
    )
    proc = lift_alloc(proc, alloc_stmt, n_lifts=n_lifts)
    proc = fission(proc, bind_stmt.after(), n_lifts=n_lifts)
    return proc


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
        if isinstance(enclosing_scope, ForSeqCursor):
            proc = expand_dim(
                proc,
                alloc_cursor,
                expr_to_string(enclosing_scope.hi()),
                enclosing_scope.name(),
            )
        proc = lift_alloc(proc, alloc_cursor)
    return proc


@dataclass
class auto_divide_loop_cursors:
    outer_loop_cursor: ForSeqCursor
    inner_loop_cursor: ForSeqCursor
    tail_loop_cursor: ForSeqCursor


def auto_divide_loop(proc, loop_cursor, div_const, tail="guard", perfect=False):
    loop_cursor = proc.forward(loop_cursor)
    loop_iter = loop_cursor.name()
    proc = divide_loop(
        proc,
        loop_cursor,
        div_const,
        (loop_iter + "o", loop_iter + "i"),
        tail=tail,
        perfect=perfect,
    )
    outer_loop_cursor = proc.forward(loop_cursor)
    inner_loop_cursor = outer_loop_cursor.body()[0]

    if perfect == True or tail == "guard":
        tail_loop_cursor = InvalidCursor()
    elif tail == "cut":
        tail_loop_cursor = outer_loop_cursor.next()
    else:
        tail_loop_cursor = outer_loop_cursor.next().body()[0]

    return proc, auto_divide_loop_cursors(
        outer_loop_cursor, inner_loop_cursor, tail_loop_cursor
    )


def scalar_loop_to_simd_loops(proc, loop_cursor, vec_width, memory_type):
    """
    for i in seq(0, hi):
        lhs(i) = (e_0(i), e_1(i), ..., e_n(i));

    ----->

    for io in seq(0, hi / vec_width):
        reg0: precision[vec_width]
        for ii in seq(0, vec_width):
            reg0[ii] = e_0(ii)

        reg1: precision[vec_width]
        for ii in seq(0, vec_width):
            reg1[ii] = e_1(ii)

        ....

        regn: precision[vec_width]
        for ii in seq(0, vec_width):
            regn[ii] = e_n(ii)

        for ii in seq(0, vec_width):
            lhs(io * vec_width + ii) = e_n(ii);

    for i in seq(0, hi % vec_width):
        lhs(i + delta) = (e_0(i + delta), e_1(i + delta), ..., e_n(i + delta));
    """

    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError(
            "vectorize_to_loops loop_cursor must be a ForSeqCursor"
        )

    loop_cursor = proc.forward(loop_cursor)

    if is_already_divided(loop_cursor, vec_width):
        inner_loop_cursor = loop_cursor.body()[0]
    else:
        proc, cursors = auto_divide_loop(proc, loop_cursor, vec_width, tail=tail)
        inner_loop_cursor = cursors.inner_loop_cursor

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    stmts = []

    def fission_stmts(proc, body, depth=1):
        body_list = list(body)
        for stmt in body_list[:-1]:
            if isinstance(stmt, AllocCursor):
                proc = parallelize_and_lift_alloc(proc, stmt, n_lifts=depth)
                proc = set_memory(proc, stmt, memory_type)
            else:
                forwarded_stmt = proc.forward(stmt)
                stmts.append(stmt)
                proc = fission(proc, forwarded_stmt.after(), n_lifts=depth)
                forwarded_stmt = proc.forward(stmt)
                if isinstance(forwarded_stmt, IfCursor):
                    proc = fission_stmts(proc, forwarded_stmt.body(), depth + 1)
                elif isinstance(forwarded_stmt, ForSeqCursor):
                    raise BLAS_SchedulingError("This is an inner loop vectorizer")
        forwarded_stmt = body_list[-1]
        stmts.append(forwarded_stmt)
        if isinstance(forwarded_stmt, IfCursor):
            proc = fission_stmts(proc, forwarded_stmt.body(), depth + 1)
        return proc

    proc = fission_stmts(proc, inner_loop_cursor.body())

    def detect_madd(expr):
        return (
            isinstance(expr, BinaryOpCursor)
            and expr.op() == "*"
            and isinstance(expr.parent(), ReduceCursor)
        )

    def get_expr_subtree_cursors(expr, stmt, alias):
        stmt_lhs_in_mem = False
        if isinstance(expr.parent(), StmtCursor):
            stmt_lhs_in_mem = (
                get_declaration(proc, stmt, stmt.name()).mem() == memory_type
            )

        if isinstance(expr, ReadCursor):
            expr_in_mem = get_declaration(proc, stmt, expr.name()).mem() == memory_type
            if alias or not (stmt_lhs_in_mem or expr_in_mem):
                return [expr]
            else:
                return []
        elif isinstance(expr, BinaryOpCursor):
            lhs = expr.lhs()
            rhs = expr.rhs()
            children_alias = (
                isinstance(lhs, ReadCursor)
                and isinstance(rhs, ReadCursor)
                and lhs.name() == rhs.name()
            )
            lhs_alias_with_stmt_lhs = False
            rhs_alias_with_stmt_lhs = False
            if isinstance(expr.parent(), StmtCursor):
                lhs_alias_with_stmt_lhs = (
                    isinstance(lhs, ReadCursor) and lhs.name() == expr.parent().name()
                )
                rhs_alias_with_stmt_lhs = (
                    isinstance(rhs, ReadCursor) and rhs.name() == expr.parent().name()
                )

            lhs_cursors = get_expr_subtree_cursors(
                lhs, stmt, lhs_alias_with_stmt_lhs or children_alias
            )
            rhs_cursors = get_expr_subtree_cursors(rhs, stmt, rhs_alias_with_stmt_lhs)

            if not detect_madd(expr) and not stmt_lhs_in_mem:
                return lhs_cursors + rhs_cursors + [expr]
            else:
                return lhs_cursors + rhs_cursors
        elif isinstance(expr, UnaryMinusCursor):
            return get_expr_subtree_cursors(expr.arg(), stmt, False) + [expr]
        elif isinstance(expr, BuiltInFunctionCursor):
            exprs = []
            for arg in expr.args():
                exprs = exprs + get_expr_subtree_cursors(arg, stmt, False)
            exprs = exprs + [expr]
            return exprs
        else:
            return [expr]

    is_assign_or_reduce = lambda stmt: isinstance(stmt, (AssignCursor, ReduceCursor))
    inner_loop_cursor = proc.forward(inner_loop_cursor)

    def vectorize_rhs(proc, stmt, depth=1):
        if not is_assign_or_reduce(stmt):
            return proc
        flat_rhs = get_expr_subtree_cursors(stmt.rhs(), stmt, False)

        for expr in flat_rhs:
            proc = stage_expr(proc, expr, f"reg", memory_type, depth)

        return proc

    def vectorize_stmt(proc, stmt, depth=1):
        proc = vectorize_rhs(proc, stmt, depth)
        if isinstance(stmt, ReduceCursor):
            alloc_stmt = get_declaration(proc, stmt, stmt.name())
            if alloc_stmt.mem() != memory_type:

                lhs_reg = "reg"
                proc = stage_mem(
                    proc, stmt, f"{stmt.name()}{expr_to_string(stmt.idx())}", lhs_reg
                )
                forwarded_stmt = proc.forward(stmt)

                alloc_cursor = forwarded_stmt.prev().prev()
                if depth > 1:
                    proc = lift_alloc(proc, alloc_cursor, n_lifts=depth - 1)
                proc = parallelize_and_lift_alloc(proc, alloc_cursor)

                forwarded_stmt = proc.forward(stmt)
                proc = fission(proc, forwarded_stmt.after(), n_lifts=depth)
                forwarded_stmt = proc.forward(forwarded_stmt)
                proc = fission(proc, forwarded_stmt.before(), n_lifts=depth)

                proc = set_memory(proc, alloc_cursor, memory_type)
        elif isinstance(stmt, IfCursor):
            assert len(stmt.body()) == 1
            if not isinstance(stmt.orelse(), InvalidCursor):
                raise BLAS_SchedulingError("Not implemented yet")
            proc = vectorize_stmt(proc, stmt.body()[0], depth + 1)
        elif isinstance(stmt, AssignCursor):
            # This will be a store
            pass
        else:
            raise BLAS_SchedulingError("Not implemented yet")
        return proc

    for stmt in stmts:
        forwarded_stmt = proc.forward(stmt)
        if isinstance(forwarded_stmt, IfCursor):
            continue
        inner_loop = get_enclosing_loop(forwarded_stmt)
        if isinstance(inner_loop, ForSeqCursor):
            assert len(inner_loop.body()) == 1
            proc = vectorize_stmt(proc, inner_loop.body()[0])

    return proc


def interleave_execution(proc, loop_cursor, interleave_factor):
    """
    for i in seq(0, n):
        S1
        S2
        S3

    ----->

    for io in seq(0, n / interleave_factor):
        S1
        ... x interleave_factor
        S2
        ... x interleave_factor
        S3
        ... x interleave_factor
    """
    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForSeqCursor")

    loop_cursor = proc.forward(loop_cursor)

    if not (
        isinstance(loop_cursor.hi(), LiteralCursor)
        and loop_cursor.hi().value() == interleave_factor
    ):
        proc = divide_loop(
            proc,
            loop_cursor,
            interleave_factor,
            (loop_cursor.name() + "o", loop_cursor.name() + "i"),
            tail="cut",
        )

        outer_loop_cursor = proc.forward(loop_cursor)
        inner_loop_cursor = outer_loop_cursor.body()[0]
    else:
        inner_loop_cursor = loop_cursor

    inner_loop_stmts = list(inner_loop_cursor.body())

    for stmt in inner_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = parallelize_and_lift_alloc(proc, stmt)

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    inner_loop_stmts = list(inner_loop_cursor.body())

    for stmt in inner_loop_stmts[:-1]:
        forwarded_stmt = proc.forward(stmt)
        proc = fission(proc, forwarded_stmt.after())
        forwarded_stmt = proc.forward(stmt).parent()
        proc = unroll_loop(proc, forwarded_stmt)

    last_loop = proc.forward(inner_loop_stmts[-1]).parent()
    proc = unroll_loop(proc, last_loop)

    return proc


def hoist_stmt(proc, stmt_cursor):
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
    if not isinstance(stmt_cursor, StmtCursor):
        raise BLAS_SchedulingError("Cannot hoist cursor that are not statements")

    if isinstance(stmt_cursor, AllocCursor):
        return lift_alloc(proc, stmt_cursor)

    enclosing_loop = get_enclosing_loop(stmt_cursor)
    if enclosing_loop.name() in get_stmt_dependencies(stmt_cursor):
        raise BLAS_SchedulingError(
            "Cannot hoist cursor to a statement that depends on enclosing loop"
        )

    stmt_cursor = proc.forward(stmt_cursor)
    while not isinstance(stmt_cursor.prev(), InvalidCursor):
        proc = reorder_stmts(proc, stmt_cursor.expand(1, 0))
        stmt_cursor = proc.forward(stmt_cursor)

    proc = fission(proc, stmt_cursor.after())
    stmt_cursor = proc.forward(stmt_cursor)
    proc = remove_loop(proc, stmt_cursor.parent())
    return proc


def apply_to_block(proc, block_cursor, stmt_scheduling_op):
    if not isinstance(block_cursor, BlockCursor):
        raise BLAS_SchedulingError("cannot apply to a non-block cursor")

    for stmt in block_cursor:
        try:
            proc = stmt_scheduling_op(proc, stmt)
        except:
            pass

    return proc


def parallelize_reduction(
    proc,
    loop_cursor,
    reduction_stmt,
    parallel_factor,
    memory=DRAM,
    tail="cut",
):
    """
    for i in seq(0, n):
        x[...] += y[i]

    ----->

    xReg: f32[parallel_factor]
    for ii in seq(0, parallel_factor):
        xReg[ii][...] = 0.0
    for io in seq(0, n / parallel_factor):
        for ii in seq(0, vec_width):
            xReg[ii][..] += y[io * parallel_factor + ii]
    for ii in seq(0, parallel_factor):
        x[...] += xReg[ii]

    Returns: (proc, allocation cursors)
    """
    # Check arguments pre-condition
    if not isinstance(loop_cursor, ForSeqCursor):
        raise BLAS_SchedulingError("vectorize loop_cursor must be a ForSeqCursor")

    if not isinstance(parallel_factor, int):
        raise BLAS_SchedulingError("parallel_factor must be an integer")

    if parallel_factor <= 1:
        return proc, None

    reduction_stmt = proc.forward(reduction_stmt)

    reduction_buffer_window = reduction_stmt.name()
    for idx in reduction_stmt.idx():
        reduction_buffer_window += f"[{expr_to_string(idx)}]"

    # Forward input cursors
    loop_cursor = proc.forward(loop_cursor)

    # Divide the loop if necessary
    if is_already_divided(loop_cursor, parallel_factor):
        outer_loop_cursor = loop_cursor
    else:
        proc, cursors = auto_divide_loop(proc, loop_cursor, parallel_factor, tail=tail)
        outer_loop_cursor = cursors.outer_loop_cursor

    proc = reorder_loops(proc, outer_loop_cursor)
    outer_loop_cursor = proc.forward(outer_loop_cursor)

    proc = simplify(
        stage_mem(proc, outer_loop_cursor, reduction_buffer_window, "reg", accum=True)
    )
    outer_loop_cursor = proc.forward(outer_loop_cursor)
    alloc_cursor = outer_loop_cursor.prev().prev()
    proc = set_memory(proc, alloc_cursor, memory)

    outer_loop_cursor = proc.forward(outer_loop_cursor)
    proc = parallelize_and_lift_alloc(proc, outer_loop_cursor.prev().prev())
    proc = fission(proc, outer_loop_cursor.before())
    proc = fission(proc, outer_loop_cursor.after())
    outer_loop_cursor = proc.forward(outer_loop_cursor)
    proc = reorder_loops(proc, outer_loop_cursor.parent())
    outer_loop_cursor = proc.forward(outer_loop_cursor)

    return proc


def interleave_outer_loop_with_inner_loop(
    proc, outer_loop_cursor, inner_loop_cursor, interleave_factor
):
    """
    for i in seq(0, hi):
        B1;
        for j in seq(0, hi'):
            B;
        B2;

    --->

    for io in seq(0, hi / interleave_factor):
        for ii in seq(0, interleave_factor):
            B1;
        for j in seq(0, hi'):
            for ii in seq(0, interleave_factor):
                B;
        for ii in seq(0, interleave_factor):
            B2;

    for io in seq(0, hi % interleave_factor):
        B1;
        for j in seq(0, hi'):
            B;
        B2;
    """
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

    for stmt in middle_loop_stmts:
        if isinstance(stmt, AllocCursor):
            proc = parallelize_and_lift_alloc(proc, stmt)

    inner_loop_cursor = proc.forward(inner_loop_cursor)

    if not isinstance(inner_loop_cursor.prev(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.before())

    inner_loop_cursor = proc.forward(inner_loop_cursor)
    if not isinstance(inner_loop_cursor.next(), InvalidCursor):
        proc = fission(proc, inner_loop_cursor.after())

    proc = simplify(proc)
    inner_loop_cursor = proc.forward(inner_loop_cursor)
    proc = reorder_loops(proc, inner_loop_cursor.parent())

    return proc


def get_reduction_stmts(proc, loop):
    for stmt in post_order_stmts(loop.body()):
        if isinstance(stmt, ReduceCursor):

            def get_idx_deps(stmt):
                for idx in stmt.idx():
                    yield from get_expr_dependencies(idx)

            if not loop.name() in get_idx_deps(stmt):
                yield stmt


def parallelize_loop_reductions(proc, loop, parallel_factor, memory=DRAM):
    for reduction in get_reduction_stmts(proc, loop):
        proc = parallelize_reduction(
            proc, loop, reduction, parallel_factor, memory=memory
        )
    return proc


def vectorize(
    proc,
    loop_cursor,
    vec_width,
    memory,
    instructions,
):
    loop_cursor = proc.forward(loop_cursor)
    proc, cursors = auto_divide_loop(proc, loop_cursor, vec_width, tail="cut")
    proc = parallelize_loop_reductions(
        proc, cursors.outer_loop_cursor, vec_width, memory
    )
    proc = scalar_loop_to_simd_loops(proc, cursors.outer_loop_cursor, vec_width, memory)
    proc = replace_all(proc, instructions)
    return proc


def tile_loops_top_down(proc, loop_tile_pairs):

    loop_tile_pairs = [(proc.forward(i[0]), i[1]) for i in loop_tile_pairs]

    inner_loops = []
    for i in range(len(loop_tile_pairs)):
        outer_loop = loop_tile_pairs[i][0]
        tile_size = loop_tile_pairs[i][1]
        new_names = (outer_loop.name() + "o", outer_loop.name() + "i")
        if isinstance(outer_loop.hi(), LiteralCursor) and (
            outer_loop.hi().value() % tile_size == 0
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
                proc, inner_loop, loop, tile_size
            )
    return proc, [proc.forward(l) for l in inner_loops]


def tile_loops_bottom_up(proc, outer_most_loop, tiles):
    loop = outer_most_loop
    for i in tiles[:-1]:
        if not len(loop.body()) == 1:
            raise BLAS_SchedulingError("All loop must have a body length of 1")
        if not isinstance(loop.body()[0], ForSeqCursor):
            raise BLAS_SchedulingError("Did not find a nested loop")

    loops = []
    loop = outer_most_loop
    for i in tiles:
        loops.append((loop, i))
        loop = loop.body()[0]

    def get_depth(loop):
        if not isinstance(loop, (ForSeqCursor, IfCursor)):
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
            if isinstance(child, ForSeqCursor):
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
            proc, cursors = auto_divide_loop(proc, loop, tile, tail="cut_and_guard")
            proc = push_loop_in(proc, cursors.inner_loop_cursor, depth + 1)
            proc = push_loop_in(proc, cursors.tail_loop_cursor, depth + 1)
        else:
            proc = push_loop_in(proc, loop, depth + 1)

    return proc


def auto_stage_mem(proc, cursor, new_buff_name, n_lifts=1, accum=False):
    if not isinstance(cursor, (ReadCursor, ReduceCursor, AssignCursor)):
        raise BLAS_SchedulingError("auto_stage_mem expects a read a cursor")

    cursor = proc.forward(cursor)

    lo = []
    hi = []
    loop = get_enclosing_loop(cursor)
    loops = [loop]
    for _ in range(n_lifts - 1):
        loop = get_enclosing_loop(loop)
        loops.append(loop)

    subst = {}
    for i in range(len(loops) - 1, -1, -1):
        loop = loops[i]
        subst[loop.name()] = f"(({expr_to_string(loop.hi(), subst)})-1)"

    for idx in cursor.idx():
        hi.append(expr_to_string(idx, subst))

    for key in subst:
        subst[key] = 0

    for idx in cursor.idx():
        lo.append(expr_to_string(idx, subst))

    def ith_idx(i):
        if lo[i] == hi[i]:
            return lo[i]
        else:
            return f"{lo[i]}:(({hi[i]})+1)"

    window = ",".join([ith_idx(i) for i in range(len(cursor.idx()))])
    window = f"{cursor.name()}[{window}]"
    return stage_mem(proc, loops[-1], window, new_buff_name, accum=accum)


def ordered_stage_expr(proc, expr_cursors, new_buff_name, precision, n_lifts=1):
    if not isinstance(expr_cursors, list):
        expr_cursors = [expr_cursors]

    if not all([isinstance(cursor, ExprCursor) for cursor in expr_cursors]):
        raise BLAS_SchedulingError("auto_stage_mem expects a read a cursor")

    expr_cursors = [proc.forward(c) for c in expr_cursors]
    original_stmt = get_statement(expr_cursors[0])

    proc = bind_expr(proc, expr_cursors, new_buff_name, cse=True)
    original_stmt = proc.forward(original_stmt)
    assign_cursor = original_stmt.prev()
    alloc_cursor = assign_cursor.prev()
    expr_cursor = assign_cursor.rhs()
    deps = list(get_expr_dependencies(expr_cursor))

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

        if not isinstance(parent, ForSeqCursor):
            raise BLAS_SchedulingError("Not implemented yet")
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
        if isinstance(child_stmt, ForSeqCursor):
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
        if isinstance(child_stmt, (ForSeqCursor, IfCursor)):
            proc = lift_all_ifs(proc, child_stmt, depth + 1)
        return proc

    proc = lift_all_ifs(proc, scopes_nest)

    return proc


def eliminate_dead_code_pass(proc):
    def is_loop_const_bound(loop):
        return isinstance(loop.lo(), LiteralCursor) and isinstance(
            loop.hi(), LiteralCursor
        )

    def visit(proc, block):
        for s in block:
            if isinstance(s, ForSeqCursor):
                if not is_loop_const_bound(s):
                    try:
                        proc = eliminate_dead_code(proc, s)
                    except:
                        proc = visit(proc, s.body())
                else:
                    proc = visit(proc, s.body())
            elif isinstance(s, IfCursor):
                proc = visit(proc, s.body())
                if not isinstance(s.orelse(), InvalidCursor):
                    proc = visit(proc, s.orelse())
                try:
                    proc = eliminate_dead_code(proc, s)
                except:
                    pass

        return proc

    return visit(proc, proc.body())


def post_order_stmts(block):
    for stmt in block:
        if isinstance(stmt, ForSeqCursor):
            yield from post_order_stmts(stmt.body())
        elif isinstance(stmt, IfCursor):
            yield from post_order_stmts(stmt.body())
            if not isinstance(stmt.orelse(), InvalidCursor):
                yield from post_order_stmts(stmt.orelse())
        yield stmt


def is_inner_loop(loop):
    return all(
        not isinstance(stmt, ForSeqCursor) for stmt in post_order_stmts(loop.body())
    )


def name_exists(proc, name):
    return any(arg.name() == name for arg in proc.args()) or any(
        isinstance(stmt, AllocCursor) and stmt.name() == name
        for stmt in post_order_stmts(proc.body())
    )


def get_reg_name(proc):
    counter = 0
    while True:
        counter += 1
        new_name = f"reg_{counter}"
        if not name_exists(proc, new_name):
            yield new_name


def stage_computation(proc, loop):
    def post_order_data_exprs(expr):
        if isinstance(expr, UnaryMinusCursor):
            yield from post_order_data_exprs(expr.arg())
            yield expr
        elif isinstance(expr, BinaryOpCursor):
            yield from post_order_data_exprs(expr.lhs())
            yield from post_order_data_exprs(expr.rhs())
            yield expr
        elif isinstance(expr, BuiltInFunctionCursor):
            for arg in expr.args():
                yield from post_order_data_exprs(arg)
            yield expr
        elif isinstance(expr, (ReadCursor, LiteralCursor)):
            yield expr

    name_generator = get_reg_name(proc)

    for stmt in post_order_stmts(loop.body()):
        if isinstance(stmt, (AssignCursor, ReduceCursor)):
            for expr in post_order_data_exprs(stmt.rhs()):
                proc = bind_expr(proc, [expr], next(name_generator))
        if isinstance(stmt, ReduceCursor):
            lhs = f"{stmt.name()}{expr_to_string(stmt.idx())}"
            proc = stage_mem(proc, stmt, lhs, next(name_generator))

    return proc
