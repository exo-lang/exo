from __future__ import annotations

from exo.API_cursors import *
from exo.LoopIR import get_reads_of_expr, LoopIR  # TODO: get rid of this

from .range_analysis import bounds_inference
from .scheduling import *
from .inspection import *


# Assumptions made by Halide scheduling ops:
# - unique buffer names
# - loop nests are top-level
# - each consumer uses a contiguous region of producer values
# - each loop corresponds to a single buffer dimension, e.g. no arr[x, x]
# - for each dimension of a buffer, the loops which access along that
# dimension are always nested in decreasing stride size order.


# For better generality, this code should think about how to handle prologue loops and tail
# cases, which we currently only have very limited support for. I think user-written annotations
# attached to LoopIR could be very helpful for this, since part of the current bottleneck is that
# we need to either hard-code what each loopnest represents, or re-identify it via complex inspection.


def _get_reads_of_expr(expr: ExprCursor) -> list[str]:
    return [name.name() for (name, _) in get_reads_of_expr(expr._impl._node)]


def get_affected_write_dim(
    proc: Procedure, buffer_name: str, loop: ForCursor
) -> list[int]:
    """
    Return which dimension of buffer are affected by [iter_sym]. Raises
    an error if there are multiple.
    """
    dims = set()
    iter = loop.name()
    # TODO: we should scope this to [loop], but currently .find() can only be called on procs
    for c in proc.find(f"{buffer_name} = _", many=True):
        for idx, idx_expr in enumerate(c.idx()):
            idx_vars = _get_reads_of_expr(idx_expr)
            if iter in idx_vars:
                dims.add(idx)

    if len(dims) > 1:
        raise ValueError(f"{iter} affects multiple indices into {buffer_name}")

    return list(dims)[0]


def _divide_with_recompute(
    proc, loop_cursor: ForCursor, outer_hi, outer_stride: int, new_iters: list[str]
):
    """
    Wrapper around `divide_with_recompute` which will not perform the operation
    if it generates a trivial inner loop of size 1.
    """
    temp_proc = divide_with_recompute(
        proc, loop_cursor, outer_hi, outer_stride, new_iters
    )
    temp_proc = simplify(temp_proc)
    if is_literal(temp_proc, temp_proc.forward(loop_cursor).body()[0].hi(), 1):
        return proc
    return temp_proc


def _simplify_with_preds(proc):
    # TODO: Try to simplify the expressions without needing this.
    # This would also remove the need for the LoopIR import.
    for pred in proc._loopir_proc.preds:
        if (
            isinstance(pred, LoopIR.BinOp)
            and pred.op == "=="
            and isinstance(pred.rhs, LoopIR.Const)
        ):
            try:
                proc = rewrite_expr(proc, f"{pred.lhs}", pred.rhs.val)
            except:
                pass
    return simplify(proc)


# -------------------------------------------------------------#
#                   Halide Scheduling Ops                      #
# -------------------------------------------------------------#
# TODO: add returning relevant cursors to all the scheduling ops


def compute_at(proc: Procedure, producer_assign: AssignCursor, target_loop: ForCursor):
    """
    Computes the necessary values of producer at the level of [target_loop]
    in the consumer loop nest by fusing the two loop nests.

    NOTE: This implementation assumes that both loop nests are at the top-level, but
    it is easy to extend if we replace `get_top_level_stmt` with an LCA function of
    producer_assign and target_loop.

    NOTE: This implementation assumes that consumer[i] uses a contiguous range of values,
    e.g. producer[i:i+k].

    Example transformation:
        for i in _:
            for j in _:
                producer[_] = ... # producer_assign
        for io in _:
            for jo in _: # target_loop
                for ii in _:
                    for ji in _:
                        consumer[_] = ...
    ->
        for io in _:
            for jo in _:
                for ii in _:
                    for ji in _:
                        producer[_] = ...
                for ii in _:
                    for ji in _:
                        consumer[_] = ...
    """
    producer_assign = proc.forward(producer_assign)
    target_loop = proc.forward(target_loop)

    producer = producer_assign.name()
    p_loop = get_top_level_stmt(proc, producer_assign)
    c_loops = [target_loop] + list(get_parents(proc, target_loop, up_to=None))

    assert p_loop.next() == c_loops[-1], "loop nests must be consecutive"

    for c_loop in reversed(c_loops):
        producer_assign = proc.forward(producer_assign)
        c_loop = proc.forward(c_loop)

        # Infer bounds of producer that the are consumed to determine cut-factor
        buffer_dim = get_affected_write_dim(proc, producer, c_loop)
        bounds = bounds_inference(proc, c_loop, producer, buffer_dim, include=["R"])
        N_c = c_loop.hi()._impl._node  # TODO: remove this ._impl._node
        w_c = bounds.get_stride_of(c_loop._impl._node.iter)

        # Identify the producer loop corresponding to the consumer loop based on dimension affected
        dim_vars = _get_reads_of_expr(producer_assign.idx()[buffer_dim])
        p_loop = match_depth(producer_assign, c_loop)
        while p_loop.name() not in dim_vars:
            p_loop = p_loop.body()[0]

        # Reorder producer loop to be directly before consumer loop level
        while p_loop.parent() != c_loop.parent():
            proc = reorder_loops(proc, p_loop.parent())
            p_loop = proc.forward(p_loop)
            c_loop = proc.forward(c_loop)
        while p_loop.next() != c_loop:
            proc = reorder_stmts(proc, p_loop.expand(0, 1))
            p_loop = proc.forward(p_loop)
            c_loop = proc.forward(c_loop)

        # TODO: think about this hard-coded naming convention
        new_iters = [f"{c_loop.name()}", f"{c_loop.name()}i"]
        proc = _divide_with_recompute(proc, p_loop, f"{N_c}", w_c, new_iters)
        proc = fuse(proc, p_loop, c_loop, unsafe_disable_check=True)

        proc = _simplify_with_preds(proc)

    return proc


def compute_at_with_prologue(
    proc: Procedure, producer_assign: AssignCursor, target_loop: ForCursor
):
    """
    Computes the necessary values of producer at the level of [target_loop]
    in the consumer loop nest by fusing the two loop nests.

    TODO: make it work for multiple levels of consumer loops

    Example transformation:
        for i in _:
            producer[_] = ... # producer_assign
        for i in _:
            consumer[_] = ... # uses producer[i:i+k]
    ->
        for i in _:
            if i == 0: # prologue
                for ii in seq(0, k):
                    producer[_] = ...
            producer[i + k] = ...
            consumer[i] = ...
    """
    producer_assign = proc.forward(producer_assign)
    c_loop = proc.forward(target_loop)
    p_loop = get_enclosing_loop_by_name(proc, producer_assign, c_loop.name())
    producer = producer_assign.name()

    # Reorder producer loop up to consumer loop level and adjacent
    while p_loop.parent() != c_loop.parent():
        proc = reorder_loops(proc, p_loop.parent())
        p_loop = proc.forward(p_loop)
        c_loop = proc.forward(c_loop)
    while p_loop.next() != c_loop:
        proc = reorder_stmts(proc, p_loop.expand(0, 1))
        p_loop = proc.forward(p_loop)
        c_loop = proc.forward(c_loop)

    # bounds inference
    buffer_dim = get_affected_write_dim(proc, producer, c_loop)
    bounds = bounds_inference(proc, c_loop, producer, buffer_dim, include=["R"])
    w_p = bounds.get_size()

    # separate out prologue
    proc = cut_loop(proc, p_loop, w_p - 1)
    prologue_loop = proc.forward(p_loop)
    main_loop = prologue_loop.next()

    # fuse main loop
    proc = shift_loop(proc, main_loop, 0)
    proc = simplify(proc)
    proc = fuse(proc, main_loop, main_loop.next())

    # fuse prologue loop
    # proc = add_loop(proc, prologue_loop, "y_i", str(N_c), guard=True)
    # prologue_loop = proc.forward(prologue_loop).parent()
    # proc = fuse(proc, prologue_loop, prologue_loop.next())

    return proc


def store_at(proc: Procedure, producer_alloc: AllocCursor, target_loop: ForCursor):
    """
    Moves [producer]'s allocation into [target_loop] and reduces the dimensions
    as necessary.
    """
    producer_alloc = proc.forward(producer_alloc)
    target_loop = proc.forward(target_loop)

    producer = producer_alloc.name()

    def loops_between(producer_alloc, target_loop):
        """
        producer_alloc
        ...
        for i in _:
            for j in _:
                for k in _: <- target_loop
        loops_between(producer_alloc, target_loop) -> [i, j, k]
        """
        top_loop = match_depth(target_loop, producer_alloc)
        return reversed(
            [target_loop] + list(get_parents(proc, target_loop, up_to=top_loop))
        )

    for loop in loops_between(producer_alloc, target_loop):
        loop = proc.forward(loop)
        producer_alloc = proc.forward(producer_alloc)

        buffer_dim = get_affected_write_dim(proc, producer, loop)

        bounds = bounds_inference(proc, loop, producer, buffer_dim)
        lo, _ = bounds.get_bounds()
        size = bounds.get_size()

        while producer_alloc.next() != loop:
            proc = reorder_stmts(proc, producer_alloc.expand(0, 1))
            producer_alloc = proc.forward(producer_alloc)
            loop = proc.forward(loop)

        proc = sink_alloc(proc, producer_alloc)
        proc = resize_dim(proc, producer_alloc, buffer_dim, size, lo)
        proc = simplify(proc)

    return simplify(proc)


def compute_and_store_at(proc: Procedure, producer: str, target_loop: ForCursor):
    """
    Calling Halide's compute_at without a corresponding store_at is implicitly a combination
    of compute_at and store_at.

    Args:
        producer/consumer   - name of the buffers for the producer/consumer stages
        target_loop         - loop level to compute at
    """
    target_loop = proc.forward(target_loop)
    producer_assign = proc.find(f"{producer} = _")
    proc = compute_at(proc, producer_assign, target_loop)

    producer_alloc = proc.find(f"{producer} : _")
    target_loop = proc.forward(target_loop.body()[0]).parent()
    proc = store_at(proc, producer_alloc, target_loop)

    return proc


def tile(
    proc: Procedure,
    i_loop: ForCursor,
    j_loop: ForCursor,
    new_i_iters: list[str],
    new_j_iters: list[str],
    i_tile_size: int,
    j_tile_size: int,
    perfect: bool = True,
):
    assert j_loop.parent() == i_loop
    assert perfect, "only perfect tiling is currently supported"

    i_loop = proc.forward(i_loop)
    j_loop = proc.forward(j_loop)

    proc = divide_loop(proc, i_loop, i_tile_size, new_i_iters, perfect=perfect)
    proc = divide_loop(proc, j_loop, j_tile_size, new_j_iters, perfect=perfect)
    ii_loop = proc.forward(i_loop).body()[0]
    proc = reorder_loops(proc, ii_loop)
    return proc


def split(
    proc: Procedure,
    loop: ForCursor,
    o_iter: str,
    i_iter: str,
    split_factor: int,
    tail: str,
):
    loop = proc.forward(loop)
    if tail == "perfect":
        proc = divide_loop(proc, loop, split_factor, [o_iter, i_iter], perfect=True)
    else:
        proc = divide_loop(proc, loop, split_factor, [o_iter, i_iter], tail=tail)
    return proc


# -------------------------------------------------------------#
#                    Halide-like Interface                     #
# -------------------------------------------------------------#


def halide_tile(p, buffer, y, x, yi, xi, yTile, xTile):
    assign = p.find(f"{buffer} = _")
    y_loop = get_enclosing_loop_by_name(p, assign, y)
    x_loop = get_enclosing_loop_by_name(p, assign, x)

    return tile(p, y_loop, x_loop, [y, yi], [x, xi], yTile, xTile, perfect=True)


def halide_split(
    p, stage: str, x: str, xo: str, xi: str, split_factor: int, tail: str = "perfect"
):
    """
    tail can be {"perfect", "cut", "guard", "cut_and_guard"}
    """
    loop = get_enclosing_loop_by_name(p, p.find(f"{stage} = _"), x)
    return split(p, loop, xo, xi, split_factor, tail)


def halide_compute_at(
    p, producer: str, consumer: str, loop: str, divide_with_recompute: bool = True
):
    producer_assign = p.find(f"{producer} = _")
    target_loop = get_enclosing_loop_by_name(p, p.find(f"{consumer} = _"), loop)
    if divide_with_recompute:
        return compute_at(p, producer_assign, target_loop)
    else:
        return compute_at_with_prologue(p, producer_assign, target_loop)


def halide_store_at(p, producer: str, consumer: str, loop: str):
    target_loop = get_enclosing_loop_by_name(p, p.find(f"{consumer} = _"), loop)
    producer_alloc = p.find(f"{producer} : _")
    return store_at(p, producer_alloc, target_loop)


def halide_compute_and_store_at(p, producer: str, consumer: str, loop: str):
    target_loop = get_enclosing_loop_by_name(p, p.find(f"{consumer} = _"), loop)
    return compute_and_store_at(p, producer, target_loop)


def halide_fully_inline(p, producer: str, consumer: str):
    loop = get_enclosing_loop(p, p.find(f"{consumer} = _"))
    p = compute_and_store_at(p, producer, loop)

    # TODO: currently assumes consumer only uses one producer value
    p = inline_assign(p, p.find(f"{producer} = _"))
    p = delete_buffer(p, p.find(f"{producer}: _"))

    return p


def halide_parallel(p, loop: str):
    return parallelize_loop(p, p.find_loop(loop))


__all__ = [
    "compute_at",
    "compute_at_with_prologue",
    "store_at",
    "compute_and_store_at",
    "tile",
    "split",
    "halide_tile",
    "halide_split",
    "halide_compute_at",
    "halide_store_at",
    "halide_compute_and_store_at",
    "halide_fully_inline",
    "halide_parallel",
]
