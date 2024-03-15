from __future__ import annotations

from exo.API_cursors import *
from exo.LoopIR import get_reads_of_expr, LoopIR  # TODO: get rid of this

from .range_analysis import bounds_inference
from .scheduling import *
from .inspection import *


def get_affected_dim(proc, buffer_name: str, iter_sym) -> list[int]:
    """
    Return which dimension of buffer are affected by [iter_sym]. Raises
    an error if there are multiple.
    """
    dims = set()
    # TODO: this only matches against writes
    for c in proc.find(f"{buffer_name}[_] = _", many=True):
        for idx, idx_expr in enumerate(c.idx()):
            idx_vars = [
                name.name() for (name, _) in get_reads_of_expr(idx_expr._impl._node)
            ]
            if iter_sym in idx_vars:
                dims.add(idx)

    if len(dims) > 1:
        raise ValueError(f"{iter_sym} affects multiple indices into {buffer_name}")

    return list(dims)[0]


def fuse_at(proc, producer: str, consumer: str, target_loop):
    """
    Computes the necessary values of producer at the level of [target_loop]
    in the consumer loop nest by fusing the two loop nests.

    Example transformation:
        for i in _:
            for j in _:
                producer[_] = ...
        for io in _:
            for jo in _:
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

    TODO: add returning relevant cursors
    """

    target_loop = proc.forward(target_loop)
    p_assign = proc.find(f"{producer}[_] = _")
    p_loop = get_top_level_stmt(proc, p_assign)
    c_loops = [target_loop] + list(get_parents(proc, target_loop, up_to=None))

    assert p_loop.next() == c_loops[-1], "loop nests must be consecutive"

    for c_loop in reversed(c_loops):
        c_loop = proc.forward(c_loop)
        p_loop = get_enclosing_loop_by_name(proc, proc.forward(p_assign), c_loop.name())
        assert p_loop.next() == c_loop

        # infer bounds of consumer to determine cut-factor
        N_c = c_loop.hi()._impl._node
        buffer_dim = get_affected_dim(proc, consumer, c_loop.name())
        w_c = bounds_inference(proc, c_loop, consumer, buffer_dim).get_size()

        p_iter = p_loop.name()
        new_iters = [f"{p_iter}", f"{p_iter}i"]

        proc = divide_with_recompute(proc, p_loop, f"{N_c}", w_c, new_iters)
        proc = fuse(proc, p_loop, c_loop, unsafe_disable_check=True)

        # TODO: rethink the reorder thing here
        p_inner_loop = proc.forward(p_loop).body()[0]
        while isinstance(p_inner_loop.body()[0], ForCursor):
            proc = reorder_loops(proc, p_inner_loop)
            p_inner_loop = proc.forward(p_inner_loop)

        # TODO: Try to simplify the expressions without needing this.
        # This would also remove the need for the LoopIR import.
        for pred in p_assign._impl.get_root().preds:
            if (
                isinstance(pred, LoopIR.BinOp)
                and pred.op == "=="
                and isinstance(pred.rhs, LoopIR.Const)
            ):
                try:
                    proc = rewrite_expr(proc, f"{pred.lhs}", pred.rhs.val)
                except:
                    pass
        proc = simplify(proc)

    return simplify(proc)


def store_at(proc, producer, target_loop):
    """
    Moves [producer]'s allocation into [target_loop] and reduces the dimensions
    as necessary.
    """
    target_loop = proc.forward(target_loop)
    producer_alloc = proc.find(f"{producer}:_")

    def loops_between(producer_alloc, target_loop):
        """
        producer_alloc
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
        buffer_dim = get_affected_dim(proc, producer, loop.name())

        loop = proc.forward(loop)
        producer_alloc = proc.forward(producer_alloc)

        bounds = bounds_inference(proc, loop, producer, buffer_dim)
        lo, _ = bounds.get_bounds()
        size = bounds.get_size()

        # TODO: think more about this reorder strategy
        while producer_alloc.next() != loop:
            reorder_stmts(proc, producer_alloc.expand(0, 1))
            producer_alloc = proc.forward(producer_alloc)

        proc = sink_alloc(proc, producer_alloc)
        proc = resize_dim(proc, producer_alloc, buffer_dim, size, lo)

    return simplify(proc)


def compute_at(proc, producer, consumer, target_loop):
    """
    Halide's compute_at is a combination of fuse_at and store_at.
    """
    target_loop = proc.forward(target_loop)
    proc = fuse_at(proc, producer, consumer, target_loop)

    target_loop = proc.forward(target_loop.body()[0]).parent()
    proc = store_at(proc, producer, target_loop)

    return proc


def tile(
    proc,
    i_loop,
    j_loop,
    new_i_iters,
    new_j_iters,
    i_tile_size,
    j_tile_size,
    perfect=True,
):
    assert j_loop.parent() == i_loop
    assert perfect, "only perfect tiling is supported"
    proc = divide_loop(proc, i_loop, i_tile_size, new_i_iters, perfect=perfect)
    proc = divide_loop(proc, j_loop, j_tile_size, new_j_iters, perfect=perfect)
    ii_loop = proc.forward(i_loop).body()[0]
    proc = reorder_loops(proc, ii_loop)
    return proc
