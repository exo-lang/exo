from functools import wraps as _wraps


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Expose the built-in Scheduling operators here

from ..API import (
    SchedulingError,
)

from ..API_scheduling import (
    is_atomic_scheduling_op,
    # argument processing
    sched_op,
    BlockCursorA,
    ProcA,
    BoolA,
    ArgOrAllocCursorA,
    FormattedExprStr,
    # basic operations
    simplify,
    rename,
    make_instr,
    #
    # general statement and expression operations
    insert_pass,
    delete_pass,
    reorder_stmts,
    rewrite_expr,
    bind_expr,
    commute_expr,
    #
    # subprocedure oriented operations
    extract_subproc,
    inline,
    replace,
    call_eqv,
    #
    # precision, memory, and window annotation setting
    set_precision,
    set_window,
    set_memory,
    #
    # Configuration modifying operations
    bind_config,
    delete_config,
    write_config,
    #
    # buffer and window oriented operations
    expand_dim,
    resize_dim,
    rearrange_dim,
    bound_alloc,
    divide_dim,
    mult_dim,
    sink_alloc,
    lift_alloc,
    delete_buffer,
    reuse_buffer,
    inline_window,
    stage_window,
    stage_mem,
    unroll_buffer,
    #
    # loop rewriting
    parallelize_loop,
    divide_with_recompute,
    divide_loop,
    mult_loops,
    cut_loop,
    join_loops,
    shift_loop,
    reorder_loops,
    merge_writes,
    inline_assign,
    lift_reduce_constant,
    fission,
    fuse,
    remove_loop,
    add_loop,
    unroll_loop,
    #
    # guard rewriting
    lift_scope,
    eliminate_dead_code,
    specialize,
    #
    # deprecated scheduling operations
    add_unsafe_guard,
    bound_and_guard,
    #
    # to be replaced by stdlib compositions eventually
    autofission,
    autolift_alloc,
)

# TODO: remove
from ..LoopIR import LoopIR
from .analysis import (
    check_call_mem_types,
)

from .range_analysis import bounds_inference, get_affected_dim
from ..API_cursors import InvalidCursor

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Higher-order Scheduling operations


def repeat(sched, n_times=None, verbose=False):
    """
    TODO: Documentation
    """
    if n_times is not None and (not isinstance(n_times, int) or n_times < 1):
        raise TypeError("expected n_times to be None or a positive int")

    @_wraps(sched)
    def repeated_sched(proc, *args, **kwargs):
        def do_iter():
            nonlocal proc
            local_args = args.copy() if isinstance(args, list) else args
            local_kwargs = kwargs.copy()
            proc = sched(proc, *local_args, **local_kwargs)

        if n_times is None:
            try:
                while True:
                    do_iter()
            except (SchedulingError, TypeError, ValueError) as err:
                if verbose:
                    print("repeat ended with error", err)
        else:
            for i in range(n_times):
                do_iter()

        return proc

    repeated_sched.__name__ = f"repeat_{sched.__name__}"
    return repeated_sched


_sched_seq_err = """
sched_seq(proc, sched_list) expects an iterable object `sched_list`
containing a sequence of scheduling operations to apply to `proc`.
Each scheduling operation must either be a Python Callable object
(such as a function) which will be passed one argument---the `proc`---or
the scheduling operation must be a list/tuple whose first entry is
a scheduling function/operation and whose remaining entries are arguments
to be passed to that scheduling function.

e.g.
proc = sched_seq(proc,[
    (rename, 'foo'),
    (set_memory, 'A', SPECIAL_DRAM),
    ...
])
"""


def sched_seq(proc, sched_list):
    for s in sched_list:
        if callable(s):
            proc = s(proc)
        elif isinstance(s, (list, tuple)):
            if len(s) == 0:
                raise TypeError(
                    _sched_seq_err
                    + "\n"
                    + "expected scheduling operation list/tuple "
                    + "to have at least one entry"
                )
            s_call = s[0]
            s_args = s[1:]
            proc = s_call(proc, *s_args)
        else:
            raise TypeError(_sched_seq_err)
    return proc


def loop_hack(sched, find_func, verbose=False):
    """
    DEPRECATED

    This higher level method is a patch to help migrate code to the
    new interface, but before the cursor forwarding implementation is
    complete.  It should be eliminated and replaced with a cursor-friendly
    version once forwarding is finished being implemented
    """

    @_wraps(sched)
    def loop_hack_sched(proc, *args, **kwargs):
        match_len = len(find_func(proc))
        for i in range(0, match_len):
            cursor = find_func(proc)[i]
            proc = sched(proc, cursor, *args, **kwargs)

        return proc

    loop_hack_sched.__name__ = f"loop_hack_{sched.__name__}"
    return loop_hack_sched


import exo.API_cursors as _PC
from ..API import Procedure as _Procedure
from ..LoopIR_unification import UnificationError as _UnificationError


class MemoryError(Exception):
    def __init__(self, msg):
        self._err_msg = str(msg)

    def __str__(self):
        return self._err_msg


@sched_op([BlockCursorA, ProcA, BoolA])
def call_site_mem_aware_replace(proc, block_cursor, subproc, quiet=False):
    proc = replace(proc, block_cursor, subproc, quiet=quiet)

    def check_all_calls(body_cursor):
        check_passed = True
        for cursor in body_cursor:
            if isinstance(cursor, _PC.CallCursor):
                check_passed = check_passed and check_call_mem_types(cursor)
            elif isinstance(cursor, _PC.IfCursor):
                check_passed = check_passed and check_all_calls(cursor.body())
                if type(cursor.orelse()) is not _PC.InvalidCursor:
                    check_passed = check_passed and check_all_calls(cursor.orelse())
            elif isinstance(cursor, _PC.ForCursor):
                check_passed = check_passed and check_all_calls(cursor.body())
        return check_passed

    if not check_all_calls(proc.body()):
        raise MemoryError(
            "replace failed due to memory type mismatch between block and subproc"
        )

    return proc


def _replace_helper(proc, subprocs, mem_aware, once):

    if not isinstance(subprocs, list):
        subprocs = [subprocs]

    for subproc in subprocs:
        assert isinstance(subproc, _Procedure), "expected Procedure as 2nd argument"

    patterns = {
        _PC.AssignCursor: "_ = _",
        _PC.ReduceCursor: "_ += _",
        _PC.AssignConfigCursor: "TODO",
        _PC.PassCursor: "TODO",
        _PC.IfCursor: "TODO",
        _PC.ForCursor: "for _ in _: _",
        _PC.AllocCursor: "TODO",
        _PC.CallCursor: "TODO",
        _PC.WindowStmtCursor: "TODO",
    }

    for subproc in subprocs:
        body = subproc.body()
        pattern = patterns[type(body[0])]
        i = 0
        while True:
            try:
                block = proc.find(f"{pattern} #{i}").expand(0, len(body) - 1)
                if len(block) != len(body):
                    raise _UnificationError("Unification failed due to length mismatch")

                if mem_aware:
                    proc = call_site_mem_aware_replace(proc, block, subproc, quiet=True)
                else:
                    proc = replace(proc, block, subproc, quiet=True)
                if once:
                    break
            except (TypeError, SchedulingError) as e:
                if "failed to find matches" in str(e):
                    break
                raise
            except (
                _UnificationError,
                MemoryError,
                NotImplementedError,
            ):
                i += 1

    return proc


def replace_all(proc, subprocs, mem_aware=True):
    """
    Givin a proc and subprocs, replace the body of proc with subproc
    as much as possible.

    args:
        subprocs  - list of subprocedures (or instruction definitions) to
                    be replaced with
        mem_aware - if True, replace will only suceed when memory annotation
                    also matches
    """

    return _replace_helper(proc, subprocs, mem_aware, False)


def replace_once(proc, subprocs, mem_aware=True):
    """
    Givin a proc and subprocs, replace the body of proc with subproc only once.

    args:
        subprocs  - list of subprocedures (or instruction definitions) to
                    be replaced with
        mem_aware - if True, replace will only suceed when memory annotation
                    also matches
    """

    return _replace_helper(proc, subprocs, mem_aware, True)


def lift_if(proc, cursor, n_lifts=1):
    """
    Move the indicated If-statement upwards through other control-flow
    for a total of n_lifts times.

    args:
        cursor       - cursor to the innermost if statement to lift up
        n_lifts      - number of times to lift the if statement up

    rewrite: (one example)
        `for i in _:`
        `    if p:`
        `        s1`
        `    else:`
        `        s2`
        ->
        `if p:`
        `    for i in _:`
        `        s1`
        `else:`
        `    for i in _:`
        `        s2`
    """
    orig_proc = proc
    for i in range(n_lifts):
        try:
            proc = lift_scope(proc, cursor)
        except SchedulingError as e:
            raise SchedulingError(
                f"Could not fully lift if statement! {n_lifts-i} lift(s) remain! {str(e)}",
                orig=orig_proc,
                proc=proc,
            ) from e
    return proc


def apply(p, cursors, schedop):
    for c in cursors:
        p = schedop(p, c)
    return p


def fuse_at(proc, producer: str, consumer: str, target_loop, reorder=True):
    """
    This version of compute_at will only go down one-level of for loops

    TODO: bounds currently assumes that bounds is of the form [0, 1, ..., n-1]
    TODO: remove the [reorder] hack.
    """

    target_loop = proc.forward(target_loop)
    p_assign = proc.find(f"{producer}[_] = _")
    p_loop = _PC.get_top_level_stmt(p_assign)
    c_loops = _PC.get_ancestors(target_loop, up_to=None)

    for c_loop in reversed(c_loops):
        c_loop = proc.forward(c_loop)
        p_loop = _PC.get_enclosing_loop(proc.forward(p_assign), c_loop.name())

        # infer bounds of consumer to determine cut-factor
        N_c = c_loop.hi()._impl._node
        buffer_dim = get_affected_dim(proc, consumer, c_loop.name())
        w_c = bounds_inference(proc, c_loop, consumer, buffer_dim).get_size()

        p_iter = p_loop.name()
        new_iters = [f"{p_iter}", f"{p_iter}i"]

        assert p_loop.next() == c_loop
        proc = divide_with_recompute(proc, p_loop, f"{N_c}", w_c, new_iters)
        proc = fuse(proc, p_loop, c_loop, unsafe_disable_check=True)

        p_inner_loop = proc.forward(p_loop).body()[0]
        while isinstance(p_inner_loop.body()[0], _PC.ForCursor):
            proc = reorder_loops(proc, p_inner_loop)
            p_inner_loop = proc.forward(p_inner_loop)

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


def store_at(proc, producer, consumer, target_loop):
    """
    Moves [producer]'s allocation into

    TODO: bounds
     - currently assumes that bounds is of the form [0, 1, ..., n-1]
     - bounds should be automatically inferred, not manually passed
    """
    target_loop = proc.forward(target_loop)
    producer_alloc = proc.find(f"{producer}:_")

    def loops_between(producer_alloc, target_loop):
        top_loop = _PC.match_level(target_loop, producer_alloc)
        return _PC.get_ancestors(target_loop, up_to=top_loop)

    for loop in reversed(loops_between(producer_alloc, target_loop)):
        buffer_dim = get_affected_dim(proc, producer, loop.name())

        loop = proc.forward(loop)
        producer_alloc = proc.forward(producer_alloc)

        lo, hi = bounds_inference(proc, loop, producer, buffer_dim).get_bounds()

        while producer_alloc.next() != loop:
            reorder_stmts(proc, producer_alloc.expand(0, 1))
            producer_alloc = proc.forward(producer_alloc)

        proc = sink_alloc(proc, producer_alloc)
        proc = resize_dim(proc, producer_alloc, buffer_dim, lo, hi)

    return simplify(proc)


def compute_at(proc, producer, consumer, target_loop):
    """
    Combination of fuse_at and store_at.
    """
    target_loop = proc.forward(target_loop)
    proc = fuse_at(proc, producer, consumer, target_loop)
    target_loop = proc.forward(target_loop.body()[0]).parent()
    proc = store_at(proc, producer, consumer, target_loop)
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
    # TODO: fix this for perfect=False
    proc = divide_loop(proc, i_loop, i_tile_size, new_i_iters, perfect=perfect)
    proc = divide_loop(proc, j_loop, j_tile_size, new_j_iters, perfect=perfect)
    ii_loop = proc.forward(i_loop).body()[0]
    proc = reorder_loops(proc, ii_loop)
    return proc
