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
    rearrange_dim,
    bound_alloc,
    divide_dim,
    mult_dim,
    lift_alloc,
    reuse_buffer,
    inline_window,
    stage_window,
    stage_mem,
    #
    # loop rewriting
    divide_loop,
    mult_loops,
    cut_loop,
    reorder_loops,
    merge_writes,
    fission,
    fuse,
    remove_loop,
    add_loop,
    unroll_loop,
    #
    # guard rewriting
    lift_scope,
    assert_if,
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

from .analysis import (
    check_call_mem_types,
)

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


from ..API_cursors import public_cursors as _PC
from ..API import Procedure as _Procedure
from ..LoopIR_unification import UnificationError as _UnificationError


class MemoryError(Exception):
    def __init__(self, msg):
        self._err_msg = str(msg)

    def __str__(self):
        return self._err_msg


@sched_op([BlockCursorA, ProcA, BoolA])
def mem_aware_replace(proc, block_cursor, subproc, quiet=False):
    def has_alloc(body):
        if not isinstance(body, _PC.BlockCursor):
            return
        check = False
        for stmt in body:
            stype = type(stmt)
            if stype is _PC.AllocCursor:
                return True
            elif stype is _PC.IfCursor:
                check = check or has_alloc(stmt.body())
                check = check or has_alloc(stmt.orelse())
            elif stype is _PC.ForSeqCursor:
                check = check or has_alloc(stmt.body())
        return check

    if has_alloc(block_cursor):
        raise MemoryError(
            "reasoning about memories allocated within a block is not supported"
        )

    if has_alloc(subproc):
        raise MemoryError(
            "reasoning about memories allocated within a procedure is not supported"
        )

    proc = replace(proc, block_cursor, subproc, quiet=quiet)

    # TODO: specifically pass a cursor to the new call statement with forwarding
    # this check might throw an error for an unrelated reason
    # if not check_call_mem_types(forwarded_cursor):
    #     raise MemoryError(
    #         "replace failed due to memory type mismatch between block and subproc"
    #     )

    def check_all_calls(body_cursor):
        check_passed = True
        for cursor in body_cursor:
            if isinstance(cursor, _PC.CallCursor):
                check_passed = check_passed and check_call_mem_types(cursor)
            elif isinstance(cursor, _PC.IfCursor):
                check_passed = check_passed and check_all_calls(cursor.body())
                check_passed = check_passed and check_all_calls(cursor.orlese())
            elif isinstance(cursor, _PC.ForSeqCursor):
                check_passed = check_passed and check_all_calls(cursor.body())
        return check_passed

    if not check_all_calls(proc.body()):
        raise MemoryError(
            "replace failed due to memory type mismatch between block and subproc"
        )

    return proc


def replace_all(proc, subproc):
    """
    DEPRECATED ?
    Is there a better way to write this out of primitives?
    Does this simply require that we have better introspection facilities?
    """
    assert isinstance(subproc, _Procedure), "expected Procedure as 2nd argument"
    body = subproc.body()
    assert len(body) == 1, (
        "replace_all only supports single statement " "subprocedure bodies right now"
    )

    patterns = {
        _PC.AssignCursor: "_ = _",
        _PC.ReduceCursor: "_ += _",
        _PC.AssignConfigCursor: "TODO",
        _PC.PassCursor: "TODO",
        _PC.IfCursor: "TODO",
        _PC.ForSeqCursor: "for _ in _: _",
        _PC.AllocCursor: "TODO",
        _PC.CallCursor: "TODO",
        _PC.WindowStmtCursor: "TODO",
    }

    pattern = patterns[type(body[0])]
    i = 0
    while True:
        try:
            proc = mem_aware_replace(proc, f"{pattern} #{i}", subproc, quiet=True)
        except (TypeError, SchedulingError) as e:
            if "failed to find matches" in str(e):
                return proc
            raise
        except (_UnificationError, MemoryError):
            i += 1


def replace_all_arch(proc, arch_instrs):
    """
    try to replace every procedure in `arch_instrs`
    with every statement in the `proc`

    Args:
        proc (Procedure): proc to replace statements in
        arch_instrs (Iterable): an iterable containing Procedures

    Returns:
        Procedure: the result after procedure after attempting all the replacements
    """
    for instr in arch_instrs:
        proc = replace_all(proc, instr)
    return proc


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


def replace_all_mockup(proc, l):
    # Get all instructions in global scope
    # for each statements, we rum replace on all instructions
    # run memoryanalysis on every succussful replace
    proc = replace_all(proc, l[0])
    proc = replace_all(proc, l[1])
    proc = replace(proc, "for i0 in _:_", l[2])
    proc = replace(proc, "for i0 in _:_", l[3])
    return proc
