
from functools import wraps as _wraps

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Expose the built-in Scheduling operators here

from ..API import (
    SchedulingError,
)

from ..API_scheduling import (
    is_atomic_scheduling_op,
    # basic operations
    simplify,
    rename,
    make_instr,
    #
    # general statement operations
    insert_pass,
    delete_pass,
    reorder_stmts,
    #
    # subprocedure oriented operations
    extract_subproc,
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
    lift_alloc,
    reuse_buffer,
    inline_window,
    #
    # loop rewriting
    divide_loop,
    reorder_loops,
    fission,
    fusion,
    remove_loop,
    add_loop,
    #
    # guard rewriting
    lift_if,
    assert_if,
    specialize,
    #
    # deprecated scheduling operations
    add_unsafe_guard,
    double_fission,
    #
    # to be replaced by stdlib compositions eventually
    autofission,
    autolift_alloc,
    bound_and_guard,
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
            cursor  = find_func(proc)[i]
            proc    = sched(proc, cursor, *args, **kwargs)

        return proc

    loop_hack_sched.__name__ = f"loop_hack_{sched.__name__}"
    return loop_hack_sched


