
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
    reuse_buffer,
    inline_window,
    #
    # loop rewriting
    divide_loop,
    reorder_loops,
    fission,
    #
    # deprecated scheduling operations
    add_unsafe_guard,
    double_fission,
    #
    # to be replaced by stdlib compositions eventually
    autofission,
)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Higher-order Scheduling operations

def repeat(sched, verbose=False):
    """
    TODO: Documentation
    """
    @_wraps(sched)
    def repeated_sched(proc, *args, **kwargs):
        try:
            while True:
                local_args = args.copy() if isinstance(args, list) else args
                local_kwargs = kwargs.copy()
                proc = sched(proc, *local_args, **local_kwargs)
        except (SchedulingError, TypeError, ValueError) as err:
            if verbose:
                print("repeat ended with error", err)

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


