from exo import DRAM

import exo.API_cursors as _PC
from ..API import Procedure, SchedulingError


def get_observed_stmts(stmt_cursor):
    """
    Generator of observed statement by this statement cursor

    Args:
        stmt_cursor (StmtCursor): a statement cursor

    Raises:
        TypeError: if stmt_cursor is not of the correct type

    Yields:
        StmtCursor: cursors to all the previous statements

    def proc(...):
        s1;
        s2;
        if ... : (s3)
            s4;
            s5;
        for ... : (s6) <- stmt_cursor.parent()
            s7; <- stmt_cursor.prev()
            s8; <- stmt_cursor
            s9;

        yields: s7, s6, s3; s2; s1
    """
    if not isinstance(stmt_cursor, _PC.StmtCursor):
        raise TypeError("stmt_cursor must be an instance of StmtCursor")

    cursor = stmt_cursor
    while True:
        prev_cursor = cursor.prev()

        # This means cursor was the first statement in a body
        if type(prev_cursor) is _PC.InvalidCursor:
            prev_cursor = cursor.parent()

            # This means cursor was the first statement in the proc
            if type(prev_cursor) is _PC.InvalidCursor:
                return

        cursor = prev_cursor
        yield cursor


def check_call_mem_types(call_cursor):
    """
    check memory consistency between the called procedure
    and the arguments passed at the call site.

    Args:
        call_cursor (CallCursor): call to check memory consistency at

    Raises:
        TypeError: if the cursor provided isn't a `CallCursor`

    Returns:
        bool: whether memory consistency holds at the call site
    """
    if not isinstance(call_cursor, _PC.CallCursor):
        raise TypeError("call_cursor must be a CallCursor")

    ###################################################################
    # build an env of symbols this call statement observes.
    # e.g. {x: DRAM, y: Neon}
    ###################################################################
    env = {}
    caller = call_cursor.proc()

    # Add proc parameters to env
    for arg in caller.args():
        if arg.type().is_numeric():
            mem = arg.mem()
            env[arg.name()] = mem

    # Search through observed statement to find allocations
    for stmt_cursor in get_observed_stmts(call_cursor):
        # Found a buffer allocation, record memory type
        if type(stmt_cursor) is _PC.AllocCursor:
            mem = stmt_cursor.mem()
            env[stmt_cursor.name()] = mem

    ###################################################################
    # Check memory consistency at call site
    ###################################################################
    call_args = call_cursor.args()
    callee_parameters = call_cursor.subproc().args()
    for ca, sa in zip(call_args, callee_parameters):
        if sa.type().is_numeric():
            smem = sa.mem()
            cmem = env[ca.name()]
            # Check if the argument memory type is a subclass of the callee's parameter
            if not issubclass(cmem, smem):
                return False

    return True
