from exo import DRAM

from ..API_cursors import public_cursors as _PC
from ..API import Procedure, SchedulingError


class SymProperties:
    """
    An object that holds information about the properties of a `Sym`
    """

    def __init__(self, mem=None):
        self.mem = mem

    def get_mem(self):
        """
        The memory type of the `Sym` or `None`
        if the symbol is not a buffer.

        Returns:
            Memory | None: memory type of the symbol
        """
        return self.mem

    def __repr__(self) -> str:
        return f"{{mem: {self.mem}}}"


def build_env(stmt_cursor):
    """
    A function that build the environment observed by a statement

    Args:
        stmt_cursor (StmtCursor): a cursor to the statement to build the environment

    Raises:
        TypeError: if `stmt_cursor` is not of type `StmtCursor`

    Returns:
        dict: a mapping from `Sym` to `SymProperties` where the keys
            are all the symbols in the environment observed by the statement
    """
    if not isinstance(stmt_cursor, _PC.StmtCursor):
        raise TypeError("cursor to build the environment around must be a StmtCursor")

    env = {}
    proc = stmt_cursor.proc()

    # Add proc arguments to env
    for arg in proc._loopir_proc.args:
        if arg.type.is_numeric():
            mem = arg.mem
            mem = mem if mem is not None else DRAM
            env[arg.name] = SymProperties(mem=mem)

    def build_s(cursor):
        prev_cursor = cursor.prev()

        # This means cursor was the first statement in a body
        if type(prev_cursor) is _PC.InvalidCursor:
            prev_cursor = cursor.parent()

        # This means cursor was the first statement in the proc
        if type(prev_cursor) is _PC.InvalidCursor:
            return

        if type(prev_cursor) is _PC.AllocCursor:
            mem = prev_cursor.mem()
            mem = mem if mem is not None else DRAM
            env[prev_cursor.name()] = SymProperties(mem=mem)

        if type(prev_cursor) is _PC.ForSeqCursor:
            env[prev_cursor.name()] = SymProperties(mem=None)

        return build_s(prev_cursor)

    build_s(stmt_cursor)

    return env


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

    env = build_env(call_cursor)

    def get_e_mem(e_cursor):
        if isinstance(e_cursor, (_PC.ReadCursor, _PC.WindowExprCursor)):
            return env[e_cursor.name()].get_mem()
        else:
            assert False

    for ca, sa in zip(call_cursor.args(), call_cursor.subproc()._loopir_proc.args):
        if sa.type.is_numeric():
            smem = sa.mem if sa.mem else DRAM
            cmem = get_e_mem(ca)
            if not issubclass(cmem, smem):
                return False

    return True
