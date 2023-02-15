# from __future__ import annotations
#
# import weakref
# from abc import ABC, abstractmethod
from dataclasses import dataclass

# from enum import Enum, auto
# from functools import cached_property
from typing import Optional, Iterable, Union, List, Any

# from weakref import ReferenceType
#
from . import API
from .LoopIR import LoopIR
from .configs import Config
from .memory import Memory

from . import internal_cursors as C
from .prelude import Sym

# expose this particular exception as part of the API
from .internal_cursors import InvalidCursorError


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# General Cursor Interface


@dataclass
class Cursor:
    """
    This is the base class for all cursors.  Cursors are objects that are
    used in scheduling to point to different parts of a Procedure's AST.
    You can think of a cursor as defined by the data pair
        (Procedure, Location)

    You can navigate a cursor around within a Procedure using its various
    methods.  However, note that a valid Cursor pointing into one Procedure
    p1 is not inherently a valid Cursor pointing into another Procedure p2,
    even if p2 was created from p1 using scheduling transformations.

    If p2 was created from p1, then we can `update` a Cursor c1 pointing
    to p1 to a cursor c2 pointing to p2.  (TODO: implement `update`)

    The sub-class hierarchy looks like:
    - Cursor
        - InvalidCursor         - used to indicate cursors that no longer
                                  point to any sensible/valid object.
        - StmtCursorPrototype
            - StmtCursor        - individual statement
                - ... (cases below)
            - BlockCursor       - contiguous, non-empty sequence of statements
            - GapCursor         - the "gap" between two statements, before the
                                  first statement, or after the last statement
                                  (think of this as a blinking vertical line)
        - ExprCursorPrototype
            - ExprCursor        - individual expression
                - ... (cases below)
            - ExprListCursor    - contiguous (maybe empty) seq. of expressions

    The grammar for statements and expressions as exposed by cursors is:
        Stmt ::= Assign( name : Sym, idx : ExprList, rhs : Expr )
               | Reduce( name : Sym, idx : ExprList, rhs : Expr )
               | AssignConfig( config : Config, field : str, rhs : Expr )
               | Pass()
               | If( cond : Expr, body : Block, orelse : Block? )
               | ForSeq( name : Sym, hi : Expr, body : Block )
               | Alloc( name : Sym, mem : Memory? )
               | Call( subproc : Procedure, args : ExprList )
               | WindowStmt( name : Sym, winexpr : WindowExpr )

        Expr ::= Read( name : Sym, idx : ExprList )
               | ReadConfig( config : Config, field : str )
               | Literal( value : bool, int, or float )
               | UnaryMinus( arg : Expr )
               | BinaryOp( op : str, lhs : Expr, rhs : Expr )
               | BuiltIn( name : str, args : ExprList )
               | WindowExpr( name : str, idx : *(see below) )
               | BuiltIn( name : str, args : ExprList )

        The `idx` argument of `WindowExpr` is a list containing either
        `Expr` or `(Expr,Expr)` (a pair of expressions) at each position.
        The single expressions correspond to point accesses; the pairs to
        interval slicing/windowing.
    """

    _impl: C.Cursor

    def __init__(self, impl):
        if not isinstance(impl, C.Cursor):
            raise TypeError(
                "Do not try to directly construct a Cursor.  "
                "Use the provided methods to obtain cursors "
                "from Procedures, and from other Cursors"
            )
        self._impl = impl

    def __str__(self):
        return f"<{type(self).__name__}(...)>"

    # -------------------------------------------------------------------- #
    # methods copied from the underlying implementation

    def proc(self):
        """
        Get the Procedure object that this Cursor points into
        """
        return self._impl.proc()

    def __bool__(self):
        """
        Invalid Cursors behave like False when tested. All other Cursors
        behave like True.  So, one can test a cursor `if c: ...` to
        check for validity/invalidity
        """
        return True

    def parent(self):
        """
        Get a Cursor to the parent node in the syntax tree.

        Raises InvalidCursorError if no parent exists
        """
        impl_parent = self._impl.parent()
        if isinstance(impl_parent._node, LoopIR.w_access):
            impl_parent = impl_parent.parent()
        elif isinstance(impl_parent._node, LoopIR.proc):
            return InvalidCursor()
        return new_Cursor(impl_parent)

    def prev(self, dist=1):
        """
        If this is a statement Cursor, return a statement cursor to
            the previous statement in the block (or dist-many previous)
        If this is a gap Cursor, return a gap cursor to
            the previous gap in the block (or dist-many previous)

        Returns InvalidCursor() if there is no such cursor to point to.
        """
        if dist < 1:
            raise ValueError(f"dist must have positive value; was {dist}")
        try:
            return new_Cursor(self._impl.prev(dist))
        except InvalidCursorError:
            return InvalidCursor()

    def next(self, dist=1):
        """
        If this is a statement Cursor, return a statement cursor to
            the next statement in the block (or dist-many next)
        If this is a gap Cursor, return a gap cursor to
            the next gap in the block (or dist-many next)

        Returns InvalidCursor() if there is no such cursor to point to.
        """
        if dist < 1:
            raise ValueError(f"dist must have positive value; was {dist}")
        try:
            return new_Cursor(self._impl.next(dist))
        except InvalidCursorError:
            return InvalidCursor()


class InvalidCursor(Cursor):
    def __init__(self):
        """Invalid cursors have no data"""
        self._impl = None

    def __bool__(self):
        """
        Invalid Cursors behave like False when tested. All other Cursors
        behave like True.  So, one can test a cursor `if c: ...` to
        check for validity/invalidity
        """
        return False

    def proc(self):
        raise InvalidCursorError("Cannot get the Procedure " "of an invalid cursor")

    def parent(self) -> Cursor:
        """The parent of an invalid cursor is an invalid cursor"""
        return self

    def before(self, dist=1) -> Cursor:
        """navigating to before an invalid cursor is still invalid"""
        return self

    def after(self, dist=1) -> Cursor:
        """navigating to after an invalid cursor is still invalid"""
        return self

    def next(self, dist=1) -> Cursor:
        """navigating to the next cursor is still invalid"""
        return self

    def prev(self, dist=1) -> Cursor:
        """navigating to the previous cursor is still invalid"""
        return self


class StmtCursorPrototype(Cursor):
    """
    Prototype for all cursors used to point to statements:
        StmtCursor, BlockCursor, GapCursor
    See `help(Cursor)` for more details.
    """

    def before(self, dist=1) -> Cursor:
        """
        If this is a statement or block Cursor, return a gap Cursor
            pointing to immediately before the first statement.
        If this is a gap Cursor, return a statement Cursor, pointing to
            the statement immediately before the gap.

        If dist > 1, then return the gap/statement dist-many spots before
            the cursor, rather than immediately (1-many) before the cursor

        Returns InvalidCursor() if there is no such cursor to point to.
        """
        if dist < 1:
            raise ValueError(f"dist must have positive value; was {dist}")
        try:
            return new_Cursor(self._impl.before(dist))
        except InvalidCursorError:
            return InvalidCursor()

    def after(self, dist=1) -> Cursor:
        """
        If this is a statement or block Cursor, return a gap Cursor
            pointing to immediately after the first statement.
        If this is a gap Cursor, return a statement Cursor, pointing to
            the statement immediately after the gap.

        If dist > 1, then return the gap/statement dist-many spots after
            the cursor, rather than immediately (1-many) after the cursor

        Returns InvalidCursor() if there is no such cursor to point to.
        """
        if dist < 1:
            raise ValueError(f"dist must have positive value; was {dist}")
        try:
            return new_Cursor(self._impl.after(dist))
        except InvalidCursorError:
            return InvalidCursor()

    def block_all(self):
        """
        Return a BlockCursor for the entire statement block that this
        Stmt, Block, or Gap is contained in
        """
        return BlockCursor(self._impl._whole_block())


class StmtCursor(StmtCursorPrototype):
    """
    Cursor pointing to an individual statement or expression.
    See `help(Cursor)` for more details.
    """

    def as_block(self):
        """Return a Block containing only this one statement"""
        return BlockCursor(self._impl.as_block())

    def expand(self, arg1=None, arg2=None):
        """Shorthand for stmt_cursor.as_block().expand(...)"""
        return self.as_block().expand(arg1, arg2)


class BlockCursor(StmtCursorPrototype):
    """
    Cursor pointing to a contiguous sequence of statements.
    See `help(Cursor)` for more details.
    """

    def as_block(self):
        """Return this Block; included for symmetry with StmtCursor"""
        return self

    def expand(self, arg1=None, arg2=None):
        """
        Expand the block cursor.

        Calling convention with zero arguments
            curosr.expand()     - make the block as big as possible
        Calling convention with one argument
            cursor.expand(n)    - adds n statements to the end of the block
            cursor.expand(-n)   - adds n statements to the start of the block

        Calling convention with two arguments
            cursor.expand(n,m)  - adds n statements to the start of the block
                                   and m statements to the end of the block
        """
        if arg1 is None and arg2 is not None:
            raise TypeError("Don't supply arg2 without arg1")
        elif arg1 is None and arg2 is None:
            return BlockCursor(self._impl.expand())
        elif arg2 is None:
            if not isinstance(arg1, int):
                raise TypeError("expected an integer argument")
            if arg1 < 0:
                return BlockCursor(self._impl.expand(-arg1, 0))
            else:
                return BlockCursor(self._impl.expand(0, arg1))
        else:  # arg2 is defined
            if (
                not isinstance(arg1, int)
                or not isinstance(arg2, int)
                or arg1 < 0
                or arg2 < 0
            ):
                raise TypeError(
                    "expected one integer argument "
                    "or two non-negative integer arguments"
                )
            return BlockCursor(self._impl.expand(arg1, arg2))

    def __iter__(self):
        """
        iterate over all statement cursors contained in the block
        """
        for stmt_impl in iter(self._impl):
            yield new_Cursor(stmt_impl)

    def __getitem__(self, i) -> StmtCursor:
        """
        get a cursor to the i-th statement
        """
        return new_Cursor(self._impl[i])

    def __len__(self) -> int:
        """
        get the number of statements in the block
        """
        return len(self._impl)


class GapCursor(StmtCursorPrototype):
    """
    Cursor pointing to a gap before, after, or between statements.
    See `help(Cursor)` for more details.
    """


class ExprCursorPrototype(Cursor):
    """
    Prototype for all cursors used to point to statements:
        StmtCursor, BlockCursor, GapCursor
    See `help(Cursor)` for more details.
    """

    def list_all(self):
        """
        Return an ExprListCursor for the list of expressions that this
        expression is contained in, if this expression is contained in a
        list.  Otherwise, return an InvalidCursor
        """
        try:
            return ExprListCursor(self._impl._whole_block())
        except InvalidCursorError:
            return InvalidCursor()


class ExprCursor(ExprCursorPrototype):
    """
    Cursor pointing to an individual statement or expression.
    See `help(Cursor)` for more details.
    """


class ExprListCursor(Cursor):
    """
    Cursor pointing to a contiguous sequence of expressions.
    See `help(Cursor)` for more details.
    """

    def __iter__(self):
        """
        iterate over all expression cursors contained in the argument list
        """
        for stmt_impl in iter(self._impl):
            yield new_Cursor(stmt_impl)

    def __getitem__(self, i) -> ExprCursor:
        """
        get a cursor to the i-th argument
        """
        return new_Cursor(self._impl[i])

    def __len__(self) -> int:
        """
        get the number of arguments
        """
        return len(self._impl)


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Specific Statement Cursor Types


class AssignCursor(StmtCursor):
    """
    Cursor pointing to an assignment statement:
        `name [ idx ] = rhs`
    """

    def name(self) -> Sym:
        return self._impl._node.name

    def idx(self) -> ExprListCursor:
        return ExprListCursor(self._impl._child_block("idx"))

    def rhs(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("rhs"))


class ReduceCursor(StmtCursor):
    """
    Cursor pointing to a reduction statement:
        `name [ idx ] += rhs`
    """

    def name(self) -> Sym:
        return self._impl._node.name

    def idx(self) -> ExprListCursor:
        return ExprListCursor(self._impl._child_block("idx"))

    def rhs(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("rhs"))


class AssignConfigCursor(StmtCursor):
    """
    Cursor pointing to a configuration assignment statement:
        `config.field = rhs`
    """

    def config(self) -> Config:
        return self._impl._node.config

    def field(self) -> str:
        return self._impl._node.field

    def rhs(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("rhs"))


class PassCursor(StmtCursor):
    """
    Cursor pointing to a no-op statement:
        `pass`
    """


class IfCursor(StmtCursor):
    """
    Cursor pointing to an if statement:
        ```
        if condition:
            body
        ```
    or
        ```
        if condition:
            body
        else:
            orelse
        ```
    Returns an invalid cursor if `orelse` isn't present.
    """

    def cond(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("cond"))

    def body(self) -> BlockCursor:
        return BlockCursor(self._impl.body())

    def orelse(self) -> Cursor:
        orelse = self._impl.orelse()
        return BlockCursor(orelse) if len(orelse) > 0 else InvalidCursor()


class ForSeqCursor(StmtCursor):
    """
    Cursor pointing to a loop statement:
        ```
        for name in seq(0,hi):
            body
        ```
    """

    def name(self) -> Sym:
        return self._impl._node.iter

    def hi(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("hi"))

    def body(self) -> BlockCursor:
        return BlockCursor(self._impl.body())


class AllocCursor(StmtCursor):
    """
    Cursor pointing to a buffer definition statement:
        ```
        name : type @ mem
        ```
    """

    def name(self) -> Sym:
        return self._impl._node.name

    def mem(self) -> Optional[Memory]:
        return self._impl._node.mem


class CallCursor(StmtCursor):
    """
    Cursor pointing to a sub-procedure call statement:
        ```
        subproc( args )
        ```
    """

    def subproc(self):
        return API.Procedure(self._impl._node.f)

    def args(self) -> ExprListCursor:
        return ExprListCursor(self._impl._child_block("args"))


class WindowStmtCursor(StmtCursor):
    """
    Cursor pointing to a window declaration statement:
        ```
        name = winexpr
        ```
    """

    def name(self) -> Sym:
        return self._impl._node.name

    def winexpr(self) -> ExprCursor:
        return WindowExprCursor(self._impl._child_node("rhs"))


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Specific Expression Cursor Types


class ReadCursor(ExprCursor):
    """
    Cursor pointing to a read expression:
        `name`
    or
        `name [ idx ]`
    """

    def name(self) -> Sym:
        return self._impl._node.name

    def idx(self) -> ExprListCursor:
        return ExprListCursor(self._impl._child_block("idx"))


class ReadConfigCursor(ExprCursor):
    """
    Cursor pointing to a Config read expression:
        `config.field`
    """

    def config(self) -> Config:
        return self._impl._node.config

    def field(self) -> str:
        return self._impl._node.field


class LiteralCursor(ExprCursor):
    """
    Cursor pointing to a literal expression:
        `value`

    `value` should have Python type `bool`, `int` or `float`.
    If `value` has type `float` then it is a data-value literal.
    Otherwise, it should be a control-value literal.
    """

    def value(self) -> Any:
        n = self._impl._node
        assert (
            (n.type == T.bool and type(n.val) == bool)
            or (n.type.is_indexable() and type(n.val) == int)
            or (n.type.is_real_scalar() and type(n.val) == float)
        )
        return n.val


class UnaryMinusCursor(ExprCursor):
    """
    Cursor pointing to a unary minus-sign expression:
        `- arg`
    """

    def arg(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("arg"))


class BinaryOpCursor(ExprCursor):
    """
    Cursor pointing to an in-fix binary operation expression:
        `lhs op rhs`
    where `op` is one of:
        + - * / % < > <= >= == and or
    """

    def op(self) -> str:
        return self._impl._node.op

    def lhs(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("lhs"))

    def rhs(self) -> ExprCursor:
        return new_Cursor(self._impl._child_node("rhs"))


class BuiltInFunctionCursor(ExprCursor):
    """
    Cursor pointing to the call to some built-in function
        `name ( args )`
    """

    def name(self) -> str:
        return self._impl._node.f.name()

    def args(self) -> ExprListCursor:
        return ExprListCursor(self._impl._child_block("args"))


class WindowExprCursor(ExprCursor):
    """
    Cursor pointing to a windowing expression:
        `name [ w_args ]`

    Note that w_args is not an argument cursor.  Instead it is a list
    of "w-expressions" which are either an ExprCursor, or a pair of
    ExprCursors.  The pair represents a windowing interval; the single
    expression represents a point-access in that dimension.
    """

    def name(self) -> str:
        return self._impl._node.f.name()

    def idx(self) -> List:
        def convert_w(w):
            if isinstance(w._node, LoopIR.Interval):
                return (
                    new_Cursor(w._child_node("lo")),
                    new_Cursor(w._child_node("hi")),
                )
            else:
                return new_Cursor(w._child_node("pt"))

        return [convert_w(w) for w in self._impl._child_block("idx")]


class StrideExprCursor(ExprCursor):
    """
    Cursor pointing to a stride expression:
        `stride ( name , dim )`
    (note that stride is a keyword, and not data/a sub-expression)
    `name` is the name of some buffer or window
    """

    def name(self) -> Sym:
        return self._impl._node.name

    def dim(self) -> int:
        return self._impl._node.dim


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# List of objects to expose


class public_cursors:
    pass


for c in [
    Cursor,
    InvalidCursor,
    StmtCursorPrototype,
    StmtCursor,
    BlockCursor,
    GapCursor,
    ExprCursorPrototype,
    ExprCursor,
    ExprListCursor,
    #
    AssignCursor,
    ReduceCursor,
    AssignConfigCursor,
    PassCursor,
    IfCursor,
    ForSeqCursor,
    AllocCursor,
    CallCursor,
    WindowStmtCursor,
    #
    ReadCursor,
    ReadConfigCursor,
    LiteralCursor,
    UnaryMinusCursor,
    BinaryOpCursor,
    BuiltInFunctionCursor,
    WindowExprCursor,
    StrideExprCursor,
    #
    InvalidCursorError,
]:
    setattr(public_cursors, c.__name__, c)

# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Internal Functions; Not for Exposure to Users

# helper function to dispatch to constructors
def new_Cursor(impl):
    assert isinstance(impl, C.Cursor)

    # dispatch to the correct constructor...
    if isinstance(impl, C.Gap):
        return GapCursor(impl)

    elif isinstance(impl, C.Block):
        # TODO: Rename internal Cursor type to Sequence?
        assert len(impl) > 0
        n0 = impl[0]._node
        if isinstance(n0, LoopIR.stmt):
            assert all(isinstance(c._node, LoopIR.stmt) for c in impl)
            return BlockCursor(impl)
        elif isinstance(n0, LoopIR.expr):
            assert all(isinstance(c._node, LoopIR.expr) for c in impl)
            return ExprListCursor(impl)
        else:
            assert False, "bad case"

    elif isinstance(impl, C.Node):
        n = impl._node

        # statements
        if isinstance(n, LoopIR.Assign):
            return AssignCursor(impl)
        elif isinstance(n, LoopIR.Reduce):
            return ReduceCursor(impl)
        elif isinstance(n, LoopIR.WriteConfig):
            return AssignConfigCursor(impl)
        elif isinstance(n, LoopIR.Pass):
            return PassCursor(impl)
        elif isinstance(n, LoopIR.If):
            return IfCursor(impl)
        elif isinstance(n, LoopIR.Seq):
            return ForSeqCursor(impl)
        elif isinstance(n, LoopIR.Alloc):
            return AllocCursor(impl)
        elif isinstance(n, LoopIR.Call):
            return CallCursor(impl)
        elif isinstance(n, LoopIR.WindowStmt):
            return WindowStmtCursor(impl)

        # expressions
        elif isinstance(n, LoopIR.Read):
            return ReadCursor(impl)
        elif isinstance(n, LoopIR.ReadConfig):
            return ReadConfigCursor(impl)
        elif isinstance(n, LoopIR.Const):
            return LiteralCursor(impl)
        elif isinstance(n, LoopIR.USub):
            return UnaryMinusCursor(impl)
        elif isinstance(n, LoopIR.BinOp):
            return BinaryOpCursor(impl)
        elif isinstance(n, LoopIR.BuiltIn):
            return BuiltInFunctionCursor(impl)
        elif isinstance(n, LoopIR.WindowExpr):
            return WindowExprCursor(impl)
        elif isinstance(n, LoopIR.StrideExpr):
            return StrideExprCursor(impl)

        else:
            assert False, f"bad case: {type(n)}"

    else:
        assert False, f"bad case: {type(impl)}"
