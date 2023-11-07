from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing import Optional, List, Any

from . import API
from .LoopIR import LoopIR
from .configs import Config
from .memory import Memory

from . import internal_cursors as C
from .prelude import Sym

# expose this particular exception as part of the API
from .internal_cursors import InvalidCursorError
from .LoopIR_pprint import _print_cursor


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# General Cursor Interface


@dataclass
class Cursor(ABC):
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
            - GapCursor         - the space before or after a particular statement
        - ExprCursorPrototype
            - ExprCursor        - individual expression
                - ... (cases below)
            - ExprListCursor    - contiguous (maybe empty) seq. of expressions

    The grammar for statements and expressions as exposed by cursors is:
        Stmt ::= Assign( name : str, idx : ExprList, rhs : Expr )
               | Reduce( name : str, idx : ExprList, rhs : Expr )
               | AssignConfig( config : Config, field : str, rhs : Expr )
               | Pass()
               | If( cond : Expr, body : Block, orelse : Block? )
               | ForSeq( name : str, hi : Expr, body : Block )
               | Alloc( name : str, mem : Memory? )
               | Call( subproc : Procedure, args : ExprList )
               | WindowStmt( name : str, winexpr : WindowExpr )

        Expr ::= Read( name : str, idx : ExprList )
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
    _proc: API.Procedure

    def __init__(self, impl, proc):
        if not isinstance(impl, C.Cursor):
            raise TypeError(
                "Do not try to directly construct a Cursor. "
                "Use the provided methods to obtain cursors "
                "from Procedures and from other Cursors"
            )
        self._impl = impl
        self._proc = proc

    def __str__(self):
        return f"{_print_cursor(self._impl)}"

    # -------------------------------------------------------------------- #
    # methods copied from the underlying implementation

    def proc(self):
        """
        Get the Procedure object that this Cursor points into
        """
        return self._proc

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
        return lift_cursor(impl_parent, self._proc)

    def _child_node(self, *args, **kwargs):
        return lift_cursor(self._impl._child_node(*args, **kwargs), self._proc)

    def _child_block(self, *args, **kwargs):
        return lift_cursor(self._impl._child_block(*args, **kwargs), self._proc)


class InvalidCursor(Cursor):
    # noinspection PyMissingConstructor
    # we can't call the Cursor constructor since it checks the type of _impl
    def __init__(self):
        """Invalid cursors have no data"""
        self._impl = None
        self._proc = None

    def __bool__(self):
        """
        Invalid Cursors behave like False when tested. All other Cursors
        behave like True.  So, one can test a cursor `if c: ...` to
        check for validity/invalidity
        """
        return False

    def proc(self):
        raise InvalidCursorError("Cannot get the Procedure of an invalid cursor")

    def parent(self) -> Cursor:
        """The parent of an invalid cursor is an invalid cursor"""
        return self


class ListCursorPrototype(Cursor):
    def __iter__(self):
        """
        iterate over all cursors contained in the list
        """
        assert isinstance(self._impl, C.Block)
        yield from (lift_cursor(stmt_impl, self._proc) for stmt_impl in self._impl)

    def __getitem__(self, i) -> Cursor:
        """
        get a cursor to the i-th item
        """
        assert isinstance(self._impl, C.Block)
        return lift_cursor(self._impl[i], self._proc)

    def __len__(self) -> int:
        """
        get the number of items in the list
        """
        assert isinstance(self._impl, C.Block)
        return len(self._impl)


class ArgCursor(Cursor):
    """
    Cursor pointing to an argument of a procedure.
        ```
        name : type @ mem
        ```
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.fnarg)

        return self._impl._node.name.name()

    def mem(self) -> Optional[Memory]:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.fnarg)
        assert not self._impl._node.type.is_indexable()

        return self._impl._node.mem

    def is_tensor(self) -> bool:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.fnarg)

        return isinstance(self._impl._node.type, LoopIR.Tensor)

    def shape(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.fnarg)
        assert self.is_tensor()

        return ExprListCursor(
            self._impl._child_node("type")._child_block("hi"), self._proc
        )

    def type(self) -> API.ExoType:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.fnarg)

        return loopir_type_to_exotype(self._impl._node.type.basetype())


class StmtCursorPrototype(Cursor):
    """
    A base class that is mostly useful for testing whether some
    cursor is pointing to the statement fragment of the IR language
    or the expression fragment.
    """


class StmtCursor(StmtCursorPrototype):
    """
    Cursor pointing to an individual statement. See `help(Cursor)` for more details.
    """

    def before(self) -> GapCursor:
        """
        Get a cursor pointing to the gap immediately before this statement.

        Gaps are anchored to the statement they were created from. This
        means that if you move the statement, the gap will move with it
        when the cursor is forwarded.
        """
        assert isinstance(self._impl, C.Node)
        return lift_cursor(self._impl.before(), self._proc)

    def after(self) -> GapCursor:
        """
        Get a cursor pointing to the gap immediately after this statement.

        Gaps are anchored to the statement they were created from. This
        means that if you move the statement, the gap will move with it
        when the cursor is forwarded.
        """
        assert isinstance(self._impl, C.Node)
        return lift_cursor(self._impl.after(), self._proc)

    def prev(self, dist=1):
        """
        Return a statement cursor to the previous statement in the
        block (or dist-many previous)

        Returns InvalidCursor() if there is no such cursor to point to.
        """
        assert isinstance(self._impl, C.Node)

        try:
            return lift_cursor(self._impl.prev(dist), self._proc)
        except InvalidCursorError:
            return InvalidCursor()

    def next(self, dist=1):
        """
        Return a statement cursor to the next statement in the
        block (or dist-many next)

        Returns InvalidCursor() if there is no such cursor to point to.
        """
        assert isinstance(self._impl, C.Node)

        try:
            return lift_cursor(self._impl.next(dist), self._proc)
        except InvalidCursorError:
            return InvalidCursor()

    def as_block(self):
        """Return a Block containing only this one statement"""
        assert isinstance(self._impl, C.Node)
        return BlockCursor(self._impl.as_block(), self._proc)

    def expand(self, delta_lo=None, delta_hi=None):
        """Shorthand for stmt_cursor.as_block().expand(...)"""
        return self.as_block().expand(delta_lo, delta_hi)


class BlockCursor(StmtCursorPrototype, ListCursorPrototype):
    """
    Cursor pointing to a contiguous sequence of statements.
    See `help(Cursor)` for more details.
    """

    def as_block(self):
        """Return this Block; included for symmetry with StmtCursor"""
        return self

    def expand(self, delta_lo=None, delta_hi=None):
        """
        Expand the block cursor.

        When delta_lo (delta_hi) is not None, it is interpreted as a
        number of statements to add to the lower (upper) bound of the
        block.  When delta_lo (delta_hi) is None, the corresponding
        bound is expanded as far as possible.

        Both arguments must be non-negative if they are defined.
        """
        if delta_lo is not None and delta_lo < 0:
            raise ValueError("delta_lo must be non-negative")

        if delta_hi is not None and delta_hi < 0:
            raise ValueError("delta_hi must be non-negative")

        assert isinstance(self._impl, C.Block)
        return BlockCursor(self._impl.expand(delta_lo, delta_hi), self._proc)

    def anchor(self) -> StmtCursor:
        """
        Get a cursor pointing to the node to which this gap is anchored.
        """
        assert isinstance(self._impl, C.Block)
        return lift_cursor(self._impl.parent(), self._proc)

    def before(self) -> GapCursor:
        """
        Get a cursor pointing to the gap before the first statement in
        this block.

        Gaps are anchored to the statement they were created from. This
        means that if you move the statement, the gap will move with it
        when the cursor is forwarded.
        """
        assert isinstance(self._impl, C.Block)
        return lift_cursor(self._impl.before(), self._proc)

    def after(self) -> GapCursor:
        """
        Get a cursor pointing to the gap after the last statement in
        this block.

        Gaps are anchored to the statement they were created from. This
        means that if you move the statement, the gap will move with it
        when the cursor is forwarded.
        """
        assert isinstance(self._impl, C.Block)
        return lift_cursor(self._impl.after(), self._proc)


class GapCursor(StmtCursorPrototype):
    """
    Cursor pointing to a gap before, after, or between statements.
    See `help(Cursor)` for more details.
    """

    def anchor(self) -> StmtCursor:
        """
        Get a cursor pointing to the node to which this gap is anchored.
        """
        assert isinstance(self._impl, C.Gap)
        return lift_cursor(self._impl.anchor(), self._proc)

    def type(self) -> C.GapType:
        """
        Get the type of this gap.
        """
        assert isinstance(self._impl, C.Gap)
        return self._impl.type()


class ExprCursorPrototype(Cursor):
    """
    Prototype for all cursors used to point to statements:
        StmtCursor, BlockCursor, GapCursor
    See `help(Cursor)` for more details.
    """


class ExprCursor(ExprCursorPrototype):
    """
    Cursor pointing to an individual statement or expression.
    See `help(Cursor)` for more details.
    """

    def type(self) -> API.ExoType:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.expr)
        return loopir_type_to_exotype(self._impl._node.type.basetype())


class ExprListCursor(ListCursorPrototype):
    """
    Cursor pointing to a contiguous sequence of expressions.
    See `help(Cursor)` for more details.
    """


class ArgListCursor(ListCursorPrototype):
    """
    Cursor pointing to a contiguous sequence of function arguments.
    See `help(Cursor)` for more details.
    """


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Specific Statement Cursor Types


class AssignCursor(StmtCursor):
    """
    Cursor pointing to an assignment statement:
        `name [ idx ] = rhs`
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Assign)
        return self._impl._node.name.name()

    def idx(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Assign)
        return ExprListCursor(self._impl._child_block("idx"), self._proc)

    def rhs(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Assign)
        return self._child_node("rhs")

    def type(self) -> API.ExoType:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Assign)
        return loopir_type_to_exotype(self._impl._node.type.basetype())


class ReduceCursor(StmtCursor):
    """
    Cursor pointing to a reduction statement:
        `name [ idx ] += rhs`
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Reduce)
        return self._impl._node.name.name()

    def idx(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Reduce)
        return ExprListCursor(self._impl._child_block("idx"), self._proc)

    def rhs(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Reduce)
        return self._child_node("rhs")


class AssignConfigCursor(StmtCursor):
    """
    Cursor pointing to a configuration assignment statement:
        `config.field = rhs`
    """

    def config(self) -> Config:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WriteConfig)
        return self._impl._node.config

    def field(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WriteConfig)
        return self._impl._node.field

    def rhs(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WriteConfig)
        return self._child_node("rhs")


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
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.If)

        return self._child_node("cond")

    def body(self) -> BlockCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.If)

        return BlockCursor(self._impl._child_block("body"), self._proc)

    def orelse(self) -> Cursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.If)

        orelse = self._impl._child_block("orelse")
        return BlockCursor(orelse, self._proc) if len(orelse) > 0 else InvalidCursor()


class ForSeqCursor(StmtCursor):
    """
    Cursor pointing to a loop statement:
        ```
        for name in seq(0,hi):
            body
        ```
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Seq)

        return self._impl._node.iter.name()

    def lo(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Seq)

        return self._child_node("lo")

    def hi(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Seq)

        return self._child_node("hi")

    def body(self) -> BlockCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Seq)

        return BlockCursor(self._impl._child_block("body"), self._proc)


def loopir_type_to_exotype(typ: LoopIR.Type) -> API.ExoType:
    if isinstance(typ, LoopIR.Num):
        return API.ExoType.R
    elif isinstance(typ, LoopIR.F32):
        return API.ExoType.F32
    elif isinstance(typ, LoopIR.F64):
        return API.ExoType.F64
    elif isinstance(typ, LoopIR.INT8):
        return API.ExoType.I8
    elif isinstance(typ, LoopIR.INT32):
        return API.ExoType.I32
    elif isinstance(typ, LoopIR.Bool):
        return API.ExoType.Bool
    elif isinstance(typ, LoopIR.Size):
        return API.ExoType.Size
    elif isinstance(typ, LoopIR.Index):
        return API.ExoType.Index
    else:
        raise NotImplementedError(f"Unsupported {typ}")


class AllocCursor(StmtCursor):
    """
    Cursor pointing to a buffer definition statement:
        ```
        name : type @ mem
        ```
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Alloc)

        return self._impl._node.name.name()

    def mem(self) -> Optional[Memory]:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Alloc)

        return self._impl._node.mem

    def shape(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Alloc)
        assert isinstance(self._impl._node.type, LoopIR.Tensor)

        return ExprListCursor(
            self._impl._child_node("type")._child_block("hi"), self._proc
        )

    def type(self) -> API.ExoType:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Alloc)

        return loopir_type_to_exotype(self._impl._node.type.basetype())


class CallCursor(StmtCursor):
    """
    Cursor pointing to a sub-procedure call statement:
        ```
        subproc( args )
        ```
    """

    def subproc(self):
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Call)

        return API.Procedure(self._impl._node.f)

    def args(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Call)

        return ExprListCursor(self._impl._child_block("args"), self._proc)


class WindowStmtCursor(StmtCursor):
    """
    Cursor pointing to a window declaration statement:
        ```
        name = winexpr
        ```
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WindowStmt)

        return self._impl._node.name.name()

    def winexpr(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WindowStmt)

        return WindowExprCursor(self._impl._child_node("rhs"), self._proc)


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

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Read)

        return self._impl._node.name.name()

    def idx(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Read)

        return ExprListCursor(self._impl._child_block("idx"), self._proc)


class ReadConfigCursor(ExprCursor):
    """
    Cursor pointing to a Config read expression:
        `config.field`
    """

    def config(self) -> Config:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.ReadConfig)

        return self._impl._node.config

    def field(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.ReadConfig)

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
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.Const)

        n = self._impl._node
        assert (
            (n.type.is_bool() and type(n.val) == bool)
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
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.USub)

        return self._child_node("arg")


class BinaryOpCursor(ExprCursor):
    """
    Cursor pointing to an in-fix binary operation expression:
        `lhs op rhs`
    where `op` is one of:
        + - * / % < > <= >= == and or
    """

    def op(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.BinOp)

        return self._impl._node.op

    def lhs(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.BinOp)

        return self._child_node("lhs")

    def rhs(self) -> ExprCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.BinOp)

        return self._child_node("rhs")


class BuiltInFunctionCursor(ExprCursor):
    """
    Cursor pointing to the call to some built-in function
        `name ( args )`
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.BuiltIn)

        return self._impl._node.f.name()

    def args(self) -> ExprListCursor:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.BuiltIn)

        return ExprListCursor(self._impl._child_block("args"), self._proc)


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
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WindowExpr)

        return self._impl._node.name.name()

    def idx(self) -> List:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.WindowExpr)

        def convert_w(w):
            if isinstance(w._node, LoopIR.Interval):
                return (
                    lift_cursor(w._child_node("lo"), self._proc),
                    lift_cursor(w._child_node("hi"), self._proc),
                )
            else:
                return lift_cursor(w._child_node("pt"), self._proc)

        return [convert_w(w) for w in self._impl._child_block("idx")]


class StrideExprCursor(ExprCursor):
    """
    Cursor pointing to a stride expression:
        `stride ( name , dim )`
    (note that stride is a keyword, and not data/a sub-expression)
    `name` is the name of some buffer or window
    """

    def name(self) -> str:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.StrideExpr)

        return self._impl._node.name.name()

    def dim(self) -> int:
        assert isinstance(self._impl, C.Node)
        assert isinstance(self._impl._node, LoopIR.StrideExpr)

        return self._impl._node.dim


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# High-level cursor navigation functions


class CursorNavigationError(Exception):
    pass


def validate_cursors(c1, c2):
    """
    Checks that [c1] and [c2] are both valid cursors, and that they originate from the
    same proc.
    """
    assert not isinstance(c1, InvalidCursor), "first cursor was an InvalidCursor"
    assert not isinstance(c2, InvalidCursor), "second cursor was an InvalidCursor"
    assert c1.proc() == c2.proc(), "cursors originate from different procs"


def match_level(cursor, cursor_to_match):
    """
    Lifts [cursor] through the AST until [cursor] and [cursor_to_match] are at
    the same level, e.g. have the same parent. Returns an InvalidCursorError if
    [cursor_to_match]'s parent is not an ancestor of [cursor].
    """
    validate_cursors(cursor, cursor_to_match)

    while cursor.parent() != cursor_to_match.parent():
        cursor = cursor.parent()
        if isinstance(cursor, InvalidCursor):
            raise CursorNavigationError(
                "cursor_to_match's parent is not an ancestor of cursor"
            )

    return cursor


def get_stmt_within_scope(cursor, scope):
    """
    Gets the statement containing [cursor] that is directly in the provided [scope]
    """
    validate_cursors(cursor, scope)
    assert isinstance(
        scope, (ForSeqCursor, IfCursor)
    ), "scope was not an for loop or if statement"

    try:
        return match_level(cursor, scope.body()[0])
    except CursorNavigationError:
        raise CursorNavigationError("scope is not an ancestor of cursor")


def get_enclosing_loop(cursor, loop_iter=None):
    """
    Gets the enclosing loop with the given [loop_iter]. If [loop_iter] is None,
    returns the innermost loop enclosing [cursor].
    """
    assert not isinstance(cursor, InvalidCursor), "first cursor was an InvalidCursor"

    match_iter = (
        lambda x: x.name() == loop_iter if loop_iter is not None else lambda x: True
    )

    while not (isinstance(cursor, ForSeqCursor) and match_iter(cursor)):
        cursor = cursor.parent()
        if isinstance(cursor, InvalidCursor):
            raise CursorNavigationError("no enclosing loop found")

    return cursor


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Internal Functions; Not for Exposure to Users

# helper function to dispatch to constructors
def lift_cursor(impl, proc):
    assert isinstance(impl, C.Cursor)
    assert isinstance(proc, API.Procedure)

    # dispatch to the correct constructor...
    if isinstance(impl, C.Gap):
        return GapCursor(impl, proc)

    elif isinstance(impl, C.Block):
        # TODO: Rename internal Cursor type to Sequence?
        assert len(impl) > 0
        n0 = impl[0]._node
        if isinstance(n0, LoopIR.stmt):
            assert all(isinstance(c._node, LoopIR.stmt) for c in impl)
            return BlockCursor(impl, proc)
        elif isinstance(n0, LoopIR.expr):
            assert all(isinstance(c._node, LoopIR.expr) for c in impl)
            return ExprListCursor(impl, proc)
        elif isinstance(n0, LoopIR.fnarg):
            assert all(isinstance(c._node, LoopIR.fnarg) for c in impl)
            return ArgListCursor(impl, proc)
        else:
            assert False, "bad case"

    elif isinstance(impl, C.Node):
        n = impl._node

        # procedure arguments
        if isinstance(n, LoopIR.fnarg):
            return ArgCursor(impl, proc)

        # statements
        elif isinstance(n, LoopIR.Assign):
            return AssignCursor(impl, proc)
        elif isinstance(n, LoopIR.Reduce):
            return ReduceCursor(impl, proc)
        elif isinstance(n, LoopIR.WriteConfig):
            return AssignConfigCursor(impl, proc)
        elif isinstance(n, LoopIR.Pass):
            return PassCursor(impl, proc)
        elif isinstance(n, LoopIR.If):
            return IfCursor(impl, proc)
        elif isinstance(n, LoopIR.Seq):
            return ForSeqCursor(impl, proc)
        elif isinstance(n, LoopIR.Alloc):
            return AllocCursor(impl, proc)
        elif isinstance(n, LoopIR.Call):
            return CallCursor(impl, proc)
        elif isinstance(n, LoopIR.WindowStmt):
            return WindowStmtCursor(impl, proc)

        # expressions
        elif isinstance(n, LoopIR.Read):
            return ReadCursor(impl, proc)
        elif isinstance(n, LoopIR.ReadConfig):
            return ReadConfigCursor(impl, proc)
        elif isinstance(n, LoopIR.Const):
            return LiteralCursor(impl, proc)
        elif isinstance(n, LoopIR.USub):
            return UnaryMinusCursor(impl, proc)
        elif isinstance(n, LoopIR.BinOp):
            return BinaryOpCursor(impl, proc)
        elif isinstance(n, LoopIR.BuiltIn):
            return BuiltInFunctionCursor(impl, proc)
        elif isinstance(n, LoopIR.WindowExpr):
            return WindowExprCursor(impl, proc)
        elif isinstance(n, LoopIR.StrideExpr):
            return StrideExprCursor(impl, proc)

        else:
            assert False, f"bad case: {type(n)}"

    else:
        assert False, f"bad case: {type(impl)}"


# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# List of objects to expose

__all__ = [
    "Cursor",
    "InvalidCursor",
    "ArgCursor",
    "StmtCursorPrototype",
    "StmtCursor",
    "BlockCursor",
    "GapCursor",
    "ExprCursorPrototype",
    "ExprCursor",
    "ExprListCursor",
    #
    "AssignCursor",
    "ReduceCursor",
    "AssignConfigCursor",
    "PassCursor",
    "IfCursor",
    "ForSeqCursor",
    "AllocCursor",
    "CallCursor",
    "WindowStmtCursor",
    #
    "ReadCursor",
    "ReadConfigCursor",
    "LiteralCursor",
    "UnaryMinusCursor",
    "BinaryOpCursor",
    "BuiltInFunctionCursor",
    "WindowExprCursor",
    "StrideExprCursor",
    #
    "InvalidCursorError",
    #
    "CursorNavigationError",
    "match_level",
    "get_stmt_within_scope",
    "get_enclosing_loop",
]
