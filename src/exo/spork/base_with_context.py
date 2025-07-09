class BaseWithContext(object):
    """
    Base type for all X that can appear in Exo object code of the form with X:

    BaseWithContext() can also be used as a hole in pattern matching.
    This is a "temporary" hack until with statement handling in LoopIR
    is fixed (i.e. not smuggling them as if statements).
    """

    __slots__ = []

    def __str__(self):
        if type(self) is BaseWithContext:
            return "_"
        else:
            return repr(self)


def is_if_holding_with(node, AST):
    """
    Check if the AST node is a with statement disguised as an if

    statement with a constant "condition" holding the BaseWithContext.
    This is how we're handling the IR for with (until we fix it).
    Such "if statements" must have an empty orelse.
    """
    if isinstance(node, AST.If):
        cond = node.cond
        if isinstance(cond, AST.Const):
            if isinstance(cond.val, BaseWithContext):
                assert not node.orelse
                return True
    return False
