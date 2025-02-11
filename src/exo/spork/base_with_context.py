from typing import Dict


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


class ExtWithContext(BaseWithContext):
    """Special with context used for code generation of accelerator device code in different files

    The compiled body of the with statement is redirected to a file
    with body_ext as its file extension (the period is not included).
    e.g. body_ext="cu" would redirect to a .cu CUDA file.

    """

    __slots__ = ["launch", "body_prefix", "body_suffix", "body_ext", "ext_snippets"]

    # Launch syntax for the "caller" of the accelerator code.
    # e.g. for CUDA, may look like "foobar<<<...>>>"
    launch: str

    # The compiled body code is wrapped by the prefix and suffix
    # and inserted to the file with the given body_ext file extension.
    body_prefix: str
    body_suffix: str
    body_ext: str

    # For each (K,V) pair, insert code V into the file with extension K.
    # The LoopIR compiler shall do this before the main body code is inserted.
    ext_snippets: Dict[str, str]

    def __init__(self, launch, body_prefix, body_suffix, body_ext, ext_snippets):
        self.launch = launch
        self.body_prefix = body_prefix
        self.body_suffix = body_suffix
        self.body_ext = body_ext
        self.ext_snippets = ext_snippets

    def __repr__(self):
        return f"ExtWithContext({self.launch!r}, {self.body_prefix!r}, {self.body_suffix!r}, {self.body_ext!r}, {self.ext_snippets!r})"
