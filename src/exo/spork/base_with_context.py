from ..core.prelude import Sym
from typing import Dict, Set


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

    We further affect code lowering of the subtree by customizing:

    * reserved_names: set of string C variable names that must not be used
      to lower the names of Exo variables

    * force_names: generated C names for Sym [if not provided, we use defaults]

    * force_const: set of Syms that must lower to const values in subtree
      [for syms not in set, the compiler's deduction is accepted]

    * scalar_refs: set of Syms that lower to a pointer to a scalar
      `T * {sym C name}` if sym in scalar_refs else `T {sym C name}`

      [we discard the outer compiler's scalar_refs set and replace with this
      completely, even if empty ... the purpose of this feature is to work
      around C's inability to pass scalars by-reference across function
      boundaries, so the set of scalar_refs for the generated accelerator
      function isn't correlated with that of the outer C function]

    NOTE: Except scalar_refs, for nested ExtWithContext, the effects of
    these overrides are combined, with the inner context taking priority.
    """

    __slots__ = [
        "launch",
        "body_prefix",
        "body_suffix",
        "body_ext",
        "ext_snippets",
        "reserved_names",
        "force_names",
        "force_const",
        "scalar_refs",
    ]

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

    def __init__(
        self,
        launch: str,
        body_prefix: str,
        body_suffix: str,
        body_ext: str,
        ext_snippets: Dict[str, str],
        reserved_names: Set[str],
        force_names: Dict[Sym, str],
        force_const: Set[Sym],
        scalar_refs: Set[Sym],
    ):
        self.launch = launch
        self.body_prefix = body_prefix
        self.body_suffix = body_suffix
        self.body_ext = body_ext
        self.ext_snippets = ext_snippets
        self.reserved_names = reserved_names
        self.force_names = force_names
        self.force_const = force_const
        self.scalar_refs = scalar_refs

        assert all(isinstance(s, str) for s in reserved_names)
        assert all(isinstance(k, Sym) for k in force_names)
        assert all(isinstance(v, str) for v in force_names.values())
        assert all(isinstance(s, Sym) for s in force_const)
        assert all(isinstance(s, Sym) for s in scalar_refs)

    def __repr__(self):
        return f"ExtWithContext({self.launch!r}, {self.body_prefix!r}, {self.body_suffix!r}, {self.body_ext!r}, {self.ext_snippets!r}, {self.reserved_names!r}, {self.force_names!r}, {self.force_const!r}, {self.scalar_refs!r})"
