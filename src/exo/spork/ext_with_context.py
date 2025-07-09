from .base_with_context import BaseWithContext
from ..core.prelude import Sym
from dataclasses import dataclass
from typing import Dict, Set
from .lowered_barrier import LoweredBarrier


@dataclass(slots=True)
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

    reserved_names: Set[str]  # TODO rethink this
    force_names: Dict[Sym, str]
    force_const: Set[Sym]
    scalar_refs: Set[Sym]
    lowered_barriers: Dict[Sym, LoweredBarrier]

    def __post_init__(self):
        assert all(isinstance(s, str) for s in self.reserved_names)
        assert all(isinstance(k, Sym) for k in self.force_names)
        assert all(isinstance(v, str) for v in self.force_names.values())
        assert all(isinstance(s, Sym) for s in self.force_const)
        assert all(isinstance(s, Sym) for s in self.scalar_refs)
        for k, v in self.lowered_barriers.items():
            assert isinstance(k, Sym) and isinstance(v, LoweredBarrier)
