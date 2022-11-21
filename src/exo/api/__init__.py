from __future__ import annotations

from .procedure import Procedure
from .decorators import proc, instr, config
from .compiler import compile_procs, compile_procs_to_strings

__all__ = [
    "Procedure",
    "compile_procs",
    "compile_procs_to_strings",
    "proc",
    "instr",
    "config",
]
