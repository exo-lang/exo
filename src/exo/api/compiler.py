from __future__ import annotations

from pathlib import Path

import exo.api as API
from exo.LoopIR_compiler import run_compile

__all__ = ["compile_procs", "compile_procs_to_strings"]


def compile_procs(proc_list, basedir: Path, c_file: str, h_file: str):
    c_data, h_data = compile_procs_to_strings(proc_list, h_file)
    (basedir / c_file).write_text(c_data)
    (basedir / h_file).write_text(h_data)


def compile_procs_to_strings(proc_list, h_file_name: str):
    assert isinstance(proc_list, list)
    assert all(isinstance(p, API.Procedure) for p in proc_list)
    return run_compile([p._loopir_proc for p in proc_list], h_file_name)
