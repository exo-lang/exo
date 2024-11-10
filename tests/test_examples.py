from __future__ import annotations

from pathlib import Path

import pytest

import exo
import exo.main

REPO_ROOT = Path(__file__).parent.parent.resolve()


def _test_app(module_file: Path):
    module_file = module_file.resolve(strict=True)
    mod = exo.main.load_user_code(module_file)
    procs = exo.main.get_procs_from_module(mod)

    c_file, h_file = exo.compile_procs_to_strings(procs, "test_case.h")

    return f"{h_file}\n{c_file}"


# ---------------------------------------------------------------------------- #


def test_avx2_matmul(golden):
    module_file = REPO_ROOT / "examples" / "avx2_matmul" / "x86_matmul.py"
    assert _test_app(module_file) == golden


def test_cursors(golden):
    module_file = REPO_ROOT / "examples" / "cursors" / "cursors.py"
    assert _test_app(module_file) == golden


def test_rvm_conv1d(golden):
    module_file = REPO_ROOT / "examples" / "rvm_conv1d" / "exo" / "conv1d.py"
    assert _test_app(module_file) == golden


def test_quiz1(golden):
    module_file = REPO_ROOT / "examples" / "quiz1" / "quiz1.py"
    assert _test_app(module_file) == golden


def test_quiz3(golden):
    module_file = REPO_ROOT / "examples" / "quiz3" / "quiz3.py"
    assert _test_app(module_file) == golden
