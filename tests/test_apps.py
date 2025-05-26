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


def test_x86_sgemm(golden):
    module_file = REPO_ROOT / "apps" / "x86" / "sgemm" / "sgemm.py"
    assert _test_app(module_file) == golden


def test_x86_conv(golden):
    module_file = REPO_ROOT / "apps" / "x86" / "conv" / "conv.py"
    assert _test_app(module_file) == golden


def test_neon_sgemm(golden):
    module_file = REPO_ROOT / "apps" / "aarch64" / "sgemm" / "sgemm.py"
    assert _test_app(module_file) == golden


@pytest.mark.slow
def test_gemmini_matmul(golden):
    module_file = REPO_ROOT / "apps" / "gemmini" / "src" / "exo" / "matmul.py"
    assert _test_app(module_file) == golden


@pytest.mark.slow
def test_gemmini_conv(golden):
    module_file = REPO_ROOT / "apps" / "gemmini" / "src" / "exo" / "conv.py"
    assert _test_app(module_file) == golden


def test_blur(golden):
    module_file = REPO_ROOT / "apps" / "x86" / "halide" / "blur" / "blur.py"
    assert _test_app(module_file) == golden


@pytest.mark.slow
def test_unsharp(golden):
    module_file = REPO_ROOT / "apps" / "x86" / "halide" / "unsharp" / "unsharp.py"
    assert _test_app(module_file) == golden
