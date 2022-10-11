from __future__ import annotations

from pathlib import Path

import exo
import exo.main

REPO_ROOT = Path(__file__).parent.parent.resolve()


def _test_app(module_file):
    mod = exo.main.load_user_code(module_file)
    procs = exo.main.get_procs_from_module(mod)

    c_file, h_file = exo.compile_procs_to_strings(procs, 'test_case.h')

    return f'{h_file}\n{c_file}'


def test_x86_sgemm(golden):
    module_file = REPO_ROOT / 'apps' / 'x86_demo' / 'sgemm' / 'sgemm.py'
    assert _test_app(module_file.resolve(strict=True)) == golden


def test_x86_conv(golden):
    module_file = REPO_ROOT / 'apps' / 'x86_demo' / 'conv' / 'conv.py'
    assert _test_app(module_file.resolve(strict=True)) == golden


def test_neon_sgemm(golden):
    module_file = REPO_ROOT / 'apps' / 'neon_dev' / 'sgemm' / 'sgemm.py'
    assert _test_app(module_file.resolve(strict=True)) == golden
