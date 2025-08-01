import contextlib
import sys
from pathlib import Path
from unittest import mock

import pytest

from exo.main import exocc

TEST_ROOT = Path(__file__).parent.resolve()


def test_build_only_foo(tmp_path):
    with mock.patch.dict(sys.modules):
        exocc("-o", str(tmp_path / "foo"), str(TEST_ROOT / "exocc_test" / "foo.py"))
        with open(tmp_path / "foo" / "foo.h") as header:
            contents = header.read()
            assert "foo" in contents and "bar" not in contents


def test_build_only_bar(tmp_path):
    with mock.patch.dict(sys.modules):
        exocc("-o", str(tmp_path / "bar"), str(TEST_ROOT / "exocc_test" / "bar.py"))
        with open(tmp_path / "bar" / "bar.h") as header:
            contents = header.read()
            assert "bar" in contents and "foo" not in contents


def test_build_both_explicit(tmp_path):
    with mock.patch.dict(sys.modules):
        exocc(
            "-o",
            str(tmp_path / "exocc_test"),
            "--stem",
            "exocc_test",
            "-p",
            str(TEST_ROOT / "exocc_test"),
            str(TEST_ROOT / "exocc_test" / "foo.py"),
            str(TEST_ROOT / "exocc_test" / "bar.py"),
        )
        with open(tmp_path / "exocc_test" / "exocc_test.h") as header:
            contents = header.read()
            assert "foo" in contents and "bar" in contents


def test_build_both_chdir(tmp_path):
    with (
        contextlib.chdir(TEST_ROOT / "exocc_test"),
        mock.patch.dict(sys.modules),
    ):
        exocc(
            "-o",
            str(tmp_path / "exocc_test"),
            "--stem",
            "exocc_test",
            str(TEST_ROOT / "exocc_test" / "foo.py"),
            str(TEST_ROOT / "exocc_test" / "bar.py"),
        )
        with open(tmp_path / "exocc_test" / "exocc_test.h") as header:
            contents = header.read()
            assert "foo" in contents and "bar" in contents


def test_build_both_needs_pythonpath(tmp_path):
    with (
        pytest.raises(ImportError, match="No module named 'common'"),
        mock.patch.dict(sys.modules),
    ):
        exocc(
            "-o",
            str(tmp_path / "exocc_test"),
            "--stem",
            "exocc_test",
            str(TEST_ROOT / "exocc_test" / "foo.py"),
            str(TEST_ROOT / "exocc_test" / "bar.py"),
        )
