from pathlib import Path

from exo.main import main as exocc

TEST_ROOT = Path(__file__).parent.resolve()


def test_build_only_foo(tmp_path):
    exocc("-o", str(tmp_path / "foo"), str(TEST_ROOT / "exocc_test" / "foo.py"))
    with open(tmp_path / "foo" / "foo.h") as header:
        contents = header.read()
        assert "foo" in contents and "bar" not in contents


def test_build_only_bar(tmp_path):
    exocc("-o", str(tmp_path / "bar"), str(TEST_ROOT / "exocc_test" / "bar.py"))
    with open(tmp_path / "bar" / "bar.h") as header:
        contents = header.read()
        assert "bar" in contents and "foo" not in contents


def test_build_both(tmp_path):
    exocc("-o", str(tmp_path / "exocc_test"), str(TEST_ROOT / "exocc_test"))
    with open(tmp_path / "exocc_test" / "exocc_test.h") as header:
        contents = header.read()
        assert "foo" in contents and "bar" in contents
