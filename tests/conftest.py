import ctypes
import distutils.spawn
import functools
import os
import platform
import re
import shlex
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any, Dict, Union, List, Set

import numpy as np
import pytest
from _pytest.config import argparsing, Config
from _pytest.nodes import Node

from exo import Procedure, compile_procs


# ---------------------------------------------------------------------------- #
# Pytest hooks                                                                 #
# ---------------------------------------------------------------------------- #


def pytest_addoption(parser: argparsing.Parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden outputs.",
    )


def pytest_configure(config: Config):
    config.addinivalue_line(
        "markers", "isa(name): mark test to run only when required ISA is available"
    )


def pytest_runtest_setup(item: Node):
    for mark in item.iter_markers(name="isa"):
        isa = mark.args[0].lower()
        if isa not in get_cpu_features():
            pytest.skip(f"skipping test because {isa} is not available")


# ---------------------------------------------------------------------------- #
# Pytest fixtures                                                              #
# ---------------------------------------------------------------------------- #


@pytest.fixture
def golden(request):
    """
    A fixture to load the golden output for the requesting test. Should be
    checked against some actual output in at least one assertion.
    """

    basedir = request.config.rootpath / "tests"
    testpath = Path(request.fspath)

    p = (
        basedir
        / "golden"
        / (
            testpath.relative_to(basedir).with_suffix("") / request.node.name
        ).with_suffix(".txt")
    )

    text = p.read_text() if p.exists() else None
    yield GoldenOutput(p, text, request.config)


@pytest.fixture
def compiler(tmp_path, request):
    return Compiler(tmp_path, request.node.name)


@pytest.fixture
def sde64():
    sde = distutils.spawn.find_executable(
        "sde64", os.getenv("SDE_PATH")
    ) or distutils.spawn.find_executable("sde64")
    if not sde:
        pytest.skip("could not find SDE")

    def run(cmd, **kwargs):
        if not isinstance(cmd, list):
            cmd = shlex.split(str(cmd))
        return subprocess.run([sde, "-future", "--", *cmd], check=True, **kwargs)

    return run


# ---------------------------------------------------------------------------- #
# Implementation classes                                                       #
# ---------------------------------------------------------------------------- #


class GoldenOutput(str):
    _missing = "\0"

    def __new__(cls, path, text, config):
        return str.__new__(cls, cls._missing if text is None else text)

    def __init__(self, path, _, config):
        self.path = path
        self.update = config.getoption("--update-golden")
        self.verbose = config.getoption("verbose")

    def __eq__(self, actual):
        if isinstance(actual, GoldenOutput) and self.path != actual.path:
            return False

        if super().__eq__(self._missing) and not self.update:
            # Hides this stack frame in the PyTest traceback.
            __tracebackhide__ = True

            message = f"golden output missing: {self.path}.\n"

            if self.verbose:
                message += (
                    f"Actual output:\n"
                    f"{actual}\n"
                    f"Did you forget to run with --update-golden?"
                )
            else:
                message += "Run with -v (verbose) to see actual output."

            pytest.fail(message)

        equal = super().__eq__(actual)
        if not equal and self.update:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(actual))
        return equal or self.update


@dataclass
class ProcWrapper:
    fn_ptr: Any  # CDLL's internal _FuncPtr

    @staticmethod
    def _convert(arg):
        if arg is None:
            return ctypes.POINTER(ctypes.c_int)()
        if isinstance(arg, np.ndarray):
            return arg.ctypes.data_as(ctypes.c_void_p)
        if isinstance(arg, int):
            return arg

        raise ValueError(f"unrecognized type {type(arg)}")

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ValueError("cannot call with kwargs")
        args = [self._convert(arg) for arg in args]
        return self.fn_ptr(*args)


@dataclass
class LibWrapper:
    dll: ctypes.CDLL
    default_proc: str

    def __getattr__(self, item):
        return ProcWrapper(getattr(self.dll, item))

    def __call__(self, *args, **kwargs):
        fn_ptr = getattr(self.dll, self.default_proc)
        return ProcWrapper(fn_ptr)(*args, **kwargs)


@dataclass
class Compiler:
    workdir: Path
    basename: str

    def compile(
        self,
        procs: Union[Procedure, List[Procedure]],
        *,
        test_files: Optional[Dict[str, str]] = None,
        include_dir=None,
        additional_file=None,
        compile_only: bool = False,
        skip_on_fail: bool = False,
        **kwargs,
    ):
        test_files = test_files or {}
        if isinstance(procs, Procedure):
            procs = [procs]

        compile_procs(procs, self.workdir, f"{self.basename}.c", f"{self.basename}.h")

        atl = self.workdir / f"{self.basename}_pretty.atl"
        atl.write_text("\n".join(map(str, procs)))

        (self.workdir / "CMakeLists.txt").write_text(
            self._generate_cml(test_files, include_dir, additional_file)
        )

        self._run_command(
            [
                f"ctest",
                f"-C",
                f"Release",
                f"--build-and-test",
                f"{self.workdir}",
                f'{self.workdir / "build"}',
                f"--build-generator",
                f'{os.getenv("CMAKE_GENERATOR", default="Ninja")}',
                f"--build-options",
                *(f"-D{arg}={value}" for arg, value in kwargs.items()),
            ],
            skip_on_fail,
        )

        artifact_path = Path(
            (self.workdir / "build" / "Release" / "artifact_path.txt").read_text()
        ).resolve(strict=True)

        if compile_only or test_files:
            return artifact_path

        return LibWrapper(ctypes.CDLL(artifact_path), procs[0].name())

    @staticmethod
    def _run_command(build_command, skip_on_fail):
        skip = False
        try:
            subprocess.run(build_command, check=True)
        except subprocess.CalledProcessError:
            if skip_on_fail:
                skip = True
            else:
                raise
        if skip:
            pytest.skip("Compile failure converted to skip")

    def _generate_cml(
        self, test_files: Dict[str, str], include_dir=None, additional_file=None
    ):
        additional_file = f'"{additional_file}"' if additional_file else ""

        cml_body = textwrap.dedent(
            f"""
            cmake_minimum_required(VERSION 3.21)
            project({self.basename} LANGUAGES C)
            
            option(BUILD_SHARED_LIBS "Build shared libraries by default" ON)
            
            add_library({self.basename} "{self.basename}.c" {additional_file})
            add_library({self.basename}::{self.basename} ALIAS {self.basename})
            
            """
        )

        if include_dir:
            cml_body += textwrap.dedent(
                f"""
                target_include_directories({self.basename} PRIVATE {include_dir})
                """
            )

        lib_name = f"{self.basename}::{self.basename}"
        if not test_files:
            artifact = lib_name
        else:
            artifact = "main::main"

            for filename, contents in test_files.items():
                (self.workdir / filename).write_text(contents)

            cml_body += textwrap.dedent(
                f"""
                add_executable(main {' '.join(test_files.keys())})
                add_executable({artifact} ALIAS main)
                target_link_libraries(main PRIVATE {lib_name})
                
                """
            )

        cml_body += textwrap.dedent(
            f"""
            file(GENERATE OUTPUT "$<CONFIG>/artifact_path.txt"
                 CONTENT "$<TARGET_FILE:{artifact}>")
            """
        )
        return cml_body


@functools.cache
def get_cpu_features() -> Set[str]:
    def get_cpuinfo_string() -> str:
        if cpuinfo := os.getenv("EXO_OVERRIDE_CPUINFO"):
            return cpuinfo

        if platform.system() == "Linux":
            try:
                cpuinfo = Path("/proc/cpuinfo").read_text()
                if m := re.search(r"^flags\s*:(.+)$", cpuinfo, re.MULTILINE):
                    return m.group(1)
            except IOError:
                return ""
        elif platform.system() == "Darwin":
            x86_features = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True,
            ).stdout.decode()

            arm_features = subprocess.run(
                ["sysctl", "hw.optional"], capture_output=True
            ).stdout.decode()
            arm_features = re.findall(
                r"^hw\.optional\.([^:]+): [^0]\d*$", arm_features, re.MULTILINE
            )

            return x86_features + " " + " ".join(arm_features)
        elif platform.system() == "Windows":
            return ""  # TODO: implement checking for Windows
        else:
            return ""

    return set(get_cpuinfo_string().lower().split())
