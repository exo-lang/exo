import ctypes
import os
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pytest
from _pytest.config import argparsing


def pytest_addoption(parser: argparsing.Parser):
    parser.addoption(
        "--update-golden",
        action="store_true",
        default=False,
        help="Update golden outputs.",
    )


def _nodeid_to_path(nodeid: str) -> Path:
    nodeid = nodeid.replace('::', '/')
    nodeid = nodeid.replace('.py/', '/')
    return Path(nodeid)


@dataclass
class GoldenOutput:
    path: Path
    text: Optional[str]
    update: bool

    def _compare(self, other):
        if isinstance(other, GoldenOutput):
            return self.path == other.path and self.text == other.text
        elif isinstance(other, str):
            return self.text == other
        else:
            return False

    def __str__(self):
        if isinstance(self.text, str):
            return self.text
        raise ValueError(f'No golden output for {self.path}')

    def __eq__(self, other):
        result = self._compare(other)
        if not result and self.update:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(str(other))
        return result or self.update


@pytest.fixture
def golden(request):
    """
    A fixture to load the golden output for the requesting test. Should be
    checked against some actual output in at least one assertion.
    """

    basedir = Path(request.config.invocation_dir)
    testpath = Path(request.fspath)

    p = (Path('golden') /
         testpath.relative_to(basedir).with_suffix('') /
         request.node.name).with_suffix('.txt')

    update = request.config.getoption("--update-golden")
    if p.exists():
        yield GoldenOutput(p, p.read_text(), update)
    else:
        yield GoldenOutput(p, None, update)


@dataclass
class ProcWrapper:
    fn_ptr: Any  # CDLL's internal _FuncPtr

    @staticmethod
    def _convert(arg):
        if isinstance(arg, np.ndarray):
            return arg.ctypes.get_as_parameter()
        if isinstance(arg, int):
            return arg
        if isinstance(arg, type(None)):
            return ctypes.POINTER(ctypes.c_int)()

        raise ValueError(f'unrecognized type {type(arg)}')

    def __call__(self, *args, **kwargs):
        if kwargs:
            raise ValueError('cannot call DllWrapper with kwargs')
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

    def compile(self, proc, *, compile_only=False, skip_on_fail=False,
                **kwargs):
        atl = self.workdir / f'{proc.name()}_pretty.atl'
        atl.write_text(str(proc))

        proc.compile_c(self.workdir, self.basename)

        cml = (self.workdir / 'CMakeLists.txt')
        cml.write_text(textwrap.dedent(
            f'''
            cmake_minimum_required(VERSION 3.22)
            project({self.basename} LANGUAGES C)
            
            add_library({self.basename} SHARED "{self.basename}.c")
            
            file(GENERATE OUTPUT "$<CONFIG>/lib_path.txt"
                 CONTENT "$<TARGET_FILE:{self.basename}>")
            '''
        ))

        cm_build = [
            f'ctest',
            f'-C',
            f'Release',
            f'--build-and-test',
            f'{self.workdir}',
            f'{self.workdir / "build"}',
            f'--build-generator',
            f'{os.getenv("CMAKE_GENERATOR", default="Ninja")}',
            f'--build-options',
            *(f'-D{arg}={value}'
              for arg, value in kwargs.items())
        ]

        skip = False

        try:
            subprocess.run(cm_build, check=True)
        except subprocess.CalledProcessError:
            if skip_on_fail:
                skip = True
            else:
                raise

        if skip:
            pytest.skip('Compile failure converted to skip')

        if not compile_only:
            lib_rsp = self.workdir / 'build' / 'Release' / 'lib_path.txt'
            return LibWrapper(
                ctypes.CDLL(lib_rsp.read_text()),
                proc.name()
            )
        else:
            return None


@pytest.fixture
def compiler(tmpdir, request):
    return Compiler(Path(tmpdir), request.node.name)
