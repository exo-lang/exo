import ctypes
import distutils.spawn
import functools
import os
import platform
import re
import shlex
import subprocess
import textwrap
import warnings
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path, PurePath
from typing import Optional, Any, Dict, Union, List, Set, Callable

import numpy as np
import pytest
from _pytest.config import argparsing, Config
from _pytest.nodes import Node

from exo import Procedure, compile_procs, ext_compile_procs
from exo.spork import excut


class CudaRun(Enum):
    cpu = auto()
    Sm80 = auto()
    Sm90a = auto()


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
    parser.addoption(
        "--cuda-run-cpu",
        dest="cuda_run",
        action="append_const",
        const=CudaRun.cpu,
        help="Enable compiler test fixture (implied if no --cuda-run-* flags given)",
    )
    parser.addoption(
        "--cuda-run-Sm80",
        dest="cuda_run",
        action="append_const",
        const=CudaRun.Sm80,
        help="Enable compiler_Sm80 test fixture (for tests requiring sm_80 device)",
    )
    parser.addoption(
        "--cuda-run-Sm90a",
        dest="cuda_run",
        action="append_const",
        const=CudaRun.Sm90a,
        help="Enable compiler_Sm90a test fixture (for tests requiring sm_90a device)",
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
    """C and (optional) CUDA compiler test fixture

    Use this test fixture for all tests requiring compiling C, as well as tests
    with CUDA where you do NOT intend to actually execute the CUDA code.

    """

    cuda_run = request.config.getoption("cuda_run")
    if not cuda_run or CudaRun.cpu in cuda_run:
        return Compiler(tmp_path, request.node.name, None)
    else:
        pytest.skip("--cuda-run-cpu not given")


@pytest.fixture
def compiler_Sm80(tmp_path, request):
    """Test fixture for compiling and executing code for sm_80 CUDA devices"""
    cuda_run = request.config.getoption("cuda_run")
    if cuda_run and CudaRun.Sm80 in cuda_run:
        return Compiler(tmp_path, request.node.name, "80")
    else:
        pytest.skip("--cuda-run-Sm80 not given")


@pytest.fixture
def compiler_Sm90a(tmp_path, request):
    """Test fixture for compiling and executing code for sm_90a CUDA devices"""
    cuda_run = request.config.getoption("cuda_run")
    if cuda_run and CudaRun.Sm90a in cuda_run:
        return Compiler(tmp_path, request.node.name, "90a")
    else:
        pytest.skip("--cuda-run-Sm90a not given")


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
        if isinstance(arg, PurePath):
            return ctypes.c_char_p(bytes(str(arg), "utf-8"))

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
    workdir: Path
    basename: Path

    def __getattr__(self, item):
        return ProcWrapper(getattr(self.dll, item))

    def __call__(self, *args, **kwargs):
        fn_ptr = getattr(self.dll, self.default_proc)
        return ProcWrapper(fn_ptr)(*args, **kwargs)

    def read_source_by_ext(self, ext) -> str:
        assert ext in ("c", "h", "cu", "cuh"), "Update this if needed"
        with open(str(self.workdir / self.basename) + "." + ext) as f:
            text = f.read()
            return text


@dataclass(slots=True)
class CudaTestSource:
    h_src: str
    c_src: str
    cu_src: str
    cuh_src: str

    def compare_golden(self, golden: str):
        assert self._make_golden() == golden

    def _make_golden(self):
        return f"""// ########################################################################
// .h
// ########################################################################
{self.h_src}

// ########################################################################
// .c
// ########################################################################
{self.c_src}

// ########################################################################
// .cuh
// ########################################################################
{self.cuh_src}

// ########################################################################
// .cu
// ########################################################################
{self.cu_src}
"""


@dataclass
class Compiler:
    workdir: Path
    basename: str
    cuda_sm: str

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

        file_exts = ext_compile_procs(procs, self.workdir, self.basename)

        atl = self.workdir / f"{self.basename}_pretty.atl"
        atl.write_text("\n".join(map(str, procs)))

        assert file_exts == ["c", "h"]
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

        return LibWrapper(
            ctypes.CDLL(artifact_path), procs[0].name(), self.workdir, self.basename
        )

    def nvcc_compile(
        self,
        procs: Union[Procedure, List[Procedure]],
        *,
        excut=False,
        include_dir=None,
        additional_files=(),
        skip_on_fail: bool = False,
        compiler_flags=(),
        _cpu_test_sm=None,
    ):
        sm = _cpu_test_sm or self.cuda_sm
        if sm is None:
            raise ValueError("Use compiler_Sm80 fixture")
        if isinstance(procs, Procedure):
            procs = [procs]

        file_exts = ext_compile_procs(procs, self.workdir, self.basename)
        assert file_exts == ["c", "cu", "cuh", "excut_str_table", "h"]

        # Directly use nvcc ($EXO_NVCC)
        # This pretty much only works on Linux.
        nvcc = os.getenv("EXO_NVCC", default=None)
        if nvcc is None:
            if _cpu_test_sm is None:
                raise ValueError("Define EXO_NVCC environment variable")
            else:
                pytest.skip("EXO_NVCC environment variable not defined")
        artifact_path = str(self.workdir / (self.basename + ".so"))
        args = [
            nvcc,
            f"-arch=compute_{sm}",
            f"-code=sm_{sm},compute_{sm}",
            "-std=c++20",
            "-lineinfo",
            "-O3",
            str(self.workdir / (self.basename + ".c")),
            str(self.workdir / (self.basename + ".cu")),
            "--compiler-options",
            "'-fPIC'",
            "--shared",
            "-o",
            artifact_path,
            "-lcuda",
        ]
        args.extend(compiler_flags)
        if ccbin := os.getenv("EXO_CCBIN", default=None):
            args.append("-ccbin")
            args.append(ccbin)
        if include_dir is not None:
            args.append("-I")
            args.append(include_dir)
        args.extend(additional_files)
        if excut:
            args.append("-DEXO_EXCUT_bENABLE_LOG=1")
            args.append("-Itests/cuda/excut")
            args.append("tests/cuda/excut/exo_excut.cu")

        self._run_command(args, skip_on_fail)
        return LibWrapper(
            ctypes.CDLL(artifact_path), procs[0].name(), self.workdir, self.basename
        )

    def cuda_cpu_test(
        self,
        mkproc: Callable[[], Union[Procedure, List[Procedure]]],
        golden=None,
        sm=None,
        **kwargs,
    ) -> CudaTestSource:
        """CPU-only CUDA test (no CUDA device required)

        We call mkproc(**kwargs) and compile the resulting proc(s) into C+CUDA code.

        Optionally, we compare with a reference str if the golden arg is provided,
        and we try to compile with nvcc as an sm_{sm}-architecture CUDA module
        if the sm argument is provided.

        NOTE: if sm is not provided, then the generated CUDA code is
        never compiled, but it is still desirable to use the compiler
        test fixture, since this is how
        --cuda-run-Sm80/--cuda-run-Sm90a filter out non-executed tests.
        For now I centralize this filtering to the (mutually
        exclusive) compiler* fixtures to avoid accidents where some
        test is filtered out on all code paths.

        """

        procs = mkproc(**kwargs)
        assert procs is not None, f"Forgot return in {mkproc}?"
        if isinstance(procs, Procedure):
            procs = [procs]

        file_exts = ext_compile_procs(procs, self.workdir, self.basename)
        assert file_exts == ["c", "cu", "cuh", "excut_str_table", "h"]
        read_source = LibWrapper.read_source_by_ext
        sources = CudaTestSource(
            h_src=read_source(self, "h"),
            c_src=read_source(self, "c"),
            cu_src=read_source(self, "cu"),
            cuh_src=read_source(self, "cuh"),
        )
        if golden is not None:
            sources.compare_golden(golden)
        if sm:
            self.nvcc_compile(procs, _cpu_test_sm=sm)
        return sources

    def excut_test(
        self,
        mkproc: Callable[[], Union[Procedure, List[Procedure]]],
        mkref: Callable[[excut.ExcutReferenceGenerator], None],
        *args,
        **kwargs,
    ) -> "CudaTestContext":
        """Helper for simple CUDA excut tests.

        Generate proc(s) with mkproc(**kwargs) and generate excut
        reference actions with mkref(xrg, **kwargs) where you use the
        member functions of xrg: ExcutReferenceGenerator to create the
        reference actions.

        The procs get compiled to a CUDA library and we execute it with
        *args (plus the implicit null ctx arg...) and compare the
        output excut trace with the reference.

        More advanced executed CUDA tests may use cuda_test_context(...) directly.
        Be sure to pass excut=True if you intend to use excut action logging.

        """
        procs = mkproc(**kwargs)
        assert procs is not None, f"Forgot return in {mkproc}?"
        if isinstance(procs, Procedure):
            procs = [procs]
        cu = CudaTestContext(self, procs, excut=True)
        cu(None, *args)
        cu.excut_concordance(mkref, **kwargs)
        return cu

    def cuda_test_context(self, *args, **kwargs) -> "CudaTestContext":
        return CudaTestContext(self, *args, **kwargs)

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


@dataclass(slots=True)
class CudaTestContext(CudaTestSource):
    """Wrapper around a compiled CUDA library.

    There is bunch of extra context state that only is used if excut
    is enabled (excut=True). In this case, the library gets compiled
    to log excut actions (trace) to file.

    IMPORTANT: in the excut=True code path, calls to the library may
    be repeated, so side effects (e.g. +=) may be duplicated.
    Generally, do NOT enable excut if you plan to test the tensor
    outputs for correctness (i.e. CUDA runtime tests should assert
    correctness either by examining the excut outputs, or by examining
    the numpy/etc. outputs, but not both).

    """

    compiler: Compiler
    fn: LibWrapper
    enable_excut: bool
    trace_actions: List[excut.ExcutAction]
    ref_actions: List[excut.ExcutAction]
    xrg: excut.ExcutReferenceGenerator
    deductions: dict

    _saved_excut_buffer_size = 1 << 28

    def __init__(self, compiler, *args, **kwargs):
        # Note in particular excut=bool keyword argument
        # controls whether excut logging is enabled.
        self.compiler = compiler
        self.enable_excut = kwargs.get("excut", False)
        self.fn = compiler.nvcc_compile(*args, **kwargs)
        self.trace_actions = None
        self.c_src = self.fn.read_source_by_ext("c")
        self.cu_src = self.fn.read_source_by_ext("cu")
        self.cuh_src = self.fn.read_source_by_ext("cuh")
        self.h_src = self.fn.read_source_by_ext("h")

    def __call__(self, *args):
        self.run(self.fn.default_proc, *args)

    def run(self, proc_name, *args):
        assert args[0] == None, "Expect ctxt=None for CUDA"
        c_proc = getattr(self.fn, proc_name)
        if self.enable_excut:
            self._run_trace_excut(c_proc, *args)
        else:
            c_proc(*args)

    def excut_concordance(
        self,
        make_reference: Callable[[excut.ExcutReferenceGenerator], None],
        ref_filename="excut_ref.json",
        **kwargs,
    ):
        """Compare previously generated trace with newly generated reference

        make_reference(**kwargs) must create the reference actions using the
        member functions of ExcutReferenceGenerator.
        Either kwargs or functools.partial could be very useful to you.

        var = self.get_var(varname: str) gets an existing variable

        var2 = var[idx] and var2 = var + offset adds indices/offsets

        var2(self) gets the deduced integer value.

        """
        assert self.enable_excut
        trace_path = self.compiler.workdir / "excut_trace.json"
        ref_path = self.compiler.workdir / ref_filename
        xrg = excut.ExcutReferenceGenerator()
        make_reference(xrg, **kwargs)
        with open(str(ref_path), "w") as f:
            xrg.write_json(f)
        self.ref_actions = excut.parse_json_file(str(ref_path))
        assert self.trace_actions is not None, "Need to run CUDA function first"
        deductions = excut.require_concordance(
            self.ref_actions, self.trace_actions, xrg.varname_set
        )
        self.xrg = xrg
        self.deductions = deductions

    def get_var(self, varname: str) -> excut.ExcutVariableArg:
        return self.xrg.get_var(varname)

    def __getitem__(self, excut_id: excut.ExcutVariableID):
        """For ExcutVariableArg.__call__"""
        return self.deductions[excut_id]

    @classmethod
    def set_excut_buffer_size(cls, n_bytes):
        cls._saved_excut_buffer_size = n_bytes

    def _run_trace_excut(self, c_proc, *args):
        for i in range(3):
            trace_filename = self.compiler.workdir / "excut_trace.json"
            self.fn.exo_excut_begin_log_file(
                trace_filename, self._saved_excut_buffer_size
            )
            c_proc(*args)
            self.fn.exo_excut_end_log_file()
            try:
                self.trace_actions = excut.parse_json_file(trace_filename)
                return
            except excut.ExcutOutOfCudaMemory as e:
                mib = e.bytes_needed / 1048576
                self.set_excut_buffer_size(e.bytes_needed)
                warnings.warn(f"excut: out of CUDA memory; now requesting {mib} MiB")
        raise MemoryError(
            "Excut internal error, trace out-of-cuda-memory after 3 tries"
        )
