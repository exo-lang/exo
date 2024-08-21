[![CI](https://github.com/exo-lang/exo/actions/workflows/main.yml/badge.svg)](https://github.com/exo-lang/exo/actions/workflows/main.yml)
![GitHub last commit](https://img.shields.io/github/last-commit/exo-lang/exo)
[![codecov](https://codecov.io/gh/exo-lang/exo/branch/master/graph/badge.svg?token=BFIZ0WKP4I)](https://codecov.io/gh/exo-lang/exo)

# Basics

## Install Exo

We support Python versions 3.9 and above.
If you're just using Exo, install it using `pip`:
```sh
$ pip install exo-lang
```

## Compile Exo

Exo files can be directly excuted with Python:
```sh
$ python exo_file.py
```

To generate generate C and header files, use `exocc` command:
```sh
$ exocc exo_file.py
```
Running the command will generate two files: `exo_file.c` and `exo_file.h`. These files will be created in a directory called `exo_file/` by default.
You can use optional arguments to customize the output:
- The `-o` argument allows you to specify a different directory name.
- The `--stem` argument allows you to specify custom names for the C file and header file.


# Examples

Take a look at [examples](examples/README.md) for scheduling examples, and [API documentation](docs/API.md) for complete scheduling interface documentation.


# Build Exo from source

We make active use of newer Python 3.x features. Please use Python 3.9 or 3.10 if you're getting errors about unsupported features.

Setting up Exo for development is like any other Python project. We
_strongly_ recommend you use a virtual environment.

```
$ git clone git@github.com:exo-lang/exo.git
$ cd exo/
$ git submodule update --init --recursive
$ python -m venv ~/.venv/exo
$ source ~/.venv/exo/bin/activate
(exo) $ python -m pip install -U pip setuptools wheel
(exo) $ python -m pip install -r requirements.txt
(exo) $ pre-commit install
```

This will make sure you have the submodules checked out and that the pre-commit
scripts (that run an autoformatter, maybe other tools in the future) run.

Finally, you can build and install Exo.

```
(exo) $ python -m build .
(exo) $ pip install dist/*.whl
```


## PySMT

Depending on your setup, getting PySMT to work correctly may be difficult. You
need to independently install a solver such as Z3 or CVC4, and even then getting
the PySMT library to correctly locate that solver may be difficult. We have
included the `z3-solver` package as a requirement, which will hopefully avoid
this issue, but you can also install z3 (or your choice of solver)
independently.

# Notes for Testing

## Dependencies

### Build system (required)

The Exo test harness generates C code and as such needs to compile and link
using an unknown (i.e. system) compiler. To do this, it generates CMake build
files and invokes CMake behind the scenes.

Therefore, you must have CMake **3.21** or newer installed.

By default, CMake will use [Ninja](https://ninja-build.org) as its backend, but
this may be overridden by setting the environment variable `CMAKE_GENERATOR`
to `Unix Makefiles`, in case you do not wish to install Ninja.

### SDE (optional)

For testing x86 features on processors which don't support them (e.g., AVX-512
or AMX), we rely on
the [Intel Software Development Emulator](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html)
as an optional dependency. Tests which rely on this (namely for AMX) look
for `sde64` either in the path defined by the `SDE_PATH` environment variable or
in the system `PATH`, and are skipped if it is not available.

## Running tests

To run the tests, simply type

```
pytest
```

in the root of the project.

## Running Coverage Testing

To run pytest with coverage tests, execute

```
pytest --cov=./ --cov-report=html
```

Then, if you want to see annotated source files, open `./htmlcov/index.html`.


# Repository structure

In this repository, folders are structured as follows:

1. `src/exo` is where the core Exo implementation resides.
    - **APIs.** Documentation for the APIs can be found in the [API documentation](docs/API.md).
      - `API.py` defines a stable API for top-level decorators (`proc`, `instr`, and `config`).
      - `API_scheduling.py` defines a API for scheduling primitives.
      - `API_cursors.py` defines a API for scheduling primitives.  
    - **Standard libraries.** These could be user-defined, but we provide them for convenience.
      - `libs/` contains some common memory definitions (`memories.py`) and custom malloc implementations. 
      - `platforms/` contains instruction definitions that are part of the release.
      - `stdlib/` contains user-level scheduling functions such as `vectorize`.
    - Other files are implementation details of Exo (e.g., `typecheck.py` implements typecheck), are not exposed to users.
2. `apps/` contains some sample applications written in Exo.
3. `dependencies/` contains submodules that Exo's apps and testing depends on.
4. `examples/` contains a step-by-step example of scheduling basic matrix multiplication on AVX2.
5. `tests/` contains the Exo test suite.
6. `docs/` contains additional Exo documentation.


# Contact

Please contact [exo@mit.edu](mailto:exo@mit.edu) if you have any questions.


# Publication

The first paper on Exo was published at PLDI '22. You can download the
paper [from ACM Digital Library](https://dl.acm.org/doi/abs/10.1145/3519939.3523446).
If you use Exo, please cite both the compiler and the paper!

```
@inproceedings{pldi22:exo,
  title        = {Exocompilation for Productive Programming of Hardware Accelerators},
  author       = {
    Ikarashi, Yuka and Bernstein, Gilbert Louis and Reinking, Alex and Genc,
    Hasan and Ragan-Kelley, Jonathan
  },
  year         = 2022,
  booktitle    = {
    Proceedings of the 43rd ACM SIGPLAN International Conference on Programming
    Language Design and Implementation
  },
  location     = {San Diego, CA, USA},
  publisher    = {Association for Computing Machinery},
  address      = {New York, NY, USA},
  series       = {PLDI 2022},
  pages        = {703–718},
  doi          = {10.1145/3519939.3523446},
  isbn         = 9781450392655,
  url          = {https://doi.org/10.1145/3519939.3523446},
  abstract     = {
    High-performance kernel libraries are critical to exploiting accelerators
    and specialized instructions in many applications. Because compilers are
    difficult to extend to support diverse and rapidly-evolving hardware
    targets, and automatic optimization is often insufficient to guarantee
    state-of-the-art performance, these libraries are commonly still coded and
    optimized by hand, at great expense, in low-level C and assembly. To better
    support development of high-performance libraries for specialized hardware,
    we propose a new programming language, Exo, based on the principle of
    exocompilation: externalizing target-specific code generation support and
    optimization policies to user-level code. Exo allows custom hardware
    instructions, specialized memories, and accelerator configuration state to
    be defined in user libraries. It builds on the idea of user scheduling to
    externalize hardware mapping and optimization decisions. Schedules are
    defined as composable rewrites within the language, and we develop a set of
    effect analyses which guarantee program equivalence and memory safety
    through these transformations. We show that Exo enables rapid development
    of state-of-the-art matrix-matrix multiply and convolutional neural network
    kernels, for both an embedded neural accelerator and x86 with AVX-512
    extensions, in a few dozen lines of code each.
  },
  numpages     = 16,
  keywords     = {
    program optimization, user-schedulable languages, user-extensible backend
    &amp; scheduling, instruction abstraction, scheduling, hardware
    accelerators
  }
}
```
