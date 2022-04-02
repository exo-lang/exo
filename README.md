[![CI](https://github.com/ChezJrk/exo/actions/workflows/main.yml/badge.svg)](https://github.com/ChezJrk/exo/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/ChezJrk/exo/branch/master/graph/badge.svg?token=BFIZ0WKP4I)](https://codecov.io/gh/ChezJrk/exo)

# Setup

We make active use of newer Python 3.x features, so please use the same version
of Python as our CI if you're getting errors about unsupported features.

Setting up Exo for development is like any other Python project. We _
strongly_ recommend you use a virtual environment.

```
$ python -m venv ~/.venv/exo
$ . ~/.venv/exo
(exo) $ python -m pip install -U pip setuptools wheel
(exo) $ python -m pip install -r requirements.txt
```

## PySMT

Depending on your setup, getting PySMT to work correctly may be difficult. You
need to independently install a solver such as Z3 or CVC4, and even then getting
the PySMT library to correctly locate that solver may be difficult. We have
included the `z3-solver` package as a requirement, which will hopefully avoid
this issue, but you can also install z3 (or your choice of solver)
independently.

## Submodules

After pulling or updating the repository, be sure to update the submodules.

```
git submodule update --init --recursive
```

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
