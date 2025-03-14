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
In case of `ModuleNotFoundError: No module named 'attrs'` please upgrade your attrs module by `pip install --upgrade attrs`.

## Compile Exo

Exo files can be directly executed with Python:
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

---

# Learn about Exo

Take a look at the [examples](examples/README.md) directory for scheduling examples and the [documentation](docs/README.md) directory for various documentation about Exo.


# Contact

Please contact [exo@mit.edu](mailto:exo@mit.edu) or [yuka@csail.mit.edu](mailto:yuka@csail.mit.edu) if you have any questions.


# Publication

Exo's major contributions and ideas are published in the following two papers.
The gist of its design principles and features is summarized in [Design.md](./docs/Design.md).

- [Exocompilation for Productive Programming of Hardware Accelerators](https://dl.acm.org/doi/abs/10.1145/3519939.3523446)\
  Yuka Ikarashi\*, Gilbert Louis Bernstein\*, Alex Reinking, Hasan Genc, Jonathan Ragan-Kelley\
  PLDI 2022\
  The full version with appendices can be found [here](https://people.csail.mit.edu/yuka/pdf/exo_pldi2022_full.pdf).
- [Exo 2: Growing a Scheduling Language](https://arxiv.org/abs/2411.07211)\
  Yuka Ikarashi, Kevin Qian, Samir Droubi, Alex Reinking, Gilbert Bernstein, Jonathan Ragan-Kelley\
  ASPLOS 2025\
  The full version with appendices can be found [here](https://arxiv.org/abs/2411.07211).

If you use Exo, please cite both the compiler and the papers!

