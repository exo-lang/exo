[![CI](https://github.com/ChezJrk/exo/actions/workflows/main.yml/badge.svg)](https://github.com/ChezJrk/exo/actions/workflows/main.yml)
[![codecov](https://codecov.io/gh/ChezJrk/exo/branch/master/graph/badge.svg?token=BFIZ0WKP4I)](https://codecov.io/gh/ChezJrk/exo)

# Install Exo

## Using Exo

If you're just using Exo, install it using `pip`:
```sh
$ pip install exo-lang
```

## Developing Exo

If you plan to work on the compiler directly, clone this repository and run the following commands:

```sh
$ git submodule update --init --recursive
$ python3.9 -m venv ~/.venv/exo
$ source ~/.venv/exo/bin/activate
```

This will checkout all the required submodules and enable the Exo virtual environment. Next, install the compiler:
```sh
$ python -m pip install -U pip setuptools wheel
$ python -m pip install -r requirements.txt
$ pre-commit install
```

This will make sure you have the submodules checked out and that the pre-commit
scripts (that run an autoformatter, maybe other tools in the future) run.

If you're feeling ambitious, you can also [install Exo from source](#build-exo-from-source).

# Examples

Take a look at `exo/examples` for scheduling examples.

# Exo's scheduling API

## Top-level Python function decorator

1. `@proc` - decorates a Python function which is parsed and compiled as Exo. Replaces
   the function with a `Procedure` object.
2. `@instr` - same as `@proc`, but accepts a hardware instruction as a format string.
3. `@config` - decorates a Python class which is parsed and compiled as an Exo
   configuration object

## Procedure object methods

**Introspection operations**

- `.name()` returns the procedure name.
- `.check_effects()` forces Exo to run effect checking on the procedure.
- `.show_effects()` prints the effects of the procedure.
- `.show_effect(stmt)` prints the effect of the `stmt` in the procedure.
- `.is_instr()` returns `true` if the procedure has a hardware instruction string.
- `.get_instr()` returns the hardware instruction string.
- `.get_ast()` returns a `QAST`, which is an AST representation suitable for
  introspection.

**Execution / interpretation operations**

- `.compile_c(directory, filename)` compiles the procedure into C and stores
  in `filename` in the `directory`.
- `.interpret(**args)` runs Exo interpreter on the procedure.

## Scheduling operations on Procedure objects

**Buffer related operations**

| Operation                                                   | Description                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `.data_reuse(buf1, buf2)`                                   | Reuses a buffer `buf1` in the use site of `buf2` and removes the allocation of `buf2`.                                                                                                                                                                                        |
| `.inline_window(win_stmt)`                                  | Removes the window statement `win_stmt`, which is an alias to the window, and inlines the windowing in its use site.                                                                                                                                                          |
| `.expand_dim(stmt, alloc_dim, indexing)`                    | Expands the dimension of the allocation statement `stmt` with dimension `alloc_dim` of indexing `indexing`.                                                                                                                                                                   |
| `.bind_expr(new_name, expr)`                                | Binds the right hand side expression `expr` to a newly allocated buffer named `new_name`                                                                                                                                                                                      |
| `.stage_mem(win_expr, new_name, stmt_start, stmt_end=None)` | Stages the buffer `win_expr` to the new window expression `new_name` in statement block (`stmt_start` to `stmt_end`), and adds an initialization loop and a write-back loop.                                                                                                  |
| `.rearrange_dim(alloc, dimensions)`                         | Takes an allocation statement and a list of integers to map the dimension. It rearranges the dimensions of `alloc` in `dimension` order. E.g., if `alloc` were `foo[N,M,K]` and the `dimension` were `[2,0,1]`, it would become `foo[K,N,M]` after this operation.            |
| `.lift_alloc(alloc, n_lifts=1, keep_dims=False)`            | Lifts the allocation statement `alloc` out of `n_lifts` number of scopes. If and For statements are the only statements in Exo which introduce a scope. When lifting the allocation out of a for loop, it will expand its dimension to the loop bound if `keep_dims` is True. |

**Loop related operations**

| Operation                                                           | Description                                                                                                                                                                                                                                                                                                                       |
|---------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `.split(loop, split_const, iter_vars, tail='guard', perfect=False)` | Splits `loop` into an outer and an inner loop. The inner loop bound is `split_const` and the outer and inner loop names are specified by a list of strings `iter_vars`. If `perfect` is True, it will not introduce a tail case. `tail` specifies the tail strategies, where the options are `guard`, `cut`, and `cut_and_guard`. |
| `.fuse_loop(loop1, loop2)`                                          | Fuses two adjacent loops with a common iteration variable.                                                                                                                                                                                                                                                                        |
| `.partition_loop(loop, num)`                                        | Partitions `loop` into two loops, the first running between `0` and `num` and the second between `num+1` and `loop`'s original bound.                                                                                                                                                                                             |
| `.reorder(loop1, loop2)`                                            | Reorders two nested loops. `loop2` should be nested directly inside `loop1`. `loop1` will be nested inside `loop2` after this operation.                                                                                                                                                                                          |
| `.unroll(loop)`                                                     | Unrolls the loop. The loop needs to have a constant bound.                                                                                                                                                                                                                                                                        |
| `.fission_after(stmt, n_lifts=1)`                                   | Fissions the `n_lifts` number of loops around the `stmt`. The fissioned loops around the `stmt` need to be directly nested with each other and the statements before and after the `stmt` should not have any allocation dependencies.                                                                                            |
| `.remove_loop(loop)`                                                | Replaces the loop with its body if the body is idempotent. The system must be able to prove that the loop runs at least once.                                                                                                                                                                                                     |

**Config related operations**

| Operation                                       | Description                                                                                                                                                                |
|-------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `.bind_config(expr, config, field)`             | Binds the right hand side `expr` to `config.field`. It will replace the use site of `expr` with `config.field` and introduces a config statement of `config.field = expr`. |
| `.configwrite_root(config, field, expr)`        | Inserts the config statement `config.field = expr` in the beginning of the procedure.                                                                                      |
| `.configwrite_after(stmt, config, field, expr)` | Inserts the config statement `config.field = expr` after `stmt`.                                                                                                           |
| `.delete_config(stmt)`                          | Deletes the configuration statement.                                                                                                                                       |

**Other scheduling operations**

| Operation                        | Description                                                                                                                                                                                  |
|----------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `.add_assertion(assertion)`      | Asserts the truth of the expression `assertion` at the beginning of the procedure.                                                                                                           |
| `.lift_if(if, n_lifts=1)`        | Lifts the if statement `if` out of `n_lifts` number of scopes. This is similar to `reorder()`, but for if statements.                                                                        |
| `.eliminate_dead_code(stmt)`     | Eliminates `if` statement if condition is always True or False. Eliminates `for` statement if condition is always False.                                                                     |
| `.delete_pass()`                 | Deletes a `Pass` statement in the procedure.                                                                                                                                                 |
| `.reorder_stmts(stmt1, stmt2)`   | Reorder two adjacent statements `stmt1` and `stmt2`. After this operation, the order will be `stmt2` `stmt1`.                                                                                |
| `.reorder_before(stmt)`          | Move the statement `stmt` before the previous statement. This is a shorthand for `reorder_stmts()`.                                                                                          |
| `.replace(subproc, stmt)`        | Replace the statement with a call to `subproc`. This operation is one of our contributions and is explained in detail in the paper.                                                          |
| `.replace_all(subproc)`          | Eagerly replace every matching statement with a call to `subproc`.                                                                                                                           |
| `.inline(call_site)`             | Inline the function call.                                                                                                                                                                    |
| `.is_eq(another_proc)`           | Returns True if `another_proc` is equivalent to the procedure.                                                                                                                               |
| `.call_eqv(eqv_proc, call_site)` | Replace the function call statement of `call_site` with a call to an equivalent procedure `eqv_proc`.                                                                                        |
| `.repeat(directive, *args)`      | Continue to run the directive until it fails. The directive and its arguments are given separately, e.g. `proc.repeat(Procedure.inline, "proc_to_inline(_)")`                                |
| `.simplify()`                    | Simplify the code in the procedure body. Tries to reduce expressions to constants and eliminate dead branches and loops. Uses branch conditions to simplify expressions inside the branches. |
| `.rename(new_name)`              | Rename this procedure to `new_name`.                                                                                                                                                         |
| `.make_instr(instr_string)`      | Converts this procedure to an instruction procedure with instruction `instr_string`.                                                                                                         |
| `.partial_eval(*args, **kwargs)` | Specializes this procedure to the given argument values.                                                                                                                                     |
| `.set_precision(name, type)`     | Sets the precision type of `name` to `type`.                                                                                                                                                 |
| `.set_window(name, is_window)`   | If `is_window` is True, it sets the buffer `name` to window type, instead of a tensor type.                                                                                                  |
| `.set_memory(name, mem_type)`    | Sets a buffer `name`'s memory type to `mem_type`.                                                                                                                                            |

# Exo's repository structure

In this repository, folders are structured as follows:

1. `src/exo` is where the core Exo implementation resides.
    - `API.py` defines the stable API. Documentation for this API can be found in the
      section below.
    - `libs/` contains some common memory definitions (`memories.py`) and custom malloc
      implementations. These could be user-defined, but we provide them for convenience.
    - `platforms/` contains instruction definitions that are part of the release. These
      could be user-defined, but we provide them for convenience.
    - Other files are implementation details of Exo (e.g., `typecheck.py` implements
      typecheck), but we will not dwell on these as they are not exposed to users.
2. `apps/` contains some sample applications written in Exo.
3. `dependencies/` contains submodules that Exo's apps and testing depends on.
4. `examples/` contains a Python notebook that we used for live demos. This should be
   ignored.
5. `tests/` contains the Exo test suite.

# Build Exo from source

## Self-contained install with Python

If you don't want to use your system version of python (e.g. if it's too old),
you can install Exo and a compatible version of Python with Nix.

First, install Nix (if you don't have it) using either the
[systemwide installer](https://nixos.org/download.html) or the portable install
(no root required for portable):

```
$ wget https://github.com/DavHau/nix-portable/releases/download/v009/nix-portable
$ chmod +x nix-portable
```

Then launch a shell which includes Exo and a compatible version of Python:

```
$ git clone git@github.com:exo-lang/exo.git
$ cd exo/

# with a systemwide nix installation
$ nix --experimental-features 'nix-command flakes' develop

# or with a portable nix installation
$ PATH_TO_NIX_PORTABLE/nix-portable nix develop
```

This is a virtualenv-like environment that you will need to enter each time you
wish to use Exo.

## Manual install

We make active use of newer Python 3.x features, so please use the same version
of Python as our CI if you're getting errors about unsupported features.

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
(exo) $ python -m build
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
