# API Documentation

## Top-level Python Function Decorators

- `@proc`: Decorates a Python function to be parsed and compiled as an Exo procedure.
- `@instr`: Similar to `@proc`, but accepts a hardware instruction as a format string.
- `@config`: Decorates a Python class to be parsed and compiled as an Exo configuration object.

## Procedure Object Methods

The following are methods on Exo Procedures (functions decorated with `@proc` or `@instr`).

### Inspection Operations

- `.name()`: Returns the procedure name.
- `.is_instr()`: Returns `True` if the procedure has a hardware instruction string.
- `.get_instr()`: Returns the hardware instruction string.

### Obtaining Cursors

Cursors can be obtained by querying patterns on a procedure. All the Cursor related documentations are in [Cursors.md](Cursors.md).

### Compilation Operations

- `.compile_c(directory, filename)`: Compiles the procedure into C and stores it in `filename` within the specified `directory`.
- `.c_code_str()`: Compiles the procedure and returns a string containing declarations and C code.

### Non-equivalence Preserving Transformations

- `.unsafe_assert_eq(other_proc)`: An unsafe escape hatch asserting the equivalence of this procedure with `other_proc`.
- `.partial_eval(*args, **kwargs)`: Partially evaluates the procedure with given arguments and returns a new, partially evaluated procedure.
- `.transpose(arg_cursor)`: Transposes a 2D buffer argument in the signature and the body. Returns a new procedure and is non-equivalence preserving because the signature has changed.
- `.add_assertion(assertion)`: Adds an assertion to the procedure.
- `.is_eq(other_proc)`: Checks the equivalence of this procedure with another procedure.

## Scheduling Primitives

We have classified scheduling primitives into six categories. Here are the links to each:

- [Buffer Transformations](primitives/buffer_ops.md)
- [Loop and Scope Transformations](primitives/loop_ops.md)
- [Configuration States](primitives/config_ops.md)
- [Subprocedure Operations](primitives/subproc_ops.md)
- [Memory, Precision, and Parallelism Transformations](primitives/backend_ops.md)
- [Other Operations](primitives/other_ops.md)
