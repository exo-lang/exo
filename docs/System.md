# System overview

This document provides an overview of the Exo compilation process, as illustrated in Figure 1 of the PLDI'22 paper.

![System overview](images/system-overview.png)

## Compilation Process

The Exo compiler consists of a frontend and a backend, with user schedules applied in between. The input to the compiler is a set of Exo source files (`*.exo`), and the output is generated C code (`*.c`).

### Frontend

The frontend performs the following tasks:

1. **Type Checking**: Ensures that the program is well-typed according to Exo's type system.
2. **Bounds Checking**: Verifies that array accesses are within the specified bounds.
3. **Assert Checking**: Checks that any `assert` statements in the code are satisfied.

If any of these checks fail, the compiler reports an error and halts the compilation process.

### User Schedules

After the frontend checks, user-defined schedules are applied to optimize the program for the target hardware. Schedules are written as a sequence of rewrite rules, which transform the program while preserving its semantics.

Exo provides a set of primitive scheduling operators, such as:

- `split`: Splits a loop into two nested loops.
- `reorder`: Reorders two nested loops.
- `unroll`: Unrolls a loop by a specified factor.
- `inline`: Inlines a function call.
- `replace`: Replaces a code fragment with a semantically equivalent implementation, often used for mapping to custom instructions.

Users can compose these primitives to define higher-level scheduling operations using Python code. The Exo compiler applies the user-defined schedules to transform the program.

### Backend

After the user schedules are applied, the backend performs the following tasks:

1. **Memory/Precision Checking**: Verifies that the program correctly uses the memories and data types specified in the hardware library.
2. **Code Generation**: Generates C code from the transformed Exo program.

The backend checks are performed after scheduling to allow the schedules to modify the memory and precision annotations in the program.

## Hardware Libraries

An essential part of the Exo system is the ability to define hardware targets as user libraries. These libraries specify the details of the target accelerator, such as:

- Custom memories
- Custom instructions
- Configuration state

By defining these hardware details in libraries, Exo allows targeting new accelerators without modifying the core compiler. The schedules can then use these hardware-specific features to optimize the program for the target accelerator.

## Source Code

The source code for the Exo compiler is available on GitHub: [https://github.com/exo-lang/exo](https://github.com/exo-lang/exo)

The repository contains the implementation of the Exo language, the compiler, and a set of hardware libraries for different accelerators.

## Conclusion

The Exo system provides a productive environment for developing high-performance kernel libraries targeting specialized hardware accelerators. By combining a flexible scheduling language with the ability to define hardware targets in libraries, Exo enables achieving state-of-the-art performance with significantly less engineering effort compared to traditional approaches.
