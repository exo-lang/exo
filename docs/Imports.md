# Imports in Exo

This document provides an overview of the imports used when writing Exo.

## Table of Contents

1. [Standard Python Future Import](#1-standard-python-future-import)
2. [Core Exo Module](#2-core-exo-module)
3. [Memory Libraries](#3-memory-libraries)
4. [Instruction Libraries](#4-instruction-libraries)
5. [Frontend Syntax Utilities](#5-frontend-syntax-utilities)
6. [Standard Library Modules](#6-standard-library-modules)
   - [6.1 Scheduling Utilities](#61-scheduling-utilities)
   - [6.2 Standard Library Functions](#62-standard-library-functions)
7. [External Interfaces](#7-external-interfaces)
8. [API Cursors](#8-api-cursors)


## 1. Standard Python Future Import

```python
from __future__ import annotations
```

Enables postponed evaluation of type annotations, allowing you to use forward references in type hints without causing issues during runtime. This is necessary to support Exo's `x : f32` syntax.


## 2. Core Exo Module

```python
from exo import *
```

Imports basic classes and functions necessary for defining and manipulating high-performance computational kernels, such as `proc`, `instr`, `config`, `Memory`, `Extern`, `DRAM`, and `SchedulingError`.


## 3. Memory Libraries

Even though users can define memory definitions externally to the compiler in the user code (see [memories.md](./memories.md)), we provide memory definitions for some architectures for convinience.
The supported memory definitions can be found by looking into `src/exo/libs/memories.py`.

```python
from exo.libs.memories import DRAM_STATIC, AVX2, AVX512
```

For example, you can import `DRAM_STATIC`, `AVX2`, or `AVX512` as shown above.


## 4. Instruction Libraries

Similar to memories, we provide some hardware instruction definitions for convinience (see [instructions.md](./instructions.md) to learn how to define your own accelerator instructions).

```python
from exo.platforms.x86 import mm256_loadu_ps, mm256_setzero_ps, mm256_broadcast_ss
```

## 5. Extern Libraries

Similary, convinience extern libraries can be imported as follows. See [externs.md](./externs.md) to learn how to define your own externs.

```python
from exo.libs.externs import sin, relu
```


## 6. Frontend Syntax Utilities

```python
from exo.frontend.syntax import *
```

This module defines special symbols that are used inside Exo code.
Importing this can suppress warnings inside an IDE (like PyCharm).


## 7. Standard Library Scheduling Functions

Exo provides users with the ability to define new scheduling operations using Cursors. For convenience, we have implemented scheduling libraries (standard library) that contain common scheduling operations users may want to use, such as vectorization and tiling. Users can import the standard library as follows:

```python
from exo.stdlib.scheduling import repeat, replace_all
from exo.stdlib.stdlib import vectorize, tile_loops
```

Alternatively, users can define their own scheduling operations by composing scheduling primitives directly in their code.

## 8. API Cursors

Cursors (see [Cursors.md](./Cursors.md)) are Exo's reference mechanism that allows users to navigate and inspect object code. When users define new scheduling operators using Cursors, they may wish to write their own inspection pass. API Cursors define types that will be useful for user inspection.

```python
from exo.API_cursors import ForCursor, AssignCursor, InvalidCursor
```

These API Cursors provide specific types, such as `ForCursor` for for-loops, `AssignCursor` for assignments, and `InvalidCursor` for invalid cursors. Users can leverage these types when inspecting and manipulating code using Cursors.

