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

- **Purpose**: Enables postponed evaluation of type annotations, allowing you to use forward references in type hints without causing issues during runtime. This is necessary to support Exo's `x : f32` syntax.
- **Context**: This is a standard Python feature that improves compatibility and performance when using type hints in your code.


## 2. Core Exo Module

```python
from exo import *
```

- **Purpose**: Imports all core functionalities from the Exo language.
- **Includes**: Fundamental classes and functions necessary for defining and manipulating high-performance computational kernels, such as `proc`, `instr`, `config`, `Memory`, `Extern`, `DRAM`, and `SchedulingError`.


## 3. Memory Libraries

Even though users can define memory definitions externally to the compiler in the user code (see [./memories.md]), we provide memory definitions for some architectures as examples. The supported memories can be found by looking into `src/exo/libs/memories.py`.

```python
from exo.libs.memories import DRAM_STATIC, AVX2, AVX512
```

For example, you can import `DRAM_STATIC`, `AVX2`, or `AVX512` as shown above.


## 4. Instruction Libraries

Similar to memories, we provide some hardware instruction definitions as a library.

```python
from exo.platforms.x86 import mm256_loadu_ps, mm256_setzero_ps, mm256_broadcast_ss
```


## 5. Frontend Syntax Utilities

```python
from exo.frontend.syntax import *
```

This module defines special symbols that are used inside Exo code. You may
import this module via `from exo.syntax import *` to suppress warnings and
see documentation inside an IDE (like PyCharm).


## 6. Standard Library Scheduling Functions


```python
from exo.stdlib.scheduling import repeat, replace_all
from exo.stdlib.stdlib import vectorize, tile_loops
```




## 7. External Interfaces

```python
from exo.libs.externs import sin, relu
```

- **Purpose**: Facilitates interaction with external libraries and functions not defined within Exo.
- **Usage**: Allows for the integration of external code, such as C functions or hardware-specific routines, into Exo programs.


## 8. API Cursors

```python
from exo.API_cursors import ForCursor, AssignCursor, InvalidCursor
```

- **Purpose**: Provides cursor-based APIs for navigating and modifying code structures.
- **Usage**: Enables advanced code introspection and manipulation, useful for metaprogramming and automated optimizations.
