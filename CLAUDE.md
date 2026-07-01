# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

TODO consider moving random factoids into `endgame_exo` documentation.

## Overview

Exo is a domain-specific language for writing high-performance hardware-accelerated code. It uses a scheduling language approach where you write high-level Exo code using the `@proc` decorator and apply scheduling transformations to generate optimized C/CUDA code.

## Build and Development Commands

### Installation
```bash
python -m pip install -r requirements.txt
python -m build .
pip install dist/*.whl
```

### Compiling Exo Files
```bash
exocc exo_file.py              # Generates exo_file/exo_file.c and .h
exocc -o outdir exo_file.py    # Custom output directory
exocc --stem name exo_file.py  # Custom output file names
```

## Architecture

### Compiler Pipeline
```
Python Source (@proc) → Parser (pyparser) → UAST → Type Checker → LoopIR
    → Scheduling Transformations → Code Generation → C/CUDA
```

### Source Structure (`src/exo/`)

**Core Components:**
- `core/LoopIR.py`: The intermediate representation (IR) - procedures, statements, expressions, types.
   `ADT` compiles a grammar into a Python module of strongly-typed nodes, which enforce the types of node members.
   `@extclass(LoopIR.typename) def fname...` injects a member function `fname` into the `LoopIR.typename` class.
- `core/memory.py`: Memory abstraction framework (DRAM, GPU memory types, registers)
- `core/instr_class.py`: Instruction template framework with `@instr` decorator
- `core/configs.py`: Configuration objects for parameterizing generated code

**Frontend:**
- `frontend/pyparser.py`: Converts Python `@proc` functions to UAST
- `frontend/typecheck.py`: Type inference and validation, UAST → LoopIR
- `frontend/pattern_match.py`: Pattern matching for cursor navigation
- `frontend/boundscheck.py`: Array bounds checking

**Scheduling System:**
- `rewrite/LoopIR_scheduling.py`: All scheduling primitives (split, tile, parallelize, fuse, reorder, etc.)
- `rewrite/new_eff.py`: Effect-based legality analysis using SMT solver
- `rewrite/range_analysis.py`: Index range analysis for bounds
- `rewrite/LoopIR_unification.py`: Equivalence checking

**Backend:**
- `backend/LoopIR_compiler.py`: C/CUDA code generation

**Platform Targets:**
- `platforms/cuda.py`: CUDA backend (gmem, smem, warp intrinsics)
- `platforms/x86.py`: AVX2/AVX512 intrinsics
- `platforms/neon.py`: ARM NEON SIMD
- `platforms/rvv.py`: RISC-V Vector Extension
- `platforms/gemmini.py`: Gemmini accelerator
- `platforms/Sm80.py`, `platforms/Sm90.py`: GPU-specific (A100, H100)

**Spork Extensions (Exo 2.0 GPU/threading):**
- `spork/timelines.py`: Timeline abstraction for GPU execution ordering
- `spork/sync_types.py`: Synchronization primitives (barriers, arrive, await)
- `spork/loop_modes.py`: Loop classification (seq, par)
- `spork/coll_algebra.py`: Collective operation algebra

**User API:**
- `API.py`, `API_*.py`: User-facing decorators (`@proc`, `@instr`, `@config`)
- `API_cursors.py`: High-level cursor API for program navigation
- `main.py`: CLI entry point (`exocc`)

### Test Structure (`tests/`)

- `test_schedules.py`: Scheduling operations
- `test_typecheck.py`: Type checking
- `test_codegen.py`: Code generation
- `test_cursors.py`: Cursor navigation
- `test_x86.py`, `test_neon.py`, `test_rvv.py`: Platform-specific tests
- `cuda/`: CUDA-specific tests
- `amx/`: Intel AMX tests
- `golden/`: Expected outputs for golden tests

### Key Patterns

**Writing a Procedure:**
```python
from exo import *

@proc
def my_kernel(n: size, A: f32[n], B: f32[n]):
    for i in seq(0, n):
        B[i] = A[i] * 2.0
```

**Scheduling:**
```python
p = my_kernel
p = p.split('i', 8, ['io', 'ii'], perfect=True)
p = p.reorder('ii', 'io')
```

**Cursor Navigation:**
```python
cursor = p.find('for io in _:_')
p = p.parallelize(cursor)
```

### Distributed Memory Concepts

TODO: reconcile any contradictions with `spork_b` docs.
Almost certainly, the docs are what are actually correct.

**Permitted index expression**: A plain read of:
1. A **required iterator** (cuda_threads iterator with tile_count > 1 and thread_pitch != 0), OR
2. A cuda_threads iterator with **0 thread pitch** (tile_count = 1, e.g., `cuda_threads(0, 1, unit=...)`)

The 0-thread-pitch iterators are no-ops for distribution - they're permitted but not required.

**Thread pitch**: The difference in thread indices between consecutive loop iterations:
- `cuda_threads(0, N)` with no unit: pitch = 1
- `cuda_threads(0, N, unit=cuda_warp)`: pitch = 32
- `cuda_threads(0, N, unit=cuda_warpgroup)`: pitch = 128
- `cuda_threads(0, N, unit=cuda_cta_in_cluster)`: pitch = blockDim

**Native unit**: Memory types have a native collective unit that determines distribution requirements:
- `CudaRmem`: native unit is `cuda_thread` (must subdivide to individual threads)
- `CudaSmemLinear`: native unit is `cuda_cta_in_cluster` (can distribute by CTA in cluster)

**Common distributed memory errors**:
| Error Message | Cause |
|--------------|-------|
| `Expected single variable name, not X` | Index is expression/constant, not plain variable |
| `Expected cuda_threads-loop iterator, not X` | Index variable is from seq loop, not cuda_threads |
| `X.tile_count = N; must be M` | Iterator range doesn't match array dimension |
| `Missing subdivision on dims[N]` | Memory's native unit requires finer thread distribution |
| `X.thread_pitch (A) != Y.thread_pitch (B)` | Inconsistent thread pitches across usages |

## Documentation Resources

### AI-Optimized Spork Documentation
The Exo-GPU (Spork) documentation is available in an AI-friendly format at `../spork/docs/spork_b/`. This documentation is structured for optimal context window usage:

- **Sections and definitions**: Stored in `*.tex` files (one per section/definition)
- **Code listings**: Plain text code stored in `*.0.txt` files (preferred for reading)
- **LaTeX-formatted code**: `*.0.tex` files (avoid reading - hard to parse, auto-generated)

When looking up Spork concepts (e.g., "Split Barrier Basic Requirements", sync-tl semantics, barrier mechanisms), check this documentation directory for the relevant `.tex` and `.0.txt` files.

## Dependencies

- Python 3.9+
- CMake 3.21+ (for test harness)
- Ninja (default) or Make
- PySMT with Z3 solver (for scheduling legality checks)
