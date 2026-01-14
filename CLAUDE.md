# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Exo is a domain-specific language for writing high-performance hardware-accelerated code. It uses a scheduling language approach where you write high-level Exo code using the `@proc` decorator and apply scheduling transformations to generate optimized C/CUDA code.

## Build and Development Commands

### Installation
```bash
python -m pip install -r requirements.txt
python -m build .
pip install dist/*.whl
```

### Running Tests
```bash
# Run all tests
pytest

# Run a single test file
pytest tests/test_schedules.py

# Run a specific test
pytest tests/test_schedules.py::test_name

# Skip slow tests
pytest -m "not slow"

# Run with coverage
pytest --cov=./ --cov-report=html

# Update golden test outputs
pytest --update-golden
```

### CUDA Tests
```bash
# CPU-only compile tests (no device required)
pytest --cuda-run-cpu

# Tests requiring sm_80 GPU (e.g., A100)
pytest --cuda-run-Sm80

# Tests requiring sm_90a GPU (e.g., H100)
pytest --cuda-run-Sm90a
```

Environment variables for CUDA:
- `EXO_NVCC`: Path to nvcc compiler
- `EXO_CCBIN`: Host compiler for nvcc
- `SDE_PATH`: Path to Intel SDE for AVX-512/AMX emulation

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
- `core/LoopIR.py`: The intermediate representation (IR) - procedures, statements, expressions, types
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

### CUDA Error Testing Pattern

For testing CUDA error conditions, use the `mkproc` pattern with parameterized positive/negative cases:

```python
def mkproc_feature(valid_param=True, invalid_option=False):
    @proc
    def test_proc(...):
        with CudaDeviceFunction(blockDim=256):
            for task in cuda_tasks(0, 1):
                if invalid_option:
                    # Code that triggers an error
                    ...
                else:
                    # Valid code path
                    ...
    return test_proc

# Positive test: valid proc, compare against golden output
def test_feature_positive(compiler, golden):
    compiler.cuda_cpu_test(mkproc_feature, golden)

# Negative test: invalid proc, check error message
def test_feature_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_feature, invalid_option=True)
    assert "expected error substring" in str(exc.value)
```

Key points:
- `mkproc_*` functions return a proc, taking parameters that control valid/invalid code paths
- `compiler.cuda_cpu_test(mkproc_fn, **kwargs)` compiles and optionally runs the proc
- Positive tests use the `golden` fixture to compare generated code against expected output
- Negative tests use `pytest.raises` and assert on specific error message substrings
- Parameters to `mkproc_fn` are passed as kwargs to `cuda_cpu_test`

## Dependencies

- Python 3.9+
- CMake 3.21+ (for test harness)
- Ninja (default) or Make
- PySMT with Z3 solver (for scheduling legality checks)
