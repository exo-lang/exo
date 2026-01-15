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

For testing CUDA error conditions, use the `mkproc` pattern with parameterized positive/negative cases. **Share as much common code as possible** between positive and negative paths to ensure any error is due to the specific thing being tested, not an unrelated mistake.

**Preferred: Parameterize values used inside a single proc**
```python
def mkproc_feature(num_threads):
    """Test thread count limits - parameterize the value, not the structure"""
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, num_threads):  # Parameter controls behavior
                    foo = 1.0
    return simplify(test_proc)

def test_feature_positive(compiler):
    compiler.cuda_cpu_test(mkproc_feature, num_threads=32)  # Valid: 32 <= blockDim

def test_feature_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_feature, num_threads=64)  # Invalid: 64 > blockDim
    assert "thread" in str(exc.value).lower()
```

**Also good: Parameterize sync-tl, memory types, or other Exo objects**
```python
def mkproc_fence(first_sync_tl, second_sync_tl):
    """Test Fence sync-tl combinations - objects defined outside @proc work inside"""
    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with CudaDeviceFunction(blockDim=32):
            for task in cuda_tasks(0, 1):
                for tid in cuda_threads(0, 32):
                    Fence(first_sync_tl, second_sync_tl)  # Parameters from outside
    return simplify(test_proc)

def test_fence_valid_positive(compiler):
    compiler.cuda_cpu_test(mkproc_fence, first_sync_tl=cuda_in_order, second_sync_tl=cuda_in_order)

def test_fence_invalid_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_fence, first_sync_tl=wgmma_async, second_sync_tl=cuda_in_order)
    assert "sync-tl" in str(exc.value).lower()
```

**Last resort: Different proc structures (only when code paths must differ)**
```python
def mkproc_wgmma_fence(use_warpgroup_unit=True):
    """Only use separate branches when structure must differ"""
    device_fn = CudaDeviceFunction(blockDim=128)  # Share what you can

    if use_warpgroup_unit:
        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with device_fn:
                for task in cuda_tasks(0, 1):
                    for wg in cuda_threads(0, 1, unit=cuda_warpgroup):  # Correct unit
                        Fence(wgmma_fence_1, wgmma_fence_2)
    else:
        @proc
        def test_proc(foo: f32 @ CudaGmemLinear):
            with device_fn:
                for task in cuda_tasks(0, 1):
                    for tid in cuda_threads(0, 128):  # Wrong: individual threads
                        Fence(wgmma_fence_1, wgmma_fence_2)
    return simplify(test_proc)
```

Key points:
- **Maximize shared code** - CudaDeviceFunction, CudaWarps, integers, sync-tl objects, and memory types can be assigned outside `@proc` and used inside
- `mkproc_*` functions return a proc, taking parameters that control valid/invalid code paths
- `compiler.cuda_cpu_test(mkproc_fn, **kwargs)` compiles and optionally runs the proc
- Positive tests use the `golden` fixture to compare generated code against expected output
- Negative tests use `pytest.raises` and assert on specific error message substrings
- Parameters to `mkproc_fn` are passed as kwargs to `cuda_cpu_test`

**CudaWarps for controlling thread/warp execution:**
```python
def mkproc_warpgroup_alignment(warp_lo):
    """Use CudaWarps to control which warps execute the operation."""
    device_fn = CudaDeviceFunction(blockDim=256)
    warps = CudaWarps(warp_lo, warp_lo + 4)  # 4 warps = 1 warpgroup

    @proc
    def test_proc(foo: f32 @ CudaGmemLinear):
        with device_fn:
            for task in cuda_tasks(0, 1):
                with warps:  # Use CudaWarps with 'with' statement inside proc
                    for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
                        Fence(wgmma_fence_1, wgmma_fence_2)
    return simplify(test_proc)

def test_positive(compiler):
    compiler.cuda_cpu_test(mkproc_warpgroup_alignment, warp_lo=0)  # Aligned: 0 % 4 == 0

def test_negative(compiler):
    with pytest.raises(Exception) as exc:
        compiler.cuda_cpu_test(mkproc_warpgroup_alignment, warp_lo=1)  # Misaligned: 1 % 4 != 0
    assert "alignment" in str(exc.value).lower()
```

Note: For warpgroup operations (like wgmma fence), warp alignment IS checked at compile time. The `lo` parameter to CudaWarps must be aligned to warpgroup boundaries (multiples of 4 warps).

### CUDA Error Testing Pitfalls

Common mistakes when writing CUDA error tests:

1. **Variables must be used** - Declaring a variable is not enough; dead code elimination removes unused allocations before type/memory checks run. Always read from or write to the variable:
   ```python
   # BAD: packed is never used, type check never runs
   packed: i8[4] @ CudaRmemPacked32

   # GOOD: using the variable triggers the type check
   packed: i8[4] @ CudaRmemPacked32
   packed[0] = 0
   ```

2. **Barriers need distribution dimensions** - Use `barrier[N]` with matching thread tiling to avoid distributed memory errors:
   ```python
   # BAD: barrier not distributed, causes "distributed memory deduction failed"
   bar: barrier @ CudaMbarrier
   for tid in cuda_threads(0, 32):
       Arrive(...) >> bar

   # GOOD: barrier dimension matches thread tiling
   bar: barrier[1] @ CudaMbarrier
   for wg in cuda_threads(0, 1, unit=cuda_warpgroup):
       Arrive(...) >> bar[wg]
   ```

3. **CudaGridConstant is read-only on device** - Can only read from it in device code:
   ```python
   # BAD: writing causes "mutable access" error
   gc: f32[16] @ CudaGridConstant
   gc[0] = 1.0

   # GOOD: read from gc, write to different buffer
   dst[0] = gc[0]
   ```

4. **guarded_by syntax** - Use `barrier(guard_name)`, not subscript syntax:
   ```python
   # BAD: invalid Python syntax
   bar2: barrier @ CudaMbarrier[guarded_by=bar1]

   # GOOD: guarded_by in parentheses
   bar2: barrier(bar1) @ CudaMbarrier
   ```

5. **Proc definitions in parameterized mkproc (last resort)** - If you must have different proc structures (not just different parameter values), define procs inside if/else branches to avoid both being created. **Prefer parameterizing values over branching structures** (see CUDA Error Testing Pattern above):
   ```python
   # BAD: both procs created regardless of parameter
   def mkproc(use_valid=True):
       @proc
       def invalid_proc(): ...  # Always created!
       @proc
       def valid_proc(): ...
       return valid_proc if use_valid else invalid_proc

   # ACCEPTABLE (when structure must differ): only requested proc is created
   def mkproc(use_valid=True):
       device_fn = CudaDeviceFunction(blockDim=128)  # Share what you can!
       if use_valid:
           @proc
           def test_proc(): ...  # Valid version
       else:
           @proc
           def test_proc(): ...  # Invalid version
       return simplify(test_proc)

   # PREFERRED: parameterize values, not structure (see examples above)
   def mkproc(num_threads):
       @proc
       def test_proc(): ...  # Single proc, behavior controlled by parameter
       return simplify(test_proc)
   ```

6. **Test the right error** - Ensure test structure is valid first; structural errors mask the error you're testing for. If you get unexpected errors about distributed memory or bounds, fix the test structure before asserting on error messages.

## Dependencies

- Python 3.9+
- CMake 3.21+ (for test harness)
- Ninja (default) or Make
- PySMT with Z3 solver (for scheduling legality checks)
