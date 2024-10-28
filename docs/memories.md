# External Memory Definitions

Exo allows users to define custom memory types external to the compiler.
This feature enables modeling of specialized memory systems, such as vector machines and hardware accelerator memories, directly within your Exo code.
By defining custom memories, you can optimize your programs to target specific hardware architectures.

## Overview

- **Custom Memories**: Define your own memory types by subclassing the `Memory` class.
- **Usage**: Use custom memories as annotations in your Exo code or set them during scheduling.

## Defining Custom Memories

To define a custom memory, you need to create a class that inherits from `Memory` and implement the required methods.
Below is an example of defining an `AVX512` memory, which models the AVX-512 vector registers.

### Example: Defining AVX512 Memory

```python
class AVX512(Memory):
    @classmethod
    def global_(cls):
        return "#include <immintrin.h>"

    @classmethod
    def can_read(cls):
        return False

    @classmethod
    def alloc(cls, new_name, prim_type, shape, srcinfo):
        if not shape:
            raise MemGenError(f"{srcinfo}: AVX512 vectors are not scalar values")
        if not prim_type == "float":
            raise MemGenError(f"{srcinfo}: AVX512 vectors must be f32 (for now)")
        if not _is_const_size(shape[-1], 16):
            raise MemGenError(f"{srcinfo}: AVX512 vectors must be 16-wide")
        shape = shape[:-1]
        if shape:
            result = f'__m512 {new_name}[{"][".join(map(str, shape))}];'
        else:
            result = f"__m512 {new_name};"
        return result

    @classmethod
    def free(cls, new_name, prim_type, shape, srcinfo):
        return ""

    @classmethod
    def window(cls, basetyp, baseptr, indices, strides, srcinfo):
        assert strides[-1] == "1"
        idxs = indices[:-1] or ""
        if idxs:
            idxs = "[" + "][".join(idxs) + "]"
        return f"{baseptr}{idxs}"
```

#### Explanation of Methods

- **`global_(cls)`**: Returns any global code or headers needed. Here, it includes the AVX-512 intrinsic header.

  ```python
  @classmethod
  def global_(cls):
      return "#include <immintrin.h>"
  ```

- **`can_read(cls)`**: Controls whether the memory can be read directly. Setting it to `False` means you cannot read/write directly to this memory using standard array access.

  ```python
  @classmethod
  def can_read(cls):
      return False
  ```

- **`alloc(cls, new_name, prim_type, shape, srcinfo)`**: Defines how memory allocation is handled. For `AVX512`, it ensures that the allocated memory represents 16-wide vectors of `float` type.

  ```python
  @classmethod
  def alloc(cls, new_name, prim_type, shape, srcinfo):
      # Validation checks and allocation code
  ```

- **`free(cls, new_name, prim_type, shape, srcinfo)`**: Handles memory deallocation. For `AVX512`, no action is needed.

  ```python
  @classmethod
  def free(cls, new_name, prim_type, shape, srcinfo):
      return ""
  ```

- **`window(cls, basetyp, baseptr, indices, strides, srcinfo)`**: Defines how to access elements in the memory.

  ```python
  @classmethod
  def window(cls, basetyp, baseptr, indices, strides, srcinfo):
      # Windowing logic for memory access
  ```

## Understanding `can_read`

The `can_read` method controls whether direct array access is allowed for the memory type. When `can_read` is set to `False`, you cannot read or write to the memory using standard array indexing in Exo or the generated C code. This models hardware that requires special instructions for memory access, such as vector registers.

### Invalid Usage

Attempting to read or write directly results in an error.

```python
x: f32[16] @ AVX512
x[0] = 3.0  # Invalid when can_read() is False
```

### Valid Usage

To interact with the memory, you must use specific instructions or operations designed for that memory type (e.g., AVX-512 intrinsics).

```python
# Use AVX-512 instructions to manipulate x
x: f32[16] @ AVX512
mm512_loadu_ps(x, inp[16*i : 16*i+16])
```
- **Instructions Documentation**: [instructions.md](instructions.md)

## Using Custom Memories

There are two primary ways to use custom memories in Exo:

1. **Direct Annotation**: Annotate variables with the custom memory type using the `@` symbol.
2. **Scheduling Primitive**: Change the memory annotation during scheduling using `set_memory`.

### 1. Direct Annotation

Annotate buffers at the time of declaration.
```python
from exo import *
from exo.libs.memories import AVX512

@proc
def foo(x: f32[16] @ AVX512):
    y: f32[16] @ AVX512
    # Function body
```

- **`x: f32[16] @ AVX512`**: Declares `x` as a 16-element array of `f32` stored in `AVX512` memory.
- **`y: f32[16] @ AVX512`**: Similarly declares `y` in `AVX512` memory.

### 2. Changing Memory During Scheduling

Use the `set_memory` primitive to change the memory annotation of a variable during scheduling.
- **`set_memory(p, "C", AVX512)`**: Changes the memory of variable `C` in procedure `p` to `AVX512`.
- This is common when optimizing simple object code (e.g., GEMM) for specific hardware.

#### Documentation for `set_memory`

The `set_memory` primitive is documented in [primitives/buffer_ops.md](primitives/buffer_ops.md).


## Additional Examples

- **Memory Definitions**: More examples of custom memory definitions can be found in [src/exo/libs/memories.py](https://github.com/exo-lang/exo/blob/main/src/exo/libs/memories.py).
- **Usage in Applications**: Examples of using custom memories in real applications are available in [examples/rvm_conv1d/exo/conv1d.py](https://github.com/exo-lang/exo/blob/main/examples/rvm_conv1d/exo/conv1d.py).


