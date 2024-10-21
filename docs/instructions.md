# External Instruction Definitions

Exo allows users to define custom hardware instructions within their code using the `@proc` annotation. These user-defined instructions can be leveraged during the scheduling process to replace specific code fragments with calls to hardware-optimized instructions. This feature enables fine-grained control over code optimization and hardware acceleration, making it easier to target specific architectures like SIMD units or custom accelerators.

## Overview

- **Custom Instructions**: Define hardware-specific instructions as procedures using the `@proc` decorator.
- **Replacement**: Use the `replace` primitive to substitute code fragments with calls to these instructions.
- **Pattern Matching**: Exo uses pattern matching to unify code fragments with instruction definitions.
- **Code Generation**: Custom instructions can emit arbitrary C code, including inline assembly, with placeholders for arguments.

## Defining Custom Instructions

Custom instructions are defined as procedures annotated with `@proc` and further decorated with `@instr`. The `@instr` decorator allows you to specify the C code to be emitted when the instruction is called, including placeholders for arguments.

### Syntax

```python
@instr("C code with placeholders")
@proc
def instruction_name(args):
    # Specification of the instruction's behavior
```

- **`@instr`**: Decorator that specifies the C code to emit.
- **`@proc`**: Indicates that the function is an Exo procedure.
- **`instruction_name`**: The name of your custom instruction.
- **`args`**: Arguments to the instruction.
- **Specification**: A high-level description of what the instruction does, used for pattern matching.

### Placeholders in C Code

In the string provided to `@instr`, you can include placeholders wrapped in `{}`. These placeholders will be replaced with the names of the arguments when the code is compiled.

### Example: Defining a NEON Load Instruction

Below is an example of defining a NEON load instruction that loads four `f32` values into NEON memory.

```python
from exo import *
from exo.core.proc import instr

@instr("{dst_data} = vld1q_f32(&{src_data});")
@proc
def neon_vld_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]
```

#### Explanation

- **`@instr("{dst_data} = vld1q_f32(&{src_data});")`**: Specifies the C code to emit when this instruction is called.
  - `{dst_data}` and `{src_data}` are placeholders that will be replaced with the actual argument names.
- **`dst: [f32][4] @ Neon`**: Declares `dst` as a 4-element array of `f32` in `Neon` memory.
- **`src: [f32][4] @ DRAM`**: Declares `src` as a 4-element array of `f32` in `DRAM`.
- **Assertions**: Ensure that the strides of `src` and `dst` are 1 for correct memory access.
- **Loop**: The loop specifies the semantics of the instruction, copying elements from `src` to `dst`.

### Defining the Memory Annotation `Neon`

The `Neon` memory type can be defined similarly to how custom memories are defined, as explained in [memories.md](memories.md).

```python
class Neon(Memory):
    @classmethod
    def global_(cls):
        return "#include <arm_neon.h>"

    # Implement other required methods
```

## Using Custom Instructions

Once you've defined a custom instruction, you can use it to replace code fragments in your procedures.

### Step 1: Define Your Procedure

Define your Exo procedure as usual.

```python
@proc
def foo(src: [f32][4] @ DRAM, dst: [f32][4] @ Neon):
    for i in seq(0, 4):
        dst[i] = src[i]
```

### Step 2: Use `replace` to Substitute the Instruction

Use the `replace` primitive to substitute the loop with the custom instruction.

```python
# Instantiate the procedure
p = foo

# Replace the loop with the custom instruction
p = replace(p, "for i in _:_", neon_vld_4xf32)
```

#### Explanation

- **`replace(p, "for i in _:_", neon_vld_4xf32)`**:
  - **`p`**: The procedure in which to perform the replacement.
  - **`"for i in _:_"`**: A cursor pointing to the loop to replace.
  - **`neon_vld_4xf32`**: The instruction to replace the loop with.

### How `replace` Works

- **Pattern Matching**: Exo attempts to unify the code fragment (the loop) with the body of `neon_vld_4xf32`.
- **Automatic Argument Determination**: If successful, Exo replaces the fragment with a call to `neon_vld_4xf32`, automatically determining the correct arguments.
- **Semantics Preservation**: The specification in the instruction's body ensures that the replacement is semantically correct.

### Step 3: Compile and Generate Code

Compile your procedure to generate the optimized C code.

```python
print(p)
```

### Generated C Code

```c
void foo(float src[4], float32x4_t dst) {
    dst = vld1q_f32(&src[0]);
}
```

- **`dst = vld1q_f32(&src[0]);`**: The custom instruction is emitted as specified in the `@instr` decorator, with placeholders replaced.

## Understanding the Magic

By defining the behavior of hardware instructions in Python using Exo procedures, you can express the semantics of your accelerator or specialized hardware. The `replace` primitive allows Exo to reason about whether it's safe to offload certain computations to hardware instructions based on their specifications.

- **No Compiler Backend Needed**: The heavy lifting is done within Exo, eliminating the need for a separate compiler backend.
- **Semantics Encoding**: The instruction's body acts as a specification, encoding its semantics for Exo's pattern matching.
- **Flexible and Extensible**: Users can define any instruction and specify how it should be matched and replaced.

## The `replace` Primitive

The `replace` primitive is used to substitute a fragment of code within a procedure with a call to another procedure (e.g., a custom instruction).

### Syntax

```python
replace(proc, cursor_path, subproc)
```

- **`proc`**: The procedure containing the code to be replaced.
- **`cursor_path`**: A string or cursor pointing to the code fragment.
- **`subproc`**: The procedure whose body will replace the code fragment.

### Documentation

The `replace` primitive is documented in [primitives/subproc_ops.md](primitives/subproc_ops.md).

## Practical Example: RISC-V Matrix Multiply

### Step 1: Define the Instruction

```python
@instr("{dst} = asm_rvm_macc({src_a}, {src_b}, {dst});")
@proc
def rvm_macc(dst: f32 @ RVM, src_a: f32 @ RVM, src_b: f32 @ RVM):
    dst += src_a * src_b
```

- **`asm_rvm_macc`**: Hypothetical assembly function for RISC-V multiply-accumulate.
- **Specification**: The procedure specifies that `dst += src_a * src_b`.

### Step 2: Use the Instruction in a Procedure

```python
@proc
def matmul_rvm(A: f32[M, K], B: f32[K, N], C: f32[M, N]):
    for i in seq(0, M):
        for j in seq(0, N):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]
```

### Step 3: Optimize Using `replace`

```python
p = matmul_rvm

# Apply transformations to expose the computation pattern
...

# Replace the innermost loop with the custom instruction
p = replace(p, "for k in _:_", rvm_macc)
```

### Step 4: Compile and Generate Code

```python
print(p)
```

### Generated C Code

```c
void matmul_rvm(float A[M][K], float B[K][N], float C[M][N]) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            C[i][j] = asm_rvm_macc(A[i][k], B[k][j], C[i][j]);
        }
    }
}
```

## Further Reading and Examples

- **RVM Tutorial**: [https://exo-lang.dev/tutorial.html](https://exo-lang.dev/tutorial.html)
- **Running Code Examples**: [examples/rvm_conv1d/exo/conv1d.py](https://github.com/exo-lang/exo/blob/main/examples/rvm_conv1d/exo/conv1d.py)

## Tips and Best Practices

- **Define Clear Specifications**: Ensure that the body of your instruction accurately represents its semantics.
- **Use Assertions**: Include assertions in your instruction definitions to enforce constraints and ensure correctness.
- **Leverage Memory Annotations**: Use custom memory annotations to model hardware-specific memory behaviors (e.g., `Neon`, `RVM`).
- **Pattern Matching**: Structure your code to facilitate pattern matching with instruction definitions.
- **Test Thoroughly**: Verify that replacements are correct and that the generated code behaves as expected.

## Conclusion

By defining custom instructions and using the `replace` primitive, Exo provides a powerful mechanism to optimize code for specific hardware architectures directly within the user code. This approach offers flexibility and control, enabling developers to harness hardware acceleration without the need for extensive compiler support.

**Key Takeaways**:

- **Custom Instructions**: Define hardware-specific instructions with precise semantics.
- **Pattern Matching**: Use Exo's pattern matching to replace code fragments safely.
- **Code Generation**: Emit custom C code, including inline assembly, tailored to your hardware.
- **Optimization**: Optimize existing code by replacing computational patterns with hardware-accelerated instructions.

---

**Note**: The examples provided are illustrative and may need adjustments to fit your specific hardware and use cases. Ensure that any external functions or assembly code used in the `@instr` decorator are properly defined and compatible with your target architecture.
