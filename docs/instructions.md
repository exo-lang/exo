# External Instruction Definitions

Exo allows users to define custom hardware instructions within their code using the `@instr` annotation.
These user-defined instructions can be leveraged during the scheduling process to replace specific code fragments with calls to hardware-optimized instructions.

## Overview

- **Custom Instructions**: Define hardware-specific instructions as procedures using the `@instr` decorator.
- **Replace**: Use the `replace` primitive to substitute code fragments with calls to these instructions.
- **Code Generation**: Custom instructions can emit arbitrary C code, including inline assembly, with placeholders for arguments.

## Defining Custom Instructions

Custom instructions are defined as procedures annotated with `@instr`.
The `@instr` decorator allows you to specify the C code to be emitted when the instruction is called.

### Syntax

```python
@instr("C code")
def instruction_name(args):
    # Specification of the instruction's behavior
```
- **`@instr`**: Decorator that specifies the C code to emit. In the string provided to `@instr`, you can include placeholders wrapped in `{}`. These placeholders will be replaced with the names of the arguments when the code is compiled.
- **`instruction_name`**: The name of your custom instruction.
- **`args`**: Arguments to the instruction.
- **semantics**: Semantics of the hardware instruction, written as Exo object code.

### Example: Defining a Neon Load Instruction

Below is an example of defining a NEON load instruction that loads four `f32` values into Neon memory.

```python
from exo import *

@instr("{dst_data} = vld1q_f32(&{src_data});")
def neon_vld_4xf32(dst: [f32][4] @ Neon, src: [f32][4] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 4):
        dst[i] = src[i]
```

- **`@instr("{dst_data} = vld1q_f32(&{src_data});")`**: Specifies the C code to emit when this instruction is called.
  - `{dst_data}` and `{src_data}` are format strings that will be replaced with the actual arguments during codegen.
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

### Define Your Procedure

Define your Exo procedure as usual.

```python
@proc
def foo(src: [f32][4] @ DRAM, dst: [f32][4] @ Neon):
    ...
    for i in seq(0, ...):
        ...
        for j in seq(0, 4):
            dst[j] = src[j]
    ...
```

### Use `replace` to Substitute the Instruction

Use the `replace` primitive to substitute the loop with the custom instruction.

```python
# Replace the loop with the custom instruction
foo = replace(foo, "for j in _:_", neon_vld_4xf32)
```

- **`replace(foo, "for i in _:_", neon_vld_4xf32)`**:
  - **`foo`**: The procedure in which to perform the replacement.
  - **`"for i in _:_"`**: A cursor pointing to the loop to replace.
  - **`neon_vld_4xf32`**: The instruction to replace the loop with.

After `replace`, the procedure `foo` will look like:
```python
@proc
def foo(M: size, src: [f32][4] @ DRAM, dst: [f32][4] @ Neon):
    ...
    for i in seq(0, M/4):
        ...
        neon_vld_4xf32(dst, src)
    ...
```

#### How `replace` Works

The `replace` primitive is used to substitute a fragment of code within a procedure with a call to another procedure (e.g., a custom instruction). The syntax for `replace` is as follows:

```python
replace(proc, cursor_path, subproc)
```

- **`proc`**: The procedure containing the code to be replaced.
- **`cursor`**: A cursor pointing to the code fragment to be replaced.
- **`subproc`**: The procedure whose body will replace the code fragment.

The `replace` primitive works by performing an unification modulo linear equalities. The process can be broken down into two main steps:

1. **Pattern Matching**: The body of the sub-procedure `subproc` is unified (pattern matched) with the designated statement block `s` in the original procedure `proc`. During this process:
   - The arguments of `subproc` are treated as unknowns.
   - The free variables of `s` are treated as known symbols.
   - Any symbols introduced or bound within the body of `subproc` or within `s` are unified.

   The ASTs (Abstract Syntax Trees) of `subproc` and `s` are required to match exactly with respect to statements and all expressions that are not simply integer-typed control.

2. **Solving Linear Equations**: Any equivalences between integer-typed control expressions are recorded as a system of linear equations. These equations are then solved to determine the values of the unknowns and ensure a consistent substitution.

By following this process, the `replace` primitive effectively replaces the designated code fragment with a call to the sub-procedure, while ensuring that the substitution is valid and consistent.


### Generated C Code

`exocc` can be used to compile Exo code into C.

```c
void foo(float src[4], float32x4_t dst) {
    ...
    for (int_fast32_t i = 0; i < ...; i++) {
        ...
        dst = vld1q_f32(&src[0]);
    }
    ...
}
```

- **`dst = vld1q_f32(&src[0]);`**: The custom instruction is emitted as specified in the `@instr` decorator, with arguments replaced.

## Understanding the Magic

By defining the behavior of hardware instructions in Python using Exo procedures, you can express the semantics of your accelerator or specialized hardware. The `replace` primitive allows Exo to reason about whether it's safe to offload certain computations to hardware instructions based on their specifications.

- **No Compiler Backend Needed**: The heavy lifting is done within Exo, eliminating the need for a separate compiler backend.
- **Semantics Encoding**: The instruction's body acts as a specification, encoding its semantics for Exo's pattern matching.
- **Flexible and Extensible**: Users can define any instruction and specify how it should be matched and replaced.


## Further Reading and Examples

- **RVM Tutorial**: [https://exo-lang.dev/tutorial.html](https://exo-lang.dev/tutorial.html)
- **Running Code Examples**: [examples/rvm_conv1d/exo/conv1d.py](https://github.com/exo-lang/exo/blob/main/examples/rvm_conv1d/exo/conv1d.py)
