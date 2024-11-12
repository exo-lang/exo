# Exo Object Code Syntax

In Exo, object code can be defined using Python-like syntax with specific annotations and constructs to model low-level programming concepts.

This documentation explains Exo's object code syntax using the following example of a 1D convolution operation:

```python
@proc
def generic_conv1d(
    data: i32[IC, N] @ DRAM,
    kernels: i32[OC, IC, W] @ DRAM,
    out: i32[OC, N] @ DRAM,
):
    # Perform the convolution
    for i in seq(0, OC):
        for j in seq(0, N):
            # Zero out the output memory
            out[i, j] = 0.0
            for c in seq(0, IC):
                for r in seq(0, W):
                    y: i32
                    if j + r < N:
                        y = data[c, j + r]
                    else:
                        y = 0
                    out[i, j] += kernels[i, c, r] * y
```

## Table of Contents

- [Annotations and Decorators](#annotations-and-decorators)
  - [`@proc` Decorator](#proc-decorator)
  - [Type and Memory Annotations](#type-and-memory-annotations)
  - [Procedure Arguments](#procedure-arguments)
  - [Allocations](#allocations)
  - [Memories](#memories)
- [Loops](#loops)
  - [`for` Loop Syntax](#for-loop-syntax)
- [Conditional Statements](#conditional-statements)
- [Assignments](#assignments)
- [Understanding the Example](#understanding-the-example)

## Annotations and Decorators

### `@proc` Decorator

The `@proc` decorator is used to define an Exo procedure (analogous to a function in other programming languages). It indicates that the following function definition should be treated as Exo object code (not Python), which can be further optimized and transformed.

```python
@proc
def function_name(arguments):
    # Function body
```

### Type and Memory Annotations

In Exo, types and memory spaces are explicitly annotated. The syntax is:

```python
name: type[size] @ memory
```

- **`name`**: The variable name.
- **`type`**: The data type. Supported precision types are: `f16`, `f32`, `f64`, `i8`, `i32`, `ui8`, and `ui16`.
- **`[size]`**: The dimensions of the array (optional for scalars).
- **`@ memory`**: The memory space where the variable resides.


### Procedure Arguments

Procedure arguments are declared with their types, sizes, and memory spaces. They can have sizes that depend on other arguments.

Example from the code:

```python
data: i32[IC, N] @ DRAM
```

- **`data`**: The name of the argument.
- **`i32`**: The data type (32-bit integer).
- **`[IC, N]`**: A 2D array with dimensions `IC` and `N`.
- **`@ DRAM`**: Specifies that `data` resides in DRAM memory.

The `data` buffer above represents **tensor** types, which means the stride of the innermost dimension is 1, and the strides of other dimensions are simple multiples of the shapes of the inner dimensions.

Exo allows **window expressions** as well, which are similar to array slicing in Python. Instead of accessing the buffer point-wise (e.g., `x[i]`), users can *window* the array as `x[i:i+2]`. This will create a windowed array of size 2.
Exo procedures take tensor expressions when annotated with `x:f32[3]` syntax and take window expressions when annotated with `x:[f32][3]`, with square brackets around the types.

```python
@proc
def foo(x: [f32][3]):
    for i in seq(0, 3):
        x[i] = 0.0

@proc
def bar(y: f32[10], z: f32[20, 20]):
    foo(y[2:5])
    foo(z[1, 10:13])
```

In this example, `foo` takes a window array of size 3, and `bar` calls `foo` by slicing `y` and `z`, respectively. Running `exocc` on this will generate the following C code:

```c
#include "tmp.h"

#include <stdio.h>
#include <stdlib.h>

// bar(
//     y : f32[10] @DRAM,
//     z : f32[20, 20] @DRAM
// )
void bar(void *ctxt, float* y, float* z) {
    foo(ctxt, (struct exo_win_1f32){ &y[2], { 1 } });
    foo(ctxt, (struct exo_win_1f32){ &z[20 + 10], { 1 } });
}

// foo(
//     x : [f32][3] @DRAM
// )
void foo(void *ctxt, struct exo_win_1f32 x) {
    for (int_fast32_t i = 0; i < 3; i++) {
        x.data[i * x.strides[0]] = 0.0f;
    }
}
```

Moreover, Exo checks the consistency of tensor and window bounds in the frontend. If you modify `foo(y[2:5])` to `foo(y[2:6])` in the code above, the bounds check will fail and emit the following error:

```
TypeError: Errors occurred during effect checking:
/private/tmp/tmp.py:12:8: type-shape of calling argument may not equal the required type-shape: [Effects.BinOp(op='-', lhs=Effects.Const(val=6, type=LoopIR.Int(), srcinfo=<exo.core.prelude.SrcInfo object at 0x106318920>), rhs=Effects.Const(val=2, type=LoopIR.Int(), srcinfo=<exo.core.prelude.SrcInfo object at 0x1063188c0>), type=LoopIR.Index(), srcinfo=<exo.core.prelude.SrcInfo object at 0x106318920>)] vs. [Effects.Const(val=3, type=LoopIR.Int(), srcinfo=<exo.core.prelude.SrcInfo object at 0x105111610>)]. It could be non equal when:
   y_stride_0 = 1, z_stride_0 = 20, z_stride_1 = 1
```

#### Aliasing Limitations

When passing buffers to procedure arguments, aliasing is not allowed. Concretely, you cannot write something like:

```python
foo(y, y)
foo(y[0:5], y[2:7])
```

This limitation exists because the analysis would be imprecise if we allowed such aliasing. This is similar to how C++ compilers can perform more optimization when you use the `__restrict__` keyword to explicitly indicate that you're not aliasing your buffers.


#### Passing Tensor Window Slices to Functions Expecting Non-Window Tensors

It is not allowed to pass a _window_ to a function that expects a non-window tensor as an argument. Consider the following example:

```python
@proc
def callee(x: f32[10]):
    pass

@proc
def caller(x: f32[2, 10]):
    callee(x[0])     # Error: Passing a window slice to a function expecting a non-window tensor
    callee(x[1, :])  # Error: Passing a window slice to a function expecting a non-window tensor
```

In this code snippet, the `callee` function expects a non-window tensor `x` of shape `f32[10]`. However, in the `caller` function, we attempt to pass slices of the `x` tensor (`x[0]` and `x[1]`) to the `callee` function. These slices are windows of the original tensor, and passing them to a function expecting a non-window tensor is not allowed.

To resolve this issue, you can either:
1. Modify the `callee` function to accept a window tensor as an argument, or
2. Create a new non-window tensor from the slice before passing it to the `callee` function.


### Allocations

Variables within the procedure are declared similarly to arguments.

Example:

```python
y: i32
```

- **`y`**: The variable name.
- **`i32`**: The data type (32-bit integer).
- **No memory annotation**: Defaults to `DRAM` if memory is unspecified.

### Memories

Memory annotations in Exo are used to model different hardware memory regions, such as DRAM, caches, or specialized memories. The `@` symbol is used to specify the memory space, for example: `@DRAM`, `@AVX2`, or `@Neon`.
Memory annotations for your custom hardware accelerators can be defined externally to Exo and can be used as annotations in the same way.
While Exo provides default memory (`DRAM`) and some library memory definitions for convenience (`AVX2`, `AVX512`, `Neon`, `GEMM_SCRATCH`, etc.), it is recommended and encouraged that users define their own memory annotations for their specific hardware. For more information on defining custom memory annotations, refer to [memories.md](./memories.md).



## Loops

### `for` Loop Syntax

Exo uses explicit loop constructs to model iteration. The `for` loop syntax is:

```python
for loop_variable in seq(start, end):
    # Loop body
```

- **`loop_variable`**: The loop counter variable.
- **`seq(start, end)`**: Iterates from `start` to `end - 1`.

Example from the code:

```python
for i in seq(0, OC):
    # Iterates i from 0 to OC - 1
```

## Conditional Statements

Conditional logic is expressed using `if` and `else` statements.

Syntax:

```python
if condition:
    # True branch
else:
    # False branch
```

Example:

```python
if j + r < N:
    y = data[c, j + r]
else:
    y = 0
```

- Checks if `j + r` is less than `N`.
- Assigns `y` accordingly.

## Assignments

- **Assignment (`=`)**: Assigns a value to a variable.

  ```python
  y = data[c, j + r]
  ```

- **Reduction (`+=`)**: Adds a value to a variable and stores the result back.

  ```python
  out[i, j] += kernels[i, c, r] * y
  ```

- **Array Access**: Uses square brackets to access array elements.

  ```python
  data[c, j + r]
  ```

- **Window Statements**: Creates a slice (in other words, _window_) of the buffer and assign a new name.
  ```python
  y = x[0:3]
  ```

## Limitations

Exo has a few limitations that users should be aware of:

1. **Non-affine indexing**: Exo does not support non-affine indexing. This means that any indexing operation must be a linear combination of loop variables and constants. For example, the following expressions are not allowed:
   
   ```python
   data[i * j + r] = 0.0  # i * j is non-affine
   if n * m < 30:         # n * m is non-affine
     pass
   ```

   Exo allows quasi-affine indexing by division (e.g., `i/3`) and modulo (e.g., `i%3`) by constants.

   To work around this limitation, you may need to restructure your code or use additional variables to represent the non-affine expressions.

2. **Value-dependent control flow**: Exo separates control values from buffer values, which means that it is not possible to write value-dependent control flow. For instance, the following code is not allowed:
   
   ```python
   if data[i] < 3.0:
     pass
   ```

   If you need to express such operations, consider using externs (see [externs documentation](./externs.md)).


## Understanding the Example

Let's break down the example code step by step.

### Procedure Definition

```python
@proc
def generic_conv1d(
    data: i32[IC, N] @ DRAM,
    kernels: i32[OC, IC, W] @ DRAM,
    out: i32[OC, N] @ DRAM,
):
```

- **`generic_conv1d`**: The procedure name.
- **Arguments**:
  - **`data`**: Input data array of shape `[IC, N]` in DRAM.
  - **`kernels`**: Kernel weights array of shape `[OC, IC, W]` in DRAM.
  - **`out`**: Output data array of shape `[OC, N]` in DRAM.
- **Variables**:
  - **`IC`**, **`OC`**, **`N`**, **`W`**: Dimensions, assumed to be defined elsewhere or passed as parameters.

### Loop Nest

```python
for i in seq(0, OC):
    for j in seq(0, N):
        # Zero out the output memory
        out[i, j] = 0.0
        for c in seq(0, IC):
            for r in seq(0, W):
                y: i32
                if j + r < N:
                    y = data[c, j + r]
                else:
                    y = 0
                out[i, j] += kernels[i, c, r] * y
```

#### Outer Loops

- **`for i in seq(0, OC):`**: Iterates over the output channels.
- **`for j in seq(0, N):`**: Iterates over the spatial dimension of the output.

#### Initialization

- **`out[i, j] = 0.0`**: Initializes the output element at `(i, j)` to zero.

#### Inner Loops

- **`for c in seq(0, IC):`**: Iterates over the input channels.
- **`for r in seq(0, W):`**: Iterates over the kernel width.

#### Conditional Data Access

```python
y: i32
if j + r < N:
    y = data[c, j + r]
else:
    y = 0
```

- **Purpose**: Handles boundary conditions where the kernel extends beyond the input data.
- **`y`**: Temporary variable to hold the input data or zero.
- **Condition**:
  - **If `j + r < N`**: Valid index; assign `data[c, j + r]` to `y`.
  - **Else**: Out-of-bounds; assign `0` to `y`.

#### Accumulation

```python
out[i, j] += kernels[i, c, r] * y
```

- **Operation**: Accumulates the product of the kernel weight and the input data into the output.
- **`kernels[i, c, r]`**: Kernel weight for output channel `i`, input channel `c`, at position `r`.
- **`y`**: The input data value or zero.
