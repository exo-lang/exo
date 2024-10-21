# Exo Object Code Syntax

Exo is a programming language designed for performance-critical code, providing fine-grained control over code generation and optimization. In Exo, object code can be defined using Python-like syntax with specific annotations and constructs to model low-level programming concepts.

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
- [Variable Declarations](#variable-declarations)
- [Memory Spaces](#memory-spaces)
- [Loops](#loops)
  - [`for` Loop Syntax](#for-loop-syntax)
- [Conditional Statements](#conditional-statements)
- [Operations and Assignments](#operations-and-assignments)
- [Understanding the Example](#understanding-the-example)
- [Conclusion](#conclusion)

## Annotations and Decorators

### `@proc` Decorator

The `@proc` decorator is used to define an Exo procedure (analogous to a function in other programming languages). It indicates that the following function definition should be treated as Exo object code, which can be further optimized and transformed.

```python
@proc
def function_name(arguments):
    # Function body
```

### Type and Memory Annotations

In Exo, types and memory spaces are explicitly annotated to provide precise control over data representation and placement. The syntax for annotations is:

```python
name: type[size] @ memory
```

- **`name`**: The variable name.
- **`type`**: The data type (e.g., `i32`, `f32`).
- **`[size]`**: The dimensions of the array (optional for scalars).
- **`@ memory`**: The memory space where the variable resides.

## Procedure Arguments

Procedure arguments are declared with their types, sizes, and memory spaces. They can have dependent sizes based on other arguments.

Example from the code:

```python
data: i32[IC, N] @ DRAM
```

- **`data`**: The name of the argument.
- **`i32`**: The data type (32-bit integer).
- **`[IC, N]`**: A 2D array with dimensions `IC` and `N`.
- **`@ DRAM`**: Specifies that `data` resides in DRAM memory.

## Variable Declarations

Variables within the procedure are declared similarly to arguments but without the `@` annotation if they reside in default memory.

Example:

```python
y: i32
```

- **`y`**: The variable name.
- **`i32`**: The data type (32-bit integer).
- **No memory annotation**: Defaults to a standard memory space (e.g., registers).

## Memory Spaces

Memory spaces in Exo are used to model different hardware memory regions, such as DRAM, caches, or specialized memories. The `@` symbol is used to specify the memory space.

Common memory spaces:

- **`@ DRAM`**: Main memory.
- **`@ SRAM`**: Static RAM or cache.
- **`@ Registers`**: CPU registers.

Example:

```python
out: i32[OC, N] @ DRAM
```

- **`out`**: Output array.
- **Resides in DRAM memory.**

## Loops

### `for` Loop Syntax

Exo uses explicit loop constructs to model iteration. The `for` loop syntax is:

```python
for loop_variable in seq(start, end):
    # Loop body
```

- **`loop_variable`**: The loop counter variable.
- **`seq(start, end)`**: Generates a sequence from `start` to `end - 1`.

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

## Operations and Assignments

- **Assignment (`=`)**: Assigns a value to a variable.

  ```python
  y = data[c, j + r]
  ```

- **In-place Addition (`+=`)**: Adds a value to a variable and stores the result back.

  ```python
  out[i, j] += kernels[i, c, r] * y
  ```

- **Array Access**: Uses square brackets to access array elements.

  ```python
  data[c, j + r]
  ```

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

## Conclusion

This example demonstrates how Exo's object code syntax allows for precise and expressive definitions of computations, particularly for performance-critical operations like convolutions. By understanding the annotations, loops, and operations, you can write efficient Exo procedures that can be further optimized and transformed for specific hardware targets.

### Key Points

- **Annotations**: Use `name: type[size] @ memory` to declare variables with explicit types and memory spaces.
- **Loops**: Utilize `for` loops with `seq(start, end)` for controlled iteration.
- **Conditionals**: Implement boundary checks and other logic using `if` and `else`.
- **Operations**: Perform computations using standard arithmetic operators, with support for in-place updates.

### Further Reading

- **Exo Documentation**: Explore more about Exo's syntax and capabilities in the official documentation.
- **Optimizations**: Learn how to apply scheduling primitives and transformations to optimize Exo procedures.

By leveraging Exo's powerful syntax and features, you can develop high-performance code tailored to specific hardware architectures, enabling efficient execution of complex algorithms.
