# Externs

Externs in Exo provide a mechanism to interface with external functions and libraries directly from your Exo code. By defining custom extern functions, you can extend the capabilities of Exo and leverage existing code written in other languages like C or C++. Externs can be used as expressions in your code, particularly on the right-hand side (RHS) of assignment and reduction statements.

## Defining Externs in User Code

Extern functions are defined by subclassing the `Extern` class provided by Exo. This allows you to specify how the extern function should behave, including type checking, compilation, and any global code it might require.

### Step-by-Step Guide

#### 1. Import the Extern Class

Before you can define an extern function, you need to import the `Extern` class and the `_EErr` exception from `exo.core.extern`.

```python
from exo.core.extern import Extern, _EErr
```

- `Extern`: The base class for creating custom extern functions.
- `_EErr`: An exception class used for error handling during type checking.

#### 2. Subclass the Extern Class

Create a new class that inherits from `Extern`. This class represents your custom extern function.

```python
class _Sin(Extern):
    # Implementation details will go here
```

#### 3. Implement Required Methods

Your subclass must implement several methods to define the behavior of the extern function.

##### `__init__(self)`

Initialize your extern function with its name.

```python
def __init__(self):
    super().__init__("sin")
```

- `"sin"`: The name of the external function as it will appear in the generated code.

##### `typecheck(self, args)`

Define how the function checks the types of its arguments.

```python
def typecheck(self, args):
    if len(args) != 1:
        raise _EErr(f"expected 1 argument, got {len(args)}")

    arg_type = args[0].type
    if not arg_type.is_real_scalar():
        raise _EErr(
            f"expected argument to be a real scalar value, but got type {arg_type}"
        )
    return arg_type
```

- Checks that there is exactly one argument.
- Ensures the argument is a real scalar type (e.g., `float`, `double`).
- Returns the type of the argument as the return type of the function.

##### `compile(self, args, prim_type)`

Define how the function is compiled into target code.
- `args`: list of arguments as C strings
- `prim_type`: A C string representing the primitive data type. It could be one of the following C strings, mapping from LoopIR types to C strings:
  - `f16` -> `"_Float16"`
  - `f32` -> `"float"`
  - `f64` -> `"double"`
  - `i8`  -> `"int8_t"`
  - `ui8` -> `"uint8_t"`
  - `ui16`-> `"uint16_t"`
  - `i32` -> `"int32_t"`

```python
def compile(self, args, prim_type):
    return f"sin(({prim_type}){args[0]})"
```

- Generates the code that calls the external function, ensuring proper casting to the primitive type.

##### `globl(self, prim_type)`

Provide any global code or headers needed.

```python
def globl(self, prim_type):
    return "#include <math.h>"
```

- Includes necessary headers required for the external function (e.g., `<math.h>` for mathematical functions).

#### 4. Instantiate the Extern Function

Create an instance of your extern class to make it usable in your code.

```python
sin = _Sin()
```

- `sin` now represents the extern function and can be used like any other expression in Exo.

## Using Externs as Expressions

Externs can be used as expressions on the RHS of assignment and reduction statements. This allows you to incorporate external functions seamlessly into your Exo computations.

Note that externs (and Exo procedures) do not allow aliasing in their arguments. This restriction is in place to prevent externs from having side effects on the input arguments.

### Example: Using `sin` in an Expression

Here's a complete example demonstrating how to define and use the `sin` extern function within an expression.

```python
from __future__ import annotations
from exo import *
from exo.core.extern import Extern, _EErr

class _Sin(Extern):
    def __init__(self):
        super().__init__("sin")

    def typecheck(self, args):
        if len(args) != 1:
            raise _EErr(f"expected 1 argument, got {len(args)}")

        arg_type = args[0].type
        if not arg_type.is_real_scalar():
            raise _EErr(
                f"expected argument to be a real scalar value, but got type {arg_type}"
            )
        return arg_type

    def compile(self, args, prim_type):
        return f"sin(({prim_type}){args[0]})"

    def globl(self, prim_type):
        return "#include <math.h>"

    def interpret(self, args):
        import math
        return math.sin(args[0])

# Instantiate the extern function
sin = _Sin()

# Define an Exo procedure using the extern function in an expression
@proc
def foo(x: f32):
    x = sin(x) * 3.0

print(foo)
```

### Output

When you run the code above with `exocc`, the generated C code will be:
```c
#include <math.h>
// foo(
//     x : f32 @DRAM
// )
void foo( void *ctxt, float* x ) {
  *x = sin((float)*x) * 3.0f;
}
```
