# External Inspection Functions

Inspection is a metaprogramming feature that enables metaprograms (like schedules) to dynamically examine the properties of object code. Exo provides inspection through [Cursors](./Cursors.md), allowing users to examine standard AST properties such as variable names, literal expression values, and annotations (e.g., memory spaces and precisions) at scheduling time. Cursors also support local AST navigation, for example, accessing loop bounds (`loop.hi()`) and bodies (`loop.body()`). Inspection functions can be written externally from the Exo compiler, giving users the ability to customize them according to their needs.
For convenience, standard library inspection functions are provided as `exo.stdlib.inspection` module.

Cursor types (such as `ForCursor` and `IfCursor`) are defined in `exo.API_cursors`, so you should import it when writing inspection functions:

```python
from exo.API_cursors import *
```

Here are some simple inspection functions:

```python
def is_loop(proc, loop):
    loop = proc.forward(loop)
    return isinstance(loop, ForCursor)

def get_top_level_stmt(proc, c):
    c = proc.forward(c)

    while not isinstance(c.parent(), InvalidCursor):
        c = c.parent()
    return c
```

Explanation:
- The `is_loop` function takes a `proc` object and a `loop` cursor as input. It forwards the `loop` cursor using `proc.forward(loop)` and checks if the resulting cursor is an instance of `ForCursor`. This function determines whether the given cursor points to a loop statement.
- The `get_top_level_stmt` function takes a `proc` object and a cursor `c` as input. It forwards the cursor `c` using `proc.forward(c)` and then iteratively moves the cursor to its parent using `c.parent()` until it reaches an `InvalidCursor`, which means the cursor reached the outer-most level of the procedure. This function finds the top-level statement that wraps the given cursor.

Exo also exposes `ExoType` for expression types (defined in `src/exo/API_types.py`), which users can access using constructs like `ExoType.F16` and branch on it.

```python
class ExoType(Enum):
    F16 = auto()
    F32 = auto()
    F64 = auto()
    UI8 = auto()
    I8 = auto()
    UI16 = auto()
    I32 = auto()
    R = auto()
    Index = auto()
    Bool = auto()
    Size = auto()
    Int = auto()
    Stride = auto()
```

All the Cursor types and the kind of navigation you can perform on them are documented in [Cursors.md](./Cursors.md).
