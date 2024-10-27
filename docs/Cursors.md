# Cursors

This documentation covers how to use cursors to navigate, point-to, and apply forwarding on procedures.
Throughout this document:
- `p` refers to an Exo `Procedure` object 
- `c` refers to an Exo `Cursor` object

## Obtaining Cursors

### From Procedures
An Exo `Procedure` provides methods to obtain `Cursor`s:

- `p.args()`: Returns cursors to the procedure's arguments.
- `p.body()`: Returns a `BlockCursor` selecting the entire body of the procedure.
- `p.find(pattern, many=False)`: Finds cursor(s) matching the given `pattern` string:
   - If `many=False` (default), returns the first matching cursor. 
   - If `many=True`, returns a list of all matching cursors.
- `p.find_loop(loop_pattern, many=False)`: Finds cursor(s) to a loop, expanding shorthand patterns:
   - `"name"` or `"name #n"` are expanded to `for name in _:_`
   - Works like `p.find()`, returning the first match by default unless `many=True`
- `p.find_alloc_or_arg(buf_name)`: Finds an allocation or argument cursor, expanding the name to `buf_name: _`.
- `p.find_all(pattern)`: Shorthand for `p.find(pattern, many=True)`, returning all matching cursors.

### From Cursors
A `Cursor` provides a similar method to find sub-cursors within its sub-AST:

- `c.find(pattern, many=False)`: Finds cursor(s) matching the `pattern` within the cursor's sub-AST.
  - Like `p.find()`, returns the first match by default unless `many=True`.

### Pattern Language
The `pattern` argument is a string using the following special syntax:

- `_` is a wildcard matching any statement or expression
- `#n` at the end selects the `n+1`th match instead of the first 
  - Ex. `"for i in _:_ #2"` matches the 3rd `i` loop
- `;` is a sequence of statements

Example patterns:
- `"for i in _:_"` matches a `for i in seq(0, n):...` loop
- `"if i == 0:_"` or `"if _:_"` match `if` statements 
- `"a : i8"` or `"a : _"` match an allocation of a buffer `a`
- `"a = 3.0"` or `"a = _"` match an assignment to `a`
- `"a += 3.0"` or `"a += _"` match a reduction on `a`
- `"a = 3.0 ; b = 2.0"` matches a block with those two statements

## Cursor Types

Exo defines the following `Cursor` types:

- `StmtCursor`: Cursor to a specific Exo IR statement
- `GapCursor`: Cursor to the space between statements, anchored to (before or after) a statement
- `BlockCursor`: Cursor to a block (sequence) of statements
- `ArgCursor`: Cursor to a procedure argument (no navigation)
- `InvalidCursor`: Special cursor type for invalid cursors

## Common Cursor Methods

All `Cursor` types provide these common methods:

- `c.parent()`: Returns `StmtCursor` to the parent node in Exo IR
  - Raises `InvalidCursorError` if at the root with no parent
- `c.proc()`: Returns the `Procedure` this cursor is pointing to
- `c.find(pattern, many=False)`: Finds cursors by pattern-match within `c`s sub-AST

## Statement Cursor Navigation

A `StmtCursor` (pointing to one IR statement) provides these navigation methods.

- `c.next()`: Returns `StmtCursor` to next statement 
- `c.prev()`: Returns `StmtCursor` to previous statement
- `c.before()`: Returns `GapCursor` to space immediately before this statement
- `c.after()`: Returns `GapCursor` to space immediately after this statement 
- `c.as_block()`: Returns a `BlockCursor` containing only this one statement
- `c.expand()`: Shorthand for `stmt_cursor.as_block().expand(...)`
- `c.body()`: Returns a `BlockCursor` to the body. Only works on `ForCursor` and `IfCursor`.
- `c.orelse()`: Returns a `BlockCursor` to the orelse branch. Works only on `IfCursor`.

`c.next()` / `c.prev()` return an `InvalidCursor` when there is no next/previous statement.  
`c.before()` / `c.after()` return anchored `GapCursor`s that move with their anchor statements.

Examples:
```
s1 <- c
s2 <- c.next()

s1 <- c.prev()  
s2 <- c

s1
   <- c.before()
s2 <- c  

s1
s2 <- c
   <- c.after()
```

## Other Cursor Navigation

- `GapCursor.anchor()`: Returns `StmtCursor` to the statement this gap is anchored to

- `BlockCursor.expand(delta_lo=None, delta_hi=None)`: Returns an expanded block cursor 
   - `delta_lo`/`delta_hi` specify statements to add at start/end; `None` means expand fully
   - Ex. in `s1; s2; s3`, if `c` is a `BlockCursor` pointing `s1; s2`, then `c.expand(0, 1)` returns a new `BlockCursor` pointing `s1; s2; s3`
- `BlockCursor.before()`: Returns `GapCursor` before block's first statement
- `BlockCursor.after()`: Returns `GapCursor` after block's last statement
- `BlockCursor[pt]`: Returns a `pt+1`th `StmtCursor` within the BlockCursor (e.g. `c[0]` returns `s1` when `c` is pointing to `s1;s2;...`)
- `BlockCursor[lo:hi]`: Returns a slice of `BlockCursor` from `lo` to `hi-1`. (e.g. `c[0:2]` returns `s1;s2` when `c` is pointing to `s2;s2;...`)

## Cursor inspection

`StmtCursor`s wrap the underlying Exo IR object and can be inspected.
   - Ex. check cursor type with `isinstance(c, PC.AllocCursor)`

`StmtCursor`s are one of the following types.

#### `ArgCursor`

Represents a cursor pointing to a procedure argument of the form:
```
name : type @ mem
```

Methods:
- `name() -> str`: Returns the name of the argument.
- `mem() -> Memory`: Returns the memory location of the argument.
- `is_tensor() -> bool`: Checks if the argument is a tensor.
- `shape() -> ExprListCursor`: Returns a cursor to the shape expression list.
- `type() -> API.ExoType`: Returns the type of the argument.

#### `AssignCursor`

Represents a cursor pointing to an assignment statement of the form:
```
name[idx] = rhs
```

Methods:
- `name() -> str`: Returns the name of the variable being assigned to.
- `idx() -> ExprListCursor`: Returns a cursor to the index expression list.
- `rhs() -> ExprCursor`: Returns a cursor to the right-hand side expression.
- `type() -> API.ExoType`: Returns the type of the assignment.

#### `ReduceCursor`

Represents a cursor pointing to a reduction statement of the form:
```
name[idx] += rhs
```

Methods:
- `name() -> str`: Returns the name of the variable being reduced.
- `idx() -> ExprListCursor`: Returns a cursor to the index expression list.
- `rhs() -> ExprCursor`: Returns a cursor to the right-hand side expression.


#### `AssignConfigCursor`

Represents a cursor pointing to a configuration assignment statement of the form:
```
config.field = rhs
```

Methods:
- `config() -> Config`: Returns the configuration object.
- `field() -> str`: Returns the name of the configuration field being assigned to.
- `rhs() -> ExprCursor`: Returns a cursor to the right-hand side expression.

#### `PassCursor`

Represents a cursor pointing to a no-op statement:
```
pass
```

#### `IfCursor`

Represents a cursor pointing to an if statement of the form:
```
if condition:
    body
```
or
```
if condition:
    body
else:
    orelse
```

Methods:
- `cond() -> ExprCursor`: Returns a cursor to the if condition expression.
- `body() -> BlockCursor`: Returns a cursor to the if body block.
- `orelse() -> BlockCursor | InvalidCursor`: Returns a cursor to the else block if present, otherwise returns an invalid cursor.

#### `ForCursor`

Represents a cursor pointing to a loop statement of the form:
```
for name in seq(0, hi):
    body
```

Methods:
- `name() -> str`: Returns the loop variable name.
- `lo() -> ExprCursor`: Returns a cursor to the lower bound expression (defaults to 0).
- `hi() -> ExprCursor`: Returns a cursor to the upper bound expression.
- `body() -> BlockCursor`: Returns a cursor to the loop body block.


#### `AllocCursor`

Represents a cursor pointing to a buffer allocation statement of the form:
```
name : type @ mem
```

Methods:
- `name() -> str`: Returns the name of the allocated buffer.
- `mem() -> Memory`: Returns the memory location of the buffer.
- `is_tensor() -> bool`: Checks if the allocated buffer is a tensor.
- `shape() -> ExprListCursor`: Returns a cursor to the shape expression list.
- `type() -> API.ExoType`: Returns the type of the allocated buffer.


#### `CallCursor`

Represents a cursor pointing to a sub-procedure call statement of the form:
```
subproc(args)
```

Methods:
- `subproc()`: Returns the called sub-procedure.
- `args() -> ExprListCursor`: Returns a cursor to the argument expression list.


#### `WindowStmtCursor`

Represents a cursor pointing to a window declaration statement of the form:
```
name = winexpr
```

Methods:
- `name() -> str`: Returns the name of the window.
- `winexpr() -> ExprCursor`: Returns a cursor to the window expression.


## ExoType

The `ExoType` enumeration represents user-facing various data and control types. It is a wrapper around Exo IR types.

- `F16`: Represents a 16-bit floating-point type.
- `F32`: Represents a 32-bit floating-point type.
- `F64`: Represents a 64-bit floating-point type.
- `UI8`: Represents an 8-bit unsigned integer type.
- `I8`: Represents an 8-bit signed integer type.
- `UI16`: Represents a 16-bit unsigned integer type.
- `I32`: Represents a 32-bit signed integer type.
- `R`: Represents a generic numeric type.
- `Index`: Represents an index type.
- `Bool`: Represents a boolean type.
- `Size`: Represents a size type.
- `Int`: Represents a generic integer type.
- `Stride`: Represents a stride type.

The `ExoType` provides the following utility methods:

#### `is_indexable()`

Returns `True` if the `ExoType` is one of the indexable types, which include:
- `ExoType.Index`
- `ExoType.Size`
- `ExoType.Int`
- `ExoType.Stride`

#### `is_numeric()`

Returns `True` if the `ExoType` is one of the numeric types, which include:
- `ExoType.F16`
- `ExoType.F32`
- `ExoType.F64`
- `ExoType.I8`
- `ExoType.UI8`
- `ExoType.UI16`
- `ExoType.I32`
- `ExoType.R`

#### `is_bool()`

Returns `True` if the `ExoType` is the boolean type (`ExoType.Bool`).


## Cursor Forwarding

When a procedure `p1` is transformed into a new procedure `p2` by applying scheduling primitives, any cursors pointing into `p1` need to be updated to point to the corresponding locations in `p2`. This process is called *cursor forwarding*.

To forward a cursor `c1` from `p1` to `p2`, you can use the `forward` method on the new procedure:
```python
c2 = p2.forward(c1)
```

### How Forwarding Works

Internally, each scheduling primitive returns a *forwarding function* that maps AST locations in the input procedure to locations in the output procedure.

When you call `p2.forward(c1)`, Exo composes the forwarding functions for all the scheduling steps between `c1.proc()` (the procedure `c1` points into, in this case `p1`) and `p2` (the final procedure). This composition produces a single function that can map `c1` from its original procedure to the corresponding location in `p2`.

Here's the actual implementation of the forwarding in `src/exo/API.py`:

```python
def forward(self, cur: C.Cursor):
    p = self
    fwds = []
    while p is not None and p is not cur.proc():
        fwds.append(p._forward)
        p = p._provenance_eq_Procedure

    ir = cur._impl
    for fn in reversed(fwds):
        ir = fn(ir)

    return C.lift_cursor(ir, self)
```

The key steps are:

1. Collect the forwarding functions (`p._forward`) for all procedures between `cur.proc()` and `self` (the final procedure).
2. Get the underlying Exo IR for the input cursor (`cur._impl`).
3. Apply the forwarding functions in reverse order to map the IR node to its final location.
4. Lift the mapped IR node back into a cursor in the final procedure.

So in summary, `p.forward(c)` computes and applies the composite forwarding function to map cursor `c` from its original procedure to the corresponding location in procedure `p`.

Note that a forwarding function can return an invalid cursor, and that is expected. For example, when a statement cease to exist by a rewrite, cursors pointing to the statement will be forwarded to an invalid cursor.

### Implicit and Explicit Cursor Forwarding in Scheduling Primitives

Scheduling primitives, such as `lift_alloc` and `expand_dim`, operate on a target procedure, which is passed as the first argument. When passing cursors to these primitives, the cursors should be forwarded to point to the target procedure.

Consider the following example:
```python
c = p0.find("x : _")
p1 = lift_alloc(p0, c)
p2 = expand_dim(p1, p1.forward(c), ...)
```

In the call to `expand_dim`, the cursor `c` is explicitly forwarded to `p1` using `p1.forward(c)`. This is necessary because `c` was originally obtained from `p0`, and it needs to be adjusted to point to the correct location in `p1`.

However, the scheduling primitives support *implicit forwarding* of cursors. This means that all the cursors passed to these primitives will be automatically forwarded to point to the first argument procedure. The above code can be simplified as follows:

```python
c = p0.find("x : _")
p1 = lift_alloc(p0, c)
p2 = expand_dim(p1, c, ...) # implicit forwarding!
```

In this case, `c` is implicitly forwarded to `p1` within the `expand_dim` primitive, eliminating the need for explicit forwarding.

#### Limitations of Implicit Forwarding

It is important to note that implicit forwarding does not work when navigation is applied to a forwarded cursor. Consider the following example:

```python
c = p0.find("x : _")
p1 = lift_alloc(p0, c)
p2 = reorder_scope(p1, p1.forward(c).next(), ...)
```

In this code, the navigation `.next()` is applied to the forwarded cursor `p1.forward(c)`. Attempting to change `p1.forward(c).next()` to `p1.forward(c.next())` will result in incorrect behavior. This is because navigation and forwarding are *not commutative*.

## Further Reading
More details of the design principles of Cursors can be found in our [ASPLOS '25 paper](.) or in [Kevin Qian's MEng thesis](https://dspace.mit.edu/handle/1721.1/157187).


