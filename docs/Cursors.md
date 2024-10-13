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
- `"for i in _:_"` matches a `for i in _:_` loop
- `"if i == 0:_"` or `"if _:_"` match `if` statements 
- `"a : i8"` or `"a : _"` match an allocation of buffer `a`
- `"a = 3.0"` or `"a = _"` match an assignment to `a`
- `"a += 3.0"` or `"a += _"` match a reduction on `a`
- `"a = 3.0 ; b = 2.0"` matches a block with those two statements

## Cursor Types

Exo defines the following `Cursor` types:

- `StmtCursor`: Cursor to a specific Exo IR statement
- `GapCursor`: Cursor to the space between statements, anchored to a statement
- `BlockCursor`: Cursor to a block (sequence) of statements
- `ArgCursor`: Cursor to a procedure argument (no navigation)
- `InvalidCursor`: Special cursor type for invalid cursors

## Common Cursor Methods

All `Cursor` types provide these common methods:

- `c.parents()`: Returns cursor to parent node in Exo IR
  - Raises `InvalidCursorError` if at the root with no parent
- `c.proc()`: Returns the `Procedure` this cursor is associated with
- `c.find(pattern, many=False)`: Finds matches within cursor's sub-AST

## Statement Cursor Navigation

A `StmtCursor` (pointing to one IR statement) provides these navigation methods:

- `c.next()`: Returns cursor to next statement 
- `c.prev()`: Returns cursor to previous statement
- `c.before()`: Returns `GapCursor` to space immediately before this statement
- `c.after()`: Returns `GapCursor` to space immediately after this statement 
- `c.as_block()`: Returns a `BlockCursor` containing only this one statement

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

- `GapCursor.anchor()`: Returns cursor to the statement this gap is anchored to

- `BlockCursor.expand(delta_lo=None, delta_hi=None)`: Returns an expanded block cursor 
   - `delta_lo`/`delta_hi` specify statements to add at start/end; `None` means expand fully
   - Ex. in `s1; s2; s3`, if `c` is a `BlockCursor` pointing `s1; s2`, then `c.expand(0, 1)` returns a new `BlockCursor` pointing `s1; s2; s3`
- `BlockCursor.before()`: Returns `GapCursor` before block's first statement
- `BlockCursor.after()`: Returns `GapCursor` after block's last statement

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
Returns an invalid cursor if `orelse` isn't present.

Methods:
- `cond() -> ExprCursor`: Returns a cursor to the if condition expression.
- `body() -> BlockCursor`: Returns a cursor to the if body block.
- `orelse() -> Cursor`: Returns a cursor to the else block (if present).

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


## Cursor Forwarding

When a procedure `p` is transformed into a new procedure `p'` by applying scheduling primitives, any cursors pointing into `p` need to be updated to point to the corresponding locations in `p'`. This process is called *cursor forwarding*.

To forward a cursor `c` from `p` to `p'`, you can use the `forward` method on the new procedure:
```python
c' = p'.forward(c)
```

### How Forwarding Works

Internally, each scheduling primitive returns a *forwarding function* that maps locations in the input procedure to locations in the output procedure.

When you call `p'.forward(c)`, Exo composes the forwarding functions for all the scheduling steps between `c.proc()` (the procedure `c` points into, in this case `p`) and `p'` (the final procedure). This composition produces a single function that can map `c` from its original procedure to the corresponding location in `p'`.

Here's the actual implementation of forwarding in `src/exo/API.py`:

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


