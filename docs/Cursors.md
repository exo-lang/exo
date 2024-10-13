# Cursors

Throughout this document, we use `p` for Exo procedures and `c` for Exo Cursors.

## Pattern match to obtain a Cursor

### On Exo Procedures

- `p.args()`: Returns cursors to procedure arguments.
- `p.body()`: Returns a BlockCursor selecting the entire body of the Procedure.

- `p.find(pattern, many=False)`: Finds a cursor for the given pattern. If `many=True`, returns a list of all cursors matching the pattern.
- `p.find_loop(loop_pattern, many=False)`: Finds a cursor pointing to a loop. Similar to `p.find(...)`, but if the supplied pattern is of the form 'name' or 'name #n', it will be expanded to `for name in _:_`. (e.g., `p.find_loop("i")` will find a loop `i`)
- `p.find_alloc_or_arg(buf_name)`: Finds an allocation or argument cursor. will get expanded to `buf_name: _`.
- `p.find_all(pattern)`: Finds a list of all cursors matching the pattern. Shorthand for `p.find(pattern, many=True)`.

### On Cursors

- `c.find(pattern, many=False)`: Similar to `p.find(...)`, finds a cursor for the given pattern, inside the Cursor `c` to restrict the pattern matching to the sub-AST.

### Pattern language

Pattern language is an argument to `p.find(...)`. `_` denotes wildcard and will match against any statements.
Supported patterns include:
- `"for i in _:_"` will match on the loop `i`
- `"if i == 0:_"` or `"if _:_"` will match on the if statements with respective conditions
- `"a : i8"` or `"a : _"` will match on the allocation for a buffer `a`
- `"a = 3.0"` or `"a = _"` will match on the assignment for `a`
- `"a += 3.0"` or `"a += _"` will match on the reduction for `a`
- `"a = 3.0 ; b = 2.0"` will match against a block of size two

If unspecified and `many=False`, `p.find(...)` will return the first match of the pattern.
Putting `# <num>` at the end of the pattern will let you pattern match for the `<num>+1`th match.
For example, `"for i in _:_ #2"` will match against the third `i` loop in the procedure (or a cursor).


## Types of Cursors
- Stmt Cursors: cursors pointing to statements
- Gap Cursors: cursors pointing to gaps (between statements). Gap cursors are attached to the anchor statement cursor.
- Block Cursors

- Arg Cursors: arg cursors don't have navigation, and can be only obtained by `p.args()`

- Invalid Cursors: Invalid cursors are invalid!

## Methods on all types of cursors

- `c.parents()`
        Get a Cursor to the parent node in the syntax tree.

        Raises InvalidCursorError if no parent exists
- `c.proc()`
        Get the Procedure object that this Cursor points into

- `c.find(pattern, many=False)`: As noted above


## Stmt Cursor navigation

#### `c.next()` (returns a _node_ cursor):
Return a statement cursor to the next statement in the
block (or dist-many next)

Returns InvalidCursor() if there is no such cursor to point to.
```
s1 <- c
s2 <- c.next()
```

#### `c.prev()` (returns a _node_ cursor):
Return a statement cursor to the previous statement in the
block (or dist-many previous)

Returns InvalidCursor() if there is no such cursor to point to.
```
s1 <- c.prev()
s2 <- c
```

#### `c.before()` (returns a _gap_ cursor):
Get a cursor pointing to the gap immediately before this statement.
Gaps are anchored to the statement they were created from. This
means that if you move the statement, the gap will move with it
when the cursor is forwarded.
```
s1
   <- c.before()
s2 <- c
```

#### `c.after()` (returns a _gap_ cursor):
Get a cursor pointing to the gap immediately after this statement.

Gaps are anchored to the statement they were created from. This
means that if you move the statement, the gap will move with it
when the cursor is forwarded.
```
s1
s2 <- c
   <- c.after()
```

#### `c.as_block()` (returns a _block_ cursor):
Return a Block containing only this one statement

## Gap Cursor navigation

#### `c.anchor()`
Get a cursor pointing to the node to which this gap is anchored.

## Block Cursor navigation

#### `c.expand(delta_lo=None, delta_hi=None)` (returns a _block_ cursor):
Expand the block cursor.

When `delta_lo (delta_hi)` is not None, it is interpreted as a
number of statements to add to the lower (upper) bound of the
block.  When `delta_lo (delta_hi)` is None, the corresponding
bound is expanded as far as possible.

Both arguments must be non-negative if they are defined.


For example, when `s1; s2; s3`, and a block cursor `c` is pointing to a block `s1; s2`, `c.expand(0, 1)` will return a block cursor pointing to `s1; s2; s3`.


#### `c.before()` (return a _gap_ cursor):
Get a cursor pointing to the gap before the first statement in
this block.

Gaps are anchored to the statement they were created from. This
means that if you move the statement, the gap will move with it
when the cursor is forwarded.


#### `c.after()` (returns a _gap_ cursor):
Get a cursor pointing to the gap after the last statement in
this block.

Gaps are anchored to the statement they were created from. This
means that if you move the statement, the gap will move with it
when the cursor is forwarded.




## Cursor inspection

Stmt Cursors act as a wrapper to Exo's IR, and users can inspect the object 

Not sure if we should expand on all the possible Cursor types here...
`isinstance(c, PC.AlloCursor)`


## Cursor forwarding

`p.forward(...)` provide some examples


Each scheduling primitive returns a forwarding function.
Procedure objects has a pointer to the previous procedure, and the forwarding function from the previous procedure to the current procedure.
When `p.forward(...)` is called, all the forwarding function up until the Cursor's procedure will get applied.

Literally the code in src/exo/API.py
```
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


