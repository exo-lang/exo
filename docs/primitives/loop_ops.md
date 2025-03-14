
## Loop and scope related primitives

#### `divide_with_recompute(proc, loop_cursor, outer_hi, outer_stride, new_iters)`
Divides a loop into the provided `outer_hi` by `outer_stride` dimensions,
and then adds extra compute so that the inner loop will fully cover the
original loop's range.

```
rewrite:
    for i in seq(0, hi):
        s
      -->
    for io in seq(0, outer_hi):
        for ii in seq(0, outer_stride + (hi - outer_hi * outer_stride)):
            s[ i -> outer_stride * io + ii ]
```

#### `divide_loop(proc, loop_cursor, div_const, new_iters, tail="guard", perfect=False)`
Divides a loop into an outer and inner loop, where the inner loop iterates over the range 0 to `div_const`.
Old Name: In Halide and TVM, this was called "split"
```
args:
    loop_cursor     - cursor pointing to the loop to split ;
                      can also be specified using the special shorthands
                      pattern: <loop-iterator-name>
                           or: <loop-iterator-name> #<int>
    div_const       - integer > 1 specifying what to "divide by"
    new_iters       - list or tuple of two strings specifying the new outer and inner iteration variable names
    tail (opt)      - specifies the strategy for handling the "remainder" of the loop division (called the tail of the loop).
                      value can be "cut", "guard", or "cut_and_guard".  Default value: "guard"
    perfect (opt)   - Boolean (default False) that can be set to true to assert that you know the remainder will always
                      be zero (i.e. there is no tail).  You will get an error if the compiler cannot verify this fact itself.

rewrite:
    divide(..., div_const=q, new_iters=['hi','lo'], tail='cut')
    for i in seq(0,e):
        s
      -->
    for hi in seq(0,e / q):
        for lo in seq(0, q):
            s[ i -> q*hi + lo ]
    for lo in seq(0,e - q * (e / q)):
        s[ i -> q * (e / q) + lo ]
```


#### `mult_loops(proc, nested_loops, new_iter_name)`
Performs the inverse operation to `divide_loop`.  Take two loops,
the innermost of which has a literal bound. (e.g. 5, 8, etc.) and
replace them by a single loop that iterates over the product of their
iteration spaces (e.g. 5*n, 8*n, etc.)
```
args:
    nested_loops    - cursor pointing to a loop whose body is also a loop
    new_iter_name   - string with name of the new iteration variable

rewrite:
    for i in seq(0,e):
        for j in seq(0,c):    # c is a literal integer
            s
      -->
    for k in seq(0,e*c):      # k is new_iter_name
        s[ i -> k/c, j -> k%c ]
```


#### `join_loops(proc, loop1_cursor, loop2_cursor)`
Joins two loops with identical bodies and consecutive iteration spaces
into one loop.
```
args:
    loop1_cursor     - cursor pointing to the first loop
    loop2_cursor     - cursor pointing to the second loop

rewrite:
    for i in seq(lo, mid):
        s
    for i in seq(mid, hi):
        s
      -->
    for i in seq(lo, hi):
        s
```

#### `cut_loop(proc, loop_cursor, cut_point)`
Cut a loop into two loops.

First loop iterates from `lo` to `cut_point` and
the second iterating from `cut_point` to `hi`.

We must have:
    `lo` <= `cut_point` <= `hi`
```
args:
    loop_cursor     - cursor pointing to the loop to split
    cut_point       - expression representing iteration to cut at

rewrite:
    for i in seq(0,n):
        s
      -->
    for i in seq(0,cut):
        s
    for i in seq(cut, n):
        s
```

#### `shift_loop(proc, loop_cursor, new_lo)`
Shift a loop iterations so that now it starts at `new_lo`

We must have:
    0 <= `new_lo`
```
args:
    loop_cursor     - cursor pointing to the loop to shift
    new_lo          - expression representing new loop lo

rewrite:
    for i in seq(m,n):
        s(i)
      -->
    for i in seq(new_lo, new_lo + n - m):
        s(i + (m - new_lo))
```

#### `reorder_loops(proc, nested_loops)`
Reorders two loops that are directly nested with each other.
This is the primitive loop reordering operation, out of which
other reordering operations can be built.
```
args:
    nested_loops    - cursor pointing to the outer loop of the
                      two loops to reorder; a pattern to find said
                      cursor with; or a 'name name' shorthand where
                      the first name is the iteration variable of the
                      outer loop and the second name is the iteration
                      variable of the inner loop.  An optional '#int'
                      can be added to the end of this shorthand to
                      specify which match you want,

rewrite:
    for outer in _:
        for inner in _:
            s
      -->
    for inner in _:
        for outer in _:
            s
```



#### `fission(proc, gap_cursor, n_lifts=1, unsafe_disable_checks=False)`
Fissions apart the For and If statements wrapped around
this block of statements into two copies; the first containing all
statements before the cursor, and the second all statements after the
cursor.
```
args:
    gap_cursor          - a cursor pointing to the point in the statement block that we want to fission at.
    n_lifts (optional)  - number of levels to fission upwards (default=1)

rewrite:
    for i in _:
        s1
           <- gap
        s2
      -->
    for i in _:
        s1
    for i in _:
        s2
```


#### `remove_loop(proc, loop_cursor, unsafe_disable_check=False)`
Removes the loop around some block of statements.
This operation is allowable when the block of statements in question
can be proven to be idempotent.
```
args:
    loop_cursor     - cursor pointing to the loop to remove

rewrite:
    for i in _:
        s
      -->
    s
```


#### `add_loop(proc, block_cursor, iter_name, hi_expr, guard=False, unsafe_disable_check=False)`
Adds a loop around some block of statements.
This operation is allowable when the block of statements in question
can be proven to be idempotent.
```
args:
    block_cursor    - cursor pointing to the block to wrap in a loop
    iter_name       - string name for the new iteration variable
    hi_expr         - string to be parsed into the upper bound expression for the new loop
    guard           - Boolean (default False) signaling whether to wrap the block in a `if iter_name == 0: block` condition; in which case idempotency need not be proven.

rewrite:
    s  <--- block_cursor
      -->
    for iter_name in hi_expr:
        s
```


#### `unroll_loop(proc, loop_cursor)`
Unrolls a loop with a constant, literal loop bound
```
args:
    loop_cursor     - cursor pointing to the loop to unroll

rewrite:
    for i in seq(0,3):
        s
      -->
    s[ i -> 0 ]
    s[ i -> 1 ]
    s[ i -> 2 ]
```


### Scope transformations

#### `lift_scope(proc, scope_cursor)`
Lifts the indicated For/If-statement upwards one scope.
```
args: 
    scope_cursor  - cursor to the inner scope statement to lift up

rewrite: (one example)
    for i in _:
        if p:
            s1
        else:
            s2
      -->
    if p:
        for i in _:
            s1
    else:
        for i in _:
            s2
```


#### `fuse(proc, stmt1, stmt2, unsafe_disable_check=False)`
Fuses together two loops or if-guards, provided that the loop bounds or guard conditions are compatible.
```
args:
    stmt1, stmt2        - cursors to the two loops or if-statements that are being fused

rewrite:
    for i in e: <- stmt1
        s1
    for j in e: <- stmt2
        s2
      -->
    for i in e:
        s1
        s2[ j -> i ]
or
    if cond: <- stmt1
        s1
    if cond: <- stmt2
        s2
      -->
    if cond:
        s1
        s2
```


#### `specialize(proc, block, conds)`
Duplicate a statement block multiple times, with the provided
`cond`itions indicating when each copy should be invoked.
Doing this allows one to then schedule differently the "specialized"
variants of the blocks in different ways.

If `n` conditions are given, then `n+1` specialized copies of the block
are created (with the last copy as a "default" version).
```
args:
    block           - cursor pointing to the block to duplicate/specialize
    conds           - list of strings or string to be parsed into guard conditions for the

rewrite:
    B
      -->
    if cond_0:
        B
    elif cond_1:
        B
    ...
    else:
        B
```


