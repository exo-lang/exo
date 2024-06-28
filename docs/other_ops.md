
## Other scheduling primitives

#### `rename(proc, new_name)`
Renames this procedure to `new_name`.

#### `make_instr(proc, instr_string, global_string)`
Converts this procedure to an instruction procedure with instruction `instr_string`. `c_instr`  - string representing an instruction macro, `c_global` is string representing global C code necessary for this instruction e.g. includes.


### Code simplification

#### `simplify(proc)`
Simplifies the code in the procedure body. Tries to reduce expressions to constants and eliminate dead branches and loops. Uses branch conditions to simplify expressions inside the branches.


#### `eliminate_dead_code(proc, stmt_cursor)`
if statements: eliminate branch that is never reachable
for statements: eliminate loop if its condition is always false
```
args:
    stmt_cursor       - cursor to the if or for statement

rewrite:
    if p:
        s1
    else:
        s2
      --> (if p is always True)
    s1
```


### Pass modifications

#### `delete_pass(proc)`
Deletes all the `pass` statements in the procedure.


#### `insert_pass(proc, gap_cursor)`
Inserts a `pass` statement at the indicated position.
```
args:
    gap_cursor  - where to insert the new `pass` statement

rewrite:
    s1
       <--- gap_cursor
    s2 
      -->
    s1
    pass
    s2
```


### Local statement edits

#### `inline_window(proc, winstmt_cursor)`
Eliminates use of a window by inlining its definition and expanding it at all use-sites.
```
args:
    winstmt_cursor  - cursor pointing to the WindowStmt to inline

rewrite:
    y = x[...]
    s
      -->
    s[ y -> x[...] ]
```


#### `reorder_stmts(proc, block_cursor)`
Swaps the order of two statements within a block.
```
args:
    block_cursor    - a cursor to a two statement block to reorder

rewrite:
    s1 ; s2  <-- block_cursor
      -->
    s2 ; s1
```

#### `commute_expr(proc, expr_cursors)`
Commutes the binary operation of `+` and `*`.
```
args:
    expr_cursors - a list of cursors to the binary operation

rewrite:
    a * b <-- expr_cursor
      -->
    b * a

    or

    a + b <-- expr_cursor
      -->
    b + a

```

#### `left_reassociate_expr(proc, expr)`
Reassociates the binary operations of `+` and `*`.
```
args:
    expr - the expression to reassociate

rewrite:
    a + (b + c)
        ->
    (a + b) + c
```

#### `rewrite_expr(proc, expr_cursor, new_expr)`
Replaces the `expr_cursor` with `new_expr` if the two are equivalent in the context.
```
rewrite:
    s
      -->
    s[ expr_cursor -> new_expr]
```


#### `merge_writes(proc, block_cursor)`
Merges consecutive assign and reduce statement into a single statement.
Handles all 4 cases of (assign, reduce) x (reduce, assign).
```
args:
    block_cursor  - cursor pointing to the block of two consecutive assign/reduce statement.

rewrite:
    a = b
    a = c
      -->
    a = c
    ----------------------
    a += b
    a = c
      -->
    a = c
    ----------------------
    a = b
    a += c
      -->
    a = b + c
    ----------------------
    a += b
    a += c
      -->
    a += b + c
    ----------------------
```


#### `split_write(proc, stmt)`
Splits a reduce or assign statement with an addition on the RHS into two writes.

This operation is the opposite of the last two cases of `merge_writes`.
```
args:
    stmt    - cursor pointing to the assign/reduce statement.

rewrite:
    a = b + c
      -->
    a = b
    a += c
    ----------------------
    a += b + c
      -->
    a += b
    a += c
    ----------------------

forwarding:
    - cursors to the statement and any cursors within the statement gets invalidated.
    - blocks containing the statement will forward to a new block containing the resulting block.
```

#### `fold_into_reduce(proc, assign)`
Folds an assignment into a reduction if the rhs is an addition whose lhs is equal to the lhs of the assignment.
```
args:
    assign: a cursor pointing to the assignment to fold.

rewrite:
    a = a + (expr)
      -->
    a += expr
```


#### `inline_assign(proc, alloc_cursor)`
Inlines `alloc_cursor` into any statements where it is used after this assignment.
```
rewrite:
    x = y
    s
      -->
    s[ x -> y ]
```


#### `lift_reduce_constant(proc, block_cursor):`
Lifts a constant scaling factor out of a loop.
```
args:
    block_cursor  - block of size 2 containing the zero assignment and the for loop to lift the constant out of

rewrite:
    x = 0.0
    for i in _:
        x += c * y[i]
      -->
    x = 0.0
    for i in _:
        x += y[i]
    x = c * x
```

