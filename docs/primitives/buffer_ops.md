
## Buffer Transformation Primitives


#### `bind_expr(proc, expr_cursors, new_name)`
Binds one or more numeric or data-value type expressions into a new intermediate, scalar-sized buffer. It attempts to perform common subexpression elimination while binding. The primitive will stop upon encountering a read of any buffer that the expression depends on. The precision of the new allocation matches that of the bound expression.
```
args:
    expr_cursors    - a list of cursors to multiple instances of the same expression
    new_name        - a string to name the new buffer

rewrite:
    bind_expr(..., '32.0 * x[i]', 'b')
    a = 32.0 * x[i] + 4.0
      -->
    b : R
    b = 32.0 * x[i]
    a = b + 4.0
```

#### `resize_dim(proc, buf_cursor, dim_idx, size, offset, fold: bool = False)`
Resizes the `dim_idx`-th dimension of the buffer `buf_cursor` to `size`. The `offset` argument specifies how to adjust the indices relative to the old buffer. If `fold` is set to `True`, the primitive will attempt to perform a circular buffer optimization, which ignores the `offset` argument. The primitive will fail if there are any accesses to the `dim_idx`-th dimension outside of the range `(offset, offset + size)`.
```
args:
    buf_cursor      - cursor pointing to the Alloc
    dim_idx         - which dimension to shrink
    size            - new size as a positive expression
    offset          - offset for adjusting the buffer access

rewrite:
    x : T[n, ...] ; s
      -->
    x : T[size, ...] ; s[ x[idx, ...] -> x[idx - offset, ...] ]

rewrite (if fold == True):
    x : T ; s
      -->
    x : T ; s[ x[i] -> x[i % size] ]

checks:
    The provided dimension size is checked for positivity and the
    provided indexing expression is checked to make sure it is in-bounds
```


#### `expand_dim(proc, buf_cursor, alloc_dim, indexing_expr)`
Expands the number of dimensions of a buffer variable (`buf_cursor`). After the expansion, the existing code will initially use only specific entries of the new dimension, which are selected by the provided `indexing_expr`.
```
args:
    buf_cursor      - cursor pointing to the Alloc to expand
    alloc_dim       - (string) an expression for the size of the new buffer dimension.
    indexing_expr   - (string) an expression to index the newly created dimension with.

rewrite:
    x : T[...] ; s
      -->
    x : T[alloc_dim, ...] ; s[ x[...] -> x[indexing_expr, ...] ]

checks:
    The provided dimension size is checked for positivity and the
    provided indexing expression is checked to make sure it is in-bounds
```


#### `rearrange_dim(proc, buf_cursor, permute_vector)`
Rearranges the dimensions of the specified buffer allocation according to the provided permutation (`permute_vector`).
```
    args:
        buf_cursor      - cursor pointing to an Alloc statement for an N-dimensional array
        permute_vector  - a permutation of the integers (0,1,...,N-1)

    rewrite:
        (with permute_vector = [2,0,1])
        x : T[N,M,K]
          -->
        x : T[K,N,M]
```


#### `divide_dim(proc, alloc_cursor, dim_idx, quotient)`
Divides the `dim_idx`-th buffer dimension into higher-order and lower-order dimensions, where the constant integer `quotient` specifies the size of the lower-order dimension.
```
args:
    alloc_cursor    - cursor to the allocation to divide a dimension of
    dim_idx         - the index of the dimension to divide
    quotient        - (positive int) the factor to divide by

rewrite:
    divide_dim(..., 1, 4)
    x : R[n, 12, m]
    x[i, j, k] = ...
      -->
    x : R[n, 3, 4, m]
    x[i, j / 4, j % 4, k] = ...
```

#### `mult_dim(proc, alloc_cursor, hi_dim_idx, lo_dim_idx)`
Multiplies the `hi_dim_idx`-th buffer dimension by the `low_dim_idx`-th buffer dimension to create a single buffer dimension. This operation is only allowed when the `lo_dim_idx`-th dimension is a constant integer value.
```
args:
    alloc_cursor    - cursor to the allocation to divide a dimension of
    hi_dim_idx      - the index of the higher order dimension to multiply
    lo_dim_idx      - the index of the lower order dimension to multiply

rewrite:
    mult_dim(..., 0, 2)
    x : R[n, m, 4]
    x[i, j, k] = ...
      -->
    x : R[4*n, m]
    x[4*i + k, j] = ...
```


#### `unroll_buffer(proc, alloc_cursor, dimension)`
Unrolls the buffer allocation with constant dimension.
```
args:
    alloc_cursor  - cursor to the buffer with constant dimension
    dimension     - dimension to unroll

rewrite:
    buf : T[2] <- alloc_cursor
    ...
      -->
    buf_0 : T
    buf_1 : T
    ...
```

#### `lift_alloc(proc, alloc_cursor, n_lifts=1)`
Lifts a buffer allocation up and out of various loops and conditions.
```
args:
    alloc_cursor    - cursor to the allocation to lift up
    n_lifts         - number of times to try to move the allocation up

rewrite:
    for i in _:
        buf : T <- alloc_cursor
        ...
      -->
    buf : T
    for i in _:
        ...
```


#### `sink_alloc(proc, alloc_cursor)`
Sinks a buffer allocation into a scope (for loop/if statement). This scope must come immediately after the alloc statement.
```
args:
    alloc_cursor    - cursor to the allocation to sink up

rewrite:
    buf : T       <- alloc_cursor
    for i in _:
        ...
      -->
    for i in _:
        buf : T
        ...
```

#### `reuse_buffer(proc, buf_cursor, replace_cursor)`
Reuses existing buffer (`buf_cursor`) instead of allocating a new buffer (`replace_cursor`).
```
args:
    buf_cursor      - cursor pointing to the Alloc to reuse
    replace_cursor  - cursor pointing to the Alloc to eliminate

rewrite:
    x : T ; ... ; y : T ; s
      -->
    x : T ; ... ; s[ y -> x ]

checks:
    Can only be performed if the variable x is dead at the statement y : T.
```



#### `stage_mem(proc, block_cursor, win_expr, new_buf_name, accum=False)`
Stages the window of memory specified by `win_expr` into a new buffer
before the indicated code block and move the memory back after the
indicated code block.  If code analysis allows one to omit either
the load or store between the original buffer and staging buffer, then
the load/store loops/statements will be omitted.

If code analysis determines determines that `win_expr` accesses
out-of-bounds locations of the buffer, it will generate loop nests
for the load/store stages corresponding to that window, but will add
guards within the inner loop to ensure that all accesses to the buffer
are within the buffer's bounds.

In the event that the indicated block of code strictly reduces into
the specified window, then the optional argument `accum` can be set
to initialize the staging memory to zero, accumulate into it, and
then accumulate that result back to the original buffer, rather than
loading and storing.  This is especially valuable when one's target
platform can more easily zero out memory and thereby either
reduce memory traffic outright, or at least improve locality of access.
```
args:
    block_cursor    - the block of statements to stage around
    win_expr        - (string) of the form `name[pt_or_slice*]` e.g. 'x[32, i:i+4]'
                      In this case `x` should be accessed in the block, but only at locations (32, i), (32, i+1), (32, i+2), or (32, i+3)
    new_buf_name    - the name of the newly created staging buffer
    accum           - (optional, bool) see above

rewrite:
    stage_mem(..., 'x[0:n,j-1:j]', 'xtmp')
    for i in seq(0,n-1):
        x[i,j] = 2 * x[i+1,j-1]
      -->
    for k0 in seq(0,n):
        for k1 in seq(0,2):
            xtmp[k0,k1] = x[k0,j-1+k1]
    for i in seq(0,n-1):
        xtmp[i,j-(j-1)] = 2 * xtmp[i+1,(j-1)-(j-1)]
    for k0 in seq(0,n):
        for k1 in seq(0,2):
            x[k0,j-1+k1] = xtmp[k0,k1]
```


#### `delete_buffer(proc, buf_cursor)`
Deletes `buf_cursor` if it is unused.

