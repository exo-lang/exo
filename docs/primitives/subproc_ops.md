
## Sub-procedure primitives

#### `extract_subproc(proc, block, subproc_name, include_asserts=True)`
Extracts a block as a subprocedure with the name `subproc_name`.
```
args:
    block           - the block to extract as a subprocedure.
    subproc_name    - the name of the new subprocedure.
    include_asserts - whether to include asserts about the parameters that can be inferred from the parent.

returns:
    a tuple (proc, subproc).

rewrite:
    extract_subproc(..., "sub_foo", "for i in _:_")
    def foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        for i in seq(0, 8):
            x[i, 0] += 2.0
      -->
    def foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        sub_foo(N, M, K, x)
    def sub_foo(N: size, M: size, K: size, x: R[N, K + M]):
        assert N >= 8
        for i in seq(0, 8):
            x[i, 0] += 2.0
```


#### `replace(proc, block_cursor, subproc, quiet=False)`
Attempt to match the supplied `subproc` against the supplied
statement block.  If the two can be unified, then replace the block
of statements with a call to `subproc`.
```
args:
    block_cursor    - Cursor or pattern pointing to block of statements
    subproc         - Procedure object to replace this block with a call to
    quiet           - (bool) control how much this operation prints out debug info
```


#### `call_eqv(proc, call_cursor, eqv_proc):`
Swaps out the indicated call with a call to `eqv_proc` instead.
This operation can only be performed if the current procedures being
called and `eqv_proc` are equivalent due to being scheduled
from the same procedure (or one scheduled from the ).
```
args:
    call_cursor     - Cursor or pattern pointing to a Call statement
    eqv_proc        - Procedure object for the procedure to be substituted in

rewrite:
    orig_proc(...)
      -->
    eqv_proc(...)
```


#### `inline(proc, call_cursor)`
Inline the sub-procedure call. `call_cursor` is Cursor or pattern pointing to a Call statement whose body we want to inline
