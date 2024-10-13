# Cursors

Throughout this document, we use `p` for Exo procedures and `c` for Exo Cursors.

## Pattern match a Cursor

### On Exo Procedures

- `p.find(pattern, many=False)`: Finds a cursor for the given pattern. If `many=True`, returns a list of all cursors matching the pattern.
- `p.find_loop(loop_pattern, many=False)`: Finds a cursor pointing to a loop. Similar to `p.find(...)`, but if the supplied pattern is of the form 'name' or 'name #n', it will be expanded to `for name in _:_`. (e.g., `p.find_loop("i")` will find a loop `i`)
- `p.find_alloc_or_arg(buf_name)`: Finds an allocation or argument cursor. will get expanded to `buf_name: _`.
- `p.find_all(pattern)`: Finds a list of all cursors matching the pattern. Shorthand for `p.find(pattern, many=True)`.

- `.args()`: Returns cursors to procedure arguments.
- `.body()`: Returns a BlockCursor selecting the entire body of the Procedure.

### On Cursors

- `c.find()`

### Pattern language

`p.find(...)` works on a given pattern. `_` denotes wildcard and will match against any statements.
Supported patterns include:
- `"for i in _:_"` will match on the loop `i`
- `"if i == 0:_"` or `"if _:_"` will match on the if statements with respective conditions
- `"a : i8"` or `"a : _"` will match on the allocation for a buffer `a`
- `"a = 3.0"` or `"a = _"` will match on the assignment for `a`
- `"a += 3.0"` or `"a += _"` will match on the reduction for `a`
- `"a = 3.0 ; b = 2.0"` will match against a block of size two


## Types of Cursors
- Stmt Cursors
- Gap Cursors
- Block Cursors

- Arg Cursors
arg cursors don't have navigation, and can be only obtained by `p.args()`

- Invalid Cursors
Invalid cursors are invalid! wtf

There should be navigation and inspection for each types of Cursors..

## Cursor navigation

- `c.next()` (returns a _node_ cursor)
```
s1 <- c
s2 <- c.next()
```

- `c.prev()` (returns a _node_ cursor)
```
s1 <- c.prev()
s2 <- c
```

- `c.before()` (returns a _gap_ cursor)
```
s1
   <- c.before()
s2 <- c
```

- `c.after()` (returns a _gap_ cursor)
```
s1
s2 <- c
   <- c.after()
```

Gap cursors are attached to the anchor statement cursor, so `c.before().before()` is not the same as `c.prev()` (Check!!)

- `c.anchor()`
- `c.parents()`
- `c.proc()`



## Cursor inspection

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


