
## Precision, memory and parallelism primitives

Consistency of precision types, memory annotations, and window annotations is performed as back-end checks after all scheduling is complete, immediately prior to code generation.
In contrast, all the other scheduling primitives are safety-checked within their rewrite process.

#### ` set_precision(proc, cursor, typ)`
Sets the precision annotation on a given buffer to the provided base-type precision.
```
args:
    name    - string w/ optional count, e.g. "x" or "x #3"
    typ     - string representing base data type

rewrite:
    name : _[...]
      -->
    name : typ[...]
```

#### `set_window(proc, cursor, is_window=True)`
Sets the annotation on a given buffer to indicate that it should be a window (True) or should not be a window (False).
```
args:
    name        - string w/ optional count, e.g. "x" or "x #3"
    is_window   - boolean representing whether a buffer is a window

rewrite when is_window = True:
    name : R[...]
      -->
    name : [R][...]
```

#### `set_memory(proc, cursor, memory_type)`
Sets the memory annotation on a given buffer to the provided memory.
```
args:
    name    - string w/ optional count, e.g. "x" or "x #3"
    mem     - new Memory object

rewrite:
    name : _ @ _
      -->
    name : _ @ mem
```

#### `parallelize_loop(proc, loop_cursor)`
Parallelizes the loop pointed by `loop_cursor`. Lowers to OpenMP by default.

