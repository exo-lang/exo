# Quiz3!!

This quiz explores fixing subtle cursor navigation bugs.

## Correct Output
This code makes the optimization of shrinking the `blur_x` memory allocation from (H+2, W) to (34, 256). Since the code has been tiled, we don't need to store the entire intermediate `blur_x` buffer in memory. Instead, we can just reuse the same intermediate buffer for each tile.

To do so, the schedule tries to sink the allocation within the tile, reduce the memory size to the bare minimum necessary for computing that tile, and then lift the allocation back up to the top level scope.
```python
def tile_and_fused_blur(W: size, H: size, blur_y: ui16[H, W] @ DRAM,
                        inp: ui16[H + 2, W + 2] @ DRAM):
    assert H % 32 == 0
    assert W % 256 == 0
    blur_x: ui16[34, 256] @ DRAM
    for yo in seq(0, H / 32):
        for xo in seq(0, W / 256):
            for yi in seq(0, 34):
                for xi in seq(0, 256):
                    blur_x[yi + 32 * yo - 32 * yo, xi + 256 * xo - 256 *
                           xo] = (inp[yi + 32 * yo, xi + 256 * xo] +
                                  inp[yi + 32 * yo, 1 + xi + 256 * xo] +
                                  inp[yi + 32 * yo, 2 + xi + 256 * xo]) / 3.0
            for yi in seq(0, 32):
                for xi in seq(0, 256):
                    blur_y[yi + 32 * yo, xi +
                           256 * xo] = (blur_x[yi + 32 * yo - 32 * yo,
                                               xi + 256 * xo - 256 * xo] +
                                        blur_x[1 + yi + 32 * yo - 32 * yo,
                                               xi + 256 * xo - 256 * xo] +
                                        blur_x[2 + yi + 32 * yo - 32 * yo,
                                               xi + 256 * xo - 256 * xo]) / 3.0
```

## Incorrect Output
This output is partially correct: it manages to reduce the height dimension from `H+2` to `34`. However, it fails to reduce the memory usage in the width direction.
```python
def tile_and_fused_blur(W: size, H: size, blur_y: ui16[H, W] @ DRAM,
                        inp: ui16[H + 2, W + 2] @ DRAM):
    assert H % 32 == 0
    assert W % 256 == 0
    blur_x: ui16[34, W] @ DRAM
    for yo in seq(0, H / 32):
        for xo in seq(0, W / 256):
            for yi in seq(0, 34):
                for xi in seq(0, 256):
                    blur_x[yi + 32 * yo - 32 * yo, xi + 256 *
                           xo] = (inp[yi + 32 * yo, xi + 256 * xo] +
                                  inp[yi + 32 * yo, 1 + xi + 256 * xo] +
                                  inp[yi + 32 * yo, 2 + xi + 256 * xo]) / 3.0
            for yi in seq(0, 32):
                for xi in seq(0, 256):
                    blur_y[yi + 32 * yo, xi + 256 * xo] = (
                        blur_x[yi + 32 * yo - 32 * yo, xi + 256 * xo] +
                        blur_x[1 + yi + 32 * yo - 32 * yo, xi + 256 * xo] +
                        blur_x[2 + yi + 32 * yo - 32 * yo,
                               xi + 256 * xo]) / 3.0
```

---

## Solution

To understand the bug, let's insert print statements in these places:

```python
print(xo_loop)
loops_to_lower_allocation_into = get_loops_at_or_above(xo_loop)
for i, loop in enumerate(loops_to_lower_allocation_into):
    print(i, loop)
    ...
```

The `xo_loop` points to:
```python
for yo in seq(0, H / 32):
    for xo in seq(0, W / 256):  # <-- NODE
        ...
```

And the first (and only) iteration of the `loop` points to:
```python
for yo in seq(0, H / 32):  # <-- NODE
    for xo in seq(0, W / 256):
      ...
```

This reveals that the implementation of `get_loops_at_or_above` has a bug because it only contains "loops above" the `xo_loop` (which is `yo` loop), not including the `xo_loop` itself.

To fix this bug, change `loops = []` to `loops = [cursor]` in the implementation of `get_loops_at_or_above`.
