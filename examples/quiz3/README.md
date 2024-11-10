# Quiz3!!

## Correct Output
This code makes the optimization of shrinking the `blur_x` memory allocation from (H+2, W) to (34, 256). Since the code has been tiled, we don't need to store the entire intermediate `blur_x` buffer in memory. Instead, we can just reuse the same intermediate buffer for each tile.

To do so, the schedule tries to sink the allocation within the tile, reduce the memory size to the bare minimum necessary for computing that tile, and then lift the allocation back up to the top level scope.
```
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
This output is partially correct: it manages to reduce the height dimension from H+2 to 34. However, it wasn't able to reduce the memory in the width direction.
```
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
Change `loops = []` to `loops = [cursor]`.

