from __future__ import annotations

from exo import *
from exo.platforms.neon import *
from exo.stdlib.scheduling import *


@proc
def filter1D(ow: size, kw: size, x: f32[ow + kw - 1], y: f32[ow], w: f32[kw]):
    for o in seq(0, ow):
        sum: f32
        sum = 0.0
        for k in seq(0, kw):
            sum += x[o + k] * w[k]
        y[o] = sum


# divide
filter1D = divide_loop(filter1D, "o", 4, ["outXo", "outXi"], tail="cut_and_guard")

# stage sum
filter1D = expand_dim(filter1D, "sum:_", "4", "outXi")
filter1D = autolift_alloc(filter1D, "sum:_")
filter1D = autofission(filter1D, filter1D.find("sum[_] = _").after())
filter1D = autofission(filter1D, filter1D.find("y[_] = _").before())

# stage x
filter1D = reorder_loops(filter1D, "outXi k")
filter1D = bind_expr(filter1D, "x[_]", "xX4")
filter1D = expand_dim(filter1D, "xX4:_", "4", "outXi")
filter1D = autolift_alloc(filter1D, "xX4:_")
filter1D = autofission(filter1D, filter1D.find("xX4[_] = _").after())

# set memories & precision
filter1D = set_memory(filter1D, "sum #1", Neon)
filter1D = set_memory(filter1D, "xX4", Neon)
filter1D = set_precision(filter1D, "xX4", "f32")

# replace
filter1D = replace(filter1D, "for outXi in _:_", neon_zero_4xf32)
filter1D = replace(filter1D, "for outXi in _:_", neon_vld_4xf32)
filter1D = replace(filter1D, "for outXi in _:_", neon_vfmadd_4xf32_1xf32)
filter1D = replace(filter1D, "for outXi in _:_", neon_vst_4xf32)

print(filter1D)

__all__ = ["filter1D"]
