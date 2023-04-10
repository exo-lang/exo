from __future__ import annotations

from exo import *
from exo.platforms.neon import *
from exo.stdlib.scheduling import *

neon_instructions = [
    neon_zero_4xf32,
    neon_vfmadd_4xf32_1xf32,
    neon_vld_4xf32,
    neon_vst_4xf32,
]


@proc
def filter1D(ow: size, kw: size, x: f32[ow + kw - 1], y: f32[ow], w: f32[kw]):
    for o in seq(0, ow):
        y[o] = 0.0
        for k in seq(0, kw):
            y[o] += x[o + k] * w[k]


# divide
filter1D = divide_loop(filter1D, "o", 4, ["outXo", "outXi"], tail="cut_and_guard")

# stage sum
filter1D = simplify(
    stage_mem(filter1D, "for outXi in _:_", "y[4*outXo:4*outXo+4]", "sum")
)
filter1D = fission(filter1D, filter1D.find("sum[_] = 0.0").after())
filter1D = reorder_loops(filter1D, "outXi k")

# stage x
filter1D = simplify(
    stage_mem(filter1D, "for outXi in _:_ #1", "x[k+4 * outXo: k+4*outXo + 4]", "xX4")
)

# set memories & precision
filter1D = set_memory(filter1D, "sum", Neon)
filter1D = set_memory(filter1D, "xX4", Neon)

# replace
filter1D = replace_all(filter1D, neon_instructions)

print(filter1D)

__all__ = ["filter1D"]
