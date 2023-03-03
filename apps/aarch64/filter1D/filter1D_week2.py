from __future__ import annotations

from exo import *
from exo.platforms.neon import *
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *


class Neon:
    mem = Neon4f
    vec_width = 4
    instructions = [
        neon_zero_4xf32,
        neon_vfmadd_4xf32_1xf32,
        neon_vld_4xf32,
        neon_vst_4xf32,
    ]


class AVX2:
    mem = AVX2
    vec_width = 8
    instructions = [
        mm256_setzero_ps,
        mm256_fmadd_ps_broadcast,
        mm256_loadu_ps,
        mm256_storeu_ps,
    ]


@proc
def filter1D(ow: size, kw: size, x: f32[ow + kw - 1], y: f32[ow], w: f32[kw]):
    for o in seq(0, ow):
        y[o] = 0.0
        for k in seq(0, kw):
            y[o] += x[o + k] * w[k]


arch = Neon

VW = arch.vec_width

# divide
filter1D = divide_loop(filter1D, "o", VW, ["outXo", "outXi"], tail="cut_and_guard")

# stage sum
filter1D = simplify(
    stage_mem(filter1D, "for outXi in _:_", f"y[{VW}*outXo:{VW}*outXo+{VW}]", "sum")
)
filter1D = fission(filter1D, filter1D.find("sum[_] = 0.0").after())
filter1D = reorder_loops(filter1D, "outXi k")

# stage x
filter1D = simplify(
    stage_mem(
        filter1D, "for outXi in _:_#1", f"x[k+{VW}*outXo:k+{VW}*outXo+{VW}]", f"xX{VW}"
    )
)

# set memories & precision
filter1D = set_memory(filter1D, "sum", arch.mem)
filter1D = set_memory(filter1D, f"xX{VW}", arch.mem)

# replace
filter1D = replace_all(filter1D, arch.instructions)

print(filter1D)

__all__ = ["filter1D"]
