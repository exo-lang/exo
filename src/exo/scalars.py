# Public, optional exo.scalars module.
# Defines f16, f32, etc. objects of ScalarInfo type.
# Also exo_inf

from .core.LoopIR import (
    ScalarInfo,
    e4m3,
    e5m2,
    e8m0,
    bf16,
    f16,
    f32,
    f64,
    i8,
    ui8,
    ui16,
    i32,
)

# David Zhao Akeley 2026-03-17: magical infinity value
# Note, it's hard wired in various parsers and pretty-printers
# that this will always be named "inf". We added this years
# after Exo was originally created, so this is very fragile.
inf = float("inf")
