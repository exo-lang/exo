from __future__ import annotations

from exo import *
from exo.stdlib.scheduling import *


@proc
def scaled_add(N: size, a: f32[N], b: f32[N], c: f32[N]):
    assert N % 8 == 0
    for i in seq(0, N):
        c[i] = 2 * a[i] + 3 * b[i]


def stage_exprs(p, num_vectors, assign):
    if isinstance(assign.rhs(), BinaryOpCursor):
        p = bind_expr(p, assign.rhs().lhs(), "vec")
        num_vectors += 1
        p, num_vectors = stage_exprs(p, num_vectors, p.forward(assign).prev())

        p = bind_expr(p, assign.rhs().rhs(), "vec")
        num_vectors += 1
        p, num_vectors = stage_exprs(p, num_vectors, p.forward(assign).prev())
    return p, num_vectors


def wrong_schedule(p):
    p = rename(p, "scaled_add_scheduled")
    num_vectors = 0

    p = divide_loop(p, "i", 8, ["io", "ii"], perfect=True)

    p, num_vectors = stage_exprs(p, num_vectors, p.find("c[_] = _"))

    for i in range(num_vectors):
        vector_reg = p.find(f"vec: _ #{i}")
        p = expand_dim(p, vector_reg, 8, "ii")
        p = lift_alloc(p, vector_reg)

        vector_assign = p.find(f"vec = _ #{i}")
        p = fission(p, vector_assign.after())

    return p


w = wrong_schedule(scaled_add)
print(w)
