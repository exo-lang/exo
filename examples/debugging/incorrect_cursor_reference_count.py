from __future__ import annotations

from exo import *
from exo.libs.memories import AVX2
from exo.stdlib.scheduling import *


@proc
def scaled_add(N: size, a: f32[N], b: f32[N], c: f32[N]):
    assert N % 8 == 0
    for i in seq(0, N):
        c[i] = 2 * a[i] + 3 * b[i]


def correct_schedule(p):
    num_vectors = 0

    def stage_exprs(assign):
        nonlocal num_vectors
        nonlocal p

        if isinstance(assign.rhs(), BinaryOpCursor):
            p = bind_expr(p, assign.rhs().lhs(), "vec")
            num_vectors += 1
            stage_exprs(p.forward(assign).prev())

            p = bind_expr(p, assign.rhs().rhs(), "vec")
            num_vectors += 1
            stage_exprs(p.forward(assign).prev())

    p = divide_loop(p, "i", 8, ["io", "ii"], perfect=True)

    stage_exprs(p.find("c[_] = _"))

    for i in range(num_vectors):
        vector_reg = p.find(f"vec: _ #{i}")
        p = set_memory(p, vector_reg, AVX2)
        p = expand_dim(p, vector_reg, 8, "ii")
        p = lift_alloc(p, vector_reg)

    for i in range(num_vectors):
        vector_assign = p.find(f"vec = _ #{i}")
        p = fission(p, vector_assign.after())

    return p


def wrong_schedule(p):
    num_vectors = 0

    def stage_exprs(assign):
        nonlocal num_vectors
        nonlocal p

        if isinstance(assign.rhs(), BinaryOpCursor):
            p = bind_expr(p, assign.rhs().lhs(), "vec")
            num_vectors += 1
            stage_exprs(p.forward(assign).prev())

            p = bind_expr(p, assign.rhs().rhs(), "vec")
            num_vectors += 1
            stage_exprs(p.forward(assign).prev())

    p = divide_loop(p, "i", 8, ["io", "ii"], perfect=True)

    stage_exprs(p.find("c[_] = _"))

    for i in reversed(range(num_vectors)):
        vector_reg = p.find(f"vec: _ #{i}")
        p = set_memory(p, vector_reg, AVX2)
        p = expand_dim(p, vector_reg, 8, "ii")
        p = lift_alloc(p, vector_reg)

    for i in range(num_vectors):
        vector_assign = p.find(f"vec = _ #{i}")
        p = fission(p, vector_assign.after())

    return p


print(correct_schedule(scaled_add))
print(wrong_schedule(scaled_add))
