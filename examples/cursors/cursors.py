from __future__ import annotations
from exo import *
from exo.API_scheduling import *

"""
Cursor Example

This example introduces the concept of Cursors in Exo 2 paper and demonstrates
how to use scheduling operators with them to manipulate loops and optimize code.

Cursors allow you to select and refer to parts of the code such as expressions,
statements, and code blocks. They also support spatial navigation within a procedure
to proximate locations.

Key concepts covered:
- Finding cursors using patterns
- Navigating using cursors
- Applying scheduling primitives with cursors
- Forwarding cursors after transformations
"""


"""
1: Basic loop example using Exo 2

GEMV kernel: y = A * x
Args:
    M (size): Number of rows in matrix A
    N (size): Number of columns in matrix A
    A (tensor): M x N matrix stored in DRAM
    x (tensor): N-dimensional vector stored in DRAM
    y (tensor): M-dimensional vector stored in DRAM
"""


@proc
def gemv(M: size, N: size, A: f32[M, N], x: f32[N], y: f32[M]):
    assert M % 8 == 0
    assert N % 8 == 0

    for i in seq(0, M):
        for j in seq(0, N):
            y[i] += A[i, j] * x[j]


print("1: Original GEMV kernel")
print(gemv)
print()


"""
2: Finding cursors
"""
# Find a cursor to the i loop by name
i_loop = gemv.find_loop("i")

# Find the same i loop by pattern
i_loop2 = gemv.find("for i in _: _")

# Check that two cursors are pointing to the same 'i' loop
assert i_loop == i_loop2

print("2: i_loop points to:")
print(i_loop)
print()


"""
3: Navigating with cursors
"""
# Find cursors to key parts of the code
j_loop = i_loop.body()[0]  # j is the only statement in i's body
C_store = j_loop.body()[0]  # y[i] = ... is the only statement in j's body
j_loop_parent = j_loop.parent()  # The parent of the j loop

# Check that j_loop's parent is indeed pointing to the i_loop
assert i_loop == j_loop_parent

print("3: j_loop points to:")
print(j_loop)
print()


"""
4: Applying scheduling primitives & Cursor forwarding
"""
# First, rename the gemv
g = rename(gemv, "gemv_scheduled")

# Divide the i loop by 8
g = divide_loop(g, i_loop, 8, ["io", "ii"], perfect=True)

# Divide the j loop by 8
g = divide_loop(g, j_loop, 8, ["jo", "ji"], perfect=True)

# Now, we want to reorder ii and jo loops, by lifting the scope of j_loop
# We can still use the j_loop cursor!
g1 = lift_scope(g, j_loop)
g2 = lift_scope(g, g.forward(j_loop))

# Assert that g1 and g2 are the same (`j_loop` is implicitly forwarded in the first line)
assert g1 == g2

print("4: Tiled gemv")
print(g1)
print("4: g.forward(j_loop) points to:")
print(g.forward(j_loop))
print()


"""
5: Defining a new scheduling operator
"""


def tile_2D(p, i_lp, j_lp, i_itrs, j_itrs, i_sz, j_sz):
    """
    Perform a 2D tiling of the i and j loops.
    Args:
        p: Procedure to be tiled
        i_lp: Name of the i loop
        j_lp: Name of the j loop
        i_itrs: New iterators for the i loop
        j_itrs: New iterators for the j loop
        i_sz: Tile size for the i loop
        j_sz: Tile size for the j loop
    """
    p = divide_loop(p, i_lp, i_sz, i_itrs, perfect=True)
    p = divide_loop(p, j_lp, j_sz, j_itrs, perfect=True)
    p = lift_scope(p, j_itrs[0])
    return p


# Example usage of tile_2D to perform 2D tiling on the gemv kernel.
final_g = tile_2D(gemv, i_loop, j_loop, ["io", "ii"], ["jo", "ji"], 8, 8)

print("5: tile_2D applied gemv:")
print(final_g)


__all__ = ["final_g"]
