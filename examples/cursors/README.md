# Cursor step-by-step example!

"""
Exo Cursor Tutorial
===================

This tutorial demonstrates how to use cursors in the Exo scheduling language
to navigate and transform Exo object code.

Cursors allow you to select and refer to parts of the code such as expressions,
statements, and code blocks. They support spatial navigation within a procedure
to proximate locations.

Key concepts covered:
- Finding cursors using patterns
- Navigating using cursors
- Applying scheduling primitives with cursors
- Forwarding cursors after transformations

Example 1: Finding cursors
--------------------------
"""

# Assume we have an Exo procedure p with this loop nest:
# for i in seq(0, n):
#   for j in seq(0, m):
#     C[i,j] = A[i,k] * B[k,j]

# Find a cursor to the i loop by name
i_loop = p.find_loop('i')

# Find the same i loop by pattern
i_loop2 = p.find('for i in _: _')
assert i_loop == i_loop2

"""
Example 2: Navigating with cursors
----------------------------------
"""
# Find cursors to key parts of the code
j_loop = i_loop.body()[0]   # j is the only statement in i's body
C_store = j_loop.body()[0]  # C[i,j] = ... is the only statement in j's body
A_load = C_store.rhs().lhs() # A[i,k] in the RHS of the C[i,j] = ... statement
i_loop_parent = i_loop.parent() # The parent scope of the i loop

"""
Example 3: Applying scheduling primitives
-----------------------------------------
"""
# Divide the i loop by 4
p = divide_loop(p, i_loop, 4, ['io','ii'], perfect=True)

# Reorder the j loop to before the ii loop
p = reorder_loops(p, [j_loop, ii_loop])

"""
Example 4: Forwarding cursors
-----------------------------
"""
# After dividing the i loop, the original i_loop cursor is invalid
# We need to "forward" the cursor to the new procedure
with proc.undo():
  # Undo puts i_loop back in a valid state
  assert i_loop.is_valid()

  # Divide the i loop again
  p = divide_loop(p, i_loop, 4, ['io','ii'], perfect=True)

assert not i_loop.is_valid() # No longer valid after divide_loop

i_loop = p.forward(i_loop) # Forward the cursor to the new proc
assert i_loop.is_valid() # Now valid again in new proc

# Additional navigation is done relative to the new proc
ii_loop = i_loop.body()[1]

"""
This covers the key cursor concepts from the Exo 2 paper. Cursors
enable powerfully composable ways to refer to and transform code!
"""






To create an Exo Cursor tutorial in Python using the code examples from the paper, here's a Python file outline with documentation and code examples inspired by the paper's description of Exo 2.

```python
"""
Exo Cursor Tutorial - Python Version

This tutorial introduces the concept of Cursors in Exo 2 and demonstrates
how to use scheduling operators with them to manipulate loops and optimize code.

Cursors in Exo allow you to refer to parts of code by their structure or name
and perform scheduling operations such as loop tiling and vectorization.

"""

# Example 1: Basic loop example using Exo 2

def gemv(M: int, N: int, A: list, x: list, y: list):
    """
    GEMV kernel: y = A * x
    Args:
        M (int): Number of rows in matrix A
        N (int): Number of columns in matrix A
        A (list): M x N matrix stored in DRAM
        x (list): N-dimensional vector stored in DRAM
        y (list): M-dimensional vector stored in DRAM
    """
    # Ensure dimensions are multiples of 8
    assert M % 8 == 0
    assert N % 8 == 0

    for i in range(M):
        for j in range(N):
            y[i] += A[i][j] * x[j]

# Now we perform some scheduling operations

def schedule_gemv(gemv):
    """
    Example scheduling of the gemv function using Exo 2-style transformations.
    We will tile the loops to improve cache locality.
    """
    # Divide the 'i' loop into two: io (outer loop) and ii (inner loop)
    g = divide_loop(gemv, 'i', 8, ['io', 'ii'], perfect=True)

    # Divide the 'j' loop similarly
    g = divide_loop(g, 'j', 8, ['jo', 'ji'], perfect=True)

    # Lift the 'jo' loop outside
    g = lift_scope(g, 'jo')

    return g

# Cursors example

def cursor_example():
    """
    Example of how to use cursors in Exo 2 to locate loops
    and apply transformations.
    """
    # Define gemv kernel
    g = gemv

    # Find the 'i' loop
    loop_0 = g.find_loop('i')  # Find by name
    loop_1 = g.find('for i in _: _')  # Find by pattern

    # Verify both references point to the same loop
    assert(loop_0 == loop_1)

    # Now we can apply scheduling to this loop
    g = divide_loop(g, loop_0, 8, ['io', 'ii'], perfect=True)

    return g

# Helper function for tiling
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

# Example of using tile_2D function
def tile_example():
    """
    Example usage of tile_2D to perform 2D tiling on the gemv kernel.
    """
    g = gemv
    g = tile_2D(g, 'i', 'j', ['io', 'ii'], ['jo', 'ji'], 8, 8)
    return g


if __name__ == "__main__":
    # Original GEMV kernel
    gemv(8, 8, [[0.5 for _ in range(8)] for _ in range(8)], [0.5 for _ in range(8)], [0 for _ in range(8)])

    # Apply scheduling
    scheduled_gemv = schedule_gemv(gemv)

    # Run example cursor operations
    cursor_example()

    # Run tiling example
    tile_example()
```

### Key Points:
- **`gemv`**: Implements the original matrix-vector multiplication.
- **`schedule_gemv`**: Demonstrates basic loop tiling for better performance.
- **`cursor_example`**: Shows how to find loops using cursors and apply transformations.
- **`tile_2D`**: A helper function to generalize the 2D tiling operation.
- **`tile_example`**: Applies 2D tiling on the `gemv` kernel.

This is a basic tutorial to demonstrate Exo Cursor usage and scheduling optimizations. You can expand it with more complex optimizations based on the examples from the paper【5†source】.
