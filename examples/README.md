# Scheduling Example

Please [install Exo](https://github.com/exo-lang/exo#install-exo) before proceeding with this example.
This tutorial assumes some familiarity with SIMD instructions.

Exo provides *scheduling operators* to transform program and rewrite them to make use of complex hardware instructions.
We'll show you how to take a matrix multiplication kernel and transform it into an implementation that can make use of [AVX2](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions) vector instructions.

> The complete code, with the scheduling operations commented out, can be found in `exo/examples/x86_matmul.py`.

## Basic Implementation

To start off, let's implement a basic matrix multiplication kernel in Exo:
```py
from __future__ import annotation
from exo import *

@proc
def rank_k_reduce_6x16(K: size, C: f32[6, 16] @ DRAM, A: f32[6, K] @ DRAM,
                       B: f32[K, 16] @ DRAM):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]

print(rank_k_reduce_6x16)
```

This implements matrix multiplication between a $$ 6\times K $$ and a $$ K \times 16 $$.

However, the program does not take advantage of AVX2 instructions yet, and it is not obvious whether a vectorizing compiler can automatically discover the right way to parallelize this program.

## Scheduling Walkthrough

Scheduling plays a central role in Exo's machinery to generate high-performance kernels.
Instead of relying on automated compiler passes, we can specify *program rewrites* that allow Exo to generate high-performance code.

Looking at our kernel, we can see that the contraction dimension `k` is amenable to streaming.
This means that we want to perform vectorized computation using the `i` and `j` iterators.
At high-level, we're going to perform the following set of rewrites to enable our vectorized computation:
- Reorder to the loops to expose streaming behavior
- Decompose the loading and storing of the output
- Vectorize the inner loop computation

While doing this, we also have to contend with the restriction that the AVX2 instruction set exposes *16 vector registers* which means if our computation attempts to use any more than that, we'll have register spillage and lose out on the performance.

### Reordering Loops

The first step is reordering the loops in our program so that the streaming nature of the computation is better expressed.
We can do this using the `reorder_loops`.
Also, just to keep things easy to follow, we're going to rename our kernel to `rank_k_reduce_6x16_scheduled`:

First, let's import the scheduling primitives at the top of the file:
```py
from exo.stdlib.scheduling import *
```

Next, we can add the *scheduling commands* which act upon a give kernel and return a new kernel for us. Kernels in Exo are also called procs so we'll use those names interchangeably:
```py
avx = rename(rank_k_reduce_6x16, "rank_k_reduce_6x16_scheduled")
avx = reorder_loops(avx, 'j k')
avx = reorder_loops(avx, 'i k')

print(avx)
```

The `rename` command is straightforward: it renamed our proc.
The `reorder_loops` command is more interesting, it takes a pattern or a *cursor* to the loops that should be reordered.
For example, the pattern `j k` is the same as:
```
for j in _: _
  for k in _: _
```
This tells Exo to find a program fragment that matches those two, immediately nested loop nests, and reorder them.
The `j k` is a shorthand syntax for exactly that pattern.

Finally, the `print(avx)` shows us the resulting program's loop nests. Note that they have been reordered!
```
...
    for k in seq(0, K):
        for i in seq(0, 6):
            for j in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]

```

### Vectorizing the Output

The reordered loops let us better see the opportunity to expose vectorizing in our program.
At a high-level, we produce our outputs as a $$ 6\times 16 $$ matrix which can be represented by 12, 8-wide vectors.
Even though we're streaming in the `k` dimension, we know that the output will always be this size, so it is useful to allocate these registers and directly perform the computation on them.

To do this, we will use some more complicated scheduling operations in Exo. We encourage you to step through the transformation done by each operation by printing out `avx`:

```py
avx = divide_loop(avx, 'for j in _: _', 8, ['jo', 'ji'], perfect=True)
avx = stage_mem(avx, 'for k in _:_', 'C[0:6, 0:16]', 'C_reg')
avx = simplify(avx)
```

We perform three transformations:
- `divide_loop` splits the innermost `j` loop into two loops so that we have a `for _ in seq(0, 8)` which represents the size of our vectors.
- `stage_mem` replaces the use of the output memory `C` with `C_reg` and generates loops to load and store values from and to the memory.
- `simplify` simplifies simple constant expressions

Note that in the result, we have a new memory `C_reg: f32[6, 16] @ DRAM`.
This is not quite in the shape we want; a vector register should have a size of 8 so that we can map it to the AVX2 instructions.
The next set of transformations will address this:

```py
avx = divide_dim(avx, 'C_reg:_', 1, 8)
avx = repeat(divide_loop)(avx, 'for i1 in _: _', 8, ['i2', 'i3'], perfect=True)
avx = simplify(avx)
```

The `divide_dim` operation splits the last dimension of `C_reg` into two dimensions the latter of which has 8 elements.
Next, we use the `divide_loop` operator to split apart the loops that operate on the memory `C_reg` and see our first *higher-order scheduling operator* `repeat` which applies a scheduling operator till no new matches are found.
The final `simplify` simplifies the index expressions.

These changes give us a couple of loop nests amenable for mapping onto vector instructions:
```py
    ...
            for i3 in seq(0, 8):
                C_reg[i0, i2, i3] = C[i0, i3 + 8 * i2]
    ...
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg[i, jo, ji] += A[i, k] * B[k, ji + 8 * jo]
    ...
            for i3 in seq(0, 8):
                C[i0, i3 + 8 * i2] = C_reg[i0, i2, i3]
```

In order of appearance, they perform a load from `C` into `C_reg`, performs the computation on `C_reg`, and store the results into `C` from `C_reg`.
The second loop nest cannot be vectorized just yet but the other two are vectorizable.

### Instruction Mapping

Exo support *instruction mapping* which takes a particular program fragment and replaces it with an equivalent instruction.
For example, we can take the following loop nest:
```py
for i3 in seq(0, 8):
    C_reg[i0, i2, i3] = C[i0, i3 + 8 * i2]
```
And turn it into the AVX2 `mm256_loadu_ps`.

To do this, we import the AVX2 instructions and use the `replace_all` operator to replace all matching loop nests:
```py
from exo.platforms.x86 import *
...
avx = set_memory(avx, 'C_reg:_', AVX2)
avx = replace_all(avx, mm256_loadu_ps)
avx = simplify(avx)
print(avx)
```

This transforms the above loop nest into:
```
mm256_loadu_ps(C_reg[i0, i2, 0:8], C[i0, 8 * i2:8 + 8 * i2])
```

The `set_memory` operator marks the `C_reg` memory as an AVX2 vector register explicitly and `replace_all` attempts to rewrite all loops in the code that implement a load into the `mm256_loadu_ps` instruction.

The latter is a bit magical! How does the scheduling operator know what the semantics of the instruction are and when it is safe to rewrite loops to the instructions?
This is the final part of Exo's magic: the definitions of these instructions are *externalized*, i.e., provided by you:
```py
@instr("{dst_data} = _mm256_loadu_ps(&{src_data});")
def mm256_loadu_ps(dst: [f32][8] @ AVX2, src: [f32][8] @ DRAM):
    assert stride(src, 0) == 1
    assert stride(dst, 0) == 1

    for i in seq(0, 8):
        dst[i] = src[i]
```
The definition implements the semantics of the instruction using plain old python code and the `replace_all` command knows how to replace them using this definition.

### Take a Breather

Congratulations on getting through a whirlwind tour of Exo's capabilities. To review, we've seen a couple of concepts that work in tandem to make enable productive performance engineering:
- *Cursors* provide a way to talk about a particular program fragment.
- *Scheduling operators* allow you to rewrite programs.
- *Instruction mapping* uses user-level instruction definitions to rewrite program fragments to backend instructions.

Next, please uncomment the code in the first block by deleting the multi-line string
markers (`"""`). Now, you will see that `stage_mem()` stages `C` to a buffer
called `C_reg`.

```python
# First block:
def rank_k_reduce_6x16_scheduled(K: size, C: f32[6, 16] @ DRAM,
                                 A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C_reg: f32 @ AVX2
                C_reg = C[i, j]
                C_reg += A[i, k] * B[k, j]
                C[i, j] = C_reg
```

Please uncomment the code in the second block. You will see that the `j` loop
is `divide_loop()` into two loops `jo` and `ji`, and loops are `reorder_loops()`ed so that the `k`
loop becomes outermost.

```python
# Second block:
def rank_k_reduce_6x16_scheduled(K: size, C: f32[6, 16] @ DRAM,
                                 A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM):
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg: f32 @ AVX2
                    C_reg = C[i, 8 * jo + ji]
                    C_reg += A[i, k] * B[k, 8 * jo + ji]
                    C[i, 8 * jo + ji] = C_reg
```

Please uncomment the code in the third block. Please notice that

- The allocation of `C_reg` is lifted by `autolift_alloc()`
- `C_reg` initialization, reduction, and write back are `autofission()`ed into three
  separate blocks.

```python
# Third block:
def rank_k_reduce_6x16_scheduled(K: size, C: f32[6, 16] @ DRAM,
                                 A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM):
    C_reg: f32[1 + K, 6, 2, 8] @ AVX2
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg[k, i, jo, ji] = C[i, ji + 8 * jo]
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg[k, i, jo, ji] += A[i, k] * B[k, ji + 8 * jo]
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C[i, ji + 8 * jo] = C_reg[k, i, jo, ji]
```

Please uncomment the code in the fourth block. `A` is bound to 8 wide AVX2 vector
register `a_vec` by `bind_expr()`.

```python
# Fourth block:
def rank_k_reduce_6x16_scheduled(K: size, C: f32[6, 16] @ DRAM,
                                 A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM):
    C_reg: f32[1 + K, 6, 2, 8] @ AVX2
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg[k, i, jo, ji] = C[i, ji + 8 * jo]
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                a_vec: R[8] @ AVX2
                for ji in seq(0, 8):
                    a_vec[ji] = A[i, k]
                for ji in seq(0, 8):
                    C_reg[k, i, jo, ji] += a_vec[ji] * B[k, ji + 8 * jo]
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C[i, ji + 8 * jo] = C_reg[k, i, jo, ji]
```

Please uncomment the code in the fifth block. The same schedule for `A` is applied
to `B`.

```python
# Fifth block:
def rank_k_reduce_6x16_scheduled(K: size, C: f32[6, 16] @ DRAM,
                                 A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM):
    C_reg: f32[1 + K, 6, 2, 8] @ AVX2
    for k in seq(0, K):
        for i in seq(1, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg[k, i, jo, ji] = C[i, ji + 8 * jo]
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                a_vec: R[8] @ AVX2
                for ji in seq(0, 8):
                    a_vec[ji] = A[i, k]
                b_vec: R[8] @ AVX2
                for ji in seq(0, 8):
                    b_vec[ji] = B[k, ji + 8 * jo]
                for ji in seq(0, 8):
                    C_reg[k, i, jo, ji] += a_vec[ji] * b_vec[ji]
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C[i, ji + 8 * jo] = C_reg[k, i, jo, ji]
```

Finally, please uncomment the sixth block. The sixth block replaces the statements with
equivalent calls to AVX2 instructions. `set_memory()` sets `C_reg`, `a_vec`, and `b_vec`'s
memory to AVX2 to use it as an AVX vector, which is denoted by `@ AVX2`.
These AVX2 hardware instructions could be defined
by users, but are part of Exo's standard library; the sources may be
found [here](https://github.com/ChezJrk/exo/blob/master/src/exo/platforms/x86.py#L8).
Please look at the definition of `mm256_loadu_ps` (for example), and notice that it has
a similar structure to the first `ji` loop in the fifth block. We will replace the
statement with the call to AVX2 instruction procedures to get the final schedule.

```python
# Sixth block:
def rank_k_reduce_6x16_scheduled(K: size, C: f32[6, 16] @ DRAM,
                                 A: f32[6, K] @ DRAM, B: f32[K, 16] @ DRAM):
    C_reg: f32[1 + K, 6, 2, 8] @ AVX2
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                mm256_loadu_ps(C_reg[k + 0, i + 0, jo + 0, 0:8],
                               C[i + 0, 8 * jo + 0:8 * jo + 8])
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                a_vec: R[8] @ AVX2
                mm256_broadcast_ss(a_vec, A[i + 0, k + 0:k + 1])
                b_vec: R[8] @ AVX2
                mm256_loadu_ps(b_vec[0:8], B[k + 0, 8 * jo + 0:8 * jo + 8])
                mm256_fmadd_ps(C_reg[k + 0, i + 0, jo + 0, 0:8], a_vec, b_vec)
    for k in seq(0, K):
        for i in seq(0, 6):
            for jo in seq(0, 2):
                mm256_storeu_ps(C[i + 0, 8 * jo + 0:8 * jo + 8],
                                C_reg[k + 0, i + 0, jo + 0, 0:8])
```

We suggest you to attempt the following exercise:

- Modify the original algorithm so that the `k` loop becomes outermost. Adjust the
  scheduling operations so that the resulting code matches the output of the sixth
  block.

## Compiling

Finally, the code can be compiled and run on your machine if you have AVX2 instructions.
We provided a main function in `main.c` to call these procedures and to time them.
Please run `make` or compile manually:

```
$ exocc -o . --stem avx2_matmul x86_matmul.py
$ gcc -o avx2_matmul -march=native main.c avx2_matmul.c
```

It should generate something like:

```
$ ./avx2_matmul
Time taken for original matmul: 0 seconds 649 milliseconds
Time taken for scheduled matmul: 0 seconds 291 milliseconds
```

Even on this small example, we can see the benefit of AVX2 instructions.
