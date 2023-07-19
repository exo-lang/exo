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

However, these microkernels function as the inner loops of highly optimized linear algebra computations. For example, [BLIS][] (an open-source [BLAS][] library) is architected around re-implementing such microkernels for each new target architecture that they support. The goal of Exo is to make this specialization process dramatically easier.

For our example, we want to specialize the kernel to use the AVX2 instructions; it is likely the case that a vectorizing compiler cannot automatically transform this kernel.

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

The `rename` command is straightforward: it renamed our proc. Most of the time, we want access to both our original kernel and the optimized kernel so we recommend rename them.
The `reorder_loops` command is more interesting, it takes a pattern or a *cursor* to the loops that should be reordered.
For example, the pattern `j k` is the same as:
```py
for j in _:
  for k in _: _
```
This tells Exo to find a program fragment that matches those two, immediately nested loop nests, and reorder them.
The `j k` is a shorthand syntax for exactly that pattern.

Finally, the `print(avx)` shows us the resulting program's loop nests. Note that they have been reordered!
```py
...
    for k in seq(0, K):
        for i in seq(0, 6):
            for j in seq(0, 16):
                C[i, j] += A[i, k] * B[k, j]

```

> When scheduling a new program, we often leave the `print(...)` command at the bottom and keep running the program to the see the output of the last scheduling step.

### Vectorizing the Output

The reordered loops let us better see the opportunity to expose vectorizing in our program.
At a high-level, we produce our outputs as a $6\times 16$ matrix which can be stored in 12, 8-wide vectors.
Since the size of the `k` dimension is unknown, we have to keep iterating on it, but we can make use of a register blocking strategy to vectorize our computation.

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
Next, we use the `divide_loop` operator to split apart the loops that operate on the memory `C_reg` and see our first *higher-order scheduling operator* `repeat` which applies a scheduling operator till the scheduling operation fails.
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
```py
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

### Vectorizing the Computation

Next, we're going to vectorize the innermost computation. However, we have to work with our original constraint: AVX2 exposes 16 vector registers, and we've consumed 12 of those for our output memory. The rest of computation needs to be staged carefully so that we don't end up taking more than 4 registers.

The scheduling will follow a similar pattern to the previous sections: we want to stage memories `A` and `B` using vector registers and replace their uses from the computational kernel.

Let's start off with `B` which is the larger of the two:
```py
# B is easy, it is just two vector loads
avx = stage_mem(avx, 'for i in _:_', 'B[k, 0:16]', 'B_reg')
avx = simplify(avx)
avx = divide_loop(avx, 'for i0 in _: _ #1', 8, ['io', 'ii'], perfect=True)
avx = divide_dim(avx, 'B_reg:_', 0, 8)
avx = set_memory(avx, 'B_reg:_', AVX2)
avx = simplify(avx)
avx = replace_all(avx, mm256_loadu_ps)
avx = simplify(avx)
print(avx)
```

We'll not be going into the details of each scheduling operate since you've already seen all of them before, but we encourage you to step through them and printing out `avx` after each operation.

The rewritten program exposes the reuse pattern available for the data in `B`:
```py
...
    for k in seq(0, K):
        B_reg: f32[2, 8] @ AVX2
        for io in seq(0, 2):
            mm256_loadu_ps(B_reg[io, 0:8], B[k, 8 * io:8 + 8 * io])
        for i in seq(0, 6):
            for jo in seq(0, 2):
                for ji in seq(0, 8):
                    C_reg[i, jo, ji] += A[i, k] * B_reg[jo, ji]
```
For each `k` value, we get to load 16 values from `B` (two vector register's worth) and perform the computation using those.

Next, we need to stage `A`:
```py
avx = bind_expr(avx, 'A[i, k]', 'A_reg')
avx = expand_dim(avx, 'A_reg', 8, 'ji')
avx = lift_alloc(avx, 'A_reg', n_lifts=2)
avx = fission(avx, avx.find('A_reg[ji] = _').after(), n_lifts=2)
avx = remove_loop(avx, 'for jo in _: _')
avx = set_memory(avx, 'A_reg:_', AVX2)
avx = replace_all(avx, mm256_broadcast_ss)
print(avx)
```

Staging `A` is a little more complex because unlike `C` and `B`, its reuse pattern is different: each value of `A` is broadcast into `A_reg` which is then used to perform the innermost computation. There are a couple of new scheduling operators:
- `lift_alloc`: Move an variable definition through the specified number of loops.
- `fission`: Splits apart the loop using the given cursor.
- `remove_loop`: Eliminates an unused loop.

Finally, we can vectorize the computation:
```py
avx = replace_all(avx, mm256_fmadd_ps)
print(avx)
```
This is perhaps a bit underwhelming however, under the hood, Exo has been performing analyses, automatic rewriting of loop bounds and indexing expressions to make the process easier. The analysis serve as guard rails for the powerful rewrite rules and are topic of another tutorial.

## Compiling

Finally, the code can be compiled and run on your machine if you have AVX2 instructions.
We provided a main function in `main.c` to call these procedures and to time them.
Please run `make` or compile manually:

```sh
$ exocc -o . --stem avx2_matmul x86_matmul.py
$ gcc -o avx2_matmul -march=native main.c avx2_matmul.c
```

This will print out the results of running kernel with and without the AVX instructions.

[blas]: https://www.netlib.org/blas/
[blis]: https://github.com/flame/blis