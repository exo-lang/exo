# Scheduling example

Please install exo by `pip install exo-lang` before running this example.

We provided a sample user code in `exo/examples/x86_matmul.py`.
`rank_k_reduce_6x16` is a microkernel for AVX2 SGEMM application. We chose to use AVX2
so that users who do not have AVX512 machines can run this example. We chose the
SGEMM microkernel application because it is relatively simple but contains all the
important scheduling operators. Please run the code as follows.

```
$ cd examples
$ python x86_matmul.py
```

## Scheduling walk-through

Let's walk through the scheduling transforms step by step. Without any
modification, `python x86_matmul.py` will print the original, simple algorithm that we
will start with.

```python
# Original algorithm:
def rank_k_reduce_6x16(K: size, C: f32[6, 16] @ DRAM, A: f32[6, K] @ DRAM,
                       B: f32[K, 16] @ DRAM):
    for i in seq(0, 6):
        for j in seq(0, 16):
            for k in seq(0, K):
                C[i, j] += A[i, k] * B[k, j]
```

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
