# Quiz 1

Throughout the quiz, we provide incorrect code and the correct output as a reference. Your goal is to understand the code and fix the bug to match the correct output!

You can execute `quiz1.py` by running `exocc quiz1.py`. Without modification, it will show the incorrect output.

## Incorrect Output

The following output is incorrect because it does not make calls to vector intrinsics. While it matches the structure of SIMD vector code, it is still being executed one element at a time:

```python
def double(N: size, inp: f32[N] @ DRAM, out: f32[N] @ DRAM):
    assert N % 8 == 0
    two_vec: R[8] @ DRAM
    for ii in seq(0, 8):
        two_vec[ii] = 2.0
    for io in seq(0, N / 8):
        out_vec: f32[8] @ DRAM
        inp_vec: f32[8] @ DRAM
        for i0 in seq(0, 8):
            inp_vec[i0] = inp[i0 + 8 * io]
        for ii in seq(0, 8):
            out_vec[ii] = two_vec[ii] * inp_vec[ii]
        for i0 in seq(0, 8):
            out[i0 + 8 * io] = out_vec[i0]
```

## Correct Output

The correct output optimizes the function to use vectorized arithmetic operations to compute the result over the entire array:

```python
def double(N: size, inp: f32[N] @ DRAM, out: f32[N] @ DRAM):
    assert N % 8 == 0
    two_vec: R[8] @ AVX2
    vector_assign_two(two_vec[0:8])
    for io in seq(0, N / 8):
        out_vec: f32[8] @ AVX2
        inp_vec: f32[8] @ AVX2
        vector_load(inp_vec[0:8], inp[8 * io + 0:8 * io + 8])
        vector_multiply(out_vec[0:8], two_vec[0:8], inp_vec[0:8])
        vector_store(out[8 * io + 0:8 * io + 8], out_vec[0:8])
```

---

## Solution

Before calling `replace_all(p, avx_instrs)`, you need to set buffer memory annotations to AVX2, because `replace_all` is memory-aware and will only replace code chunks with instructions that have matching memory annotations.

Add the following code before the call to `replace_all`:

```python
    # Set the memory types to be AVX2 vectors
    for name in ["two", "out", "inp"]:
        p = set_memory(p, f"{name}_vec", AVX2)
```

This will ensure that the memory annotations are correctly set to AVX2 before replacing the code with vector intrinsics.
