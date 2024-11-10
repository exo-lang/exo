# Quiz2!

This quiz is about loop fission bugs and debugging via printing cursors.

## Incorrect output (compiler error)
As written, the schedule has a bug which attempts to incorrectly fission a loop.
```
Traceback (most recent call last):
  File "/home/yuka/.local/bin/exocc", line 8, in <module>
    sys.exit(main())
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/main.py", line 55, in main
    library = [
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/main.py", line 58, in <listcomp>
    for proc in get_procs_from_module(load_user_code(mod))
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/main.py", line 107, in load_user_code
    loader.exec_module(user_module)
  File "<frozen importlib._bootstrap_external>", line 790, in exec_module
  File "<frozen importlib._bootstrap>", line 228, in _call_with_frames_removed
  File "/home/yuka/exo/examples/quiz2/quiz2.py", line 42, in <module>
    w = wrong_schedule(scaled_add)
  File "/home/yuka/exo/examples/quiz2/quiz2.py", line 38, in wrong_schedule
    p = fission(p, vector_assign.after())
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/API_scheduling.py", line 100, in __call__
    return self.func(*bound_args.args, **bound_args.kwargs)
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/API_scheduling.py", line 2066, in fission
    ir, fwd = scheduling.DoFissionAfterSimple(
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/rewrite/LoopIR_scheduling.py", line 2385, in DoFissionAfterSimple
    alloc_check(pre, post)
  File "/home/yuka/.local/lib/python3.9/site-packages/exo/rewrite/LoopIR_scheduling.py", line 2352, in alloc_check
    raise SchedulingError(
exo.rewrite.new_eff.SchedulingError: <<<unknown directive>>>: Will not fission here, because doing so will hide the allocation of vec from a later use site.
```

## Correct Output
The correct output will divide the computation into individual, vectorizable loops.
```
def scaled_add_scheduled(N: size, a: f32[N] @ DRAM, b: f32[N] @ DRAM,
                         c: f32[N] @ DRAM):
    assert N % 8 == 0
    for io in seq(0, N / 8):
        vec: R[8] @ DRAM
        vec_1: R[8] @ DRAM
        vec_2: f32[8] @ DRAM
        vec_3: R[8] @ DRAM
        vec_4: R[8] @ DRAM
        vec_5: f32[8] @ DRAM
        for ii in seq(0, 8):
            vec_1[ii] = 2
        for ii in seq(0, 8):
            vec_2[ii] = a[8 * io + ii]
        for ii in seq(0, 8):
            vec[ii] = vec_1[ii] * vec_2[ii]
        for ii in seq(0, 8):
            vec_4[ii] = 3
        for ii in seq(0, 8):
            vec_5[ii] = b[8 * io + ii]
        for ii in seq(0, 8):
            vec_3[ii] = vec_4[ii] * vec_5[ii]
        for ii in seq(0, 8):
            c[8 * io + ii] = vec[ii] + vec_3[ii]
```

---

## Solution

To understand the bug, let's first try printing right before the error.
Put `print(vector_assign.after())` after line 37.
```
    for io in seq(0, N / 8):
        vec: R[8] @ DRAM
        for ii in seq(0, 8):
            vec_1: R @ DRAM
            vec_1 = 2
            [GAP - After]
            ...
```
This is showing the code is trying to fission at `[GAP - After]` location, which is unsafe because the `vec_1: R` allocation is in the `ii` loop and before the fissioning point, which means if `vec_1` is used after the fission point that'll be an error.

Change
```python
    for i in range(num_vectors):
        vector_reg = p.find(f"vec: _ #{i}")
        p = expand_dim(p, vector_reg, 8, "ii")
        p = lift_alloc(p, vector_reg)

        vector_assign = p.find(f"vec = _ #{i}")
        p = fission(p, vector_assign.after())
```

to
```python
    for i in range(num_vectors):
        vector_reg = p.find(f"vec: _ #{i}")
        p = expand_dim(p, vector_reg, 8, "ii")
        p = lift_alloc(p, vector_reg)

    for i in range(num_vectors):
        vector_assign = p.find(f"vec = _ #{i}")
        p = fission(p, vector_assign.after())
```

So that you lift all the allocations out of the loop before fissioning.

Here is the rewritten version with improved clarity and GitHub Markdown syntax:

## Solution

To understand the bug, let's first try printing right before the error. Add the following line after line 37:

```python
print(vector_assign.after())
```

This will output:

```
    for io in seq(0, N / 8):
        vec: R[8] @ DRAM
        for ii in seq(0, 8):
            vec_1: R @ DRAM
            vec_1 = 2
            [GAP - After]
            ...
```

The code is attempting to perform fission at the `[GAP - After]` location.
However, this is unsafe because the `vec_1: R` allocation is within the `ii` loop and before the fission point.
If `vec_1` is used after the fission point, the code will no longer be a valid Exo.

To fix this issue, modify the code as follows:

```python
    for i in range(num_vectors):
        vector_reg = p.find(f"vec: _ #{i}")
        p = expand_dim(p, vector_reg, 8, "ii")
        p = lift_alloc(p, vector_reg)

    for i in range(num_vectors):
        vector_assign = p.find(f"vec = _ #{i}")
        p = fission(p, vector_assign.after())
```

By separating the allocation lifting and fission operations into two separate loops, you ensure that all the allocations are lifted out of the loop before performing fission. This resolves the issue of unsafe fission due to the allocation being within the loop.

