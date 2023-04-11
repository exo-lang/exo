"""

# Step 1: Show Exo SGEMM
#   - don't worry about the assertions

# Step 1.1:
#   Run python and view console output

# Step 1.2:
#   Run build and observe timing comparison

# Step 1.3:
#   Show the *.C and *.H files that are produced

# --------------------------------------------------------------------------- #

# use Exo windowing concept...
def make_sgemm_win(p=SGEMM):
    p = rename(SGEMM, 'sgemm_win')
    p = set_window(p, 'A', True)
    p = set_window(p, 'B', True)
    p = set_window(p, 'C', True)
    return p
sgemm_win = make_sgemm_win()

# Step 2: Setup - Windowing, don't worry about this right now



# Constants
micro_N = 4
micro_M = 16
assert micro_M % 4 == 0

# Create the microkernel
microkernel = rename(sgemm_win, 'microkernel')
microkernel = microkernel.partial_eval(micro_N,micro_M)
microkernel = simplify(microkernel)

# Step 2.1: Setup - the micro-kernel
#   - explain what the micro-kernel approach is
# RUN PYTHON
#   - notice how arguments are specialized
# Now try commenting out `simplify` (what's different)
#   - explain procedure equivalence now

# --------------------------------------------------------------------------- #

# Step 3: Make a tiled outer procedure

def make_sgemm_tiled(p=SGEMM):
    p = rename(p, 'sgemm_tiled')
    return p
sgemm_tiled = make_sgemm_tiled()

# First, let's use a function to save ourselves from typing too much

    # tile i & j for the kernel
    p = divide_loop(p, 'i', micro_N, ['io', 'ii'], tail='cut_and_guard')

# Ok, now I'm going to use a scheduling operation called divide loop.
# Let me first go ahead and execute this.
# > RUN PYTHON
# - Ok, so we can see here that the `i` loop has been divided in two.
# - If we multiply the extents of those loops we basically get the
#   original loop extent M
# - There is also this second loop created, and note that it loops
#   over a `modulo` or "remainder" amount.

# Ok, let's go over the syntax of the scheduling operation now.
#   1.  the 1st argument is always the procedure to transform
#   2.  the 2nd argument is often a "cursor" telling us where to
#         perform the code transformation within the procedure
#   3.  the 3rd argument to `divide_loop` is the quotient for the division
#   4.  the 4th argument is some names for the new loop iteration variables
#   5.  then we've used the optional argument to specify what should happen
#       to the "remainder" of the division

# ** Show how we can modify the cursor thing to find the loop other ways

# We can repeat this process divide the main `j` loop here too

    p = divide_loop(p, 'j #0', micro_M, ['jo', 'ji'], tail='cut_and_guard')

# Adding the #0 tells us that we want to search for the first occurrence
# > RUN PYTHON

# Now, in order to complete our tiling, we want to re-order the
# ii and jo loops.  However, there's a loop tail in the way.
# We can resolve this by fissioning the body of the ii loop

    # isolate the main chunk of work
    p = fission(p, p.find('for jo in _: _').after(), n_lifts=2)

# Observe that here we find the loop statement, and then get a cursor
# that points to the gap immediately after that statement.

# > RUN PYTHON

# We've fissioned the code all the way up to the top level thanks to our
# optional `n_lifts=2` argument

# Now that we've isolated the main chunk of work, let's reorder the loops

    p = reorder_loops(p, 'ii jo')

# > RUN PYTHON
# Yup!

# INSERT PRINT AFTER MICROKERNEL

# Ok, so now let's look at the microkernel again
# > RUN PYTHON

# Observe that the body of the microkernel roughly matches
# the main inner loops of `sgemm_exo`

# One of the key ideas in Exo is the ability to replace a block of
# statements with an equivalent sub-procedure call.
# Let's just go ahead and do that

    p = replace_all(p, microkernel)

# > RUN PYTHON
# Ok, so we've replaced the inner loops with a call to `microkernel`
# However, this call looks a little funny.  It has all kinds of complicated
# indexing expressions in it.  Let's go ahead and simplify those real quick

    p = simplify(p)

# > RUN PYTHON
# This is better, but still confusing.
# Remember I said I'd come back to this strange "windowing" thing.
#   - explain windowing
#   - REPLACE infers arguments, including the indexing
#     arithmetic needed for windowing

# Alternative: explicit replace
    p = replace(p, "for ii in _:_ #0", microkernel)

# Note that the implementation of `replace_all` is entirely user-level code
# in a standard library.
# Show `scheduling.py` code

# ****
# Talk about scheduling as modulating a lowering vs. scheduling as rewriting 


# > RUN BUILD.SH
# - Ok, we've successfully factored our program,
#   but this hasn't had any performance consequences yet.


# --------------------------------------------------------------------------- #

# Step 4: Tuning up the micro-kernel

# Ok, now that we have an idea of what scheduling looks like, I'm going
# to pick up the pace slightly

def make_neon_microkernel(p=microkernel):
    p = rename(p, 'neon_microkernel')
    # Move k to the outermost loop
    p = reorder_loops(p, 'j k')
    p = reorder_loops(p, 'i k')
    # expose inner-loop for 4-wide vectorization
    p = divide_loop(p, 'j', 4, ['jo','ji'], perfect=True)
    return p
neon_microkernel = make_neon_microkernel()
print(neon_microkernel)

# > RUN PYTHON
# Ok, now we can see a slight change in the micro-kernel
# that prepares us for both
#   (1) vector instructions by dividing `j` by 4 and
#   (2) register blocking, by moving the `k` loop to the outside

# However, observe that we're still calling `microkernel` not
# `neon_microkernel`

def finish_sgemm_tiled(p=sgemm_tiled):
    p = call_eqv(p, microkernel, neon_microkernel)
    return p
sgemm_tiled = finish_sgemm_tiled()

# > RUN PYTHON

# Ok, let's try testing this

# > RUN BUILD.SH

# Great!  We already can see a significant benefit from
# changing the code around.

# In order to move more quickly, I'm going to paste a big block
# of code in and then we'll walk through it more slowly

def stage_C_microkernel(p=neon_microkernel):
    p = stage_mem(p, 'C[_] += _', 'C[i, 4 * jo + ji]', 'C_reg')
    for iname in reversed(['i','jo','ji']):
        p = expand_dim(p, 'C_reg', 4, iname, unsafe_disable_checks=True)
    p = lift_alloc(p, 'C_reg', n_lifts=4)
    p = autofission(p, p.find('C_reg[_] = _').after(), n_lifts=4)
    p = autofission(p, p.find('C[_] = _').before(), n_lifts=4)
    #
    p = replace(p, 'for ji in _: _ #0', neon_vld_4xf32)
    p = replace(p, 'for ji in _: _ #1', neon_vst_4xf32)
    p = set_memory(p, 'C_reg', Neon)
    return p
neon_microkernel = stage_C_microkernel()

# REPLACE; Look at Neon Code and explain instructions


def stage_A_B_microkernel(p=neon_microkernel):
    for buf in ('A','B'):
        p = bind_expr(p, f'{buf}[_]', f'{buf}_vec')
        p = expand_dim(p, f'{buf}_vec', 4, 'ji', unsafe_disable_checks=True)
        p = lift_alloc(p, f'{buf}_vec')
        p = fission(p, p.find(f'{buf}_vec[_] = _').after())
        p = set_memory(p, f'{buf}_vec', Neon)
    #
    p = replace_all(p, neon_vld_4xf32)
    p = replace_all(p, neon_broadcast_4xf32)
    p = replace_all(p, neon_vfmadd_4xf32_4xf32)
    p = autolift_alloc(p, 'A_vec', n_lifts=2)
    p = autofission(p, p.find('B_vec : _').before(), n_lifts=2)
    p = autolift_alloc(p, 'B_vec', n_lifts=2)
    p = autofission(p, p.find('neon_vld_4xf32(_) #1').after(), n_lifts=2)
    return p
neon_microkernel = stage_A_B_microkernel()

neon_microkernel = simplify(neon_microkernel)
print(neon_microkernel)


# > RUN BUILD.SH



# --------------------------------------------------------------------------- #

# Step 5: Further Cache Partitioning

# Don't necessarily need to do this



L1_N = 64
L1_M = 64
L1_K = 64

assert L1_N % micro_N == 0
assert L1_M % micro_M == 0
mid_N = L1_N // micro_N
mid_M = L1_M // micro_M
mid_K = L1_K


    # clean up tail case from earlier
    p = autofission(p, p.find('for ko in _: _ #0').after(), n_lifts=2)
    # actually tile for L1 cache
    p = reorder_loops(p, 'jo ko #0')
    p = reorder_loops(p, 'io ko #0')
    p = divide_loop(p, 'io #0', mid_N, ['io', 'im'], tail='cut')
    p = divide_loop(p, 'jo #0', mid_M, ['jo', 'jm'], tail='cut')
    p = fission(p, p.find('for jo in _: _ #0').after(), n_lifts=3)
    p = repeat(reorder_loops)(p, 'im jm')
    p = repeat(reorder_loops)(p, 'im jo')
    p = simplify(p)
    # stage per-tile memory at appropriate levels
    p = stage_mem(p, 'for jo in _: _ #0',
                     f"A[{L1_N}*io : {L1_N}*io + {L1_N},"
                     f"  {L1_K}*ko : {L1_K}*ko + {L1_K}]", 'Atile')
    p = lift_alloc(p, 'Atile', n_lifts=2)
    p = stage_mem(p, 'for im in _: _ #0',
                  f"B[{L1_K}*ko : {L1_K}*ko + {L1_K},"
                  f"  {L1_M}*jo : {L1_M}*jo + {L1_M}]", 'Btile')
    p = lift_alloc(p, 'Btile', n_lifts=3)
    # cleanup
    p = simplify(p)
    return p
sgemm_tiled = finish_sgemm_tiled()






# INSIDE `make_sgemm_tiled` AT THE END BEFORE REPLACE_ALL

    # tile k now, before we do the microkernel replacement
    p = divide_loop(p, 'k #0', mid_K, ['ko', 'ki'], tail='cut_and_guard')
    p = fission(p, p.find('for ko in _: _').after(), n_lifts=2)
    p = reorder_loops(p, 'ji ko')
    p = reorder_loops(p, 'ii ko')
"""
