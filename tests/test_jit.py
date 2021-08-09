from __future__ import annotations        # make Python behave
from PIL import Image                     # standard image library
import numpy as np                        # standard array library
import time                               # timers
import sys                                # add DSL library to the Python path
import pytest
import os
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc                  # import the SYS-ATL DSL

# open image as a data array
input_filename = os.path.dirname(os.path.realpath(__file__)) + "/input.png"
parrot = np.array(Image.open(input_filename), np.float32)

def take_timing(proc, *args, **kwargs):
    n_runs           = 10
    lo, total, hi    = 1.0e6, 0.0, 0.0
    cproc            = proc.jit_compile()
    cproc(*args, **kwargs)

    for _ in range(0,n_runs):
        start = time.perf_counter()
        cproc(*args, **kwargs)
        stop  = time.perf_counter()
        ms    = (stop-start)*1.0e3
        if ms < lo:
            lo = ms
        if ms > hi:
            hi = ms
        total += ms

    # report
    print(f"timings for {proc.name()} in ms:\n"+
          f"    avg:   {total / n_runs}\n"+
          f"    min:   {lo}\n"+
          f"    max:   {hi}\n")

    return (total / n_runs)

# define "blur" in our DSL
# this @proc "decorator" specifies that
# the following function is written in SYS-ATL
def gen_blur():
    @proc
    def blur(N: size, M: size, K: size,      # size args specify sizes of other arguments
             image  : R[N, M],          # an N x M array; for input
             kernel : R[K, K],
             res    : R[N, M]):        # an N x M array; for output

        for x in par(0, N):                  # zero out the `res`ult array
            for y in par(0, M):
                res[x, y] = 0.0

        for x in par(0, N):                  # loop nest specifying the blur
            for y in par(0, M):
                for i in par(0, K):
                    for j in par(0, K):
                        if x+i < N and y+j < M:
                            res[x, y] += kernel[i, j] * image[x+i, y+j]

    return blur

@pytest.mark.skip()
def test_blur():
    blur = gen_blur()

    # SYS-ATL is designed to trivially compile into C code
    blur.show_c_code()

    # now, let's collect and name all the arguments we'll use
    # to call the procedure with
    N, M       = parrot.shape
    K          = 3
    kernel     = np.array([[0.0625, 0.1250, 0.0625],
                           [0.1250, 0.2500, 0.1250],
                           [0.0625, 0.1250, 0.0625]], dtype=np.float32)
    result     = np.zeros([N,M], dtype=np.float32)
    blurargs   = [N,M,K, parrot, kernel, result] # useful shorthand
    print("the size is...", (N,M))

    # in normal use, we'll dump that C-code to a file and use
    # a makefile toolchain, but for the sake of this demo
    # we'll use a simple jit-compilation wrapper so we
    # can keep working inside of this notebook.

    c_blur = blur.jit_compile()
    c_blur(N=N,M=M,K=K, image=parrot, kernel=kernel, res=result)


    take_timing(blur, N,M,K, parrot, kernel, result)


    # observe what happens when we iterate over the image
    # in the wrong way relative to its storage order
    bad_blur = blur.rename('bad_blur')
    bad_blur = bad_blur.reorder('x','y')

    orig     = take_timing(blur, *blurargs)
    bad      = take_timing(bad_blur, *blurargs)
    print('slowdown is ', bad / orig)

    # in order to tile, we need to split some loops
    split_blur = blur.rename('split_blur')
    split_blur = split_blur.split('x #2', 8, ['xhi','xlo'])

    # notice that the split scheduling primitive
    # must introduce additional if-guards to ensure
    # that all array accesses remain in-bounds

    # Alternatively, we can tell .split(...) to
    # ensure safety via a different "tail-strategy"
    split_blur = blur.rename('split_blur')
    split_blur = split_blur.split('x #2', 8, ['xhi','xlo'], tail='cut')

    # Then, we can similarly split the inner y-loop
    split_blur = split_blur.split('y #2', 8, ['yhi','ylo'], tail='cut')

    # In order to accomplish tiling, we need to
    # wrap the two different split y loops in two
    # different copies of the `for xlo in ...` loop

    # This is an instance of loop-fissioning
    split_blur = split_blur.fission_after("for yhi in _: _")

    # now we can finish up the tiling transform by re-ordering the
    # lower order x iteration with the higher order y iteration

    # additionally, it turned out that moving the kernel iteration outside of this
    # inner loop was essential
    split_blur = (split_blur.reorder('xlo #1','yhi')
                            .reorder('ylo #1','i').reorder('xlo #1','i')
                            .reorder('ylo #1','j').reorder('xlo #1','j'))


    # now, let's go ahead and test our hypothesis:
    # that tiling will give us a performance improvement
    orig     = take_timing(blur, *blurargs)
    split    = take_timing(split_blur, *blurargs)
    print('speedup is ', orig / split)


    # We chose to split by a factor of 8,
    # but one natural aspect of scheduling is
    # to tune this factor.

    # We can quickly build a test harness to explore this parameter
    def tile_by(n_x,n_y):

        test_blur = (blur.rename('test_blur')
                         # split the loops we want to tile together
                         .split('x #2', n_x, ['xhi','xlo'], tail='cut')
                         .split('y #2', n_y, ['yhi','ylo'], tail='cut')
                         # push the `for xlo in _` loop down over the y-loop split
                         .fission_after("for yhi in _: _")
                         # complete the tiling by moving both lower-order loops
                         # beneath both higher-order loops
                         .reorder('xlo #1','yhi')
                         # finally, a magic improvement is to exchange the filter iteration order
                         .reorder('ylo #1','i').reorder('xlo #1','i')
                         .reorder('ylo #1','j').reorder('xlo #1','j')
                    )
        test    = take_timing(test_blur, *blurargs)
        print(f"({n_x:3d}, {n_y:3d}):   {test:8.3f}")

    print("params        test (ms)")
    for n_x,n_y in [(4,4),
                    (4,8),
                    (8,8),
                    (8,16),
                    (16,16),
                    (16,32),
                    (32,32),
                    (32,64),
                    (64,64),
                    (128,128)]:
        tile_by(n_x,n_y)
