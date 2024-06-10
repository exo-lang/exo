from __future__ import annotations  # make Python behave

from exo import proc
from exo.stdlib.scheduling import *

old_split = repeat(divide_loop)

# I'm going to define a 1D version of a standard convolutional layer (cf. CuDNN)
# K - number of output channels
# C - number of input channels
# W - length of the input signal/tensor
# R - width of the filter kernel
def gen_conv1d():
    @proc
    def conv1d(
        K: size,
        C: size,
        W: size,
        R: size,
        w: R[K, C, R],
        x: R[C, W],
        res: R[K, W],
    ):
        # zero out the result memory
        for k_init in seq(0, K):
            for i_init in seq(0, W):
                res[k_init, i_init] = 0.0

        # do the convolution
        for k in seq(0, K):
            for c in seq(0, C):
                for i in seq(0, W):
                    for r in seq(0, R):
                        if 0 <= i - r:
                            res[k, i] += w[k, c, r] * x[c, i - r]

    return conv1d


def test_im2col(golden):
    conv1d = gen_conv1d()

    # Let's start applying scheduling
    im2col_conv = rename(conv1d, "im2col_conv")
    im2col_conv = reorder_loops(im2col_conv, "i r")
    im2col_conv = bind_expr(im2col_conv, "x[c, i-r]", "y")

    # next, we can start to lift that allocation
    # up and out of the loop
    im2col_conv = autolift_alloc(im2col_conv, "y:R", 5, keep_dims=True)

    # Then, we can fission the loop correspondingly,
    # separating what is now a data-marshalling statement from
    # the actual compute statement in two subsequent
    # loop nests via fissioning
    im2col_conv = autofission(im2col_conv, im2col_conv.find("y[c,r,i] = _").after(), 5)

    # Now, in order to expose these two parts of the computation as
    # re-usable sub-procedures, we want a way to factor them out.
    im2col_conv, im2col = extract_subproc(im2col_conv, "for c in _: _", "im2col")
    im2col_conv, matmul = extract_subproc(im2col_conv, "for k in _: _", "matmul")

    # Given this factoring, we can then proceed
    # to schedule these sub-procedures themselves.
    tiled_matmul = rename(matmul, "tiled_matmul")
    # split the loops we want to tile together
    tiled_matmul = old_split(tiled_matmul, "k", 8, ["khi", "klo"], tail="cut")
    tiled_matmul = reorder_loops(tiled_matmul, "klo c #0")
    tiled_matmul = reorder_loops(tiled_matmul, "klo r #0")
    tiled_matmul = reorder_loops(tiled_matmul, "klo i #0")
    tiled_matmul = old_split(tiled_matmul, "c #0", 8, ["chi", "clo"], tail="cut")
    tiled_matmul = reorder_loops(tiled_matmul, "clo r #0")
    tiled_matmul = reorder_loops(tiled_matmul, "clo i #0")
    tiled_matmul = reorder_loops(tiled_matmul, "clo klo #0")
    tiled_matmul = old_split(tiled_matmul, "i #0", 8, ["ihi", "ilo"], tail="cut")
    tiled_matmul = reorder_loops(tiled_matmul, "ilo klo #0")
    tiled_matmul = reorder_loops(tiled_matmul, "ilo clo #0")

    # We can invoke another scheduling directive
    # to change which version of the matmul gets scheduled
    im2col_conv = call_eqv(im2col_conv, "matmul(_,_,_,_,_,_,_)", tiled_matmul)
    assert f"{im2col}\n{matmul}\n{im2col_conv}" == golden


"""
    filename = "test_im2col"

    f_pretty = open(os.path.join(TMP_DIR, filename + "_pretty.atl"), "w")
    f_pretty.write(str(im2col_conv))
    f_pretty.close()

    im2col.compile_c(TMP_DIR, filename)
"""
