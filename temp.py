from __future__ import annotations

from exo import *
from exo.platforms.x86 import *
from exo.stdlib.scheduling import *


@proc
def test(n: size, inp: R[n]):
    x: R[n]
    for i in seq(0, n):
        x[i] = inp[i]


print(test)
test = set_precision(test, "x", "i32")
test = autolift_alloc(test, "x : _")
test = set_memory(test, "x", AVX2)

if __name__ == "__main__":
    pass

__all__ = ["test"]
