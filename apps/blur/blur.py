from __future__ import annotations

from exo import *
from exo.libs.memories import DRAM_STATIC
from exo.platforms.x86 import *
from exo.syntax import *
from exo.stdlib.scheduling import *


@proc
def producer(n: size, f: ui8[n + 1], inp: ui8[n + 6]):
    for i in seq(0, n + 1):
        f[i] = (
            inp[i] + inp[i + 1] + inp[i + 2] + inp[i + 3] + inp[i + 4] + inp[i + 5]
        ) / 6.0


@proc
def consumer(n: size, f: ui8[n + 1], g: ui8[n]):
    for i in seq(0, n):
        g[i] = (f[i] + f[i + 1]) / 2.0


@proc
def blur(n: size, g: ui8[n], inp: ui8[n + 6]):
    f: ui8[n + 1]
    producer(n, f, inp)
    consumer(n, f, g)


blur_staged = rename(blur, "blur_staged")

blur = inline(blur, "producer(_)")
blur = inline(blur, "consumer(_)")
print(blur)

if __name__ == "__main__":
    print(blur)

__all__ = ["blur_staged"]
