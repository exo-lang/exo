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

print("blur_staged")
print(blur_staged)
print()

blur = inline(blur, "producer(_)")
blur = inline(blur, "consumer(_)")

blur_compute_at_store_root = compute_at(blur, "blur_compute_at_store_root")
print("blur_compute_at_store_root")
print(blur_compute_at_store_root)
print()

loop = blur.find_loop("i")
blur_compute_at_store_at = store_at(
    blur_compute_at_store_root, "f", "g", loop, "blur_compute_at_store_at"
)
print("blur_compute_at_store_at")
print(blur_compute_at_store_at)
print()

blur_inline = inline_assign(
    blur_compute_at_store_at,
    blur_compute_at_store_at.find("f_tmp[_] = _ #1")
    .as_block()
    .expand(delta_lo=0, delta_hi=1),
)
blur_inline = inline_assign(
    blur_inline,
    blur_inline.find("f_tmp[_] = _ #0").as_block().expand(delta_lo=0, delta_hi=1),
)
blur_inline = delete_buffer(blur_inline, "f_tmp : _")
blur_inline = rename(blur_inline, "blur_inline")

print("blur_inline")
print(blur_inline)
print()

if __name__ == "__main__":
    print(blur)

__all__ = [
    "blur_staged",
    "blur_compute_at_store_root",
    "blur_compute_at_store_at",
    "blur_inline",
]
