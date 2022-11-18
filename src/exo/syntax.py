"""
This module defines special symbols that are used inside Exo code. You may
import this module via `from exo.syntax import *` to suppress warnings and
see documentation inside an IDE (like PyCharm).
"""
from __future__ import annotations

from typing import TypeVar

# fmt: off

# With so many tiny names, letting Black put them all one-to-a-line is beyond hideous
__all__ = [
    "size", "index", "i8", "i32", "f32", "stride", "seq", "D", "E", "F", "G", "H", "I",
    "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z",
    "a", "b", "c", "d", "e", "f", "g", "h", "l", "m", "n", "o", "p", "q", "r", "s", "t",
    "u", "v", "w", "x", "y", "z"
]


# fmt: on


def _fail_dummy():
    raise SyntaxError("Exo Python dummy functions should never be called directly.")


# noinspection PyPep8Naming
class size(int):
    """
    A type representing a size in Exo.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


# noinspection PyPep8Naming
class index(int):
    """
    A type representing a index in Exo.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


# noinspection PyPep8Naming
class i8(float):
    """
    A type representing a 32-bit integer value.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


# noinspection PyPep8Naming
class i32(float):
    """
    A type representing a 32-bit integer value.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


# noinspection PyPep8Naming
class f32(float):
    """
    A type representing a 32-bit floating point value.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be instantiated")


def stride(tensor, dimension):
    """
    Given a tensor and a dimension, returns an opaque value representing the
    stride on that dimension.
    """
    _fail_dummy()


def seq(lo, hi):
    """
    Iteration range between lo (must be 0) and hi (exclusive). May only be
    used in a for-loop bound.
    """


# Common tensor names:
# A = _TypeVar('A', bound=size)
# B = _TypeVar('B', bound=size)
# C = _TypeVar('C', bound=size)
D = TypeVar("D", bound=size)
E = TypeVar("E", bound=size)
F = TypeVar("F", bound=size)
G = TypeVar("G", bound=size)
H = TypeVar("H", bound=size)
I = TypeVar("I", bound=size)
J = TypeVar("J", bound=size)
K = TypeVar("K", bound=size)
L = TypeVar("L", bound=size)
M = TypeVar("M", bound=size)
N = TypeVar("N", bound=size)
O = TypeVar("O", bound=size)
P = TypeVar("P", bound=size)
Q = TypeVar("Q", bound=size)
R = TypeVar("R", bound=size)
S = TypeVar("S", bound=size)
T = TypeVar("T", bound=size)
U = TypeVar("U", bound=size)
V = TypeVar("V", bound=size)
W = TypeVar("W", bound=size)
X = TypeVar("X", bound=size)
Y = TypeVar("Y", bound=size)
Z = TypeVar("Z", bound=size)

a = TypeVar("a", bound=size)
b = TypeVar("b", bound=size)
c = TypeVar("c", bound=size)
d = TypeVar("d", bound=size)
e = TypeVar("e", bound=size)
f = TypeVar("f", bound=size)
g = TypeVar("g", bound=size)
h = TypeVar("h", bound=size)
# Common loop variable names:
# i = _TypeVar('i', bound=size)
# j = _TypeVar('j', bound=size)
# k = _TypeVar('k', bound=size)
l = TypeVar("l", bound=size)
m = TypeVar("m", bound=size)
n = TypeVar("n", bound=size)
o = TypeVar("o", bound=size)
p = TypeVar("p", bound=size)
q = TypeVar("q", bound=size)
r = TypeVar("r", bound=size)
s = TypeVar("s", bound=size)
t = TypeVar("t", bound=size)
u = TypeVar("u", bound=size)
v = TypeVar("v", bound=size)
w = TypeVar("w", bound=size)
x = TypeVar("x", bound=size)
y = TypeVar("y", bound=size)
z = TypeVar("z", bound=size)
