"""
This module defines special symbols that are used inside Exo code. You may
import this module via `from exo.syntax import *` to suppress warnings and
see documentation inside an IDE (like PyCharm).
"""

from typing import TypeVar as _TypeVar


def _fail_dummy():
    raise SyntaxError("Exo Python dummy functions should never be called " "directly.")


# noinspection PyPep8Naming
class size(int):
    """
    A type representing a size in Exo.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be " "instantiated")


# noinspection PyPep8Naming
class f32(float):
    """
    A type representing a 32-bit floating point value.
    """

    def __init__(self, *_):
        raise SyntaxError("Exo Python dummy objects should never be " "instantiated")


def stride(tensor, dimension):
    """
    Given a tensor and a dimension, returns an opaque value representing the
    stride on that dimension.
    """
    _fail_dummy()


def par(lo, hi):
    """
    Parallel range between lo (must be 0) and hi (exclusive). May only be
    used in a for-loop bound. Iterations proceed as if in random order.
    """


# Common tensor names:
# A = _TypeVar('A', bound=size)
# B = _TypeVar('B', bound=size)
# C = _TypeVar('C', bound=size)
D = _TypeVar("D", bound=size)
E = _TypeVar("E", bound=size)
F = _TypeVar("F", bound=size)
G = _TypeVar("G", bound=size)
H = _TypeVar("H", bound=size)
I = _TypeVar("I", bound=size)
J = _TypeVar("J", bound=size)
K = _TypeVar("K", bound=size)
L = _TypeVar("L", bound=size)
M = _TypeVar("M", bound=size)
N = _TypeVar("N", bound=size)
O = _TypeVar("O", bound=size)
P = _TypeVar("P", bound=size)
Q = _TypeVar("Q", bound=size)
R = _TypeVar("R", bound=size)
S = _TypeVar("S", bound=size)
T = _TypeVar("T", bound=size)
U = _TypeVar("U", bound=size)
V = _TypeVar("V", bound=size)
W = _TypeVar("W", bound=size)
X = _TypeVar("X", bound=size)
Y = _TypeVar("Y", bound=size)
Z = _TypeVar("Z", bound=size)

a = _TypeVar("a", bound=size)
b = _TypeVar("b", bound=size)
c = _TypeVar("c", bound=size)
d = _TypeVar("d", bound=size)
e = _TypeVar("e", bound=size)
f = _TypeVar("f", bound=size)
g = _TypeVar("g", bound=size)
h = _TypeVar("h", bound=size)
# Common loop variable names:
# i = _TypeVar('i', bound=size)
# j = _TypeVar('j', bound=size)
# k = _TypeVar('k', bound=size)
l = _TypeVar("l", bound=size)
m = _TypeVar("m", bound=size)
n = _TypeVar("n", bound=size)
o = _TypeVar("o", bound=size)
p = _TypeVar("p", bound=size)
q = _TypeVar("q", bound=size)
r = _TypeVar("r", bound=size)
s = _TypeVar("s", bound=size)
t = _TypeVar("t", bound=size)
u = _TypeVar("u", bound=size)
v = _TypeVar("v", bound=size)
w = _TypeVar("w", bound=size)
x = _TypeVar("x", bound=size)
y = _TypeVar("y", bound=size)
z = _TypeVar("z", bound=size)
