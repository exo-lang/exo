from __future__ import annotations

import ctypes
import os
import subprocess

import numpy as np
import scipy.stats as st
from PIL import Image

# Figure out the filesystem location of this file
# and then all other resources can be located relative to it.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
TMP_DIR = os.path.join(_TEST_DIR, 'tmp')

# Initialize by creating a tmp directory
if not os.path.isdir(TMP_DIR):
    os.mkdir(TMP_DIR)

# Dump image here
IMAGE = np.asarray(Image.open(os.path.join(_TEST_DIR, "input.png")),
                   dtype="float32")


def gkern(kernlen=5, nsig=1, typ=np.float32):
    x = np.linspace(-nsig, nsig, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return np.asarray(kern2d / kern2d.sum(), dtype=typ)


def cvt_c(n_array, typ=np.float32):
    if typ is np.float32:
        return n_array.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    elif typ is np.float64:
        return n_array.ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    assert False


def nparray(arg, typ=np.float32):
    return np.array(arg, dtype=typ)


def nprand(size, typ=np.float32):
    return np.random.uniform(size=size).astype(typ)


def generate_lib(filename, *, extra_flags=""):
    c_file = os.path.join(TMP_DIR, filename + '.c')
    h_file = os.path.join(TMP_DIR, filename + '.h')
    so_file = os.path.join(TMP_DIR, filename + '.so')
    compiler = os.getenv('CC', default='clang')
    cflags = os.getenv('CFLAGS', default='-Wall')
    compile_so_cmd = (f"{compiler} {cflags} {extra_flags} -O3 -fPIC "
                      f"-shared {c_file} -I {h_file} -o {so_file}")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    return ctypes.CDLL(so_file)


__all__ = [
    'TMP_DIR',
    'IMAGE',
    'gkern',
    'cvt_c',
    'nparray',
    'nprand',
    'generate_lib',
]
