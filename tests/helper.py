from __future__ import annotations
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
from PIL import Image
import scipy.stats as st

sys.path.append(sys.path[0] + "/..")
from SYS_ATL import proc, Procedure

# Figure out the filesystem location of this file
# and then all other resources can be located relative to it.
_TEST_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.abspath(os.path.join(_TEST_DIR, '..'))
TMP_DIR = os.path.join(_TEST_DIR, 'tmp')

# Initialize by creating a tmp directory
if not os.path.isdir(TMP_DIR):
    os.mkdir(TMP_DIR)

# Dump image here
input_filename = os.path.join(_TEST_DIR, "input.png")
o_image = Image.open(input_filename)
image = np.asarray(o_image, dtype="float32")


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


def generate_lib(filename):
    c_file = os.path.join(TMP_DIR, filename + '.c')
    so_file = os.path.join(TMP_DIR, filename + '.so')
    compiler = os.getenv('CC', default='clang')
    cflags = os.getenv('CFLAGS', default='-Wall')
    compile_so_cmd = f"{compiler} {cflags} -fPIC -O3 -shared -march=native {c_file} -o {so_file}"
    subprocess.run(compile_so_cmd, check=True, shell=True)
    return ctypes.CDLL(so_file)
