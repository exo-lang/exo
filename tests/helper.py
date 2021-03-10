from __future__ import annotations
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
from PIL import Image
import scipy.stats as st
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure

# Initialize by creating a tmp directory
directory = "tmp/"
if not os.path.isdir(directory):
    os.mkdir(directory)

# Dump image here
input_filename = os.path.dirname(os.path.realpath(__file__)) + "/input.png"
o_image = Image.open(input_filename)
image = np.asarray(o_image, dtype="float32")

def gkern(kernlen=5, nsig=1, typ=np.float32):
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return np.asarray(kern2d/kern2d.sum(), dtype=typ)

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

def generate_lib(directory, filename):
    compile_so_cmd = ("clang -Wall -Werror -fPIC -O3 -shared " +
                      "-o " + directory + filename + ".so " +
                      directory + filename + ".c")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    abspath = os.path.dirname(os.path.abspath(filename))
    return ctypes.CDLL(abspath + '/' + directory + filename + ".so")
