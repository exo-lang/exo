from __future__ import annotations
import ctypes
from ctypes import *
import os
import sys
import subprocess
import numpy as np
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, Procedure

# Initialize by creating a tmp directory
directory = "tmp/"
if not os.path.isdir(directory):
    os.mkdir(directory)

c_float_p = ctypes.POINTER(ctypes.c_float)

def cvt_c(n_array):
    assert n_array.dtype == np.float32
    return n_array.ctypes.data_as(c_float_p)

def nparray(arg):
    return np.array(arg, dtype=np.float32)

def nprand(size):
    return np.random.uniform(size=size).astype(np.float32)

def test_conv1d():
    @proc
    def conv1d(n : size, m : size, r: size,
               x : R[n] @ IN, w : R[m] @ IN, res : R[r] @ OUT ):
      for i in par(0,r):
        res[i] = 0.0
      for i in par(0,r):
        for j in par(0,n):
          if j < i+1 and j >= i-m+1:
            res[i] += x[j]*w[i-j]

    assert type(conv1d) is Procedure
    filename = "uast_test_conv1d"
    conv1d.compile_c(directory, filename)

    compile_so_cmd = ("clang -Wall -Werror -fPIC -O3 -shared "+
                      "-o " + directory + filename + ".so " +
                      directory + filename + ".c")
    subprocess.run(compile_so_cmd, check=True, shell=True)
    abspath  = os.path.dirname(os.path.abspath(filename))
    test_lib = ctypes.CDLL(abspath + '/' + directory + filename + ".so")
    n_size = 5
    m_size = 3
    r_size = n_size + m_size - 1
    x = nparray([0.2, 0.5, -0.4, 1.0, 0.0])
    w = nparray([0.6, 1.9, -2.2])
    res = nprand(size=(r_size))
    res_c = cvt_c(res)
    test_lib.conv1d(c_int(n_size), c_int(m_size), c_int(r_size), cvt_c(x), cvt_c(w), res_c)
    res_c = np.ctypeslib.as_array(res_c, shape=(r_size,))
    np.testing.assert_almost_equal(res_c,nparray([0.12,0.68,0.27,-1.26,2.78,-2.2,0]))

def test_add():
  @proc
  def add(n : size, x : R[n] @ IN, y : R[n] @ IN, res : R[n] @ OUT):
    for i in par(0,n):
      res[i] = x[i] + y[i]

  assert type(add) is Procedure
  add.compile_c("tmp/", "uast_test_add")
