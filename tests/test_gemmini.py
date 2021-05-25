from __future__ import annotations
#from ctypes import *
#import os
#import subprocess
#import numpy as np
#import scipy.stats as st
import os
import sys
sys.path.append(sys.path[0]+"/..")
from SYS_ATL import proc, instr, Procedure, DRAM, compile_procs
from SYS_ATL.libs.memories import GEMM_SCRATCH, GEMM_ACCUM, MDRAM
sys.path.append(sys.path[0]+"/.")
from .helper import *
import pytest

def gen_instructions():
  # don't run this generator more than once
  if GEMMINI_INSTR.ld_i8 is not None:
    return

  _gemm_ld_i8   = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                   "1.0f, 0, 0);\n"+
                   "gemmini_extended_mvin( {src}.data, "+
                                "((uint64_t) {dst}.data), {m}, {n} );")
  @instr(_gemm_ld_i8)
  def gemmini_ld_i8(
      n   : size,
      m   : size,
      src : [i8][n, m] @ DRAM,
      dst : [i8][n, 16] @ GEMM_SCRATCH,
  ):
      assert n <= 16
      assert m <= 16
      assert stride(src, 1) == 1
      assert stride(dst, 0) == 16
      assert stride(dst, 1) == 1

      for i in par(0, n):
          for j in par(0, m):
              dst[i,j] = src[i,j]

  # in order to load i8 values into the i32 accumulator memory,
  # we must specify `shrunk=1` (3rd param of ..._config_ld)
  _gemm_ld_acc_i8 = ("gemmini_extended3_config_ld({src}.strides[0]*1, "+
                     "1.0f, 1, 0);\n"+
                     "gemmini_extended_mvin( {src}.data, "+
                                  "((uint64_t) {dst}.data), {m}, {n} );")
  gemmini_ld_acc_i8 = (gemmini_ld_i8.rename('gemmini_ld_acc_i8')
                                    .set_precision('dst', 'i32')
                                    .set_memory('dst', GEMM_ACCUM)
                                    .make_instr(_gemm_ld_acc_i8))

  _gemm_st_i8   = ("gemmini_config_st({dst}.strides[0]*1);\n"+
                   "gemmini_extended_mvout( "+
                        "((uint64_t) {dst}.data), {src}.data, {m}, {n} );")
  @instr(_gemm_st_i8)
  def gemmini_st_i8(
      n   : size,
      m   : size,
      src : [i8][n, 16] @ GEMM_SCRATCH,
      dst : [i8][n, m]  @ DRAM
  ):
      assert n <= 16
      assert m <= 16
      assert stride(dst, 1) == 1
      assert stride(src, 0) == 16
      assert stride(src, 1) == 1

      for i in par(0, n):
          for j in par(0, m):
              dst[i, j] = src[i, j]

  gemmini_st_acc_i8 = (gemmini_st_i8.rename('gemmini_st_acc_i8')
                                    .set_precision('src', 'i32')
                                    .set_memory('src', GEMM_ACCUM)
                                    .make_instr(_gemm_st_i8))

  _gemm_zero_i8 = ("gemmini_extended3_config_ld(0, 1.0f, 0, 0);\n"+
                   "gemmini_extended_mvin( 0, ((uint64_t) {dst}.data),"+
                                         "{m}, {n} );")
  @instr(_gemm_zero_i8)
  def gemmini_zero_i8(
      n   : size,
      m   : size,
      dst : [i8][n, 16] @ GEMM_SCRATCH,
  ):
      assert n <= 16
      assert m <= 16
      assert stride(dst, 0) == 16
      assert stride(dst, 1) == 1

      for i in par(0, n):
          for j in par(0, m):
              dst[i,j] = 0.0

  gemmini_zero_acc_i8 = (gemmini_zero_i8.rename('gemmini_zero_acc_i8')
                                    .set_precision('dst', 'i32')
                                    .set_memory('dst', GEMM_ACCUM)
                                    .make_instr(_gemm_zero_i8))

  @instr("gemmini_config_ex(WS, NO_ACTIVATION, 0, ACC_SCALE_IDENTITY, 0);\n"+
         "gemmini_extended_preload("+
              "(uint64_t)({B}.data), (uint64_t)({C}.data), "+
              "{M}, {K}, "+
              "{M}, {N}"+
         ");\n"+
         "gemmini_extended_compute_preloaded("+
              "(uint64_t)({A}.data), ~((uint64_t)0), "+
              "{K}, {N}, "+
              "16, 16"+
         ");")
  def gemmini_matmul_i8(
      N : size,
      M : size,
      K : size,
      A : [i8][N, 16] @ GEMM_SCRATCH,
      B : [i8][K, 16] @ GEMM_SCRATCH,
      C : [i32][N, 16] @ GEMM_ACCUM,
  ):
      assert N <= 16
      assert M <= 16
      assert K <= 16

      for i in par(0,N):
          for j in par(0,M):
              for k in par(0,K):
                  a : i32
                  b : i32
                  a = A[i,k]
                  b = B[k,j]
                  C[i, j] += a * b


  GEMMINI_INSTR.ld_i8       = gemmini_ld_i8
  GEMMINI_INSTR.st_i8       = gemmini_st_i8
  GEMMINI_INSTR.ld_acc_i8   = gemmini_ld_acc_i8
  GEMMINI_INSTR.st_acc_i8   = gemmini_st_i8
  GEMMINI_INSTR.zero_i8     = gemmini_zero_i8
  GEMMINI_INSTR.zero_acc_i8 = gemmini_zero_acc_i8
  GEMMINI_INSTR.mm_i8       = gemmini_matmul_i8
  


class GEMMINI_INSTR:
  ld_i8       = None
  ld_acc_i8   = None
  st_i8       = None
  st_acc_i8   = None
  zero_i8     = None
  zero_acc_i8 = None
  mm_i8       = None


gen_instructions()



def init_constants():
  GEMMINI_ROOT = os.getenv('GEMMINI_ROOT')
  if GEMMINI_ROOT is None:
    RISCV = os.getenv('RISCV')
    if RISCV is None:
      pytest.skip("skipping gemmini tests; could not find chipyard",
                  allow_module_level=True)
    GEMMINI_ROOT = os.path.join(RISCV,'..','generators','gemmini')
  GEMMINI_ROOT        = os.path.abspath(GEMMINI_ROOT)
  GEMMINI_ROCC_TESTS  = os.path.join(GEMMINI_ROOT,'software',
                                                  'gemmini-rocc-tests')
  ROOT                = GEMMINI_ROCC_TESTS
  BCOMMON             = os.path.join(ROOT,'riscv-tests','benchmarks','common')

  CC_BAREMETAL        = 'riscv64-unknown-elf-gcc'
  CFLAGS_BAREMETAL    = ' '.join([
                          f'-DPREALLOCATE=1',
                          f'-DMULTITHREAD=1',
                          f'-mcmodel=medany',
                          f'-std=gnu99',
                          f'-O2',
                          f'-ffast-math',
                          f'-fno-common',
                          f'-fno-builtin-printf',
                          f'-march=rv64gc -Wa,-march=rv64gcxhwacha',
                          f'-lm',
                          f'-lgcc',
                          f'-I{ROOT}/riscv-tests',
                          f'-I{ROOT}/riscv-tests/env',
                          f'-I{ROOT}',
                          f'-I{BCOMMON}',
                          f'-DID_STRING=',
                          f'-nostdlib',
                          f'-nostartfiles',
                          f'-static',
                          f'-T {BCOMMON}/test.ld',
                          f'-DBAREMETAL=1',
                        ])

  _HERE_              = os.path.dirname(os.path.abspath(__file__))
  SYSTL_ROOT          = os.path.abspath(os.path.join(_HERE_,'..'))
  TEST_ROOT           = _HERE_
  TMP_DIR             = os.path.join(TEST_ROOT,'tmp')

  GEMM_BUILD_DIR      = os.path.join(TEST_ROOT,'gemmini_build')

  COMPILE             = (f"{CC_BAREMETAL} {CFLAGS_BAREMETAL} "+
                         f"{BCOMMON}/*.c {BCOMMON}/*.S")

  # make sure the build directory exists
  os.makedirs(GEMM_BUILD_DIR, exist_ok=True)

  class GemminiBuildEnv:
    pass

  GemminiBuildEnv.GEMMINI_ROOT        = GEMMINI_ROOT
  GemminiBuildEnv.GEMMINI_ROCC_TESTS  = GEMMINI_ROCC_TESTS
  GemminiBuildEnv.ROOT                = ROOT
  GemminiBuildEnv.BCOMMON             = BCOMMON

  GemminiBuildEnv.CC_BAREMETAL        = CC_BAREMETAL
  GemminiBuildEnv.CFLAGS_BAREMETAL    = CFLAGS_BAREMETAL

  GemminiBuildEnv.SYSTL_ROOT          = SYSTL_ROOT
  GemminiBuildEnv.TEST_ROOT           = TEST_ROOT
  GemminiBuildEnv.TMP_DIR             = TMP_DIR

  GemminiBuildEnv.GEMM_BUILD_DIR      = GEMM_BUILD_DIR

  GemminiBuildEnv.COMPILE             = COMPILE

  return GemminiBuildEnv


ENV = init_constants()




def gemmini_test_template(incl_file, glob_lines, body_lines):
  lines = ['#include <stdint.h>',
           '#include <stddef.h>',
           '#include <assert.h>',
           '#include <stdlib.h>',
           '#include <stdio.h>',
           '#include <time.h>',
           '',
           '#include "include/gemmini_testutils.h"',
           f'#include "{incl_file}"',
           '',
          ]

  assert isinstance(glob_lines, list)
  lines += glob_lines

  lines +=['',
           'int main() {',
          ]
  assert isinstance(body_lines, list)


  lines += [ "    "+ln for ln in body_lines ]


  lines += ['',
            '    printf("\\nDone\\n");',
            '    ',
            '    exit(0);',
            '}',
            '',
           ]

  return '\n'.join(lines)

def gemmini_write_test_main(filename, body):
  with open(os.path.join(ENV.GEMM_BUILD_DIR, filename), "w") as F:
      F.write(body)

def gemmini_compile(mainfile, libfile, binfile):
  mainfile  = os.path.join(ENV.GEMM_BUILD_DIR, mainfile)
  libfile   = os.path.join(ENV.GEMM_BUILD_DIR, libfile)
  binfile   = os.path.join(ENV.GEMM_BUILD_DIR, binfile)
  CMD = f"{ENV.COMPILE} -I{ENV.TMP_DIR} {mainfile} {libfile} -o {binfile}"

  if 0 != subprocess.call(CMD, shell=True):
    raise OSError("Compilation Failed")

def gemmini_run(binfile):
  binfile   = os.path.join(ENV.GEMM_BUILD_DIR, binfile)
  CMD = f"spike --extension=gemmini {binfile}"

  if 0 != subprocess.call(CMD, shell=True):
    raise OSError("Spike Execution Failed")

class CaseBuilder:
  def __init__(self, test_name):
    self.test_name  = test_name
    self.glob       = []
    self.body       = []
    self.procs      = []
    self._has_gemm_alloc = False


    self.glob += ['void print_2i8(int N, int M, int8_t *data) {',
                  '    for(int i=0; i<N; i++) {',
                  '        for(int j=0; j<M; j++)',
                  '            printf("%d ", (int)data[M*i + j]);',
                  '        printf("\\n");',
                  '    }',
                  '}',
                  '',
                  'bool check_eq_2i8(int N, int M, '+
                                   'int8_t *lhs, int8_t *rhs) {',
                  '    bool flag = true;'
                  '    for(int i=0; i<N; i++) {',
                  '        for(int j=0; j<M; j++)',
                  '            if(lhs[M*i + j] != rhs[M*i + j])',
                  '                flag = false;'
                  '    }',
                  '    return flag;'
                  '}',
                  '']

  def add_proc(self, p):
    self.procs.append(p)

  def compile(self):
    path      = ENV.TMP_DIR
    lib_file  = f"{self.test_name}_lib.c"
    h_file    = f"{self.test_name}_lib.h"
    main_file = f"{self.test_name}_main.c"
    bin_file  = self.test_name

    # write lib.c and lib.h
    compile_procs(self.procs, ENV.GEMM_BUILD_DIR, lib_file, h_file)

    # write main.c
    main_src  = gemmini_test_template(h_file, self.glob, self.body)
    gemmini_write_test_main(main_file, main_src)

    gemmini_compile(main_file, lib_file, bin_file)

    return self

  def run(self):
    gemmini_run(self.test_name)



  def alloc_dram_2i8(self, name, N, M, init):
    self.glob += [f'int8_t {name}[{N}*{M}];','']
    self.body += [f'for(int i=0; i<{N}; i++) {{',
                  f'    for(int j=0; j<{M}; j++) {{',
                  f'        {name}[({M})*i + j] = {init};',
                  f'}}}}',
                  '']

  #def alloc_dram_2i32(self, name, N, M, init):
  #  self.glob += [f'int32_t {name}[{N}*{M}];','']
  #  self.body += [f'for(int i=0; i<{N}; i++) {{',
  #                f'    for(int j=0; j<{M}; j++) {{',
  #                f'        {name}[({M})*i + j] = {init};',
  #                f'}}}}',
  #                '']

  def install_gemm_allocator(self):
    if self._has_gemm_alloc:
      return
    self._has_gemm_alloc = True

    self.glob += ['void gemm_init_mem();',
                  'uint32_t gemm_malloc(long unsigned int size);',
                  'void gemm_free(uint32_t addr);',
                  '',
                  'void gemm_acc_init_mem();',
                  'uint32_t gemm_acc_malloc(long unsigned int size);',
                  'void gemm_acc_free(uint32_t addr);',
                  '']

  def alloc_gemm_1i8(self, name, N, acc=False):
    assert type(N) is int
    self.alloc_gemm_2i8(name, N,16, acc=acc)

  def alloc_gemm_2i8(self, name, N, M, acc=False):
    assert type(N) is int
    assert type(M) is int
    assert M % 16 == 0
    self.install_gemm_allocator()
    self.glob += [f'int8_t *{name};','']
    malloc    = 'gemm_acc_malloc' if acc else 'gemm_malloc'
    self.body += [f'{name} = (int8_t*)((uint64_t){malloc}({N}*{M}/16));',
                  '']

  def add_body(self, lines):
    assert type(lines) is list
    self.body += lines

  def check_2i8(self, N, M, lhs, rhs):
    self.body += [f'']


def test_load_i8_16():
  T = CaseBuilder('load_i8_16')
  T.add_body(['gemm_init_mem();',
              'gemm_acc_init_mem();',
              ''])

  ld_i8 = GEMMINI_INSTR.ld_i8
  st_i8 = GEMMINI_INSTR.st_i8

  @proc
  def dummy():
    tmp : i8[16,16] @ GEMM_ACCUM
  T.add_proc(dummy)

  @proc
  def load_store_i8_16( x : i8[16,16] @ DRAM, y : i8[16,16] @ DRAM ):
    tmp : i8[16,16] @ GEMM_SCRATCH
    ld_i8(16,16, x, tmp)
    st_i8(16,16, tmp, y)
  T.add_proc(load_store_i8_16)

  T.alloc_dram_2i8('x', 16, 16, 'i+j')
  T.alloc_dram_2i8('y', 16, 16, '0')

  T.add_body(['load_store_i8_16(x, y);',
              '',
              'if(check_eq_2i8(16,16, x, y)) {',
              '    printf("Correct\\n");',
              '} else {',
              '    printf("Results Don\'t Match\\n");',
              '    printf("Correct Result (x):\\n");',
              '    print_2i8(16,16, x);',
              '    printf("Computed Roundtrip (y):\\n");',
              '    print_2i8(16,16, y);',
              '}',
              ''])

  T.compile().run()



"""


def gen_conv1d():
    @proc
    def conv1d(K : size, C : size, W : size, R : size,
               w : R[K,C,R],
               x : R[C,W],
               res : R[K,W],
              ):
        # zero out the result memory
        for k_init in par(0,K):
            for i_init in par(0,W):
                res[k_init, i_init] = 0.0

        # do the convolution
        for k in par(0,K):
            for c in par(0,C):
                for i in par(0,W):
                    for r in par(0,R):
                        if 0 <= i-r:
                            res[k,i] += w[k,c,r] * x[c,i-r]
    return conv1d

"""



