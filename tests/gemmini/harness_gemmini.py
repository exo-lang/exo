import os
import subprocess

import pytest

from SYS_ATL import compile_procs

GEMMINI_ROOT = os.getenv('GEMMINI_ROOT')
if GEMMINI_ROOT is None:
    RISCV = os.getenv('RISCV')
    if RISCV is None:
        pass
        #pytest.skip("skipping gemmini tests; could not find chipyard",
        #            allow_module_level=True)
    else:
        GEMMINI_ROOT  = os.path.join(RISCV,'..','generators','gemmini')


if RISCV is not None:
    GEMMINI_ROOT        = os.path.abspath(GEMMINI_ROOT)
    CHIPYARD_ROOT       = os.path.abspath(os.path.join(GEMMINI_ROOT,'..','..'))
    SIMS_VCS_DIR        = os.path.join(CHIPYARD_ROOT,'sims','vcs')
    GEMMINI_ROCC_TESTS  = os.path.join(GEMMINI_ROOT,'software',
                                                    'gemmini-rocc-tests')
    ROOT                = GEMMINI_ROCC_TESTS
    BCOMMON             = os.path.join(ROOT,'riscv-tests','benchmarks',
                                            'common')

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
DIR_TEST_ROOT       = _HERE_
TMP_DIR             = os.path.join(DIR_TEST_ROOT,'tmp')

GEMM_BUILD_DIR      = os.path.join(DIR_TEST_ROOT,'gemmini_build')

if RISCV is not None:
    COMPILE             = (f"{CC_BAREMETAL} {CFLAGS_BAREMETAL} "+
                           f"{BCOMMON}/*.c {BCOMMON}/*.S")

# make sure the build directory exists
os.makedirs(GEMM_BUILD_DIR, exist_ok=True)

class ENV:
    pass


if RISCV is not None:
    ENV.GEMMINI_ROOT        = GEMMINI_ROOT
    ENV.CHIPYARD_ROOT       = CHIPYARD_ROOT
    ENV.SIMS_VCS_DIR        = SIMS_VCS_DIR
    ENV.GEMMINI_ROCC_TESTS  = GEMMINI_ROCC_TESTS
    ENV.ROOT                = ROOT
    ENV.BCOMMON             = BCOMMON

    ENV.CC_BAREMETAL        = CC_BAREMETAL
    ENV.CFLAGS_BAREMETAL    = CFLAGS_BAREMETAL

ENV.SYSTL_ROOT          = SYSTL_ROOT
ENV.DIR_TEST_ROOT       = DIR_TEST_ROOT
ENV.TMP_DIR             = TMP_DIR

ENV.GEMM_BUILD_DIR      = GEMM_BUILD_DIR

if RISCV is not None:
    ENV.COMPILE             = COMPILE




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

def gemmini_run_on_vcs(binfile):
    binfile   = os.path.join(ENV.GEMM_BUILD_DIR, binfile)
    CMD = f"{SIMS_VCS_DIR}/simv-chipyard-GemminiRocketConfig {binfile}"

    if 0 != subprocess.call(CMD, shell=True):
        raise OSError("VCS Execution Failed")

class GemmTestBuilder:
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
                      'void print_4i8(int N, int M, int K, int R, int8_t *data) {',
                      '    printf("%d %d %d %d\\n", N, M, K, R);',
                      '    for(int i=0; i<N; i++) {',
                      '        for(int j=0; j<M; j++) {',
                      '            printf("{ ");',
                      '            for(int k=0; k<K; k++) {',
                      '                printf("{ ");',
                      '                for(int r=0; r<R; r++)',
                      '                    printf("%d ", (int)data[M*K*R*i + K*R*j + R*k + r]);',
                      '                printf("}, ");',
                      '            }',
                      '            printf("}, ");',
                      '        }',
                      '        printf("\\n");',
                      '    }',
                      '}',
                      'void print_2i32(int N, int M, int32_t *data) {',
                      '    for(int i=0; i<N; i++) {',
                      '        for(int j=0; j<M; j++)',
                      '            printf("%d ", (int)data[M*i + j]);',
                      '        printf("\\n");',
                      '    }',
                      '}',
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
                      'bool check_eq_4i8(int N, int M, int K, int R, '+
                                       'int8_t *lhs, int8_t *rhs) {',
                      '    bool flag = true;'
                      '    for(int i=0; i<N; i++) {',
                      '        for(int j=0; j<M; j++)',
                      '            for(int k=0; k<K; k++)',
                      '                for(int r=0; r<R; r++)',
                      '                    if(lhs[M*K*R*i + K*R*j + R*k + r] != rhs[M*K*R*i + K*R*j + R*k + r])',
                      '                        flag = false;'
                      '    }',
                      '    return flag;'
                      '}',
                      'bool check_eq_2i32(int N, int M, '+
                                       'int32_t *lhs, int32_t *rhs) {',
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

        if RISCV is not None:
            gemmini_compile(main_file, lib_file, bin_file)

        return self

    def run(self):
        if RISCV is not None:
            gemmini_run(self.test_name)

    def vcs(self):
        if RISCV is not None:
            gemmini_run_on_vcs(self.test_name)

    def alloc_dram_4i8(self, name, N, M, K, R, init):
        self.glob += [f'static int8_t {name}[{N}*{M}*{K}*{R}];','']
        self.body += [f'for(int i=0; i<{N}; i++) {{',
                      f'    for(int j=0; j<{M}; j++) {{',
                      f'        for(int k=0; k<{K}; k++) {{',
                      f'            for(int r=0; r<{R}; r++) {{',
                      f'                {name}[({M}*{K}*{R})*i + ({K}*{R})*j + ({R})*k + r] = {init};',
                      f'}}}}}}}}',
                      '']


    def alloc_dram_2i8(self, name, N, M, init):
        self.glob += [f'static int8_t {name}[{N}*{M}];','']
        self.body += [f'for(int i=0; i<{N}; i++) {{',
                      f'    for(int j=0; j<{M}; j++) {{',
                      f'        {name}[({M})*i + j] = {init};',
                      f'}}}}',
                      '']

    def alloc_dram_2i32(self, name, N, M, init):
        self.glob += [f'static int32_t {name}[{N}*{M}];','']
        self.body += [f'for(int i=0; i<{N}; i++) {{',
                      f'    for(int j=0; j<{M}; j++) {{',
                      f'        {name}[({M})*i + j] = {init};',
                      f'}}}}',
                      '']

    def alloc_dram_f32(self, name, init):
        self.glob += [f'float {name}[1];','']
        self.body += [f'{name}[0] = {init};','']

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
        assert isinstance(N, int)
        self.alloc_gemm_2i8(name, N,16, acc=acc)

    def alloc_gemm_2i8(self, name, N, M, acc=False):
        assert isinstance(N, int)
        assert isinstance(M, int)
        assert M % 16 == 0
        self.install_gemm_allocator()
        self.glob += [f'int8_t *{name};', '']
        malloc = 'gemm_acc_malloc' if acc else 'gemm_malloc'
        self.body += [f'{name} = (int8_t*)((uint64_t){malloc}({N}*{M}/16));',
                      '']

    def start_timer(self, name):
        self.body += [f'unsigned long {name}_start = read_cycles();']

    def stop_timer(self, name, msg):
        self.body += [f'unsigned long {name}_stop = read_cycles();',
                      f'printf("{msg}: %d\\n", {name}_stop - {name}_start);',
                      '']

    def add_body(self, lines):
        assert isinstance(lines, list)
        self.body += lines
