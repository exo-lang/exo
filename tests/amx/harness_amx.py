import distutils.spawn
import os
import subprocess

import pytest

from SYS_ATL import compile_procs

SDE = (distutils.spawn.find_executable("sde64", os.getenv('SDE_PATH'))
       or distutils.spawn.find_executable("sde64"))
if not SDE:
    pytest.skip("skipping AMX tests; could not find sde",
                allow_module_level=True)

CC_BAREMETAL = os.getenv('CLANG') or os.getenv('CC', 'clang-13')
CFLAGS_BAREMETAL = ' '.join([
    f'-mamx-int8',
    f'-mamx-tile',
])

_HERE_ = os.path.dirname(os.path.abspath(__file__))
SYSTL_ROOT = os.path.abspath(os.path.join(_HERE_, '..'))
DIR_TEST_ROOT = _HERE_
TMP_DIR = os.path.join(DIR_TEST_ROOT, 'tmp')

AMX_BUILD_DIR = os.path.join(DIR_TEST_ROOT, 'amx_build')

COMPILE = (f"{CC_BAREMETAL} {CFLAGS_BAREMETAL}")

# make sure the build directory exists
os.makedirs(AMX_BUILD_DIR, exist_ok=True)


class ENV:
    pass


ENV.CC_BAREMETAL = CC_BAREMETAL
ENV.CFLAGS_BAREMETAL = CFLAGS_BAREMETAL

ENV.SYSTL_ROOT = SYSTL_ROOT
ENV.DIR_TEST_ROOT = DIR_TEST_ROOT
ENV.TMP_DIR = TMP_DIR

ENV.AMX_BUILD_DIR = AMX_BUILD_DIR

ENV.COMPILE = COMPILE


def amx_test_template(incl_file, glob_lines, body_lines):
    lines = ['#include <stdint.h>',
             '#include <stddef.h>',
             '#include <assert.h>',
             '#include <stdlib.h>',
             '#include <stdio.h>',
             '#include <time.h>',
             '#include <immintrin.h>',
             '',
             f'#include "{incl_file}"',
             '',
             ]

    assert isinstance(glob_lines, list)
    lines += glob_lines

    lines += ['',
              'int main() {',
              ]
    assert isinstance(body_lines, list)

    lines += ["    " + ln for ln in body_lines]

    lines += ['',
              '    printf("\\nDone\\n");',
              '    ',
              '    exit(0);',
              '}',
              '',
              ]

    return '\n'.join(lines)


def amx_write_test_main(filename, body):
    with open(os.path.join(ENV.AMX_BUILD_DIR, filename), "w") as F:
        F.write(body)


def amx_compile(mainfile, libfile, binfile):
    mainfile = os.path.join(ENV.AMX_BUILD_DIR, mainfile)
    libfile = os.path.join(ENV.AMX_BUILD_DIR, libfile)
    binfile = os.path.join(ENV.AMX_BUILD_DIR, binfile)
    CMD = f"{ENV.COMPILE} -I{ENV.TMP_DIR} {mainfile} {libfile} -o {binfile}"

    if 0 != subprocess.call(CMD, shell=True):
        raise OSError("Compilation Failed")


def amx_run(binfile):
    binfile = os.path.join(ENV.AMX_BUILD_DIR, binfile)
    CMD = f"{SDE} -future -- {binfile}"

    if 0 != subprocess.call(CMD, shell=True):
        raise OSError("Spike Execution Failed")


class AMXTestBuilder:
    def __init__(self, test_name):
        self.test_name = test_name
        self.glob = []
        self.body = []
        self.procs = []
        self._has_amx_alloc = False

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
                      'bool check_eq_2i8(int N, int M, ' +
                      'int8_t *lhs, int8_t *rhs) {',
                      '    bool flag = true;'
                      '    for(int i=0; i<N; i++) {',
                      '        for(int j=0; j<M; j++)',
                      '            if(lhs[M*i + j] != rhs[M*i + j])',
                      '                flag = false;'
                      '    }',
                      '    return flag;'
                      '}',
                      'bool check_eq_4i8(int N, int M, int K, int R, ' +
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
                      'bool check_eq_2i32(int N, int M, ' +
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
        path = ENV.TMP_DIR
        lib_file = f"{self.test_name}_lib.c"
        h_file = f"{self.test_name}_lib.h"
        main_file = f"{self.test_name}_main.c"
        bin_file = self.test_name

        # write lib.c and lib.h
        compile_procs(self.procs, ENV.AMX_BUILD_DIR, lib_file, h_file)

        # write main.c
        main_src = amx_test_template(h_file, self.glob, self.body)
        amx_write_test_main(main_file, main_src)

        amx_compile(main_file, lib_file, bin_file)

        return self

    def run(self):
        amx_run(self.test_name)

    def alloc_dram_4i8(self, name, N, M, K, R, init):
        self.glob += [f'int8_t {name}[{N}*{M}*{K}*{R}];', '']
        self.body += [f'for(int i=0; i<{N}; i++) {{',
                      f'    for(int j=0; j<{M}; j++) {{',
                      f'        for(int k=0; k<{K}; k++) {{',
                      f'            for(int r=0; r<{R}; r++) {{',
                      f'                {name}[({M}*{K}*{R})*i + ({K}*{R})*j + ({R})*k + r] = {init};',
                      f'}}}}}}}}',
                      '']

    def alloc_dram_2i8(self, name, N, M, init):
        self.glob += [f'int8_t {name}[{N}*{M}];', '']
        self.body += [f'for(int i=0; i<{N}; i++) {{',
                      f'    for(int j=0; j<{M}; j++) {{',
                      f'        {name}[({M})*i + j] = {init};',
                      f'}}}}',
                      '']

    def alloc_dram_2i32(self, name, N, M, init):
        self.glob += [f'int32_t {name}[{N}*{M}];', '']
        self.body += [f'for(int i=0; i<{N}; i++) {{',
                      f'    for(int j=0; j<{M}; j++) {{',
                      f'        {name}[({M})*i + j] = {init};',
                      f'}}}}',
                      '']

    def alloc_dram_f32(self, name, init):
        self.glob += [f'float {name}[1];', '']
        self.body += [f'{name}[0] = {init};', '']

    def install_amx_allocator(self):
        if self._has_amx_alloc:
            return
        self._has_amx_alloc = True

    #        self.glob += ['void gemm_init_mem();',
    #                      'uint32_t gemm_malloc(long unsigned int size);',
    #                      'void gemm_free(uint32_t addr);',
    #                      '',
    #                      'void gemm_acc_init_mem();',
    #                      'uint32_t gemm_acc_malloc(long unsigned int size);',
    #                      'void gemm_acc_free(uint32_t addr);',
    #                      '']

    def alloc_amx_1i8(self, name, N, acc=False):
        assert type(N) is int
        self.alloc_amx_2i8(name, N, 16, acc=acc)

    def alloc_amx_2i8(self, name, N, M, acc=False):
        assert type(N) is int
        assert type(M) is int
        assert M % 16 == 0
        self.install_amx_allocator()
        self.glob += [f'int8_t *{name};', '']
        malloc = 'amx_acc_malloc' if acc else 'amx_malloc'
        self.body += [f'{name} = (int8_t*)((uint64_t){malloc}({N}*{M}/16));',
                      '']

    def start_timer(self, name):
        self.body += [f'unsigned long {name}_start = read_cycles();']

    def stop_timer(self, name, msg):
        self.body += [f'unsigned long {name}_stop = read_cycles();',
                      f'printf("{msg}: %d\\n", {name}_stop - {name}_start);',
                      '']

    def add_body(self, lines):
        assert type(lines) is list
        self.body += lines
