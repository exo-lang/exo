import textwrap


class AMXTestBuilder:
    def __init__(self, basename):
        self.basename = basename
        self.glob = []
        self.body = []
        self._has_amx_alloc = False

    def __str__(self):
        return "\n".join(
            [
                "#include <stdint.h>",
                "#include <stddef.h>",
                "#include <assert.h>",
                "#include <stdlib.h>",
                "#include <stdio.h>",
                "#include <time.h>",
                "#include <immintrin.h>",
                "",
                f'#include "{self.basename}.h"',
                "",
                "void print_2i8(int N, int M, int8_t *data) {",
                "    for(int i=0; i<N; i++) {",
                "        for(int j=0; j<M; j++)",
                '            printf("%d ", (int)data[M*i + j]);',
                '        printf("\\n");',
                "    }",
                "}",
                "void print_4i8(int N, int M, int K, int R, int8_t *data) {",
                '    printf("%d %d %d %d\\n", N, M, K, R);',
                "    for(int i=0; i<N; i++) {",
                "        for(int j=0; j<M; j++) {",
                '            printf("{ ");',
                "            for(int k=0; k<K; k++) {",
                '                printf("{ ");',
                "                for(int r=0; r<R; r++)",
                '                    printf("%d ", (int)data[M*K*R*i + K*R*j + R*k + r]);',
                '                printf("}, ");',
                "            }",
                '            printf("}, ");',
                "        }",
                '        printf("\\n");',
                "    }",
                "}",
                "void print_2i32(int N, int M, int32_t *data) {",
                "    for(int i=0; i<N; i++) {",
                "        for(int j=0; j<M; j++)",
                '            printf("%d ", (int)data[M*i + j]);',
                '        printf("\\n");',
                "    }",
                "}",
                "bool check_eq_2i8(int N, int M, " + "int8_t *lhs, int8_t *rhs) {",
                "    bool flag = true;" "    for(int i=0; i<N; i++) {",
                "        for(int j=0; j<M; j++)",
                "            if(lhs[M*i + j] != rhs[M*i + j])",
                "                flag = false;" "    }",
                "    return flag;" "}",
                "bool check_eq_4i8(int N, int M, int K, int R, int8_t *lhs, int8_t *rhs) {",
                "    bool flag = true;" "    for(int i=0; i<N; i++) {",
                "        for(int j=0; j<M; j++)",
                "            for(int k=0; k<K; k++)",
                "                for(int r=0; r<R; r++)",
                "                    if(lhs[M*K*R*i + K*R*j + R*k + r] != rhs[M*K*R*i + K*R*j + R*k + r])",
                "                        flag = false;" "    }",
                "    return flag;" "}",
                "bool check_eq_2i32(int N, int M, " + "int32_t *lhs, int32_t *rhs) {",
                "    bool flag = true;" "    for(int i=0; i<N; i++) {",
                "        for(int j=0; j<M; j++)",
                "            if(lhs[M*i + j] != rhs[M*i + j])",
                "                flag = false;" "    }",
                "    return flag;" "}",
                "",
                *self.glob,
                "",
                "int main() {",
                textwrap.indent("\n".join(self.body), " " * 4),
                "",
                '    printf("\\nDone\\n");',
                "    ",
                "    exit(0);",
                "}",
                "",
            ]
        )

    def alloc_dram_4i8(self, name, N, M, K, R, init):
        self.glob += [f"int8_t {name}[{N}*{M}*{K}*{R}];", ""]
        self.body += [
            f"for(int i=0; i<{N}; i++) {{",
            f"    for(int j=0; j<{M}; j++) {{",
            f"        for(int k=0; k<{K}; k++) {{",
            f"            for(int r=0; r<{R}; r++) {{",
            f"                {name}[({M}*{K}*{R})*i + ({K}*{R})*j + ({R})*k + r] = {init};",
            f"}}}}}}}}",
            "",
        ]

    def alloc_dram_2i8(self, name, N, M, init):
        self.glob += [f"int8_t {name}[{N}*{M}];", ""]
        self.body += [
            f"for(int i=0; i<{N}; i++) {{",
            f"    for(int j=0; j<{M}; j++) {{",
            f"        {name}[({M})*i + j] = {init};",
            f"}}}}",
            "",
        ]

    def alloc_dram_2i32(self, name, N, M, init):
        self.glob += [f"int32_t {name}[{N}*{M}];", ""]
        self.body += [
            f"for(int i=0; i<{N}; i++) {{",
            f"    for(int j=0; j<{M}; j++) {{",
            f"        {name}[({M})*i + j] = {init};",
            f"}}}}",
            "",
        ]

    def alloc_dram_f32(self, name, init):
        self.glob += [f"float {name}[1];", ""]
        self.body += [f"{name}[0] = {init};", ""]

    def install_amx_allocator(self):
        if self._has_amx_alloc:
            return
        self._has_amx_alloc = True

        # self.glob += ['void gemm_init_mem();',
        #               'uint32_t gemm_malloc(long unsigned int size);',
        #               'void gemm_free(uint32_t addr);',
        #               '',
        #               'void gemm_acc_init_mem();',
        #               'uint32_t gemm_acc_malloc(long unsigned int size);',
        #               'void gemm_acc_free(uint32_t addr);',
        #               '']

    def alloc_amx_1i8(self, name, N, acc=False):
        assert type(N) is int
        self.alloc_amx_2i8(name, N, 16, acc=acc)

    def alloc_amx_2i8(self, name, N, M, acc=False):
        assert type(N) is int
        assert type(M) is int
        assert M % 16 == 0
        self.install_amx_allocator()
        self.glob += [f"int8_t *{name};", ""]
        malloc = "amx_acc_malloc" if acc else "amx_malloc"
        self.body += [f"{name} = (int8_t*)((uint64_t){malloc}({N}*{M}/16));", ""]

    def start_timer(self, name):
        self.body += [f"unsigned long {name}_start = read_cycles();"]

    def stop_timer(self, name, msg):
        self.body += [
            f"unsigned long {name}_stop = read_cycles();",
            f'printf("{msg}: %d\\n", {name}_stop - {name}_start);',
            "",
        ]

    def add_body(self, lines):
        assert type(lines) is list
        self.body += lines
