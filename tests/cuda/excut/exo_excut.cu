#include "exo_excut.h"

#include <cassert>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

namespace exo_excut {

thread_local FILE* log_file;
thread_local bool need_action_comma;
thread_local uint32_t* cuda_log;
thread_local uint32_t cuda_log_bytes;

void end_log_file();

void begin_log_file(const char* filename, uint32_t cuda_log_bytes_arg)
{
    end_log_file();

    log_file = fopen(filename, "w");
    if (log_file == nullptr) {
        fprintf(stderr, "exo_excut: could not write to \"%s\": %s (%i)", filename, strerror(errno), errno);
    }
    else {
        need_action_comma = false;
        fprintf(log_file, "[\n");
    }

    void* tmp = nullptr;
    cudaMalloc(&tmp, cuda_log_bytes_arg);
    cuda_log = (uint32_t*)tmp;
    if (cuda_log == nullptr and cuda_log_bytes_arg != 0) {
        fprintf(stderr, "exo_excut: Failed to cudaMalloc %u bytes\n", cuda_log_bytes_arg);
        cuda_log_bytes = 0;
    }
    else {
        cuda_log_bytes = cuda_log_bytes_arg;
    }
}

bool log_file_enabled()
{
    return !!log_file;
}

void end_log_file()
{
    if (log_file) {
        fprintf(log_file, "]\n");
        fclose(log_file);
        log_file = nullptr;
    }
    if (cuda_log) {
        cudaFree(cuda_log);
        cuda_log = nullptr;
    }
    cuda_log_bytes = 0;
}

thread_local bool need_arg_comma;

void begin_log_action(const char* action_name)
{
    assert(log_file_enabled());
    fprintf(log_file, "  %c[\"%s\", [", need_action_comma ? ',' : ' ', action_name);
    need_action_comma = true;
    need_arg_comma = false;
}

void log_str_arg(const char* str)
{
    // NB we currently assume the str does not require escape characters.
    fprintf(log_file, "%s\"str:%s\"", need_arg_comma ? ", " : "", str);
    need_arg_comma = true;
}

template <typename T>
void log_int_arg(T value)
{
    fprintf(log_file, "%s\"int:0x", need_arg_comma ? ", " : "");
    need_arg_comma = true;
    uint32_t words = (sizeof(T) + 3) / 4u;
    // Little endian
    for (uint32_t i = words; i > 0; ) {
        --i;
        fprintf(log_file, "%08X", uint32_t(value >> (32 * i)));
    }
    fprintf(log_file, "\"");
}

void log_int_arg(uint32_t bytes, const uint32_t* binary)
{
    fprintf(log_file, "%s\"int:0x", need_arg_comma ? ", " : "");
    need_arg_comma = true;
    uint32_t words = (bytes + 3) / 4u;
    // Little endian
    for (uint32_t i = words; i > 0; ) {
        --i;
        fprintf(log_file, "%08X", binary[i]);
    }
    fprintf(log_file, "\"");
}

void end_log_action(const char* device_name, unsigned _blockIdx, unsigned _threadIdx, const char* file, int line)
{
    // NB we assume the file name does not require escape characters.
    assert(log_file_enabled());
    fprintf(log_file, "], \"%s\", %u, %u, \"%s\", %i]\n", device_name, _blockIdx, _threadIdx, file, line);
}

exo_ExcutDeviceLog get_device_log()
{
    return {cuda_log, cuda_log_bytes / 4u};
}

}  // end namespace

extern "C" {

void exo_excut_begin_log_file(const char* filename, uint32_t cuda_log_bytes)
{
    exo_excut::begin_log_file(filename, cuda_log_bytes);
}

bool exo_excut_log_file_enabled()
{
    return exo_excut::log_file_enabled();
}

void exo_excut_end_log_file()
{
    exo_excut::end_log_file();
}

void exo_excut_begin_log_action(const char* action_name)
{
    exo_excut::begin_log_action(action_name);
}

void exo_excut_log_str_arg(const char* str)
{
    exo_excut::log_str_arg(str);
}

void exo_excut_log_int_arg(uint32_t bytes, const uint32_t* binary)
{
    exo_excut::log_int_arg(bytes, binary);
}

void exo_excut_log_ptr_arg(void* ptr)
{
    exo_excut::log_int_arg(uintptr_t(ptr));
}

void exo_excut_end_log_action(const char* device_name, unsigned _blockIdx, unsigned _threadIdx, const char* file, int line)
{
    exo_excut::end_log_action(device_name, _blockIdx, _threadIdx, file, line);
}

exo_ExcutDeviceLog exo_excut_get_device_log()
{
    return exo_excut::get_device_log();
}

}  // extern "C"

int main()
{
    exo_excut_begin_log_file("excut_foo.json", 10000);
    exo_excut_begin_log_action("foo.bar");
    exo_excut_log_str_arg("hello");
    exo_excut_log_str_arg("world");
    uint32_t values[4] = { 15, 10, 0xFEED, 0 };
    exo_excut_log_int_arg(4, &values[2]);
    exo_excut_end_log_action("cuda", 0, 1, "barf.cu", 111);
    exo_excut_begin_log_action("again");
    exo_excut_log_int_arg(8, &values[0]);
    exo_excut_end_log_action("cuda", 3, 1, "barf.cu", 112);
    exo_excut_end_log_file();
    return 0;
}
