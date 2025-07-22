#include "exo_excut.h"

#include <cassert>
#include <errno.h>
#include <memory>
#include <stdint.h>
#include <stdio.h>
#include <string.h>


#ifndef NDEBUG
#define EXO_EXCUT_ASSERT(cond) assert(cond)
#else
#define EXO_EXCUT_ASSERT(cond) if (!(cond)) { fprintf(stderr, "%s:%i: %s\n", __FILE__, __LINE__, #cond); fflush(stderr); abort(); }
#endif


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

    // Update exo_excut.h if you change the allocator!
    void* tmp = nullptr;
    cudaMallocAsync(&tmp, cuda_log_bytes_arg, cudaStreamLegacy);
    cuda_log = (uint32_t*)tmp;
    if (cuda_log == nullptr and cuda_log_bytes_arg != 0) {
        fprintf(stderr, __FILE__ ": Failed to cudaMallocAsync %u bytes\n", cuda_log_bytes_arg);
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
        cudaFreeAsync(cuda_log, cudaStreamLegacy);
        cuda_log = nullptr;
    }
    cuda_log_bytes = 0;
}

thread_local bool need_arg_comma;

void begin_log_action(const char* action_name)
{
    EXO_EXCUT_ASSERT(log_file_enabled());
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
    EXO_EXCUT_ASSERT(log_file_enabled());
    fprintf(log_file, "], \"%s\", %u, %u, \"%s\", %i]\n", device_name, _blockIdx, _threadIdx, file, line);
}

exo_ExcutDeviceLog get_device_log()
{
    return {cuda_log, cuda_log_bytes / 4u};
}

void flush_device_log(cudaStream_t stream, uint32_t gridDim, uint32_t blockDim,
                      uint32_t str_id_count, const char* const* str_table,
                      uint32_t file_id_count, const char* const* file_table)
{
    if (!log_file_enabled()) {
        return;
    }

    std::unique_ptr<uint32_t[]> data(new uint32_t[(cuda_log_bytes + 3) / 4u]);
    cudaMemcpyAsync(data.get(), cuda_log, cuda_log_bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const cudaError_t peeked_error = cudaPeekLastError();
    if (peeked_error) {
        fprintf(stderr, "%s:%i: %i (%s)\n", __FILE__, __LINE__, (int)peeked_error, cudaGetErrorName(peeked_error));
        fflush(stderr);
    }
    const uint32_t n_threads = gridDim * blockDim;
    const uint32_t words_per_thread = cuda_log_bytes / 4u / n_threads;

    auto lookup_str_id = [str_id_count, str_table](uint32_t str_id)
    {
        if (str_id < str_id_count) {
            return str_table[str_id];
        }
        return "<invalid str_id>";
    };

    auto lookup_file_id = [file_id_count, file_table](uint32_t file_id)
    {
        if (file_id < file_id_count) {
            return file_table[file_id];
        }
        return "<invalid file_id>";
    };

    auto log_out_of_cuda_memory = [&] (uint32_t blockIdx, uint32_t threadIdx, uint32_t needed)
    {
        begin_log_action("excut::out_of_cuda_memory");
        log_int_arg(4, &needed);
        end_log_action("cuda", blockIdx, threadIdx, __FILE__, __LINE__);
    };

    for (uint32_t tid = 0; tid < n_threads; ++tid) {
        const uint32_t threadIdx = tid % blockDim;
        const uint32_t blockIdx = tid / blockDim;
        const uint32_t* thread_data = &data[tid * words_per_thread];

        if (words_per_thread == 0) {
            // If there's so little memory that even the "words written" field doesn't
            // fit, then we can't calculate the memory needed. Just ask for enough
            // so next time, at least that field gets written.
            log_out_of_cuda_memory(blockIdx, threadIdx, 4u * gridDim * blockDim);
            continue;
        }
        const uint32_t n_log_words = thread_data[0];
        const uint32_t n_all_words = n_log_words + 1u;  // Extra word for holding n_log_words
        if (n_all_words > words_per_thread) {
            // Ask for more memory next time
            log_out_of_cuda_memory(blockIdx, threadIdx, 4u * gridDim * blockDim * n_all_words);
            continue;
        }

        // Translate binary trace to text
        const uint32_t* log_words = &thread_data[1];
        bool in_action = false;
        uint32_t action_file_id = 0;
        int action_line = 0;
        auto end_action = [&]
        {
            if (in_action) {
                end_log_action("cuda", blockIdx, threadIdx, lookup_file_id(action_file_id), action_line);
                in_action = false;
            }
        };

        uint32_t log_i = 0;
        while (log_i < n_log_words) {
            uint32_t cmd = log_words[log_i];
            auto high_byte = static_cast<exo_ExcutHighByte>(cmd >> 24);
            uint32_t cmd_value = cmd & 0xFFFFFF;

            switch (high_byte) {
              case exo_ExcutHighByte::action:
                end_action();
                EXO_EXCUT_ASSERT(log_i + 3 <= n_log_words);
                begin_log_action(lookup_str_id(cmd_value));
                action_file_id = log_words[log_i + 1];
                action_line = int(log_words[log_i + 2]);
                log_i += 3;
                in_action = true;
                break;
              case exo_ExcutHighByte::str_arg:
                EXO_EXCUT_ASSERT(in_action && "CUDA tracer logged arg without action");
                EXO_EXCUT_ASSERT(log_i < n_log_words);
                exo_excut_log_str_arg(lookup_str_id(cmd_value));
                log_i += 1;
                break;
              case exo_ExcutHighByte::int_arg:
                EXO_EXCUT_ASSERT(in_action && "CUDA tracer logged arg without action");
                {
                    const uint32_t int_bytes = cmd_value;
                    // Int encoded as # of bytes plus value padded to 4-byte alignment.
                    const uint32_t n_arg_words = 1u + (int_bytes + 3u) / 4u;
                    EXO_EXCUT_ASSERT(log_i + n_arg_words <= n_log_words);
                    exo_excut_log_int_arg(int_bytes, &log_words[log_i + 1]);
                    log_i += n_arg_words;
                }
                break;
              default:
                fprintf(stderr, "log_i = %u, cmd = %08X\n", log_i, cmd);
                EXO_EXCUT_ASSERT(!"CUDA tracer logged unknown exo_ExcutHighByte");
            }
        }
        end_action();
    }
}

}  // end namespace

extern "C" {

void exo_excut_begin_log_file(const char* filename, uint32_t cuda_log_bytes)
{
    exo_excut::begin_log_file(filename, cuda_log_bytes);
}

int exo_excut_log_file_enabled()
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

void exo_excut_flush_device_log(cudaStream_t exo_cudaStream, uint32_t gridDim, uint32_t blockDim,
                                uint32_t str_id_count, const char* const* str_table,
                                uint32_t file_id_count, const char* const* file_table)
{
    exo_excut::flush_device_log(exo_cudaStream, gridDim, blockDim, str_id_count, str_table, file_id_count, file_table);
}

}  // extern "C"

#if defined(EXO_EXCUT_bMAIN) && EXO_EXCUT_bMAIN

const char* const str_table[] = {
    __FILE__,
    "str_one",
    "str_two",
    "str_three",
    "str_four",
};

const uint32_t str_id_count = sizeof(str_table) / sizeof(const char*);

struct TestBlob
{
    uint32_t values[16];
};

__global__ void test_kernel(uint32_t special_block_id, uint32_t special_action_id, uint32_t special_action_count,
                            TestBlob test_blob, exo_ExcutDeviceLog device_log)
{
    exo_ExcutThreadLog log = exo_excut_begin_thread_log(device_log);
    log.log_action(1, 0, __LINE__);
    log.log_str_id_arg(2);
    log.log_str_id_arg(3);
    log.log_ptr_arg(device_log.data);

    if (blockIdx.x == special_block_id) {
        for (uint32_t i = 0; i < special_action_count; ++i) {
            log.log_action(4, 0, __LINE__);
            log.log_u32_arg(42);
            log.log_u64_arg(0x123456789ABCDEF);
            log.log_ptr_data_arg(&test_blob);
        }
    }
}

void launch_test_kernel(uint32_t gridDim, uint32_t blockDim, exo_ExcutDeviceLog device_log)
{
    TestBlob test_blob{{3, 9, 7, 9, 8, 5, 3, 5, 6, 2, 9, 5, 1, 4, 1, 3}};
    test_kernel<<<gridDim, blockDim>>>(3, 4, 3, test_blob, device_log);
    exo_excut_flush_device_log(0, gridDim, blockDim, str_id_count, str_table);
}

int main()
{
    exo_excut_begin_log_file("excut_main.json", 0x15000);

    exo_excut_begin_log_action("foo.bar");
    exo_excut_log_str_arg("hello");
    exo_excut_log_str_arg("world");
    uint32_t values[4] = { 15, 10, 0xFEED, 0 };
    exo_excut_log_int_arg(4, &values[2]);
    exo_excut_end_log_action("cpu", 0, 0, __FILE__, __LINE__);

    launch_test_kernel(1, 32, exo_excut_get_device_log());

    exo_excut_begin_log_action("again");
    exo_excut_log_int_arg(8, &values[0]);
    exo_excut_end_log_action("cuda", 3, 1, __FILE__, __LINE__);

    launch_test_kernel(4, 64, exo_excut_get_device_log());
    launch_test_kernel(8, 64, exo_excut_get_device_log());

    exo_excut_end_log_file();
    return 0;
}

#endif
