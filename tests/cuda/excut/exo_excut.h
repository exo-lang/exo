#ifndef EXO_EXCUT_H_
#define EXO_EXCUT_H_

#include <cuda_runtime.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct exo_ExcutDeviceLog
{
    uint32_t* data;
    uint32_t word_capacity;
} exo_ExcutDeviceLog;

typedef struct exo_ExcutThreadLog
{
    uint32_t* thread_data;
    uint32_t words_per_thread;
} exo_ExcutThreadLog;

void exo_excut_begin_log_file(const char* filename, uint32_t cuda_log_bytes);
bool exo_excut_log_file_enabled();
void exo_excut_end_log_file();
void exo_excut_begin_log_action(const char* action_name);
void exo_excut_log_str_arg(const char* str);
void exo_excut_log_int_arg(uint32_t bytes, const uint32_t* binary);
void exo_excut_log_ptr_arg(void* ptr);
void exo_excut_end_log_action(const char* device_name, unsigned _blockIdx, unsigned _threadIdx, const char* file, int line);
exo_ExcutDeviceLog exo_excut_get_device_log();

#ifdef __cplusplus
}  // extern "C"
#endif

#ifdef __CUDACC__
__device__ exo_ExcutThreadLog exo_excut_begin_thread_log(exo_ExcutDeviceLog device_log)
{
    uint32_t n_threads = gridDim.x * blockDim.x;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t words_per_thread = device_log.word_capacity / n_threads;
    exo_ExcutThreadLog thread_log{device_log.data + words_per_thread * tid, words_per_thread};
    if (words_per_thread != 0) {
        thread_log.thread_data[0] = 0;
    }
    return thread_log;
}
#endif

#endif
