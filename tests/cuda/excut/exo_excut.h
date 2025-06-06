#ifndef EXO_EXCUT_H_
#define EXO_EXCUT_H_

#if !defined(EXO_EXCUT_bENABLE_LOG) || !EXO_EXCUT_bENABLE_LOG
#error define EXO_EXCUT_bENABLE_LOG to 1
#endif

#include <cuda_runtime.h>
#include <stdint.h>

#define EXO_EXCUT_CUDA_INLINE __device__ __forceinline__

#ifdef __cplusplus
extern "C" {
#endif

typedef struct exo_ExcutDeviceLog
{
    uint32_t* data;
    uint32_t word_capacity;
} exo_ExcutDeviceLog;

// CPU-side API. Directly emit text to excut JSON trace file.
// This manipulates thread_local state; the same CUDA device must be used throughout.
void exo_excut_begin_log_file(const char* filename, uint32_t cuda_log_bytes);
int exo_excut_log_file_enabled();
void exo_excut_end_log_file();
void exo_excut_begin_log_action(const char* action_name);
void exo_excut_log_str_arg(const char* str);
void exo_excut_log_int_arg(uint32_t bytes, const uint32_t* binary);
void exo_excut_log_ptr_arg(void* ptr);
void exo_excut_end_log_action(const char* device_name, unsigned _blockIdx, unsigned _threadIdx, const char* file, int line);
exo_ExcutDeviceLog exo_excut_get_device_log();
void exo_excut_flush_device_log(cudaStream_t exo_cudaStream, unsigned gridDim, unsigned blockDim,
                                uint32_t str_id_count, const char* const* str_table);

#ifdef __cplusplus
}  // extern "C"
#endif

// GPU-side interface
//
// Excut is written for Exo's internal test suite, so this is just an ad-hoc
// interface, nothing too serious.
//
// The GPU logs a binary trace, which the CPU copies back and encodes in text
// format to the log file.
//
// * The log is a device array of uint32_t[word_capacity];
//
// * Threads are indexed by tid = blockDim.x * blockIdx.x + threadIdx.x.
//   words_per_thread = word_capacity / (gridDim.x * blockDim.x)
//   Each thread gets uint32_t thread_data[words_per_thread]
//   starting at offset tid * words_per_thread
//
// * thread_data[0] stores the number of log words written (initially 0)
//   thread_data[1:] stores log words. If we have more to log than will
//   fit, we skip writing the extra words, but still increment
//   log_words_written(), so the CPU can detect the out-of-memory condition.
//
// * Strings are encoded as 24-bit integer ids, indexing into a string table
//   that exocc generates.
//
// * encode(type, value) = exo_ExcutHighByte::type << 24 | value & 0xFFFFFF
//
// * Write a new action by appending to the log words:
//   [encode(action, str_id(action_name)), str_id(file_name), line_number]
//
// * After appending an action, add a string argument by appending:
//   [encode(str_arg, str_id)]
//
// * Add an integer argument T by appending encode(int_arg, sizeof(T)) followed
//   by the integer argument in little endian.
//
// * Unlike the CPU functions, there's no "end action" function required.
//
// * We can't encode sink ("_") or variables as these are only for the ref JSON

#ifdef __CUDACC__

enum class exo_ExcutHighByte
{
    action = 1,
    str_arg = 2,
    int_arg = 3,
};

struct exo_ExcutThreadLog
{
    uint32_t* thread_data;
    uint32_t words_per_thread;

    EXO_EXCUT_CUDA_INLINE uint32_t& log_words_written() const
    {
        return thread_data[0];
    }

    EXO_EXCUT_CUDA_INLINE uint32_t* _alloc_log(uint32_t word_count) const
    {
        uint32_t idx = log_words_written() + 1;
        log_words_written() += word_count;
        if (idx + word_count <= words_per_thread) {
            return &thread_data[idx];
        }
        return nullptr;  // Not enough capacity
    }

    EXO_EXCUT_CUDA_INLINE void log_action(uint32_t action_str_id, uint32_t file_str_id, uint32_t line) const
    {
        if (uint32_t* p = _alloc_log(3)) {
            p[0] = uint32_t(exo_ExcutHighByte::action) << 24 | action_str_id & 0xFFFFFF;
            p[1] = file_str_id;
            p[2] = line;
        }
    }

    EXO_EXCUT_CUDA_INLINE void log_str_id_arg(uint32_t str_id) const
    {
        if (uint32_t* p = _alloc_log(1)) {
            p[0] = uint32_t(exo_ExcutHighByte::str_arg) << 24 | str_id & 0xFFFFFF;
        }
    }

    EXO_EXCUT_CUDA_INLINE void log_u32_arg(uint32_t n) const
    {
        if (uint32_t* p = _alloc_log(2)) {
            p[0] = uint32_t(exo_ExcutHighByte::int_arg) << 24 | 4u;
            p[1] = n;
        }
    }

    EXO_EXCUT_CUDA_INLINE void log_u64_arg(uint64_t n) const
    {
        if (uint32_t* p = _alloc_log(3)) {
            p[0] = uint32_t(exo_ExcutHighByte::int_arg) << 24 | 8u;
            p[1] = uint32_t(n);
            p[2] = uint32_t(n >> 32);
        }
    }

    EXO_EXCUT_CUDA_INLINE void log_ptr_arg(void* ptr) const
    {
        static_assert(sizeof(ptr) == 8, "Assumed 64-bit");
        log_u64_arg(uint64_t(ptr));
    }

    // Log some binary T[count] object as a big integer.
    template <typename T>
    EXO_EXCUT_CUDA_INLINE void log_ptr_data_arg(const T* data, uint32_t count = 1)
    {
        static_assert(sizeof(T) % 4u == 0);
        const uint32_t n_value_words = sizeof(T) * count / 4u;
        if (uint32_t* p = _alloc_log(n_value_words + 1)) {
            p[0] = uint32_t(exo_ExcutHighByte::int_arg) << 24 | (n_value_words * 4u) & 0xFFFFFF;
            uint32_t* payload = p + 1;
            for (uint32_t i = 0; i < n_value_words; ++i) {
                payload[i] = reinterpret_cast<const uint32_t*>(data)[i];  // Strict aliasing violation
            }
        }
    }
};

EXO_EXCUT_CUDA_INLINE exo_ExcutThreadLog exo_excut_begin_thread_log(exo_ExcutDeviceLog device_log)
{
    uint32_t n_threads = gridDim.x * blockDim.x;
    uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    uint32_t words_per_thread = device_log.word_capacity / n_threads;
    exo_ExcutThreadLog thread_log{device_log.data + words_per_thread * tid, words_per_thread};
    if (words_per_thread != 0) {
        thread_log.log_words_written() = 0;
    }
    return thread_log;
}
#endif

#endif
