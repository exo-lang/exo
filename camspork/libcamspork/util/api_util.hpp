#pragma once

#include <new>
#include <stdexcept>

#define CAMSPORK_EXPORT extern "C" __attribute__ ((visibility ("default")))

// Surround body of non-noexcept C API function with these.
// Sentinel value returned in case of exception.
// Note: these use error types not included here, because that would lead to circular includes.

#define CAMSPORK_API_PROLOGUE try {

#define CAMSPORK_API_EPILOGUE(sentinel) } \
    catch (const ::camspork::RequireFail&) { \
        return sentinel; \
    } \
    catch (const std::bad_alloc&) { \
        ::camspork::thread_local_message_ref() = "out of memory"; \
        return sentinel; \
    } \
    catch (const std::runtime_error& error) { \
        try { \
            ::camspork::thread_local_message_ref() = error.what(); \
            return sentinel; \
        } catch (...) { \
            ::camspork::thread_local_message_ref() = "unknown error"; \
            return sentinel; \
        } \
    } \
    catch (...) { \
        ::camspork::thread_local_message_ref() = "unknown error"; \
        return sentinel; \
    }
