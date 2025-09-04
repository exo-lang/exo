#pragma once

#include <stdint.h>

namespace camspork
{

inline uint8_t get_low_bit_index(uint32_t n)
{
    return uint8_t(__builtin_ctz(n));
}

inline uint8_t get_low_bit_index(int32_t n)
{
    return uint8_t(__builtin_ctz(n));
}

inline uint8_t get_low_bit_index(uint64_t n)
{
    return uint8_t(__builtin_ctzl(n));
}

inline uint8_t get_low_bit_index(int64_t n)
{
    return uint8_t(__builtin_ctzl(n));
}

template <typename Int_T>
inline uint8_t pop_low_bit_index(Int_T* pn)
{
    const uint8_t index = get_low_bit_index(*pn);
    *pn &= ~(Int_T(1) << index);
    return index;
}

}
