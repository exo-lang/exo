#pragma once

#include <algorithm>
#include <utility>
#include <vector>

namespace camspork {

// Sorted vector, useful for binary-search lookups.
// Not an encapsulated class; just some convenient functions to help uphold (but not enforce) the invariant.
template <typename T>
struct SortedVector
{
    std::vector<T> data;

    size_t size() const { return data.size(); }
    const T& operator[] (size_t i) const { return data[i]; }
    T& operator[] (size_t i) { return data[i]; }
    auto begin() { return data.begin(); }
    auto begin() const { return data.begin(); }
    auto end() { return data.end(); }
    auto end() const { return data.end(); }

    template <typename MatchMe>
    const T* find_ptr(const MatchMe& match_me) const
    {
        auto iter = std::lower_bound(data.begin(), data.end(), match_me);
        if (iter == data.end() || *iter != match_me) {
            return nullptr;
        }
        else {
            return &*iter;
        }
    }

    template <typename MatchMe>
    T* find_ptr(const MatchMe& match_me)
    {
        const auto& self = *this;
        return const_cast<T*>(self.find_ptr(match_me));
    }

    // Returns did-remove flag.
    template <typename MatchMe>
    bool erase(const MatchMe& match_me)
    {
        auto iter = std::lower_bound(data.begin(), data.end(), match_me);
        if (iter == data.end() || *iter != match_me) {
            return false;
        }
        else {
            data.erase(iter);
            return true;
        }
    }

    // Returns
    // * if found: {true, iter to result}
    // * not found: {false, iter for insertion}
    template <typename MatchMe>
    std::pair<bool, typename std::vector<T>::iterator> insertion_point(const MatchMe& match_me)
    {
        auto iter = std::lower_bound(data.begin(), data.end(), match_me);
        const bool found = iter != data.end() && *iter == match_me;
        return {found, iter};
    }
};

}
