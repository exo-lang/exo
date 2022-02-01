#pragma once

#ifndef UTIL_H
#define UTIL_H

#include <random>
#include <vector>

namespace util {

template <typename T>
T absd(T a, T b) {
  return a < b ? b - a : a - b;
}

template <typename T>
bool all_close(const std::vector<T> &actual, const std::vector<T> &desired) {
  if (actual.size() != desired.size()) {
    return false;
  }

  constexpr double atol = 0.0;
  constexpr double rtol = 1e-7;

  for (decltype(actual.size()) i = 0; i < actual.size(); ++i) {
    if (absd(actual[i], desired[i]) > atol + rtol * desired[i]) {
      return false;
    }
  }
  return true;
}

template <typename T>
std::vector<T> gen_matrix(long m, long n) {
  static std::random_device rd;
  static std::mt19937 rng{rd()};
  std::uniform_real_distribution<T> rv{-1.0, 1.0};

  std::vector<T> mat(m * n);
  std::generate(std::begin(mat), std::end(mat), [&]() { return rv(rng); });

  return mat;
}

};  // namespace util

#endif
