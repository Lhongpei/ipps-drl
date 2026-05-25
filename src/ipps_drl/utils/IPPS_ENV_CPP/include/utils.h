#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <unordered_set>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <functional>

namespace detail {
// Thread-local PRNG. The previous implementation called `srand(time(nullptr))`
// before every draw, which collapses the seed to a one-second resolution and is
// not thread-safe. Using a thread-local `mt19937` gives each rollout thread its
// own independent stream.
inline std::mt19937& thread_rng() {
    thread_local std::mt19937 rng(
        std::chrono::steady_clock::now().time_since_epoch().count()
        ^ (std::hash<std::thread::id>{}(std::this_thread::get_id())));
    return rng;
}
}  // namespace detail

template <typename T>
T randSelectSet(const std::unordered_set<T>& set) {
    if (set.empty()) {
        throw std::runtime_error("The set is empty");
    }
    std::vector<T> vec(set.begin(), set.end());
    std::uniform_int_distribution<size_t> dist(0, vec.size() - 1);
    return vec[dist(detail::thread_rng())];
}
template <typename T>
T randSelectVector(const std::vector<T>& vec) {
    if (vec.empty()) {
        throw std::runtime_error("The vector is empty");
    }
    std::uniform_int_distribution<size_t> dist(0, vec.size() - 1);
    return vec[dist(detail::thread_rng())];
}

#endif // UTILS_H