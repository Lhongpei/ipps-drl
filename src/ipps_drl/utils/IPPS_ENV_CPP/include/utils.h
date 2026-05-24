#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <unordered_set>
#include <vector>
#include <cstdlib>
#include <ctime>
template <typename T>
T randSelectSet(const std::unordered_set<T>& set) {
    if (set.empty()) {
        throw std::runtime_error("The set is empty");
    }

    std::vector<T> vec(set.begin(), set.end());

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    int randomIndex = std::rand() % vec.size();

    return vec[randomIndex];
}
template <typename T>
T randSelectVector(const std::vector<T>& vec) {
    if (vec.empty()) {
        throw std::runtime_error("The vector is empty");
    }

    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    int randomIndex = std::rand() % vec.size();

    return vec[randomIndex];
}

#endif // UTILS_H