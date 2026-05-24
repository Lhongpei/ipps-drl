#ifndef LOAD_UTILS_H
#define LOAD_UTILS_H

#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>
#include <unordered_set>

struct OpeLineInfo {
    int first_value;
    std::unordered_set<int> second_value;
    std::vector<std::vector<int>> parenthesized_values;
};

std::vector<std::vector<int>> extractAllParenthesized(std::string& str);
OpeLineInfo processString(const std::string& input);
void printOpeLineInfo(const OpeLineInfo& info);

#endif // LOAD_UTILS_H