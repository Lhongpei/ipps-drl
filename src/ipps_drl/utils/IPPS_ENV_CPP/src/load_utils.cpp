#include "load_utils.h"

std::vector<std::vector<int>> extractAllParenthesized(std::string& str) {
    std::vector<std::vector<int>> result;
    size_t start = 0;
    while (true) {
        start = str.find('(', start);
        if (start == std::string::npos) break;
        size_t end = str.find(')', start);
        if (end == std::string::npos) break;

        std::string content = str.substr(start + 1, end - start - 1);
        std::istringstream iss(content);
        std::string token;
        std::vector<int> tokens;
        while (std::getline(iss, token, ',')) {
            tokens.push_back(std::stoi(token));
        }
        result.push_back(tokens);

        str.erase(start, end - start + 1);
        start = start;
    }
    return result;
}


OpeLineInfo processString(const std::string& input) {
    OpeLineInfo info;
    std::istringstream iss(input);

    iss >> info.first_value;

    std::string remainingValues;
    std::getline(iss, remainingValues);

    remainingValues.erase(remainingValues.begin(), std::find_if(remainingValues.begin(), remainingValues.end(), [](unsigned char ch) {
        return !std::isspace(ch);
    }));

    info.parenthesized_values = extractAllParenthesized(remainingValues);

    std::istringstream remainingStream(remainingValues);
    int value;
    while (remainingStream >> value) {
        info.second_value.insert(value);
    }

    for (const std::vector<int>& vec : info.parenthesized_values) {
        for (int val : vec) {
            info.second_value.insert(val);
        }
    }

    return info;
}

void printOpeLineInfo(const OpeLineInfo& info) {
    std::cout << "-------------------" << std::endl;
    std::cout << "First value: " << info.first_value << std::endl;

    std::cout << "Remaining values: ";
    for (int val : info.second_value) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < info.parenthesized_values.size(); ++i) {
        std::cout << "Parenthesized values (Group " << i + 1 << "): ";
        for (int val : info.parenthesized_values[i]) {
            std::cout << val << " ";
        }
        std::cout << std::endl;
    }

    std::cout << "-------------------" << std::endl;
}