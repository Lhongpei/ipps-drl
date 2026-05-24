#include "env.h"
#include "io.h"
#include "state.h"
#include <string>
int main()
{
    std::string file_path = "dumb_1.txt";
    vector<string> lines = readLinesFromFile(file_path);
    // State state = dealWithLines(lines);
    // Env env(lines);
}