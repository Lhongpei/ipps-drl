#ifndef IO_H_INCLUDED
#define IO_H_INCLUDED
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>
#include <queue>
#include <set>
#include <algorithm>
#include <sstream>
#include "graph.h"
#include "load_utils.h"
#include "state.h"
using namespace std;

vector<string> readLinesFromFile(const string &filePath);
State dealWithLines(const vector<string> &lines);

#endif // IO_H