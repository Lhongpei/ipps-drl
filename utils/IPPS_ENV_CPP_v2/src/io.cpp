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
#include "io.h"
#include <filesystem>
using namespace std;
namespace fs = std::filesystem;

// Function to read lines from a file

vector<string> readLinesFromFile(const string &filePath)
{

    //string root_file_path =  string(PROJECT_ROOT_DIR) + filePath;
    string root_file_path =  filePath;
    ifstream infile(root_file_path);

    // Test if the file is open
    if (!infile.is_open())
    {
        cerr << "Error: Cannot open file " << root_file_path << endl;
        return {};
    }

    vector<string> lines;
    string line;

    // 逐行读取
    while (getline(infile, line))
    {
        lines.push_back(line);
    }

    if (infile.bad())
    {
        cerr << "Warning: Errors encountered while reading file " << filePath << endl;
    }

    return lines;
}
State dealWithLines(const vector<string> &lines)
{

    // Read the number of operations and machines
    istringstream iss(lines[0]);
    int num_opes, num_mas, num_jobs;
    iss >> num_jobs >> num_mas >> num_opes;
    OpeJobProc ope_job_scheduler(num_opes, num_jobs);
    OpeMasProc ope_ma_scheduler(num_opes, num_mas);
    int line_idx = 2;
    int from;

    while (lines[line_idx].find("in") == string::npos)
    {
        OpeLineInfo out_line = processString(lines[line_idx]);
        from = out_line.first_value;
        for (const int &to : out_line.second_value)
        {
            ope_job_scheduler.addEdge(from, to);
        }
        for (vector<int> &ope_or : out_line.parenthesized_values)
        {
            ope_job_scheduler.addORPeers(ope_or.data(), int(ope_or.size()));
        }
        if (out_line.parenthesized_values.size() > 0)
        {
            ope_job_scheduler.addORConnector(from);
            ope_job_scheduler.addORDirect(from, out_line.parenthesized_values);
        }
        line_idx++;
    }
    ope_job_scheduler.updateCumulative();
    line_idx++;
    while (lines[line_idx].find("info") == string::npos)
    {
        line_idx++;
    }
    line_idx++;
    int ope_id = HAS_NOT_SET;
    string remainingValues;
    int mas_id = HAS_NOT_SET;
    int mas_num = HAS_NOT_SET;
    int proc_time = HAS_NOT_SET;
    int job_idxes = 0;
    for (int i = line_idx; i < lines.size(); i++)
    {
        istringstream iss(lines[i]);
        iss >> ope_id;
        ope_job_scheduler.setOpeJob(ope_id, job_idxes);
        getline(iss, remainingValues);
        remainingValues.erase(remainingValues.begin(), std::find_if(remainingValues.begin(), remainingValues.end(), [](unsigned char ch)
                                                                    { return !std::isspace(ch); }));
        if (remainingValues[0] == 's')
        {
            if (remainingValues[1] == 't')
            {
                ope_job_scheduler.setStartOpe(job_idxes, ope_id);
                ope_job_scheduler.setOpeType(ope_id, START);
            }
            else
            {
                ope_job_scheduler.setOpeType(ope_id, SUPERNODE);
            }

            for (int j = 0; j < num_mas; j++)
            {
                ope_ma_scheduler.addFeasiblePair(ope_id, j, 0);
            }
        }
        else if (remainingValues[0] == 'e')
        {
            ope_job_scheduler.setEndOpe(job_idxes, ope_id);
            ope_job_scheduler.setOpeType(ope_id, END);
            job_idxes++;

            for (int j = 0; j < num_mas; j++)
            {
                ope_ma_scheduler.addFeasiblePair(ope_id, j, 0);
            }
        }
        else
        {
            istringstream remainingStream(remainingValues);
            while (remainingStream >> mas_num)
            {
                for (int j = 0; j < mas_num; j++)
                {
                    remainingStream >> mas_id >> proc_time;
                    ope_ma_scheduler.addFeasiblePair(ope_id, mas_id - 1, proc_time);
                }
            }
        }
    }
    ope_job_scheduler.initFeasibleOpes();
    ope_job_scheduler.initCombInfo();
    return State(ope_job_scheduler, ope_ma_scheduler);
}
// // main function
// int main()
// {
//     string file_path = "dumb_1.txt"; // Replace with your file path
//     vector<string> lines = readLinesFromFile(file_path);
//     cout << "Contents of the file: " << endl;
//     for (const string &line : lines)
//     {
//         cout << line << endl;
//     }

//     State state = dealWithLines(lines);
//     state.ope_job_scheduler.initCombInfo();
//     state.ope_job_scheduler.printGraph();
//     state.ope_ma_scheduler.printGraph();
//     return 0;
// }