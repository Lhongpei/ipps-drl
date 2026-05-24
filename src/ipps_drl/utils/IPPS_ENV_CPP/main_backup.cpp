#include "env.h"
#include "io.h"
#include "state.h"
#include "constant.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <chrono>
namespace fs = std::filesystem;

int main()
{
    std::string folder_path = "kim";
    std::string prob_folder = string(PROJECT_ROOT_DIR) + folder_path + "/problem";
    std::string sol_folder = string(PROJECT_ROOT_DIR) + folder_path + "/o2c_sol";

    std::vector<std::string> prob_paths;
    std::vector<std::string> sol_paths;

    // if (!fs::exists(folder_path))
    // {
    //     std::cerr << "Error: Path does not exist: " << folder_path << std::endl;
    // }

    for (const auto &entry : fs::directory_iterator(prob_folder))
    {
        prob_paths.push_back(entry.path().filename().string());
    }

    for (const auto &entry : fs::directory_iterator(sol_folder))
    {
        sol_paths.push_back(entry.path().string());
    }

    for (int i = 0; i < sol_paths.size(); i++)
    {
        auto start = std::chrono::high_resolution_clock::now(); // 开始计时
        vector<string> lines = readLinesFromFile(folder_path + "/problem/" + prob_paths[i]);
        State state = dealWithLines(lines);
        Env env(lines);
        std::ifstream sol(sol_paths[i]);
        double target_makespan;
        int ope, ma;
        sol >> target_makespan;

        // //sol of dumb_1
        // vector<vector<int>> steps = {{0, 1},
        //                              {7, 1},
        //                              {9, 0},
        //                              {2, 1},
        //                              {10, 0},
        //                              {12, 0},
        //                              {1, 1},
        //                              {11, 0},
        //                              {13, 0},
        //                              {3, 0},
        //                              {-1, -1},
        //                              {5, 0},
        //                              {6, 0}};
        // // sol of dumb_2
        // vector<vector<int>> steps = {{0, 0},
        //                              {9, 1},
        //                              {2, 0},
        //                              {11, 1},
        //                              {1, 0},
        //                              {13, 0},
        //                              {3, 1},
        //                             //  {-1, -1},
        //                              {4, 0},
        //                              {14, 1},
        //                              {5, 0},
        //                              {16, 0},
        //                              {7, 1},
        //                              {15, 0},
        //                              {8, 0},
        //                              {17, 0}};

        while (sol >> ope >> ma)
        {
            env.step(ope, ma);
        }
        env.checkDone();
        cout << env.getCurMakespan() << endl;
        auto end = std::chrono::high_resolution_clock::now(); // 结束计时
        std::chrono::duration<double> elapsed = end - start;
        std::cout << "Processing time for " << prob_paths[i] << ": " << elapsed.count() << " seconds" << std::endl;
    }
    return 0;
}