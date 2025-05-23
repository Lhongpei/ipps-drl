#include "env.h"
#include "io.h"
#include "state.h"
#include "constant.h"
#include "greedy.h"
#include <filesystem>
#include <fstream>
#include <string>
#include <chrono>
namespace fs = std::filesystem;

int main() {
    std::string folder_path = "kim";
    std::string prob_folder = std::string(PROJECT_ROOT_DIR) + folder_path + "/problem";

    std::vector<std::string> prob_paths;

    for (const auto &entry : fs::directory_iterator(prob_folder)) {
        prob_paths.push_back(entry.path().filename().string());
    }

    for (int i = 0; i < prob_paths.size(); i++) {
        std::cout << "-------------------Starting Problem " << prob_paths[i] << "-------------------" << std::endl;
        auto start_total = std::chrono::high_resolution_clock::now(); // Start total time

        auto start_read = std::chrono::high_resolution_clock::now(); // Start time for reading file
        std::vector<std::string> lines = readLinesFromFile(folder_path + "/problem/" + prob_paths[i]);
        auto end_read = std::chrono::high_resolution_clock::now(); // End time for reading file
        std::chrono::duration<double> read_time = end_read - start_read;
        std::cout << "Time taken to read file " << prob_paths[i] << ": " << read_time.count() << " seconds" << std::endl;

        auto start_env = std::chrono::high_resolution_clock::now(); // Start time for environment initialization
        Env env(lines, true);
        
        std::vector<std::string> task_lines = {
            "1 10 2",
            "out",
            "0 1",
            "in",
            "info",
            "0 start",
            "1 end"
        };
        Env env3(env); 
    

        
        env3 = env; // Copy assignment operator
         // Copy constructor
        auto end_env = std::chrono::high_resolution_clock::now(); // End time for environment initialization
        std::chrono::duration<double> env_time = end_env - start_env;
        std::cout << "Time taken to initialize environment for " << prob_paths[i] << ": " << env_time.count() << " seconds" << std::endl;

        auto start_dispatch_total = std::chrono::high_resolution_clock::now(); // Start time for dispatching loop
        DispatchRule dispatch_rule = DispatchRule(FIFO, SPT, true, true, true); // Initialize dispatch rule
        auto start_loop = std::chrono::high_resolution_clock::now(); // Start time for the main loop
        double total_dispatch_time = 0.0;
        double total_step_time = 0.0;

        while (!env.isDone()) {
            auto dispatch_start = std::chrono::high_resolution_clock::now(); // Start time for dispatch
            std::tuple<int, int> ope_ma = dispatch_rule.dispatchStep(env, env.getTime());
            int ope = std::get<0>(ope_ma);
            int ma = std::get<1>(ope_ma);
            auto dispatch_end = std::chrono::high_resolution_clock::now(); // End time for dispatch
            total_dispatch_time += std::chrono::duration<double>(dispatch_end - dispatch_start).count();

            auto step_start = std::chrono::high_resolution_clock::now(); // Start time for step
            env.step(ope, ma);
            auto step_end = std::chrono::high_resolution_clock::now(); // End time for step
            total_step_time += std::chrono::duration<double>(step_end - step_start).count();
            
            Env env3(env);
            env.checkDone();
        }

        auto end_loop = std::chrono::high_resolution_clock::now(); // End time for the main loop
        std::chrono::duration<double> loop_time = end_loop - start_loop;
        std::cout << "Time taken for the main loop for " << prob_paths[i] << ": " << loop_time.count() << " seconds" << std::endl;

        auto end_dispatch_total = std::chrono::high_resolution_clock::now(); // End time for dispatching loop
        std::chrono::duration<double> dispatch_loop_time = end_dispatch_total - start_dispatch_total;
        std::cout << "Total time taken for dispatching during loop for " << prob_paths[i] << ": " << total_dispatch_time << " seconds" << std::endl;
        std::cout << "Total time taken for stepping during loop for " << prob_paths[i] << ": " << total_step_time << " seconds" << std::endl;

        auto end_total = std::chrono::high_resolution_clock::now(); // End total time
        std::chrono::duration<double> total_time = end_total - start_total;
        std::cout << "Total processing time for " << prob_paths[i] << ": " << total_time.count() << " seconds" << std::endl;

        std::cout << "Makespan for " << prob_paths[i] << ": " << env.getCurMakespan() << std::endl;
        std::cout << "-------------------Finished Problem " << prob_paths[i] << "-------------------" << std::endl;
    }   

    return 0;
}