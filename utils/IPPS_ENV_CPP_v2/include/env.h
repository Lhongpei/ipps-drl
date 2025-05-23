#ifndef ENV_H_INCLUDED
#define ENV_H_INCLUDED
#include <iostream>
#include <list>
#include <vector>
#include "state.h"
#include "graph.h"
#include "io.h"

class Env
{
private:
    State state;
    State ori_state;
    double time;
    bool done;
    vector<string> &lines;
    bool estimate_by_comb;
    double estimate_makespan;
    vector<vector<int>> job_argmin_comb;
    vector<double> estimate_proc_time;
    vector<double> estimate_job_end_time;
    vector<double> estimate_comb_remain_time;

    double old_estimate_makespan;
    vector<double> old_estimate_proc_time;
    vector<double> old_estimate_job_end_time;
    vector<double> old_estimate_comb_remain_time;

public:
    bool isDone()
    {
        return done;
    }
    void init_estimate_makespan();
    void update_estimate_makespan(int job, bool has_comb_change = false);
    Env(std::vector<std::string> &lines, bool estimate_by_comb = true);
    // 拷贝构造函数
    Env(const Env &other);

    // 拷贝赋值运算符
    Env &operator=(const Env &other);
    void step(int ope, int mas);
    void reset();
    double getCurMakespan();
    void checkDone();
    void schedule(int ope, int mas);
    void checkFinished();
    double getTime()
    {
        return time;
    }
    State &getState()
    {
        return state;
    }
    vector<vector<int>> &getJobArgminComb()
    {
        return job_argmin_comb;
    }
    vector<double> &getEstimateRemainTime()
    {
        return estimate_comb_remain_time;
    }
    void printDebugInfo() const;

};


#endif // ENV_H_INCLUDED