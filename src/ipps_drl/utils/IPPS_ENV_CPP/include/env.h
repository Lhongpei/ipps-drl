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
    // Owned copy. Stored by value (not reference) so an Env outlives the
    // Python-side `cpp_lines` local that built it, and PyEnv.copy() yields an
    // independent object safe to use from threads.
    vector<string> lines;
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
    Env(const Env &other);

    Env &operator=(const Env &other);
    void step(int ope, int mas);
    // Batched step: applies a sequence of (ope, ma) actions in order. Mirrors the
    // signature in the upstream repo (https://github.com/Lhongpei/IPPS_ENV_CPP)
    // and is required by ipps_drl.inference.mcts.
    void steps(const std::vector<std::pair<int, int>> &steps);
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
    // Lower-bound estimate of the final makespan from the current state, kept
    // in sync by step(). Cheap to read — no recomputation. Used by MCTS as a
    // C++-side substitute for the Python ``IPPSEnv.makespan_batch[0]`` cutoff
    // check, so tree descent can skip the expensive Python ``env.step``.
    double getEstimateMakespan() const
    {
        return estimate_makespan;
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