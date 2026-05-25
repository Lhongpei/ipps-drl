#ifndef GREEDY_H
#define GREEDY_H
#include <iostream>
#include "constant.h"
#include <unordered_set>
#include "graph.h"
#include "state.h"
#include "env.h"
#include <stdexcept>

class DispatchRule{
private:
    int ope_rule;
    int ma_rule;
    bool pairSPT;
    bool minComb;
    bool randomChoiceOpt;
public:
    DispatchRule(int ope_rule_type = SPT, int ma_rule_type = SPT, bool pairSPT = false, bool minComb = false, bool randomChoiceOpt = false) : 
    ope_rule(ope_rule_type), ma_rule(ma_rule_type), pairSPT(pairSPT), minComb(minComb), randomChoiceOpt(randomChoiceOpt){}

    void setTypes(int ope_rule_type, int ma_rule_type){
        this->ope_rule = ope_rule_type;
        this->ma_rule = ma_rule_type;
    }
    pair<int, int> dispatchPairSPT(Env &env, double time, bool can_wait = false);

    pair<int, int> dispatchStep(Env &Env, double time, bool can_wait = false);
};

// Pure-C++ rollout helper for MCTS evaluation. Drives ``env`` to terminal with
// the given dispatching rule and returns the final makespan. No action list is
// produced — keeps the body Python-free so the Cython wrapper can release the
// GIL and run several of these in parallel threads. On any exception inside the
// loop returns ``INF`` (defined in constant.h) rather than propagating.
double runGreedyMakespan(Env &env,
                         int ope_rule_type,
                         int ma_rule_type,
                         bool pairSPT,
                         bool minComb,
                         bool randomChoiceOpt,
                         bool can_wait);

#endif //GREEDY_H