#include "utils.h"
#include "greedy.h"
#include "state.h"
#include "env.h"    
#include "constant.h"   
#include <unordered_set>
// unordered_set<int> OpeRule::ruleFIFO(OpeJobProc &ope_job_proc)
// {
//     int ope_id = 
//     return ope_id;
// }
// // unordered_set<int> OpeRule::ruleMOR(OpeJobProc &ope_job_proc)
// // {
// //     return 0;
// // }
// // unordered_set<int> OpeRule::ruleMWKR(OpeJobProc &ope_job_proc)
// // {
// //     return 0;
// // }
// // int MaRule::ruleEFT(OpeMasProc &ope_mas_proc, unordered_set<int> &ope_set)
// // {
// //     return 0;
// // }


// int MaRule::ruleSPT(OpeMasProc &ope_mas_proc, unordered_set<int> &ope_set)
// {
//     int ma_id = ope_mas_proc.getMinTimeMa(ope);
//     return ma_id;
// }

// pair<int, int> DispatchRule::dispatch(State &state, double time)
// {  
//     unordered_set<int> well_pre_opes = state.ope_job_scheduler.getWellPrepOpes(time);
//     double min_time = INF;
//     int chosen_ope = -1;
//     int chosen_ma = -1;
//     for (int ope : well_pre_opes)
//     {
//         const unordered_set<int> &feasible_mas = state.ope_ma_scheduler.getFeasibleMas(ope);
//         for (int ma : feasible_mas)
//         {
//             double finish_time = state.ope_ma_scheduler.getFinishTime(ma);
//             if (finish_time < min_time)
//             {
//                 min_time = finish_time;
//                 chosen_ope = ope;
//                 chosen_ma = ma;
//             }
//         }
//     }
//     return make_pair(chosen_ope, chosen_ma);
// }
pair<int, int> DispatchRule::dispatchPairSPT(Env &env, double time, bool can_wait)
{
    State &state = env.getState();
    vector<vector<int>> &job_argmin_comb = env.getJobArgminComb();
    unordered_set<int> well_pre_opes = state.ope_job_scheduler.getWellPrepOpes(time);
    double min_estimate = INF;
    double min_time = INF;
    double min_mincomb_time = INF;
    int chosen_ope = -1;
    int chosen_ma = -1;
    int chosen_mincomb_ope = -1;
    int chosen_mincomb_ma = -1;
    for (int ope : well_pre_opes)
    {
        bool FLAG = false;
        if (this->minComb){
            int job_id = state.ope_job_scheduler.getOpeJob(ope);
            // A job with every comb pruned has no `job_argmin_comb[job_id][0]`
            // to compare against — skip the minComb check for this ope.
            if (!job_argmin_comb[job_id].empty()) {
                for (int comb : state.ope_job_scheduler.getOpeComb(ope))
                {
                    if(job_argmin_comb[job_id][0] == comb){
                        FLAG = true;
                        break;
                    }
                }
            }
        }
        const unordered_set<int> &feasible_mas = state.ope_ma_scheduler.getFeasibleMas(ope);
        unordered_set<int> select_mas;
        for (int ma : feasible_mas)
        {
            if (state.ope_ma_scheduler.checkMaStatus(ma) == IDLE)
            {
                select_mas.insert(ma);
            }
        }
        for (int ma : select_mas)
        {
            double proc_time = state.ope_ma_scheduler.getProcTime(ope, ma);
            if (proc_time < min_time)
            {
                min_time = proc_time;
                chosen_ope = ope;
                chosen_ma = ma;
            }
            if (this->minComb && FLAG && proc_time < min_mincomb_time)
            {
                min_mincomb_time = proc_time;
                chosen_mincomb_ope = ope;
                chosen_mincomb_ma = ma;
            }
        }
    }
    if (! this->randomChoiceOpt){
        if (this->minComb){
            if (chosen_mincomb_ope == -1 || chosen_mincomb_ma == -1){
                if (can_wait){
                    return make_pair(-1, -1);
                }
                else{
                    return make_pair(chosen_ope, chosen_ma);
                }
            }
            else{
                return make_pair(chosen_mincomb_ope, chosen_mincomb_ma);
            }
        }
        else{
            return make_pair(chosen_ope, chosen_ma);
        }
    }
        
    
    
    vector<pair<int, int>> pair_opes_mas;
    vector<pair<int, int>> pair_mincomb_opes_mas;
    for (int ope : well_pre_opes)
    {
        bool FLAG = false;
        if (this->minComb){
            int job_id = state.ope_job_scheduler.getOpeJob(ope);
            bool flag = false;
            if (!job_argmin_comb[job_id].empty()) {
                for (int comb : state.ope_job_scheduler.getOpeComb(ope))
                {
                    if(job_argmin_comb[job_id][0] == comb){
                        flag = true;
                        FLAG = true;
                        break;
                    }
                }
            }
            if(!flag) continue;
        }
        const unordered_set<int> &feasible_mas = state.ope_ma_scheduler.getFeasibleMas(ope);
        unordered_set<int> select_mas;
        for (int ma : feasible_mas)
        {
            if (state.ope_ma_scheduler.checkMaStatus(ma) == IDLE)
            {
                select_mas.insert(ma);
            }
        }
        for (int ma : select_mas)
        {
            double proc_time = state.ope_ma_scheduler.getProcTime(ope, ma);
            if (proc_time <= min_time){pair_opes_mas.push_back(make_pair(ope, ma));}
            if (FLAG && proc_time <=min_mincomb_time &&this->minComb){
                pair_mincomb_opes_mas.push_back(make_pair(ope, ma));
            }
            
        }
    }
    if (pair_mincomb_opes_mas.size() > 0){
        return randSelectVector(pair_mincomb_opes_mas);
    }
    if (can_wait){
        return make_pair(-1, -1);
    }
    return randSelectVector(pair_opes_mas);

    
}

pair<int, int> DispatchRule::dispatchStep(Env &env, double time, bool can_wait)
{
    if (this->pairSPT) return dispatchPairSPT(env, time, can_wait);
    State &state = env.getState();
    
    vector<vector<int>> &job_argmin_comb = env.getJobArgminComb();
    vector<double> &estimate_comb_remain_time = env.getEstimateRemainTime();
    
    unordered_set<int> well_pre_opes = state.ope_job_scheduler.getWellPrepOpes(time);
    double min_estimate = INF;
    double min_time = INF;
    int chosen_ope = -1;
    int chosen_ma = -1;
    if (this->ope_rule == RANDOM)
    {
        chosen_ope = randSelectSet(well_pre_opes);
    }
    else if (this->ope_rule == FIFO)
    {
        for (int ope : well_pre_opes)
        {
            if (state.ope_job_scheduler.getJobLastTime(state.ope_job_scheduler.getOpeJob(ope)) < min_time)
            {
                min_time = state.ope_job_scheduler.getJobLastTime(state.ope_job_scheduler.getOpeJob(ope));
                chosen_ope = ope;
            }
        }
    }
    else if (this->ope_rule == MWKR)
    {
        double max_remain_time = -INF;

        for (int ope : well_pre_opes)
        {
            int job = state.ope_job_scheduler.getOpeJob(ope);
            // A job whose combs were all pruned can't be ranked by remaining
            // estimated work — fall through and let the post-loop fallback
            // pick *some* ope, otherwise the rollout deadlocks on wait sentinels.
            if (job_argmin_comb[job].empty())
                continue;
            double remain = estimate_comb_remain_time[job_argmin_comb[job][0]];
            if (remain > max_remain_time)
            {
                max_remain_time = remain;
                chosen_ope = ope;
            }
        }
        // Every well-prep ope had empty argmin_comb — fall back to picking
        // any feasible ope so the rollout makes progress.
        if (chosen_ope == -1 && !well_pre_opes.empty())
            chosen_ope = *well_pre_opes.begin();
    }
    // Only wait when there's truly no ope to pick. Don't wait merely because a
    // rule couldn't rank the candidates — that deadlocks (time advances, the
    // same candidates resurface unrankable, repeat). The caller passes
    // ``can_wait=false`` in rollouts for exactly this reason.
    if (chosen_ope == -1)
        return make_pair(-1, -1);

    if (this->ma_rule == SPT)
    {
        for (int ma : state.ope_ma_scheduler.getFeasibleMas(chosen_ope))
        {
            double proc_time = state.ope_ma_scheduler.getProcTime(chosen_ope, ma);
            if (proc_time < min_estimate)
            {
                min_estimate = proc_time;
                chosen_ma = ma;
            }
        }
    }
    else if (this->ma_rule == EFT)
    {
        for (int ma : state.ope_ma_scheduler.getFeasibleMas(chosen_ope))
        {
            double finish_time = state.ope_ma_scheduler.getFinishTime(ma);
            if (finish_time < min_estimate)
            {
                min_estimate = finish_time;
                chosen_ma = ma;
            }
        }
    }
    else if (this->ma_rule == RANDOM)
    {
        chosen_ma = randSelectSet(state.ope_ma_scheduler.getFeasibleMas(chosen_ope));
    }

    return make_pair(chosen_ope, chosen_ma);
}


double runGreedyMakespan(Env &env,
                         int ope_rule_type,
                         int ma_rule_type,
                         bool pairSPT,
                         bool minComb,
                         bool randomChoiceOpt,
                         bool can_wait)
{
    DispatchRule rule(ope_rule_type, ma_rule_type, pairSPT, minComb, randomChoiceOpt);
    try {
        while (!env.isDone()) {
            double t = env.getTime();
            std::pair<int, int> action = rule.dispatchStep(env, t, can_wait);
            env.step(action.first, action.second);
            env.checkDone();
        }
        return env.getCurMakespan();
    } catch (...) {
        // Degenerate states (empty selection sets, etc.) — treat the rollout
        // as the worst outcome rather than propagating into the MCTS thread.
        return INF;
    }
}