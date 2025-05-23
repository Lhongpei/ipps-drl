#include "env.h"
#include "io.h"
#include "graph.h"
#include "load_utils.h"
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <cassert>
using namespace std;

Env::Env(vector<string> &lines, bool estimate_by_comb) : state(dealWithLines(lines)), ori_state(state), time(0), done(false), lines(lines), estimate_by_comb(estimate_by_comb)
{
    if (estimate_by_comb)
        init_estimate_makespan();
    std::cout << "Lines size when cunstructing: " << lines.size() << std::endl;
};
Env::Env(const Env &other)
    : state(other.state),
      ori_state(other.ori_state),
      time(other.time),
      done(other.done),
      lines(other.lines),
      estimate_by_comb(other.estimate_by_comb),
      estimate_makespan(other.estimate_makespan),
      job_argmin_comb(other.job_argmin_comb),
      estimate_proc_time(other.estimate_proc_time),
      estimate_job_end_time(other.estimate_job_end_time),
      estimate_comb_remain_time(other.estimate_comb_remain_time),
      old_estimate_makespan(other.old_estimate_makespan),
      old_estimate_proc_time(other.old_estimate_proc_time),
      old_estimate_job_end_time(other.old_estimate_job_end_time),
      old_estimate_comb_remain_time(other.old_estimate_comb_remain_time)
{
    assert(&other != nullptr && "Source object is null!");

    // 确保 lines 是有效的
    assert(!other.lines.empty() && "Lines in source object are empty!");

    std::cout << "Copy constructor called" << std::endl;

    std::cout << "Time: " << time << std::endl;
    std::cout << "Lines size: " << lines.size() << std::endl;
    std::cout << "estimate_makespan: " << estimate_makespan << std::endl;

    std::cout << "Job Argmin Comb size: " << job_argmin_comb.size() << std::endl;
}

// 拷贝赋值运算符
Env &Env::operator=(const Env &other)
{
    if (this != &other)
    {
        state = other.state;
        ori_state = other.ori_state;
        time = other.time;
        done = other.done;
        lines = other.lines;
        estimate_by_comb = other.estimate_by_comb;
        estimate_makespan = other.estimate_makespan;
        job_argmin_comb = other.job_argmin_comb;
        estimate_proc_time = other.estimate_proc_time;
        estimate_job_end_time = other.estimate_job_end_time;
        estimate_comb_remain_time = other.estimate_comb_remain_time;
        old_estimate_makespan = other.old_estimate_makespan;
        old_estimate_proc_time = other.old_estimate_proc_time;
        old_estimate_job_end_time = other.old_estimate_job_end_time;
        old_estimate_comb_remain_time = other.old_estimate_comb_remain_time;
    }
    return *this;
}
void Env::init_estimate_makespan()
{
    estimate_proc_time.assign(this->state.ope_job_scheduler.getNumOpes(), INF);
    job_argmin_comb.resize(this->state.ope_job_scheduler.getNumJobs());
    estimate_job_end_time.assign(this->state.ope_job_scheduler.getNumJobs(), INF);
    estimate_comb_remain_time.assign(this->state.ope_job_scheduler.getNumCombs(), INF);
    int job_num = this->state.ope_job_scheduler.getNumJobs();
    int cum_comb_num = 0;
    for (int job = 0; job < job_num; job++)
    {
        double job_end_time = INF;
        for (int comb : this->state.ope_job_scheduler.getCombByJob(job))
        {
            job_argmin_comb[job].push_back(comb);
            double comb_end_time = 0;
            for (int ope : this->state.ope_job_scheduler.getOpeByComb(comb))
            {
                if (estimate_proc_time[ope] == INF)
                    estimate_proc_time[ope] = this->state.ope_ma_scheduler.getMinTimeMa(ope);
                comb_end_time += estimate_proc_time[ope];
            }
            estimate_comb_remain_time[comb] = comb_end_time;
            if (job_end_time > comb_end_time)
            {
                job_end_time = comb_end_time;
            }
        }
        estimate_job_end_time[job] = job_end_time;
        std::sort(job_argmin_comb[job].begin(), job_argmin_comb[job].end(), [&](int a, int b)
                  { return estimate_comb_remain_time[a] < estimate_comb_remain_time[b]; });
        cum_comb_num += this->state.ope_job_scheduler.getCombByJob(job).size();
    }
    estimate_makespan = *min_element(estimate_job_end_time.begin(), estimate_job_end_time.end());

    old_estimate_makespan = estimate_makespan;
    old_estimate_proc_time.assign(estimate_proc_time.begin(), estimate_proc_time.end());
    old_estimate_job_end_time.assign(estimate_job_end_time.begin(), estimate_job_end_time.end());
    old_estimate_comb_remain_time.assign(estimate_comb_remain_time.begin(), estimate_comb_remain_time.end());
}

void Env::update_estimate_makespan(int job, bool has_comb_chage)
{
    if (job >= 0 && has_comb_chage == true)
    {
        vector<int> erase_comb_ind;
        for (int i = 0; i < job_argmin_comb[job].size(); i++)
        {
            if (!this->state.ope_job_scheduler.isCombInJob(job_argmin_comb[job][i], job))
                erase_comb_ind.push_back(i);
        }
        for (size_t i = erase_comb_ind.size(); i-- > 0;)
            job_argmin_comb[job].erase(job_argmin_comb[job].begin() + erase_comb_ind[i]);
    }
    int job_num = this->state.ope_job_scheduler.getNumJobs();
    for (int job = 0; job < job_num; job++)
    {
        double job_last_time = max(this->state.ope_job_scheduler.getJobLastTime(job), time);
        estimate_job_end_time[job] = job_last_time + estimate_comb_remain_time[job_argmin_comb[job][0]];
    }
    estimate_makespan = *max_element(estimate_job_end_time.begin(), estimate_job_end_time.end());
}
void Env::step(int ope, int mas)
{
    if (ope != -1)
    {
        this->schedule(ope, mas); // update_estimate_makespan inside
        this->time = this->state.findNextTime(this->time, false);
    }
    else
    {
        this->time = this->state.findNextTime(this->time, true);
        if (estimate_by_comb)
            update_estimate_makespan(ope);
    }

    this->checkFinished();
}

void Env::checkFinished()
{
    this->state.ope_ma_scheduler.checkIfMaFinished(this->time);
    this->state.ope_job_scheduler.checkIfOpeFinished(this->time);
}

void Env::schedule(int ope, int ma)
{
    double proc_time = this->state.ope_ma_scheduler.getProcTime(ope, ma);
    // check if the operation is feasible
    if (this->state.ope_job_scheduler.checkOpeStatus(ope) == UNFEASIBLE)
    {
        std::cerr << "Operation " << ope << " is unfeasible." << std::endl;
        // std::cerr << "Current status of " << ope << ": " <<
        //     this->state.ope_job_scheduler.checkOpeStatus(ope) << std::endl;
        std::terminate();
    }
    bool flag = false;
    if (this->state.ope_ma_scheduler.checkMaStatus(ma) != IDLE)
    {
        this->time = this->state.ope_ma_scheduler.getFinishTime(ma);
        flag = true;
    }
    if (this->state.ope_job_scheduler.checkOpeToJobStatus(ope) != IDLE)
    {
        int current_job = this->state.ope_job_scheduler.getOpeJob(ope);
        double tmp_time = this->state.ope_job_scheduler.getJobLastTime(current_job);

        if (tmp_time > this->time)
        {
            this->time = tmp_time;
        }
        flag = true;
    }
    if (flag)
        this->checkFinished();

    assert(this->state.ope_ma_scheduler.checkMaStatus(ma) == IDLE);
    this->state.ope_ma_scheduler.scheduleMa(ma, this->time, proc_time);
    assert(this->state.ope_job_scheduler.checkOpeToJobStatus(ope) == IDLE);
    bool has_comb_change;
    has_comb_change = this->state.ope_job_scheduler.scheduleOpe(ope, this->time, proc_time);

    if (estimate_by_comb)
    {
        for (int comb : this->state.ope_job_scheduler.getCombByJob(this->state.ope_job_scheduler.getOpeJob(ope)))
            estimate_comb_remain_time[comb] -= estimate_proc_time[ope];
        estimate_proc_time[ope] = proc_time;
        update_estimate_makespan(this->state.ope_job_scheduler.getOpeJob(ope), has_comb_change);
    }
}

void Env::reset()
{
    state = ori_state;
    time = 0;
    done = false;

    if (estimate_by_comb)
    {
        estimate_makespan = old_estimate_makespan;
        estimate_proc_time.assign(old_estimate_proc_time.begin(), old_estimate_proc_time.end());
        estimate_job_end_time.assign(old_estimate_job_end_time.begin(), old_estimate_job_end_time.end());
        estimate_comb_remain_time.assign(old_estimate_comb_remain_time.begin(), old_estimate_comb_remain_time.end());
    }
}
double Env::getCurMakespan()
{
    return state.ope_job_scheduler.getMaxEndTime();
}

void Env::checkDone()
{
    bool isdone = true;
    for (int i = 0; i < state.ope_job_scheduler.getNumJobs(); i++)
    {
        if (!state.ope_job_scheduler.isJobFinished(i))
        {
            isdone = false;
            break;
        }
    }
    done = isdone;
}
void Env::printDebugInfo() const
{
    std::cout << "Debugging Env Object:" << std::endl;
    std::cout << "Lines size: " << lines.size() << std::endl;

    if (!lines.empty())
    {
        std::cout << "First line (first 10 chars): " << lines.at(0).substr(0, 10) << "..." << std::endl;
    }
}
