#include "state.h"


State::State(OpeJobProc ope_job_scheduler, OpeMasProc ope_ma_scheduler) : ope_job_scheduler(ope_job_scheduler), ope_ma_scheduler(ope_ma_scheduler) {}

State::State(const State &other) : ope_job_scheduler(other.ope_job_scheduler), ope_ma_scheduler(other.ope_ma_scheduler) {}

State &State::operator=(const State &other)
{
    if (this == &other)
    {
        return *this;
    }
    ope_job_scheduler = other.ope_job_scheduler;
    ope_ma_scheduler = other.ope_ma_scheduler;
    return *this;
}

double State::findNextTime(double time, bool larger_than_time)
{
    const unordered_set<int> &feasible_opes = ope_job_scheduler.getFeasibleOpes();
    const unordered_set<int> &scheduling_opes = ope_job_scheduler.getSchedulingOpes();
    double min_time = INF;
    for (int ope : feasible_opes)
    {
        if (ope_job_scheduler.checkOpeToJobStatus(ope) == IDLE)
        {
            double ope_time = ope_ma_scheduler.findEarliestTime(ope, time);
            if (!larger_than_time && ope_time <= time )
                return time;
            if (ope_time < min_time && (!larger_than_time || ope_time > time))
                min_time = ope_time;
        }
    }
    for (int ope : scheduling_opes)
    {
        double ope_time = ope_ma_scheduler.findEarliestTime(ope, ope_job_scheduler.getEndTime(ope));
        if (ope_time < min_time && (!larger_than_time || ope_time > time))
            min_time = ope_time;
    }
    return min_time;
}


