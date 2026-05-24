#ifndef STATE_H_INCLUDED
#define STATE_H_INCLUDED
#include <iostream>
#include <list>
#include <vector>
#include "graph.h"

// cl State
// {
//     OpeJobProc ope_job_scheduler;
//     OpeMasProc ope_ma_scheduler;
// };
class State
{
    public:
        OpeJobProc ope_job_scheduler;
        OpeMasProc ope_ma_scheduler;
        State(OpeJobProc ope_job_scheduler, OpeMasProc ope_ma_scheduler);
        State(const State &other);
        State &operator=(const State &other);
        double findNextTime(double time, bool larger_than_time);
};

#endif //STATE_H_INCLUDED