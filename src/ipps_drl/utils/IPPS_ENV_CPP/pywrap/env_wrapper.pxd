from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set


cdef extern from "io.h":
    vector[cpp_string] readLinesFromFile(cpp_string path)


cdef extern from "graph.h":
    cdef cppclass OpeJobProc:
        OpeJobProc(int num_opes, int num_jobs) except +
        OpeJobProc(const OpeJobProc &other) except +
        OpeJobProc &operator=(const OpeJobProc &other) except +
        void addEdge(int a, int b) except +
        void addORPeer(int a, int b) except +
        unordered_set[int] getFeasibleOpes() except +
        unordered_set[int] getSchedulingOpes() except +
        double getMaxEndTime() except +

    cdef cppclass OpeMasProc:
        OpeMasProc(int num_opes, int num_mas) except +
        OpeMasProc(const OpeMasProc &other) except +
        OpeMasProc &operator=(const OpeMasProc &other) except +
        double getProcTime(int ope, int mas) except +
        void scheduleMa(int mas, double time, double proc_time) except +
        double getFinishTime(int mas) except +


cdef extern from "state.h":
    cdef cppclass State:
        State(OpeJobProc ope_job_scheduler, OpeMasProc ope_ma_scheduler) except +
        State(const State &other) except +
        State &operator=(const State &other) except +
        double findNextTime(double time, bint larger_than_time)
        OpeJobProc ope_job_scheduler
        OpeMasProc ope_ma_scheduler


cdef extern from "env.h":
    cdef cppclass Env:
        Env(const Env&)
        Env& operator=(const Env&)
        Env(const vector[cpp_string]& lines, bint estimate_by_comb) except +
        void step(int ope, int mas)
        void steps(const vector[pair[int, int]] &actions)
        void checkDone()
        bint isDone()
        double getCurMakespan()
        double getEstimateMakespan() const
        double getTime()
        State& getState()
        void reset()
        void printDebugInfo() const


cdef extern from "greedy.h":
    # Forward declaration so DispatchRule signatures parse without re-importing Env.
    cdef cppclass Env:
        pass

    cdef cppclass DispatchRule:
        DispatchRule(int ope_rule_type, int ma_rule_type, bint pairSPT, bint minComb, bint randomChoiceOpt) except +
        void setTypes(int ope_rule_type, int ma_rule_type)
        pair[int, int] dispatchPairSPT(Env& env, double time, bint canwait)
        pair[int, int] dispatchStep(Env& env, double time, bint canwait)

    double runGreedyMakespan(Env& env, int ope_rule_type, int ma_rule_type,
                             bint pairSPT, bint minComb, bint randomChoiceOpt,
                             bint can_wait) nogil
