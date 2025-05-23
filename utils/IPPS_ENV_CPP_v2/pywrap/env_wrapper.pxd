from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
cdef extern from "io.h":
    vector[cpp_string] readLinesFromFile(cpp_string path)

cdef extern from "graph.h":
    cdef cppclass OpeJobProc:
        # Constructors
        OpeJobProc(int num_opes, int num_jobs) except +
        OpeJobProc(const OpeJobProc &other) except +  # Copy constructor
        OpeJobProc &operator=(const OpeJobProc &other) except +  # Assignment operator
        
        # Member functions
        void addEdge(int a, int b) except +
        void addORPeer(int a, int b) except +
        unordered_set[int] getFeasibleOpes() except +
        unordered_set[int] getSchedulingOpes() except +
        double getMaxEndTime() except +

    cdef cppclass OpeMasProc:
        # Constructors
        OpeMasProc(int num_opes, int num_mas) except +
        OpeMasProc(const OpeMasProc &other) except +  # Copy constructor
        OpeMasProc &operator=(const OpeMasProc &other) except +  # Assignment operator
        
        # Member functions
        double getProcTime(int ope, int mas) except +
        void scheduleMa(int mas, double time, double proc_time) except +
        double getFinishTime(int mas) except +

cdef extern from "state.h":
    cdef cppclass State:
        # Constructors
        State(OpeJobProc ope_job_scheduler, OpeMasProc ope_ma_scheduler) except +
        State(const State &other) except +   # Copy constructor
        State &operator=(const State &other) except +  # Assignment operator
        
        # Member functions
        double findNextTime(double time, bint larger_than_time)
        
        # Public members
        OpeJobProc ope_job_scheduler
        OpeMasProc ope_ma_scheduler

cdef extern from "env.h":
    cdef cppclass Env:
        Env(const Env&)  # 拷贝构造函数
        Env& operator=(const Env&)  # 拷贝赋值运算符

        # 其他方法声明
        Env(const vector[cpp_string]& lines, bint estimate_by_comb) except +
        void step(int ope, int mas)
        void checkDone()
        bint isDone()
        double getCurMakespan()
        double getTime()
        State& getState()
        void reset()
        void printDebugInfo() const


cdef extern from "greedy.h":


    # 引入 Env 类
    cdef cppclass Env:
        pass

    cdef cppclass DispatchRule:
        # 构造函数,带有默认参数
        DispatchRule(int ope_rule_type, int ma_rule_type, bint pairSPT, bint minComb, bint randomChoiceOpt) except +

        void setTypes(int ope_rule_type, int ma_rule_type)

        pair[int, int] dispatchPairSPT(Env& env, double time,bint canwait)

        pair[int, int] dispatchStep(Env& env, double time,bint canwait)