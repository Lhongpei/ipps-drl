# env_wrapper.pyx
# cython: language_level=3
from libcpp.vector cimport vector
from libcpp.string cimport string as cpp_string
from libc.stdlib cimport malloc, free
from cpython.ref cimport PyObject
from cython.operator cimport dereference as deref
from libcpp.utility cimport pair
from libcpp.unordered_set cimport unordered_set
from cython.operator cimport dereference as deref
cdef extern from "string" namespace "std":
    cpp_string to_string(int)

cdef extern from "io.h":
    vector[cpp_string] readLinesFromFile(cpp_string path)

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



    # DispatchRule 类定义
    cdef cppclass DispatchRule:
        # 构造函数,带有默认参数
        DispatchRule(int ope_rule_type, int ma_rule_type, bint pairSPT, bint minComb, bint randomChoiceOpt) except +

        void setTypes(int ope_rule_type, int ma_rule_type)

        pair[int, int] dispatchPairSPT(Env& env, double time,bint canwait)

        pair[int, int] dispatchStep(Env& env, double time,bint canwait)
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

# Create a wrapper class in Python for State
# Import externs from C++ header
cdef extern from "graph.h":
    cdef cppclass OpeJobProc:
        OpeJobProc(int num_opes, int num_jobs) except +
        void addEdge(int a, int b) except +
        unordered_set[int] getFeasibleOpes() except +
        unordered_set[int] getSchedulingOpes() except +
        double getMaxEndTime() except +

    cdef cppclass OpeMasProc:
        OpeMasProc(int num_opes, int num_mas) except +
        double getProcTime(int ope, int mas) except +
        void scheduleMa(int mas, double time, double proc_time) except +
        double getFinishTime(int mas) except +

# Python wrappers for OpeJobProc and OpeMasProc

cdef class PyOpeJobProc:
    cdef OpeJobProc* thisptr
    
    def __cinit__(self, int num_opes, int num_jobs):
        self.thisptr = new OpeJobProc(num_opes, num_jobs)

    def add_edge(self, int a, int b):
        self.thisptr.addEdge(a, b)

    def get_feasible_opes(self):
        cdef unordered_set[int] opes = self.thisptr.getFeasibleOpes()
        return list(opes)

    def get_scheduling_opes(self):
        cdef unordered_set[int] opes = self.thisptr.getSchedulingOpes()
        return list(opes)

    def get_max_end_time(self):
        return self.thisptr.getMaxEndTime()

    def __dealloc__(self):
        del self.thisptr

cdef extern from "graph.h":
    cdef cppclass OpeMasProc:
        OpeMasProc(int num_opes, int num_mas) except +
        double getProcTime(int ope, int mas) except +
        void scheduleMa(int mas, double time, double proc_time) except +
        double getFinishTime(int mas) except +

# Python wrapper class for OpeMasProc
cdef class PyOpeMasProc:
    cdef OpeMasProc* thisptr

    def __cinit__(self, int num_opes, int num_mas):
        self.thisptr = new OpeMasProc(num_opes, num_mas)

    def get_proc_time(self, int ope, int mas):
        return self.thisptr.getProcTime(ope, mas)

    def schedule_ma(self, int mas, double time, double proc_time):
        self.thisptr.scheduleMa(mas, time, proc_time)

    def get_finish_time(self, int mas):
        return self.thisptr.getFinishTime(mas)

    def __dealloc__(self):
        del self.thisptr

cdef class PyState:
    cdef State* thisptr  # Pointer to the C++ State class
    

    def __cinit__(self, PyOpeJobProc ope_job_scheduler=None, PyOpeMasProc ope_ma_scheduler=None):
        if ope_job_scheduler is not None and ope_ma_scheduler is not None:
            self.thisptr = new State(ope_job_scheduler.thisptr[0], ope_ma_scheduler.thisptr[0])
            self.owns_ptr = True

    @staticmethod
    cdef PyState _create_wrapper(State* ptr):
        cdef PyState wrapper = PyState.__new__(PyState)
        wrapper.thisptr = ptr
        return wrapper

    def find_next_time(self, double time, bint larger_than_time):
        # Call C++ member function
        return self.thisptr.findNextTime(time, larger_than_time)

cdef class PyEnv:
    cdef Env* thisptr

    def __cinit__(self, list lines = [
            "1 10 2",
            "out",
            "0 1",
            "in",
            "info",
            "0 start",
            "1 end"
], bint is_eval=True):
        cdef vector[cpp_string] cpp_lines
        cdef bytes py_bytes
        print("hello")
        if lines is not None:
            for line in lines:
                print(line)
                if isinstance(line, str):
                    py_bytes = line.encode('utf-8')
                    cpp_lines.push_back(cpp_string(<char*>py_bytes))
                else:
                    raise TypeError("Lines must be strings")
            print("ono")
            self.thisptr = new Env(cpp_lines, is_eval)
            print("isdone")
        
        else:
            self.thisptr =  new Env(lines, is_eval)# 空指针


    def copy(self):
        """
        调用 C++ 拷贝构造函数生成新的 PyEnv 对象
        """
        print("Copying PyEnv object...")

        if not self.thisptr:
            raise ValueError("Attempting to copy an uninitialized PyEnv object")




        print("Calling C++ copy constructor...")
        
        cdef PyEnv new_env  = PyEnv( )
        print("its still fine")

        new_env.thisptr = new Env(self.thisptr[0]) 

        print("Copy successful.")
        return new_env
    def step(self, int ope, int mas):
        self.thisptr.step(ope, mas)

    def check_done(self):
        self.thisptr.checkDone()

    def is_done(self):
        return self.thisptr.isDone()

    def get_cur_makespan(self):
        return self.thisptr.getCurMakespan()

    def get_time(self):

        return self.thisptr.getTime()
    def get_state(self):

        cdef State* state_ptr = &self.thisptr.getState()
        
        return PyState._create_wrapper(state_ptr)
    def reset(self):
        self.thisptr.reset()
    def printDebugInfo(self):
        """
        Print debug information of the Env object.
        """
        if self.thisptr:
            self.thisptr.printDebugInfo()
        else:
            print("PyEnv object is not initialized.")


# Python wrapper
cdef class PyDispatchRule:
    cdef DispatchRule* thisptr

    def __cinit__(self, int ope_rule_type=1, int ma_rule_type=1, bint pairSPT=False, bint minComb=False, bint randomChoiceOpt=False):
        self.thisptr = new DispatchRule(ope_rule_type,ma_rule_type,pairSPT,minComb,randomChoiceOpt)

    def setTypes(self, int ope_rule_type, int ma_rule_type):
        self.thisptr.setTypes(ope_rule_type, ma_rule_type)

    def dispatchPairSPT(self, PyEnv env, double time,bint canwait):
        """
        调用 C++ DispatchRule 的 dispatchPairSPT 方法
        """
        cdef pair[int, int] result = self.thisptr.dispatchPairSPT(deref(env.thisptr), time,canwait)
        return (result.first, result.second)

    def dispatchStep(self, PyEnv env, double time,bint canwait):
        """
        调用 C++ DispatchRule 的 dispatchStep 方法
        """
        cdef pair[int, int] result = self.thisptr.dispatchStep(deref(env.thisptr), time, canwait)
        return (result.first, result.second)

    def __dealloc__(self):
        """
        析构函数：释放 C++ 对象指针
        """
        if self.thisptr:
            del self.thisptr

def read_lines(str path):
    cdef vector[cpp_string] lines = readLinesFromFile(path.encode('utf-8'))
    return [line.decode('utf-8') for line in lines]  # Convert to Python list

def run_greedy(PyEnv env, int ope_rule_type=1, int ma_rule_type=1, bint pairSPT=False, bint minComb=False, bint randomChoiceOpt=False,bint wait=False):
    """
    输入已有的 PyEnv 对象 env,执行 greedy 直到结束,返回 makespan。
    """
    cdef double t
    cdef int ope, ma
    cdef PyDispatchRule rule = PyDispatchRule(ope_rule_type, ma_rule_type, pairSPT,minComb,randomChoiceOpt)
    cdef list action_list = []


    while not env.is_done():
        t = env.get_time()

        result = rule.dispatchStep(env, t,wait)
        ope, ma = result[0], result[1]
        env.step(ope, ma)
        env.check_done()
        action_list.append((ope, ma))
        print(ope,ma)

    return action_list, env.get_cur_makespan()

