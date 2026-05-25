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
    cdef cppclass DispatchRule:
        DispatchRule(int ope_rule_type, int ma_rule_type, bint pairSPT, bint minComb, bint randomChoiceOpt) except +
        void setTypes(int ope_rule_type, int ma_rule_type)
        pair[int, int] dispatchPairSPT(Env& env, double time, bint canwait)
        pair[int, int] dispatchStep(Env& env, double time, bint canwait)

    double runGreedyMakespan(Env& env, int ope_rule_type, int ma_rule_type,
                             bint pairSPT, bint minComb, bint randomChoiceOpt,
                             bint can_wait) nogil

cdef extern from "state.h":
    cdef cppclass State:
        State(OpeJobProc ope_job_scheduler, OpeMasProc ope_ma_scheduler) except +
        State(const State &other) except +
        State &operator=(const State &other) except +
        double findNextTime(double time, bint larger_than_time)
        OpeJobProc ope_job_scheduler
        OpeMasProc ope_ma_scheduler

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
        # ``lines=None`` skips C++ allocation entirely. Used by ``copy()`` so it
        # can install its own pointer without wasting an Env per call (which had
        # nothing freeing it and was leaking ~5 Envs per MCTS rollout iteration).
        if lines is None:
            self.thisptr = NULL
            return
        cdef vector[cpp_string] cpp_lines
        cdef bytes py_bytes
        for line in lines:
            if not isinstance(line, str):
                raise TypeError("Lines must be strings")
            py_bytes = line.encode('utf-8')
            cpp_lines.push_back(cpp_string(<char*>py_bytes))
        self.thisptr = new Env(cpp_lines, is_eval)

    def __dealloc__(self):
        if self.thisptr != NULL:
            del self.thisptr
            self.thisptr = NULL

    def copy(self):
        """Return a new PyEnv that wraps a deep copy of the underlying C++ Env."""
        if not self.thisptr:
            raise ValueError("Attempting to copy an uninitialized PyEnv object")
        cdef PyEnv new_env = PyEnv(lines=None)
        new_env.thisptr = new Env(self.thisptr[0])
        return new_env
    def step(self, int ope, int mas):
        self.thisptr.step(ope, mas)

    def steps(self, actions):
        """Apply a batch of (operation, machine) actions sequentially.

        ``actions`` is any iterable of length-2 tuples/lists.
        """
        cdef vector[pair[int, int]] cpp_actions
        cdef int ope, mas
        for a in actions:
            ope = int(a[0])
            mas = int(a[1])
            cpp_actions.push_back(pair[int, int](ope, mas))
        self.thisptr.steps(cpp_actions)

    def check_done(self):
        self.thisptr.checkDone()

    def is_done(self):
        return self.thisptr.isDone()

    def get_cur_makespan(self):
        return self.thisptr.getCurMakespan()

    def get_estimate_makespan(self):
        """Lower-bound makespan estimate from the current state.

        Cheap: just returns the stored ``estimate_makespan`` field — updated
        incrementally by ``step()``, no recomputation. Used by MCTS to do the
        branch-cutoff check without paying for a Python ``env.step``.
        """
        return self.thisptr.getEstimateMakespan()

    def get_time(self):

        return self.thisptr.getTime()
    def get_state(self):

        cdef State* state_ptr = &self.thisptr.getState()
        
        return PyState._create_wrapper(state_ptr)
    def reset(self):
        self.thisptr.reset()
    def printDebugInfo(self):
        """Print debug information of the underlying C++ Env."""
        if self.thisptr:
            self.thisptr.printDebugInfo()
        else:
            print("PyEnv object is not initialized.")


cdef class PyDispatchRule:
    cdef DispatchRule* thisptr

    def __cinit__(self, int ope_rule_type=1, int ma_rule_type=1, bint pairSPT=False, bint minComb=False, bint randomChoiceOpt=False):
        self.thisptr = new DispatchRule(ope_rule_type,ma_rule_type,pairSPT,minComb,randomChoiceOpt)

    def setTypes(self, int ope_rule_type, int ma_rule_type):
        self.thisptr.setTypes(ope_rule_type, ma_rule_type)

    def dispatchPairSPT(self, PyEnv env, double time, bint canwait):
        cdef pair[int, int] result = self.thisptr.dispatchPairSPT(deref(env.thisptr), time, canwait)
        return (result.first, result.second)

    def dispatchStep(self, PyEnv env, double time, bint canwait):
        cdef pair[int, int] result = self.thisptr.dispatchStep(deref(env.thisptr), time, canwait)
        return (result.first, result.second)

    def __dealloc__(self):
        if self.thisptr:
            del self.thisptr


def read_lines(str path):
    cdef vector[cpp_string] lines = readLinesFromFile(path.encode('utf-8'))
    return [line.decode('utf-8') for line in lines]


def run_greedy(PyEnv env, int ope_rule_type=1, int ma_rule_type=1,
               bint pairSPT=False, bint minComb=False, bint randomChoiceOpt=False,
               bint wait=False):
    """Roll a greedy dispatching rule forward on ``env`` until it terminates.

    Returns ``(action_list, final_makespan)``. Uses the Python-side loop so the
    action list can be returned; use :func:`run_greedy_makespan` for a faster
    GIL-released variant that only returns the final makespan.
    """
    cdef double t
    cdef int ope, ma
    cdef PyDispatchRule rule = PyDispatchRule(ope_rule_type, ma_rule_type,
                                              pairSPT, minComb, randomChoiceOpt)
    cdef list action_list = []
    while not env.is_done():
        t = env.get_time()
        result = rule.dispatchStep(env, t, wait)
        ope, ma = result[0], result[1]
        env.step(ope, ma)
        env.check_done()
        action_list.append((ope, ma))
    return action_list, env.get_cur_makespan()


def run_greedy_makespan(PyEnv env, int ope_rule_type=1, int ma_rule_type=1,
                        bint pairSPT=False, bint minComb=False,
                        bint randomChoiceOpt=False, bint wait=False):
    """Roll greedy on ``env`` to terminal and return only the makespan.

    The dispatch loop runs entirely in C++ with the GIL released — multiple
    instances on independent ``PyEnv`` copies can therefore execute in parallel
    from a Python ``ThreadPoolExecutor``. Mutates ``env`` to its terminal state.
    """
    cdef double makespan
    cdef Env* env_ptr = env.thisptr
    with nogil:
        makespan = runGreedyMakespan(env_ptr[0], ope_rule_type, ma_rule_type,
                                     pairSPT, minComb, randomChoiceOpt, wait)
    return makespan

