#include <iostream>
#include <list>
#include <vector>
#include <queue>
#include <unordered_set>
#include <iostream>
#include <list>
#include <set>
#include <map>
#include <vector>
#include <algorithm>
#include <iterator>
#include <queue>
#include <bitset>
#include "constant.h"
#include <set>
#include <cassert>
#include "graph.h"
using namespace std;
void checkFeasibility(int ope, const std::vector<int> &ope_status)
{
    if (ope_status[ope] != FEASIBLE)
    {
        std::cerr << "Operation " << ope << " is not feasible." << std::endl;
        std::cerr << "Current status of all operations:" << std::endl;
        for (size_t i = 0; i < ope_status.size(); ++i)
        {
            std::cerr << "Operation " << i << ": " << ope_status[i] << std::endl;
        }
        assert(ope_status[ope] == FEASIBLE);
    }
}
void OpeJobProc::computeSuccessors(int node)
{
    std::queue<int> q;
    q.push(node);
    std::unordered_set<int> visited;
    visited.insert(node);
    while (!q.empty())
    {
        int current = q.front();
        q.pop();
        for (int neighbor : graph[current])
        {
            if (visited.find(neighbor) == visited.end())
            {
                visited.insert(neighbor);
                q.push(neighbor);
            }
        }
    }
    visited.erase(node); // Remove the node itself from the cul_successors
    cul_successors[node] = visited;
    for (int successor : visited)
    {
        cul_successors_bitset[node].set(successor);
    }
}

void OpeJobProc::computeAncestors(int node)
{
    std::queue<int> q;
    q.push(node);
    std::unordered_set<int> visited;
    visited.insert(node);
    while (!q.empty())
    {
        int current = q.front();
        q.pop();
        for (int ancestor : reverse_graph[current])
        {
            if (visited.find(ancestor) == visited.end())
            {
                visited.insert(ancestor);
                q.push(ancestor);
            }
        }
    }
    visited.erase(node); // Remove the node itself from the cul_ancestors
    cul_ancestors[node] = visited;
    for (int ancestor : visited)
    {
        cul_ancestors_bitset[node].set(ancestor);
    }
}

bool OpeJobProc::checkIfIsAncestor(int ope, int target_ope)
{
    return cul_ancestors_bitset[ope].test(target_ope);
}

bool OpeJobProc::checkIfIsSuccessor(int ope, int target_ope)
{
    return cul_successors_bitset[ope].test(target_ope);
}

vector<set<int>> OpeJobProc::union_without_or(const vector<int> &or_nb,
                                              const vector<int> &and_nb,
                                              const vector<vector<set<int>>> &or_path,
                                              const vector<vector<set<int>>> &and_path)
{
    vector<set<int>> path;
    set<int> or_common = or_dict[and_nb[0]];
    if (and_path.size() > 1)
    {
        path = and_path[0];
        for (int i = 1; i < and_nb.size(); i++)
        {
            vector<set<int>> new_path;
            int and_child = and_nb[i];
            or_common.insert(or_dict[and_child].begin(), or_dict[and_child].end());
            for (set<int> p : path)
            {
                for (set<int> q : and_path[i])
                {
                    bool flag = true;
                    set<int> tmp_p = p;
                    tmp_p.insert(q.begin(), q.end());
                    for (int or_node : or_common)
                    {
                        set<int> intersect;
                        set_intersection(tmp_p.begin(), tmp_p.end(),
                                         or_direct[or_node].begin(), or_direct[or_node].end(),
                                         inserter(intersect, intersect.begin()));
                        if (intersect.size() > 1)
                        {
                            flag = false;
                            break;
                        }
                    }
                    if (flag)
                        new_path.push_back(tmp_p);
                }
            }
            path = new_path;
        }
        vector<set<int>> new_path;
        if (or_path.size() != 0)
        {
            for (int i = 0; i < or_nb.size(); i++)
            {
                int a = or_nb[i];
                or_common.insert(or_dict[a].begin(), or_dict[a].end());
                for (set<int> p : path)
                {
                    for (set<int> q : or_path[i])
                    {
                        set<int> tmp_p = p;
                        bool flag = true;
                        tmp_p.insert(q.begin(), q.end());
                        for (int or_node : or_common)
                        {
                            set<int> intersect;
                            set_intersection(tmp_p.begin(), tmp_p.end(),
                                             or_direct[or_node].begin(), or_direct[or_node].end(),
                                             inserter(intersect, intersect.begin()));
                            if (intersect.size() > 1)
                            {
                                flag = false;
                                break;
                            }
                        }
                        if (flag)
                            new_path.push_back(tmp_p);
                    }
                }
            }
            path = new_path;
        }
    }
    else
    {
        for (vector<set<int>> a : or_path)
        {
            for (vector<set<int>> b : and_path)
            {
                for (set<int> p : a)
                {
                    for (set<int> q : b)
                    {
                        p.insert(q.begin(), q.end());
                        path.push_back(p);
                    }
                }
            }
        }
    }
    return path;
}

vector<set<int>> OpeJobProc::searchCombByRec(int node,
                                             map<int, vector<set<int>>> &dp)
{
    vector<set<int>> result = {{node}};
    if (graph[node].size() == 0)
        return result;

    vector<vector<set<int>>> or_path;
    vector<vector<set<int>>> and_path = {{{node}}};
    vector<int> or_nb;
    vector<int> and_nb = {node};
    for (int child : graph[node])
    {
        vector<set<int>> path = dp.find(child) != dp.end() ? dp[child] : searchCombByRec(child, dp);
        if (or_direct[node].find(child) != or_direct[node].end())
        {
            or_nb.push_back(child);
            or_path.push_back(path);
        }
        else
        {
            and_nb.push_back(child);
            and_path.push_back(path);
        }
    }
    result = union_without_or(or_nb, and_nb, or_path, and_path);
    dp[node] = result;
    return result;
}

void OpeJobProc::computeORDict()
{
    for (int node = 0; node < graph.size(); node++)
    {
        bool has_or_child = false;
        for (int child : cul_successors[node])
        {
            if (or_connectors.find(child) != or_connectors.end())
            {
                if (!has_or_child)
                {
                    or_dict[node] = {child};
                    has_or_child = true;
                }
                else
                {
                    or_dict[node].insert(child);
                }
            }
        }
    }
}

OpeJobProc::OpeJobProc(int num_opes, int jobs) : num_opes(num_opes), num_jobs(jobs)
{
    graph.resize(num_opes);
    reverse_graph.resize(num_opes);
    cul_ancestors.resize(num_opes);
    cul_successors.resize(num_opes);
    or_peers.resize(num_opes);
    job_to_ope.resize(num_jobs);
    ope_to_job.resize(num_opes, HAS_NOT_SET);
    ope_to_comb.resize(num_opes);
    comb_to_ope.resize(num_opes);
    job_start_ope.resize(num_jobs, HAS_NOT_SET);
    job_end_ope.resize(num_jobs, HAS_NOT_SET);
    ope_type.resize(num_opes, COMMON);
    ope_status.resize(num_opes, UNSCHEDULED);
    start_time.resize(num_opes, HAS_NOT_SET);
    end_time.resize(num_opes, HAS_NOT_SET);
    job_last_time.resize(num_jobs, 0);
    job_status.resize(num_jobs, IDLE);
    cul_ancestors_bitset.resize(num_opes);
    cul_successors_bitset.resize(num_opes);
    job_current_ope.resize(num_jobs, HAS_NOT_SET);
    max_end_time = 0.0;
    for (int i = 0; i < num_opes; ++i)
    {
        remain_opes.insert(i);
    }
}

OpeJobProc::OpeJobProc(const OpeJobProc &other)
{
    num_opes = other.num_opes;
    num_jobs = other.num_jobs;
    num_combs = other.num_combs;
    graph = other.graph;
    reverse_graph = other.reverse_graph;
    cul_ancestors = other.cul_ancestors;
    cul_successors = other.cul_successors;
    cul_ancestors_bitset = other.cul_ancestors_bitset;
    cul_successors_bitset = other.cul_successors_bitset;
    or_peers = other.or_peers;
    job_to_ope = other.job_to_ope;
    ope_to_job = other.ope_to_job;
    or_dict = other.or_dict;
    or_direct = other.or_direct;
    comb_to_ope = other.comb_to_ope;
    job_start_ope = other.job_start_ope;
    job_end_ope = other.job_end_ope;
    job_current_ope = other.job_current_ope;
    ope_type = other.ope_type;
    or_connectors = other.or_connectors;
    ope_to_comb = other.ope_to_comb;
    job_to_comb = other.job_to_comb;
    ope_status = other.ope_status;
    start_time = other.start_time;
    end_time = other.end_time;
    job_last_time = other.job_last_time;
    max_end_time = other.max_end_time;
    remain_opes = other.remain_opes;
    feasible_opes = other.feasible_opes;
    scheduling_opes = other.scheduling_opes;
    job_status = other.job_status;
}

OpeJobProc &OpeJobProc::operator=(const OpeJobProc &other)
{
    if (this == &other)
    {
        return *this;
    }
    num_opes = other.num_opes;
    num_jobs = other.num_jobs;
    num_combs = other.num_combs;
    graph = other.graph;
    reverse_graph = other.reverse_graph;
    cul_ancestors = other.cul_ancestors;
    cul_successors = other.cul_successors;
    cul_ancestors_bitset = other.cul_ancestors_bitset;
    cul_successors_bitset = other.cul_successors_bitset;
    or_peers = other.or_peers;
    job_to_ope = other.job_to_ope;
    ope_to_job = other.ope_to_job;
    or_dict = other.or_dict;
    or_direct = other.or_direct;
    comb_to_ope = other.comb_to_ope;
    job_start_ope = other.job_start_ope;
    job_end_ope = other.job_end_ope;
    job_current_ope = other.job_current_ope;
    ope_type = other.ope_type;
    or_connectors = other.or_connectors;
    ope_to_comb = other.ope_to_comb;
    job_to_comb = other.job_to_comb;
    ope_status = other.ope_status;
    start_time = other.start_time;
    end_time = other.end_time;
    job_last_time = other.job_last_time;
    max_end_time = other.max_end_time;
    remain_opes = other.remain_opes;
    feasible_opes = other.feasible_opes;
    scheduling_opes = other.scheduling_opes;
    job_status = other.job_status;
    return *this;
}

std::unordered_set<int> OpeJobProc::getFeasibleOpes()
{
    return feasible_opes;
}
std::unordered_set<int> OpeJobProc::getSchedulingOpes()
{
    return scheduling_opes;
}

std::unordered_set<int> OpeJobProc::getCombByJob(int job)
{
    return job_to_comb[job];
}

std::unordered_set<int> OpeJobProc::getOpeByComb(int comb)
{
    return comb_to_ope[comb];
}

double OpeJobProc::getEndTime(int ope)
{
    return end_time[ope];
}
double OpeJobProc::getMaxEndTime()
{
    return max_end_time;
}

int OpeJobProc::checkOpeToJobStatus(int ope)
{
    return job_status[ope_to_job[ope]];
}

void OpeJobProc::initCombInfo()
{
    vector<vector<set<int>>> all_combs;
    computeORDict();
    for (int root : job_start_ope)
    {
        map<int, vector<set<int>>> dp;
        all_combs.push_back(searchCombByRec(root, dp));
    }

    int comb_id = 0;
    for (int i = 0; i < all_combs.size(); i++)
    {
        bool flag = false;
        for (const set<int> &comb : all_combs[i])
        {
            if (flag)
                job_to_comb[i].insert(comb_id);
            else
            {
                job_to_comb.push_back({comb_id});
                flag = true;
            }
            for (int ope : comb)
            {
                ope_to_comb[ope].insert(comb_id);
                comb_to_ope[comb_id].insert(ope);
            }
            comb_id++;
        }
    }
    num_combs = comb_id;
}

// Destructor
OpeJobProc::~OpeJobProc()
{ // No need to delete vectors, they manage their own memory
}
int OpeJobProc::getNumJobs() const
{
    return num_jobs;
}

int OpeJobProc::getNumCombs() const
{
    return num_combs;
}

bool OpeJobProc::isJobFinished(int job) const
{
    return job_status[job] == FINISHED;
}

bool OpeJobProc::isCombInJob(int comb, int job) const
{
    return job_to_comb[job].find(comb) != job_to_comb[job].end();
}

void OpeJobProc::initFeasibleOpes()
{
    for (int i = 0; i < num_jobs; ++i)
    {
        for (int ope : job_start_ope)
        {
            feasible_opes.insert(ope);
            setOpeStatus(ope, FEASIBLE);
        }
    }
}

void OpeJobProc::addEdge(int from, int to)
{
    graph[from].push_back(to);
    reverse_graph[to].push_back(from); // For ancestor computation
}

void OpeJobProc::addORPeer(int a, int b)
{
    or_peers[a].insert(b);
    or_peers[b].insert(a);
}

void OpeJobProc::addORConnector(int ope)
{
    or_connectors.insert(ope);
}

void OpeJobProc::addORDirect(int ope, const vector<vector<int>> &or_childs_pairs)
{
    for (const vector<int> &or_childs_pair : or_childs_pairs)
    {
        for (const int &or_child : or_childs_pair)
        {
            or_direct[ope].insert(or_child);
        }
    }
}

int OpeJobProc::getOpeJob(int ope)
{
    return ope_to_job[ope];
}

int OpeJobProc::getCurrentOpe(int job)
{
    return job_current_ope[job];
}

double OpeJobProc::getJobLastTime(int job) const
{
    return job_last_time[job];
}

int OpeJobProc::checkOpeStatus(int ope)
{
    return ope_status[ope];
}

void OpeJobProc::addORPeers(int *nodes, int num_or_peers)
{
    for (int i = 0; i < num_or_peers; ++i)
    {
        for (int j = 0; j < num_or_peers; ++j)
        {
            if (i != j)
            {
                or_peers[nodes[i]].insert(nodes[j]);
            }
        }
    }
}

void OpeJobProc::setOpeType(int ope, int type)
{
    ope_type[ope] = type;
}

void OpeJobProc::setOpeJob(int ope, int job)
{
    job_to_ope[job].insert(ope);
    ope_to_job[ope] = job;
}

// Add an edge to the graph
void OpeJobProc::addEdgeAllInfo(int from, int to, int job, bool relation)
{
    addEdge(from, to);
    setOpeJob(from, job);
    setOpeJob(to, job);
    if (relation == OR_PEER)
    {
        addORPeer(from, to);
    }
}

// Update cumulative ancestors and successors
void OpeJobProc::updateCumulative()
{
    for (int i = 0; i < num_opes; ++i)
    {
        computeSuccessors(i);
        computeAncestors(i);
    }
}

// Get cumulative cul_ancestors for a node
const std::unordered_set<int> &OpeJobProc::getAncestors(int node) const
{
    return cul_ancestors[node];
}

// Get cumulative cul_successors for a node
const std::unordered_set<int> &OpeJobProc::getSuccessors(int node) const
{
    return cul_successors[node];
}

// Print the graph structure
void OpeJobProc::printGraph() const
{
    for (int i = 0; i < num_opes; ++i)
    {
        std::cout << "Node " << i << " cul_successors: ";
        for (int neighbor : graph[i])
        {
            std::cout << neighbor << " ";
        }
        std::cout << std::endl;
    }
    // print other info
    for (int i = 0; i < num_opes; ++i)
    {
        std::cout << "Node " << i << " job: " << ope_to_job[i] << std::endl;
        std::cout << "Node " << i << " type: " << ope_type[i] << std::endl;
        std::cout << "Node " << i << " OR peers: ";
        for (int or_peer : or_peers[i])
        {
            std::cout << or_peer << " ";
        }
        std::cout << std::endl;
    }
}

int OpeJobProc::getNumOpes() const
{
    return num_opes;
}

void OpeJobProc::setStartOpe(int job, int ope)
{
    job_start_ope[job] = ope;
}

void OpeJobProc::setEndOpe(int job, int ope)
{
    job_end_ope[job] = ope;
}

// Print cumulative ancestors and successors

void OpeJobProc::setOpeStatus(int ope, int status)
{
    ope_status[ope] = status;
}
void OpeJobProc::setJobStatus(int job, int status)
{
    job_status[job] = status;
}
bool OpeJobProc::scheduleOpe(int ope, double time, double proc_time)
{
    checkFeasibility(ope, ope_status);
    bool has_comb_change = false;
    this->job_current_ope[ope_to_job[ope]] = ope;
    start_time[ope] = time;
    feasible_opes.erase(ope);
    double end_time_this_ope = time + proc_time;
    end_time[ope] = end_time_this_ope;
    job_last_time[ope_to_job[ope]] = end_time_this_ope;

    if (end_time_this_ope > max_end_time)
    {
        max_end_time = end_time_this_ope;
    }

    if (proc_time < EPS)
    {
        finishOpe(ope);
    }
    else
    {
        setOpeStatus(ope, PROCESSING);
        scheduling_opes.insert(ope);
        setJobStatus(ope_to_job[ope], PROCESSING);
    }

    for (int or_peer : or_peers[ope])
    {
        setOpeStatus(or_peer, UNFEASIBLE);
        // BFS to set all the successors of or_peer to UNFEASIBLE
        std::queue<int> q;
        q.push(or_peer);
        while (!q.empty())
        {
            int current = q.front();
            q.pop();
            for (int successor : graph[current])
            {
                if (!checkIfIsSuccessor(ope, successor) && ope_status[successor] != UNFEASIBLE)
                {
                    setOpeStatus(successor, UNFEASIBLE);
                    remain_opes.erase(successor);
                    feasible_opes.erase(successor);
                    q.push(successor);
                }
            }
        }
        remain_opes.erase(or_peer);
        feasible_opes.erase(or_peer);
    }
    if (or_peers[ope].size() != 0)
    {
        vector<int> erase_comb;
        for (int comb : job_to_comb[ope_to_job[ope]])
        {
            if (ope_to_comb[ope].find(comb) == ope_to_comb[ope].end())
            {
                erase_comb.push_back(comb);
            }
        }
        has_comb_change = erase_comb.size();
        for (int comb : erase_comb)
        {
            job_to_comb[ope_to_job[ope]].erase(comb);
        }
    }
    return has_comb_change;
}

double OpeJobProc::getOpeJobAvaTime(int ope)
{
    int job = getOpeJob(ope);
    return job_last_time[job];
}

unordered_set<int> OpeJobProc::getWellPrepOpes(double current_time)
{
    unordered_set<int> well_prep_opes;
    for (int ope : feasible_opes)
    {
        if (getOpeJobAvaTime(ope) <= current_time)
        {
            well_prep_opes.insert(ope);
        }
    }
    return well_prep_opes;
}

void OpeJobProc::finishOpe(int ope)
{
    setOpeStatus(ope, FINISHED);
    setJobStatus(ope_to_job[ope], IDLE);
    remain_opes.erase(ope);
    scheduling_opes.erase(ope);

    if (job_end_ope[ope_to_job[ope]] == ope)
    {
        job_status[ope_to_job[ope]] = FINISHED;
    }
    else
    {
        for (int successor : graph[ope])
        {
            bool flag = true;
            for (int ancestor : reverse_graph[successor])
            {
                if (ope_status[ancestor] != FINISHED && ope_status[ancestor] != UNFEASIBLE)
                {
                    flag = false;
                    break;
                }
            }
            if (flag)
            {
                feasible_opes.insert(successor);
                setOpeStatus(successor, FEASIBLE);
            }
        }
    }
}

void OpeJobProc::checkIfOpeFinished(double time)
{
    std::vector<int> to_remove;
    for (int ope : scheduling_opes)
    {
        if (end_time[ope] - time < EPS)
        {
            to_remove.push_back(ope);
        }
    }
    for (int i = 0; i < to_remove.size(); i++)
    {
        finishOpe(to_remove[i]);
    }
}

OpeMasProc::OpeMasProc(int num_opes, int num_mas) : num_opes(num_opes), num_mas(num_mas)
{
    feasible_mas_count_for_ope.resize(num_opes, 0);
    feasible_ope_count_for_mas.resize(num_mas, 0);
    finish_time.resize(num_mas, 0.0);
    ope_feas_mas.resize(num_opes);
    mas_feas_ope.resize(num_mas);
    production_time.resize(num_opes, std::vector<double>(num_mas, 0.0));
    production_time_transpose.resize(num_mas, std::vector<double>(num_opes, 0.0));
    mas_status.resize(num_mas, IDLE);
}

OpeMasProc::OpeMasProc(const OpeMasProc &other)
{
    num_opes = other.num_opes;
    num_mas = other.num_mas;
    feasible_mas_count_for_ope = other.feasible_mas_count_for_ope;
    feasible_ope_count_for_mas = other.feasible_ope_count_for_mas;
    finish_time = other.finish_time;
    ope_feas_mas = other.ope_feas_mas;
    mas_feas_ope = other.mas_feas_ope;
    production_time = other.production_time;
    production_time_transpose = other.production_time_transpose;
    mas_status = other.mas_status;
    scheduling_mas = other.scheduling_mas;
}

OpeMasProc &OpeMasProc::operator=(const OpeMasProc &other)
{
    if (this == &other)
    {
        return *this;
    }
    num_opes = other.num_opes;
    num_mas = other.num_mas;
    feasible_mas_count_for_ope = other.feasible_mas_count_for_ope;
    feasible_ope_count_for_mas = other.feasible_ope_count_for_mas;
    finish_time = other.finish_time;
    ope_feas_mas = other.ope_feas_mas;
    mas_feas_ope = other.mas_feas_ope;
    production_time = other.production_time;
    production_time_transpose = other.production_time_transpose;
    mas_status = other.mas_status;
    scheduling_mas = other.scheduling_mas;
    return *this;
}

int OpeMasProc::getNumOpes() const
{
    return num_opes;
}
int OpeMasProc::getNumMas() const
{
    return num_mas;
}
int OpeMasProc::checkMaStatus(int mas)
{
    return mas_status[mas];
}
void OpeMasProc::addFeasiblePair(int ope, int mas, double time)
{
    ope_feas_mas[ope].insert(mas);
    mas_feas_ope[mas].insert(ope);
    production_time[ope][mas] = time;
    production_time_transpose[mas][ope] = time;
    feasible_mas_count_for_ope[ope]++;
    feasible_ope_count_for_mas[mas]++;
}

void OpeMasProc::deleteFeasibleMas(int ope, int mas)
{
    ope_feas_mas[ope].erase(mas);
    mas_feas_ope[mas].clear();
    production_time[ope][mas] = 0.0;
    production_time_transpose[mas][ope] = 0.0;
    feasible_mas_count_for_ope[ope]--;
    feasible_ope_count_for_mas[mas] = 0;
}

double OpeMasProc::calAvgTimeOpe(int ope)
{
    double sum = 0;
    for (int i : ope_feas_mas[ope])
    {
        sum += production_time[ope][i];
    }
    return sum / feasible_mas_count_for_ope[ope];
}
double OpeMasProc::getProcTime(int ope, int mas)
{
    return production_time[ope][mas];
}
double OpeMasProc::calAvgTimeMas(int mas)
{
    double sum = 0;
    for (int i : mas_feas_ope[mas])
    {
        sum += production_time_transpose[mas][i];
    }
    return sum / feasible_ope_count_for_mas[mas];
}

int OpeMasProc::getMinTimeMa(int ope, bool only_feasible)
{
    double min = INF;
    int min_mas = INF;
    for (int i : ope_feas_mas[ope])
    {
        // if (only_feasible && mas_status[i] != IDLE)
        //     continue;
        if (production_time[ope][i] < min)
        {
            min = production_time[ope][i];
            min_mas = i;
        }
    }
    // if (min_mas == -1)
    // {
    //     cerr << "No feasible machine for operation " << ope << endl;
    // }
    return min_mas;
}
void OpeMasProc::printGraph() const
{

    for (int i = 0; i < num_opes; ++i)
    {
        std::cout << "Operation " << i << " feasible machines: ";
        for (int mas : ope_feas_mas[i])
        {
            std::cout << mas << " time: " << production_time[i][mas] << std::endl;
        }

        std::cout << std::endl;
    }
}
void OpeMasProc::setMaStatus(int mas, int status)
{
    mas_status[mas] = status;
}

void OpeMasProc::scheduleMa(int mas, double time, double proc_time)
{
    if (proc_time >= EPS)
    {
        setMaStatus(mas, PROCESSING);
        scheduling_mas.insert(mas);
        finish_time[mas] = time + proc_time;
    }
}

void OpeMasProc::finishMa(int mas)
{
    setMaStatus(mas, IDLE);
    scheduling_mas.erase(mas);
}

void OpeMasProc::checkIfMaFinished(double time)
{
    std::vector<int> to_remove;
    for (int mas : scheduling_mas)
    {
        if (finish_time[mas] - time < EPS)
        {
            to_remove.push_back(mas);
        }
    }
    for (int i = 0; i < to_remove.size(); i++)
    {
        finishMa(to_remove[i]);
    }
}

// Find the earliest time for an operation to be scheduled, given the min expected time(such as current time).
double OpeMasProc::findEarliestTime(int ope, double time)
{
    double min_time = INF;

    for (int ma : ope_feas_mas[ope])
    {
        if (mas_status[ma] == IDLE)
            return time;
        else
        {
            if (finish_time[ma] < min_time)
                min_time = finish_time[ma];
        }
    }
    return min_time;
}


double OpeMasProc::getFinishTime(int mas)
{
    return finish_time[mas];
}

const unordered_set<int>& OpeMasProc::getFeasibleMas(int ope) const
{
    return ope_feas_mas[ope];
}