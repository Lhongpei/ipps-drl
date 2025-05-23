#ifndef GRAPH_H
#define GRAPH_H
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
using namespace std;

class OpeJobProc
{
private:
    // Static Attributes
    int num_opes;
    int num_jobs;
    int num_combs;
    std::vector<std::vector<int>> graph;                 // Adjacency list for the graph
    std::vector<std::vector<int>> reverse_graph;         // Adjacency list for the reverse graph
    std::vector<std::unordered_set<int>> cul_ancestors;  // Cumulative ancestors
    std::vector<std::unordered_set<int>> cul_successors; // Cumulative successors
    std::vector<bitset<MAX_OPES>> cul_ancestors_bitset;  // Cumulative ancestors
    std::vector<bitset<MAX_OPES>> cul_successors_bitset; // Cumulative successors
    std::vector<std::unordered_set<int>> or_peers;
    std::vector<std::unordered_set<int>> job_to_ope;
    std::vector<int> ope_to_job;
    std::map<int, set<int>> or_dict;
    std::map<int, set<int>> or_direct;
    std::vector<std::unordered_set<int>> comb_to_ope;
    std::vector<int> job_start_ope;
    std::vector<int> job_end_ope;
    std::vector<int> job_current_ope;
    std::vector<int> ope_type;
    std::set<int> or_connectors;
    std::vector<std::unordered_set<int>> ope_to_comb;

    // Dynamic Attributes
    std::vector<std::unordered_set<int>> job_to_comb;
    std::vector<int> ope_status;
    std::vector<double> start_time;
    std::vector<double> end_time;
    std::vector<double> job_last_time;
    double max_end_time;
    std::unordered_set<int> remain_opes;
    std::unordered_set<int> feasible_opes;
    std::unordered_set<int> scheduling_opes;
    std::vector<int> job_status;

    // Helper function to perform BFS and compute successors
    void computeSuccessors(int node);

    // Helper function to perform BFS and compute ancestors
    void computeAncestors(int node);

    vector<set<int>> union_without_or(const vector<int> &or_nb,
                                      const vector<int> &and_nb,
                                      const vector<vector<set<int>>> &or_path,
                                      const vector<vector<set<int>>> &and_path);
    vector<set<int>> searchCombByRec(int node,
                                     map<int, vector<set<int>>> &dp);

    void computeORDict();

    // Helper function to check if target_ope is an ancestor of ope
    bool checkIfIsAncestor(int ope, int target_ope);

    // Helper function to check if target_ope is a successor of ope
    bool checkIfIsSuccessor(int ope, int target_ope);

public:
    // Constructor
    OpeJobProc(int num_opes, int jobs);
    OpeJobProc(const OpeJobProc &other);
    OpeJobProc &operator=(const OpeJobProc &other);
    std::unordered_set<int> getFeasibleOpes();
    std::unordered_set<int> getSchedulingOpes();
    std::unordered_set<int> getCombByJob(int job);
    std::unordered_set<int> getOpeByComb(int comb);
    double getEndTime(int ope);
    double getMaxEndTime();

    void initCombInfo();
    // Destructor
    ~OpeJobProc();
    int getNumJobs() const;

    int getNumCombs() const;

    bool isJobFinished(int job) const;

    bool isCombInJob(int comb, int job) const;

    void initFeasibleOpes();

    void addEdge(int from, int to);

    void addORPeer(int a, int b);

    void addORConnector(int ope);

    void addORDirect(int ope, const vector<vector<int>> &or_childs_pairs);

    void addORPeers(int *nodes, int num_or_peers);

    void setOpeType(int ope, int type);
    void setOpeJob(int ope, int job);
    void setJobStatus(int job, int status);
    // Add an edge to the graph
    void addEdgeAllInfo(int from, int to, int job, bool relation);

    // Update cumulative ancestors and successors
    void updateCumulative();

    // Get cumulative ancestors for a node
    const std::unordered_set<int> &getAncestors(int node) const;

    // Get cumulative successors for a node
    const std::unordered_set<int> &getSuccessors(int node) const;

    // Print the graph structure
    void printGraph() const;

    int getNumOpes() const;

    double getJobLastTime(int job) const;

    void setStartOpe(int job, int ope);

    void setEndOpe(int job, int ope);

    // get the job of an operation
    int getOpeJob(int ope);

    // get the current operation of a job
    int getCurrentOpe(int job);

    // Print cumulative ancestors and successors
    void printCumulative() const;

    void setOpeStatus(int ope, int status);

    int checkOpeStatus(int ope);

    bool scheduleOpe(int ope, double time, double proc_time); // return whether comb set changes

    int checkOpeToJobStatus(int ope);

    void finishOpe(int ope);

    void checkIfOpeFinished(double time);

    double getOpeJobAvaTime(int ope);

    unordered_set<int> getWellPrepOpes(double current_time);

    const unordered_set<int>& getOpeComb(int ope) const {
        return ope_to_comb.at(ope);
    }
};
class OpeMasProc
{
private:
    int num_opes;
    int num_mas;
    std::vector<int> feasible_mas_count_for_ope;
    std::vector<int> feasible_ope_count_for_mas;
    std::vector<std::unordered_set<int>> ope_feas_mas;
    std::vector<std::unordered_set<int>> mas_feas_ope;
    std::vector<std::vector<double>> production_time;
    std::vector<std::vector<double>> production_time_transpose;
    std::vector<int> mas_status;
    std::unordered_set<int> scheduling_mas;
    std::vector<double> finish_time;

public:
    // Constructor
    OpeMasProc(int num_opes, int num_mas);
    OpeMasProc(const OpeMasProc &other);
    OpeMasProc &operator=(const OpeMasProc &other);
    int getNumOpes() const;
    int getNumMas() const;
    void addFeasiblePair(int ope, int mas, double time);

    void deleteFeasibleMas(int ope, int mas);

    double calAvgTimeOpe(int ope);
    double getProcTime(int ope, int mas);
    double calAvgTimeMas(int mas);

    int getMinTimeMa(int ope, bool only_feasible = true);
    void printGraph() const;
    void setMaStatus(int mas, int status);
    int checkMaStatus(int mas);

    void scheduleMa(int mas, double time, double proc_time);

    void finishMa(int mas);

    void checkIfMaFinished(double time);

    double findEarliestTime(int ope, double time);

    const unordered_set<int> &getFeasibleMas(int ope) const;
    double getFinishTime(int mas);
};

#endif // GRAPH_H