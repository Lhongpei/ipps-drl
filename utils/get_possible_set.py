import networkx as nx
import matplotlib.pyplot as plt
from itertools import product
import numpy as np
import torch
from utils.utils import getAncestors, parse_data, getAdjacent, nums_detec
from itertools import product


def read_ipps_data(file_path, matrix_proc_time, matrix_ope_ma_adj,  id_operation):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        first_line = lines[0].strip().split()
        num_jobs, num_machines, num_operation = int(
            first_line[0]), int(first_line[1]), int(first_line[2])
        machine_oper = {}
        machine_oper[(0, 0)] = []
        ope_ma_adj = {}
        process_time = {}
        for job_idx in range(num_jobs):
            for j in id_operation[job_idx]:
                machine_oper[(job_idx, j)] = []
                for k in range(num_machines):
                    if matrix_ope_ma_adj[j, k] > 0:
                        machine_oper[(job_idx, j)].append(k)
                        process_time[(job_idx, j, k)] = int(
                            matrix_proc_time[j, k].item())
                        ope_ma_adj[(job_idx, j, k)
                                   ] = matrix_ope_ma_adj[j, k].item()

    return machine_oper, process_time, ope_ma_adj


def read_ipps(graph):
    order1 = []
    for node in graph.nodes():
        successors = list(graph.successors(node))
        for i in range(len(successors)):
            for j in range(i + 1, len(successors)):
                order1.append((node, successors[i], successors[j]))

    order2 = []
    for node in nx.topological_sort(graph):
        for successor in graph.successors(node):
            order2.append((node, successor))
    return order1, order2


def read_ipps_sol(sol_path, instance_num):
    loaded_sols = np.load(sol_path, allow_pickle=True)
    loaded_sol = loaded_sols[instance_num]

    num_machines = max(action[0] for action in loaded_sol) + 1
    num_jobs = max(action[1] for action in loaded_sol) + 1
    num_operations = [max(action[2] + 1 for action in loaded_sol if action[1] == job_id)
                      for job_id in range(num_jobs)]

    durations = [max(action[4] for action in loaded_sol if action[1] == job_id)
                 for job_id in range(num_jobs)]
    makespan = max(durations)

    assignment = [[[0 for _ in range(num_machines)] for _ in range(
        operation)] for operation in num_operations]
    start_times = [[[0 for _ in range(num_machines)] for _ in range(
        operation)] for operation in num_operations]
    end_times = [[[0 for _ in range(num_machines)] for _ in range(
        operation)] for operation in num_operations]

    for action in loaded_sol:
        ma_id, job_id, ope_id, start_time, end_time = action
        assignment[job_id][ope_id][ma_id] = 1
        start_times[job_id][ope_id][ma_id] = start_time
        end_times[job_id][ope_id][ma_id] = end_time

    return start_times, end_times, assignment, durations, makespan

def build_graph(data):
    out_lines, in_lines, _ = parse_data(data)
    _, _, num_opes = nums_detec(data)
    adjacent, or_edge = getAdjacent(out_lines, num_nodes=num_opes, or_successors=False)
    graph = nx.DiGraph(adjacent.to('cpu').numpy())  
    return graph

def get_subgraphs(graph):
    connected_components = list(nx.weakly_connected_components(graph))

    subgraphs = [graph.subgraph(nodes) for nodes in connected_components]

    return subgraphs




def visualize_graph(graph):
    pos = nx.nx_agraph.graphviz_layout(graph, prog='dot')

    plt.figure(figsize=(15, 10))

    nx.draw(graph, pos, with_labels=True, node_size=2000,
            node_color="lightblue", font_size=10, font_weight="bold", arrowsize=20)

    edge_labels = {}
    for u, v, d in graph.edges(data=True):
        if 'label' in d:
            edge_labels[(u, v)] = d['label']
        else:
            edge_labels[(u, v)] = ''  
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.title("Job Workflow")
    plt.show()


def get_set(problem):
    if (isinstance(problem, str)):
        with open(problem, 'r') as file:
            data = file.readlines()
    else:
        data = problem
        data[:-1] = [line + '\n' for line in data[:-1]]
    out_lines, _, _ = parse_data(data)
    _, _, num_opes = nums_detec(data)
    adjacent, or_edge = getAdjacent(out_lines, num_nodes=num_opes, or_successors=False)
    matrix_cal_cumul=getAncestors(adjacent.t()).t()
    or_dict,or_direct=find_or_dict(num_opes,matrix_cal_cumul,or_edge)
    composite_set = get_set_core(adjacent,or_edge,or_dict,or_direct)
    return composite_set

def find_or_dict(num_opes,matrix_cal_cumul,or_edge):
    or_nodes=set()
    or_dict={}
    or_direct={}
    for u,v in or_edge:
        or_nodes.add(u)
        if u in or_direct:
            or_direct[u].add(v)
        else:
            or_direct[u]={v}

    for u in range(num_opes):
        or_dict[u]=set()
        for v in or_nodes:
            if matrix_cal_cumul[u][v]==1:
                or_dict[u].add(v)
    return or_dict,or_direct


def search_path_by_rec(G,root,or_edges,or_dict,or_direct,dp={}):
    result=[{root}]
    edges = list(G.edges(root))
    if(G.out_degree(root) == 0):
        return result
    or_path=[]
    and_path=[[{root}]]
    or_nb=[]
    and_nb=[root]
    for u,v in edges: 
        path=dp[v] if v in dp else search_path_by_rec(G,v,or_edges,or_dict,or_direct,dp)
        if (u,v) in or_edges:
            or_nb.append(v)
            or_path.append(path)
        else:
            and_nb.append(v)
            and_path.append(path)
    result=union_without_or(or_nb,and_nb,or_path,and_path,or_dict,or_direct)
    dp[root]=result
    return result

def union_without_or(or_nb,and_nb,or_path,and_path,or_dict,or_direct):
    path=[]
    root=or_dict[and_nb[0]]
    or_common=set(root)
    if len(and_path)>1:
        path=[i for i in and_path[0]]
        for i in range(1,len(and_nb)):
            new_path=[]
            a=and_nb[i]
            or_common=or_common.union(or_dict[a])
            for p,q in product(path,and_path[i]):
                flag=True
                for common_or in or_common:
                    if len(p.union(q).intersection(or_direct[common_or]))>1:
                        flag=False
                        break
                if flag:
                    new_path.append(p.union(q))
            path=new_path.copy()
        new_path=[]
        if len(or_path)!=0:
            for i,a in enumerate(or_nb):
                or_common=or_dict[a].union(or_common)
                for p,q in product(path,or_path[i]):
                    flag=True
                    for common_or in or_common:
                        if len(p.union(q).intersection(or_direct[common_or]))>1:
                            flag=False
                            break
                    if flag:
                        new_path.append(p.union(q))
            path=new_path
    else:
        for a,b in product(or_path,and_path):
            for p,q in product(a,b):
                path.append(p.union(q))
    return path


def find_root_nodes(graph):
    all_nodes = set(graph.nodes)
    child_nodes = {child for parent in graph for child in graph.successors(parent)}
    root_nodes = all_nodes - child_nodes
    return root_nodes

def get_comb_info(problem, num_jobs):
    composite_set = get_set(problem)
    num_combinations = []
    for i, job_combinations in enumerate(composite_set):
        num_combinations.append(len(job_combinations))
    num_set_operations = {}
    for i in range(num_jobs):
        for j in range(len(job_combinations)):
            num_set_operations[(i, j)] = len(job_combinations)
    id_combination = {i: [l for l in range(
        num_combinations[i])] for i in range(num_jobs)}
    id_set_operation = {i: {h: {
        j for j in composite_set[i][h]} for h in id_combination[i]} for i in range(num_jobs)}
    return id_combination, id_set_operation

def comb_matrix(problem):
    if (isinstance(problem, str)): 
        with open(problem, 'r') as file:
            lines = file.read().splitlines()
        first_line = lines[0].strip().split()
    else:
        first_line = problem[0].strip().split()
    num_jobs, num_opes = int(first_line[0]), int(first_line[2])
    out_lines, in_lines, _ = parse_data(lines)
    adjacent, or_edge = getAdjacent(out_lines, num_nodes=num_opes, or_successors=False)
    matrix_cal_cumul=getAncestors(adjacent.t()).t()
    or_dict,or_direct=find_or_dict(num_opes,matrix_cal_cumul,or_edge)
    matrix_combs_id, matrix_combs = comb_matrix_core(adjacent, or_edge, in_lines, num_jobs, num_opes,or_dict,or_direct)
    return matrix_combs_id, matrix_combs

def get_set_core(adjacent,or_edges,or_dict,or_direct):
    graph=nx.DiGraph(adjacent.to('cpu').numpy())
    subgraphs = get_subgraphs(graph)
    composite_set = []
    for subgraph in subgraphs:
        root=find_root_nodes(subgraph)
        if(len(root)!=1):
            assert "len(root)!=1"
        root=next(iter(root))
        #root=1
        composite_set.append(search_path_by_rec(subgraph,root,or_edges,or_dict,or_direct,{}))
    return composite_set

def comb_matrix_core(adjacent, or_edges, in_lines, num_jobs, num_opes,or_dict,or_direct):
    
    graph=nx.DiGraph(adjacent.to('cpu').numpy())
    subgraphs = get_subgraphs(graph)
    composite_set = []
    for subgraph in subgraphs:
        root=find_root_nodes(subgraph)
        if(len(root)!=1):
            assert "len(root)!=1"
        root=next(iter(root))
        composite_set.append(search_path_by_rec(subgraph,root,or_edges,or_dict,or_direct,{}))

    num_combs = sum(len(inner_list) for inner_list in composite_set)
    matrix_combs_id = torch.zeros((num_jobs, num_combs), dtype=torch.float)
    matrix_combs = torch.zeros((num_combs, num_opes), dtype=torch.float)
    comb_id = 0
    for job_id in range(num_jobs):
        for comb in composite_set[job_id]:
            matrix_combs_id[job_id][comb_id] = 1
            matrix_combs[comb_id][list(comb)] = 1
            comb_id += 1
    return matrix_combs_id, matrix_combs
