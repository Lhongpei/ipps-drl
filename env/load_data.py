import torch
from utils.utils import getAncestors, parse_data, getAdjacent
from utils.get_possible_set import comb_matrix_core,find_or_dict
#假设数据从一个名为"graph_data.txt"的文件中读取
file_path = 'kim_dataset/problem/problem1.txt'


def extract_numbers(lines):
    numbers = []
    for line in lines:
        for part in line.replace('(', ' ').replace(')', ' ').split():
            numbers.extend(map(int, part.split(',')))
    return numbers

def load_ipps(lines, num_mas = None):
    """
    Load IPPS data from the given lines.

    Parameters:
    - lines (list or str):
        - A list of strings representing the IPPS data.
        - Or a string representing the file path of the IPPS data.
    - num_mas (int): The number of machines.(
        default: None, 
        which means the number of machines will be determined from the data.
        )

    Returns:
    - matrices (tuple): A tuple containing the generated matrices.

    """
    if isinstance(lines, str):
        with open(lines, 'r') as file:
            lines = file.read().splitlines()
    
    return generate_matrices(lines, num_mas)

def generate_matrices(lines, given_num_mas=None):
    """
    Generate matrices based on the given input lines.

    Parameters:
        - out_lines (list): List of output lines.
        - in_lines (list): List of input lines.
        - info_lines (list): List of information lines.
        - given_num_mas (int): The number of machines.(
            default: None,
            which means the number of machines will be determined from the data.
            )

    Returns:
        tuple: A tuple containing the following matrices:
            - matrix_proc_time (torch.Tensor): Matrix representing processing time for each operation on each machine.
            - matrix_ope_ma_adj (torch.Tensor): Matrix representing the adjacency between operations and machines.
            - matrix_pre_proc (torch.Tensor): Matrix representing the adjacency between operations.
            - opes_appertain (torch.Tensor): Tensor representing the job to which each operation belongs.
            - num_ope_biases (torch.Tensor): Tensor representing the biases for the number of operations in each job.
            - nums_ope (torch.Tensor): Tensor representing the number of operations in each job.
            - matrix_cal_cumul (torch.Tensor): Matrix representing the cumulative adjacency between operations.
            - ope_or (torch.Tensor): Matrix representing the adjacency between operations for replacement.
            - ope_req_num (torch.Tensor): Tensor representing the number of required operations for each operation.
            - ope_eligible (torch.Tensor): Tensor representing the eligibility of each operation.

    """
    out_lines, in_lines, info_lines = parse_data(lines)
    num_jobs, num_mas, num_opes = map(int, info_lines[0].split())
    
    # if the number of machines is given, use it
    if given_num_mas is not None:
        num_mas = given_num_mas
        
    info_lines = info_lines[1:]
    num_nodes = len(info_lines)

    # initialize matrices
    matrix_proc_time = torch.zeros((num_opes, num_mas))
    matrix_ope_ma_adj = torch.zeros((num_opes, num_mas), dtype=torch.int)
    opes_appertain = torch.zeros(num_opes, dtype=int)
    num_ope_biases = []
    nums_ope = torch.zeros(num_jobs, dtype=int)
    matrix_cal_cumul = torch.zeros((num_opes, num_opes), dtype=torch.int)
    ope_req_num = torch.zeros((num_nodes), dtype=torch.int)
    ope_eligible = torch.zeros(num_opes, dtype=torch.int)

    # fill matrices
    
    matrix_pre_proc, or_edge, ope_or = getAdjacent(out_lines, num_opes, or_successors=True)
    matrix_cal_cumul=getAncestors(matrix_pre_proc.t()).t()
    or_dict,or_direct=find_or_dict(num_opes,matrix_cal_cumul,or_edge)
    matrix_combs_id, matrix_combs = comb_matrix_core(matrix_pre_proc, or_edge, in_lines, num_jobs, num_opes,or_dict,or_direct)

    # 计算入度并调整特定节点
    ope_req_num = matrix_pre_proc.sum(0).t()
    remain_opes = torch.ones(num_opes, dtype=int)
    for line in in_lines:
        parts = line.split()
        node = int(parts[0])
        for i in range(1, len(parts)):
            if '(' in parts[i]:
                targets = map(int, parts[i].strip('()').split(','))
                ope_req_num[node] -= (len(list(targets)) - 1)

    # 处理机器分配和处理时间
    job = -1
    for line in info_lines:
        parts = line.split()
        ope_id = int(parts[0])
        if parts[1] == 'start':
            job += 1
            matrix_ope_ma_adj[ope_id, :] = 1
            matrix_proc_time[ope_id, :] = 0
            num_ope_biases.append(ope_id)
            opes_appertain[ope_id] = job
            nums_ope[job] += 1
        elif parts[1] == 'end':
            matrix_ope_ma_adj[ope_id, :] = 1
            matrix_proc_time[ope_id, :] = 0
            opes_appertain[ope_id] = job
            nums_ope[job] += 1
            # remain_opes[ope_id] = 0
        elif parts[1] == 'supernode':
            matrix_ope_ma_adj[ope_id, :] = 1
            matrix_proc_time[ope_id, :] = 0
            opes_appertain[ope_id] = job
            nums_ope[job] += 1
        else:
            mas_info = parts[2:]
            for j in range(0, len(mas_info), 2):
                mas_id = int(mas_info[j]) - 1
                proc_time = float(mas_info[j + 1])
                matrix_proc_time[ope_id, mas_id] = proc_time
                matrix_ope_ma_adj[ope_id, mas_id] = 1
            opes_appertain[ope_id] = job
            nums_ope[job] += 1

    matrix_cal_cumul=getAncestors(matrix_pre_proc)

    # 考虑到当前节点，进行初始化更新
    for ope_id in num_ope_biases:
        ope_eligible[ope_id] = 1
    remain_opes[num_ope_biases] = 0
    ope_eligible = ope_eligible.unsqueeze(0).float().matmul(matrix_pre_proc.float()).squeeze().int()
    ope_req_num = ope_req_num - ope_eligible

    return (
        matrix_proc_time,
        matrix_ope_ma_adj,
        matrix_pre_proc,
        opes_appertain.int(),
        torch.tensor(num_ope_biases).int(),
        nums_ope.int(),
        matrix_cal_cumul,
        ope_or,
        ope_req_num,
        ope_eligible,
        matrix_combs_id,
        matrix_combs,
        remain_opes
    )

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    :param lines: List of strings, each string is a line of input data representing job-shop scheduling information
    :return: Tuple of (num_jobs, num_machines, num_opes)
    '''
    # 从第一行提取作业数和机器数
    num_jobs, num_machines, num_opes = map(int, lines[0].split())

    return num_jobs, num_machines, num_opes

def main():
    with open(file_path, 'r') as file:
        lines = file.read().splitlines()
    return load_ipps(lines)


if __name__ == '__main__':
    matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc_t, \
    opes_appertain, num_ope_biases, nums_ope, matrix_cal_cumul, \
    ope_or, ope_req_num, ope_eligible = main()
    print(matrix_proc_time)
    print(matrix_ope_ma_adj)
    print(matrix_pre_proc)
    print(matrix_pre_proc_t)
    print(opes_appertain)
    print(num_ope_biases)
    print(nums_ope)
    print(matrix_cal_cumul)
    print(ope_or)
    print(ope_req_num)
    print(ope_eligible)
