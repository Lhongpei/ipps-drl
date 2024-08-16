import torch.nn.functional as F
import torch
from collections import deque
import numpy as np
from functools import cmp_to_key
from torch.nn.utils.rnn import pad_sequence
def human_readable_size(size, decimal_places=2):
    """Convert a size in bytes to a human-readable string."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024:
            return f"{size:.{decimal_places}f} {unit}"
        size /= 1024
    return f"{size:.{decimal_places}f} TB"

def pad_2d_tensors(tensors, value=0, pad_dim=1, given_size=None):
    """
    Pad a list of 2D tensors with zeros to match the maximum size along the specified dimension.

    Args:
        tensors (list): A list of 2D tensors to be padded.
        value (int, optional): The value used for padding. Defaults to 0.
        pad_dim (int, optional): The dimension along which to pad the tensors. 
                                1 for padding along the first dimension (rows), 
                                2 for padding along the both dimensions (rows and columns).
                                Defaults to 1.

    Returns:
        torch.Tensor: A stacked tensor containing the padded tensors.

    Example:
        tensors = [torch.tensor([[1, 2], [3, 4]]), torch.tensor([[5, 6]])]
        padded_tensors = pad_2d_tensors(tensors, value=0, pad_dim=2)
        print(padded_tensors)
        # Output: tensor([[[1, 2, 0], [3, 4, 0]], [[5, 6, 0], [0, 0, 0]]])
    """

    max_row = max(t.size(0) for t in tensors)
    max_col = max(t.size(1) for t in tensors)
    if given_size is not None:
        assert len(given_size) == 2
        assert given_size[0] >= max_row
        assert given_size[1] >= max_col
        max_row = given_size[0]
        max_col = given_size[1]
    
    padded_tensors = []
    for t in tensors:
        if pad_dim == 1:
            padding = (0, 0, 0, max_row - t.size(0))
        elif pad_dim == 2:
            padding = (0, max_col - t.size(1), 0, max_row - t.size(0))
        
        padded_tensor = F.pad(t, padding, "constant", value)
        padded_tensors.append(padded_tensor)
    
    stacked_tensors = torch.stack(padded_tensors, dim=0)
    return stacked_tensors

def pad_to_given_size(tensors:torch.Tensor, size, value=0, ignore_first=True, num_dims=2)->torch.Tensor:
    
    if num_dims == 2:
        target_row = size[0]
        target_col = size[1]
        if ignore_first:
            padding = (0, target_col - tensors.size(2), 0, target_row - tensors.size(1), 0, 0)
        else:
            padding = (0, target_col - tensors.size(1), 0, target_row - tensors.size(0))
            
        padded_tensor = F.pad(tensors, padding, "constant", value)
    elif num_dims == 1:
        target_size = size
        if ignore_first:
            padding = (0, target_size - tensors.size(1), 0, 0)
        else:
            padding = (0, target_size - tensors.size(0))
        padded_tensor = F.pad(tensors, padding, "constant", value)        
    return padded_tensor

def pad_to_middle_given_size(tensors:torch.Tensor, size, value=0):

    target_size = size
    padding = (0, 0, 0, target_size - tensors.size(1), 0, 0)
    padded_tensor = F.pad(tensors, padding, "constant", value)
    return padded_tensor
            
def pad_stack_add_idxes(ori_tensors, add_tensor_list, ori_valid_num, new_valid_num, add_batch_idxes, value=0, dim=2):
    new_tensor_list = []
    j = 0
    ori_valid_num = (ori_valid_num[0][add_batch_idxes], ori_valid_num[1][add_batch_idxes]) if dim == 2 else ori_valid_num[add_batch_idxes]
    max_size = (new_valid_num[0].max(), new_valid_num[1].max()) if dim == 2 else new_valid_num.max()
    for i in range(ori_tensors.size(0)):

        padded_ori_tensor = pad_to_given_size(ori_tensors[i], max_size, value=value, num_dims=dim, ignore_first=False)
        if i in add_batch_idxes:
            if dim == 2:
                padded_ori_tensor[torch.arange(ori_valid_num[0][j], new_valid_num[0][j]).unsqueeze(-1), 
                                   torch.arange(ori_valid_num[1][j], new_valid_num[1][j])] = add_tensor_list[j].to(padded_ori_tensor.dtype)
            else:
                padded_ori_tensor[torch.arange(ori_valid_num[j], new_valid_num[j])] = add_tensor_list[j].to(padded_ori_tensor.dtype)
            j += 1
        new_tensor_list.append(padded_ori_tensor)
    return torch.stack(new_tensor_list, dim=0)

def pad_1d_tensors(tensors, value=0, given_size=None):
    """
    Pad a list of 1-dimensional tensors with a specified value.

    Args:
        tensors (list): A list of 1-dimensional tensors to be padded.
        value (int, optional): The value used for padding. Defaults to 0.

    Returns:
        torch.Tensor: A stacked tensor with padded tensors.

    """
    max_size = max(t.size(0) for t in tensors)
    if given_size is not None:
        assert given_size >= max_size
        max_size = given_size
        
    padded_tensors = []
    for t in tensors:
        padding = (0, max_size - t.size(0))
        padded_tensor = F.pad(t, padding, "constant", value)
        padded_tensors.append(padded_tensor)
    
    stacked_tensors = torch.stack(padded_tensors, dim=0)
    return stacked_tensors

def wait_noneligible(state):
    machine_avail_time = state.machines_batch[:, :, 1]
    # available_time of jobs
    job_avail_time = state.job_end_time_batch
    # remain available_time greater than current time
    expanded_time = state.time_batch.unsqueeze(1)  # shape: (batch_size, 1)

    ma_jump = machine_avail_time > expanded_time
    job_jump = job_avail_time > expanded_time

    eligible_index = (torch.any(ma_jump, dim=1) | torch.any(job_jump, dim=1)).view(-1)
    return torch.where(eligible_index[state.batch_idxes], 0, 1)


def find_action_indexes(eligible_pair, selected_pair):

    # Expand dimensions to compare all pairs at once
    eligible_pair_expanded = eligible_pair  # Shape: (2, N, 1)
    action_expanded = selected_pair.t().view(2,-1) # Shape: (2, 1, M)
    
    # Compare all pairs
    matches = (eligible_pair_expanded == action_expanded).all(dim=0)  # Shape: (N, M)
    
    # Find the index of the matches
    action_indexes = matches.nonzero()  # Shape: (M,)
    
    if action_indexes.size(0) == 0:
        return None
    
    return action_indexes[0]

def getAdjacent(out_lines, num_nodes, or_successors=True):
    """
    Constructs an adjacency matrix and a special matrix for replaceable nodes based on the given output lines.

    Args:
        out_lines (list): A list of strings representing the output lines.
        num_nodes (int): The number of nodes in the graph.
        or_successors (bool, optional): Flag indicating whether to construct the special matrix for replaceable nodes. 
            Defaults to True.

    Returns:
        tuple: A tuple containing the adjacency matrix and the special matrix for replaceable nodes.
    """
    matrix_pre_proc = torch.zeros((num_nodes, num_nodes), dtype=torch.int)
    if not or_successors:
        ope_or = []  
    else:
        ope_or = []  
        ope_or_matrix = torch.eye(num_nodes, dtype=torch.int)
    for line in out_lines:
        parts = line.split()
        source = int(parts[0])
        for part in parts[1:]:
            if '(' in part:
                
                target_nodes = list(map(int, part.strip('()').split(',')))
                matrix_pre_proc[source, target_nodes] = 1

                if not or_successors:
                    ope_or += [(source, target_node) for target_node in target_nodes]
                    
                else:
                    ope_or += [(source, target_node) for target_node in target_nodes]
                    for i in range(len(target_nodes)):
                        for j in range(len(target_nodes)):
                            if i != j:
                                ope_or_matrix[target_nodes[i], target_nodes[j]] = 1
            else:
                target = int(part)
                matrix_pre_proc[source, target] = 1
    if or_successors:
        return matrix_pre_proc, ope_or, ope_or_matrix
    return matrix_pre_proc, ope_or

def getAncestors(adj_matrix):
    """
    Calculates the ancestors of each node in a directed graph represented by an adjacency matrix.

    Parameters:
        - adj_matrix (torch.Tensor): The adjacency matrix representing the directed graph.

    Returns:
        torch.Tensor: The cumulative matrix representing the ancestors of each node.
    """
    n = adj_matrix.shape[0]
    matrix_cal_cumul = torch.zeros((n, n), dtype=torch.int)

    edges = []
    for i in range(n):
        for j in range(n):
            if adj_matrix[i][j] == 1:
                edges.append([i, j])

    father = [set() for _ in range(n)]
    degree = [0] * n
    graph = [[] for _ in range(n)]
    
    # construct graph and calculate degrees
    for x, y in edges:
        degree[y] += 1
        graph[x].append(y)
    
    # initialization of queue
    q = deque()
    for i in range(n):
        if degree[i] == 0:
            q.append(i)
    
    # BFS
    while q:
        thisNode = q.popleft()
        for nextNode in graph[thisNode]:
            father[nextNode].add(thisNode)
            father[nextNode].update(father[thisNode])
            degree[nextNode] -= 1
            if degree[nextNode] == 0:
                q.append(nextNode)
    for index,i in enumerate(father):
        if len(i)==0:
            continue
        for j in i:
            matrix_cal_cumul[j,index]=1
    return  matrix_cal_cumul

def parse_data(lines):
    lines = [line.strip() for line in lines]
    
    split_index = lines.index('in')
    out_lines = lines[2:split_index]
    
    split_index_2 = lines.index('info')
    in_lines = lines[split_index + 1:split_index_2]
    
    info_lines = [lines[0]] + lines[split_index_2 + 1:]

    return out_lines, in_lines, info_lines

def nums_detec(lines):
    '''
    Count the number of jobs, machines and operations
    :param lines: List of strings, each string is a line of input data representing job-shop scheduling information
    :return: Tuple of (num_jobs, num_machines, num_opes)
    '''
    num_jobs, num_machines, num_opes = map(int, lines[0].split())
    return num_jobs, num_machines, num_opes

def sort_schedule(schedule, matrix_cal_cumul):
    def compare(x, y):
        i, j = int(x[0]), int(y[0])
        if matrix_cal_cumul[i, j] > matrix_cal_cumul[j, i]:
            return -1
        elif matrix_cal_cumul[i, j] < matrix_cal_cumul[j, i]:
            return 1
        else:
            return 0

    schedule = sorted(schedule, key=cmp_to_key(compare))
    schedule = sorted(schedule, key=lambda x: (x[3], x[4]))
    
    return np.array(schedule)

def solutions_padding(sols):
    tensor_sol = []
    for sol in sols:
        tensor_sol.append(sol)
    return pad_sequence(tensor_sol, batch_first=True, padding_value=-1)

def solution_reader(sol, drop:list):
    process = []
    if isinstance(sol, str):
        with open(sol, 'r') as file:
            lines = file.read().splitlines()
    else:
        lines = sol

    for line in lines[1:]:
        if isinstance(line, str):
            line = list(map(int, line.split()))
            ope, ma, job, start, _ = line
        else:
            ope, ma, job, start = line
        if int(ope) in drop:
            continue
        process.append([int(ope), int(ma), int(job), int(start)])
    return torch.tensor(process, dtype=torch.long)

def flatten_padded_tensor(ptr:torch.Tensor, padded_tensor):
    """
    Flatten a padded tensor based on the given pointer tensor.

    Args:
        ptr (Any)): A 1D tensor containing the pointers to the start of each sequence.
        padded_tensor (torch.Tensor): A 2D tensor containing the padded sequences.

    Returns:
        torch.Tensor: A 1D tensor containing the flattened sequences.
    """
    length = (ptr[1:] - ptr[:-1]).unsqueeze(-1).expand_as(padded_tensor)
    mask = (torch.arange(padded_tensor.size(1)).expand_as(padded_tensor) < length).flatten()
    return padded_tensor.flatten()[mask]
    
    