import torch
from env.ipps_env import EnvState
from torch_scatter import scatter
import random
    # """
    #     Notice that ALL PDR METHOD SHOULD SET BATCH_SIZE = 1
    # """
def SPT_pairs(state:EnvState, device):
    """
    Selects the next operation to be scheduled using the Shortest Processing Time (SPT) rule.

    Args:
        state (torch.Tensor): The current state of the system.
        device (torch.device): The device on which the computation will be performed.

    Returns:
        torch.Tensor: A tensor containing the indices of the selected operation, machine, and job.
    """
    eligible = state.eligible_pairs.to(device)
    # Get the processing time of each operation
    proc_time = torch.where(eligible, state.proc_times_batch, torch.ones_like(state.proc_times_batch) * float('inf'))
    # Choose a machine with the shortest processing time
    # 将flat索引转换回原始的操作和机器索引
    indices = proc_time.view(proc_time.size(0),-1).argmin(dim=1)
    num_mas = proc_time.size(2)
    operations_indices = indices // num_mas
    machines_indices = indices % num_mas
    jobs_indices = state.opes_appertain_batch.gather(1,operations_indices.view(-1,1)).squeeze(0)
    return torch.stack([operations_indices,machines_indices,jobs_indices],dim=1).t()

def RandChoice(state:EnvState, device):
    """
    Randomly selects a machine for each operation based on the shortest processing time.

    Args:
        state (torch.Tensor): The state of the system.
        device (torch.device): The device to perform the computation on.

    Returns:
        torch.Tensor: A tensor containing the chosen operation indices, machine indices, and job indices.
    """
    eligible = state.eligible_pairs.to(device)
    # Get the processing time of each operation
    proc_time = torch.where(eligible, state.proc_times_batch, torch.ones_like(state.proc_times_batch) * float('inf'))
    # Choose a machine with the shortest processing time
    flat_proc_time = proc_time.view(proc_time.size(0),-1)
    valid_indices = torch.isfinite(flat_proc_time)
    
    # Choose a valid index for each batch
    chosen_indices = [torch.multinomial(valid_indices[b].float(), 1) for b in range(proc_time.size(0))]
    chosen_indices = torch.stack(chosen_indices).squeeze(-1)
    
    # Convert flat indices back to operation and machine indices
    operations_indices = chosen_indices // proc_time.size(2)
    machines_indices = chosen_indices % proc_time.size(2)
    
    # Get the corresponding job indices
    jobs_indices = state.opes_appertain_batch.gather(1, operations_indices.unsqueeze(-1)).squeeze(-1)

    return torch.stack([operations_indices, machines_indices, jobs_indices], dim=1).t()

def MOR(state:EnvState, device, deal_combs = 'rand'):
    """
    Selects the next operation to be scheduled using the Most Operation Remaining (MOR) rule.

    Args:
        state (torch.Tensor): The current state of the system.
        device (torch.device): The device on which the computation will be performed.

    Returns:
        torch.Tensor: A tensor containing the indices of the selected operation, machine, and job.
    """
    eligible = state.eligible_pairs.to(device)
    remain_opes = state.remain_opes_batch.to(device).float()
    if deal_combs == 'min':
        jobs_remain_opes = ((torch.bmm(state.combs_batch.float(), remain_opes.unsqueeze(-1).float())\
            .permute(0,2,1).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = torch.inf)).min(dim=-1)[0]
        
    elif deal_combs == 'mean':
        jobs_remain_opes = ((torch.bmm(state.combs_batch.float(), remain_opes.unsqueeze(-1).float())\
            .permute(0,2,1).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = torch.nan)).nanmean(dim=-1)[0]
        
    elif deal_combs == 'max':
        jobs_remain_opes = ((torch.bmm(state.combs_batch.float(), remain_opes.unsqueeze(-1).float())\
            .permute(0,2,1).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = -torch.inf)).max(dim=-1)[0]
    
    elif deal_combs == 'rand':
        jobs_remain_opes_tem = ((torch.bmm(state.combs_batch.float(), remain_opes.unsqueeze(-1).float())\
            .permute(0,2,1).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = 0))

        shape = state.combs_id_batch.shape
        selected = torch.multinomial(torch.where(state.combs_id_batch.view(-1, shape[-1])==1, 1., 0.).float(), 1)
        selected = selected.view(shape[:-1])
        jobs_remain_opes = jobs_remain_opes_tem.gather(2, selected.unsqueeze(-1)).squeeze(-1)
            
    ope_remain_opes = jobs_remain_opes.gather(1, state.opes_appertain_batch)*state.next_ope_eligible_batch
    # Choose the operation with the most remaining operations
    opes = ope_remain_opes.argmax(dim=1)
    # Get the corresponding machine and job

    jobs = state.opes_appertain_batch.gather(1, opes.unsqueeze(-1)).squeeze(-1)
    opes = torch.where(eligible.float().sum(-1)[:,opes]>0, opes, -1).view(-1)
    jobs = torch.where(opes!=-1, jobs, -1)
    return opes, jobs

def MWKR(state:EnvState, device, deal_combs = 'rand'):
    """
    Selects the next operation to be scheduled using the Most Operation Remaining (MOR) rule.

    Args:
        state (torch.Tensor): The current state of the system.
        device (torch.device): The device on which the computation will be performed.

    Returns:
        torch.Tensor: A tensor containing the indices of the selected operation, machine, and job.
    """
    eligible = state.eligible_pairs.to(device)
    if deal_combs == 'min':
        jobs_remain_time = ((state.combs_time_batch.squeeze(-2).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = torch.inf)).min(dim=-1)[0]
        
    elif deal_combs == 'mean':
        jobs_remain_time = ((state.combs_time_batch.squeeze(-2).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = torch.nan)).nanmean(dim=-1)[0]
        
    elif deal_combs == 'max':
        jobs_remain_time = ((state.combs_time_batch.squeeze(-2).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = -torch.inf)).max(dim=-1)[0]
    elif deal_combs == 'rand':
        jobs_remain_time_tem = ((state.combs_time_batch.squeeze(-2).expand_as(state.combs_id_batch)*state.combs_id_batch).where(state.combs_id_batch == 1, other = 0))
        shape = state.combs_id_batch.shape
        selected = torch.multinomial(torch.where(state.combs_id_batch.view(-1, shape[-1])==1, 1., 0.).float(), 1)
        selected = selected.view(shape[:-1])
        jobs_remain_time = jobs_remain_time_tem.gather(2, selected.unsqueeze(-1)).squeeze(-1)
            
    ope_remain_time = jobs_remain_time.gather(1, state.opes_appertain_batch)*state.next_ope_eligible_batch
    # Choose the operation with the most remaining operations
  
    opes = ope_remain_time.argmax(dim=1)

    # Get the corresponding machine and job
    jobs = state.opes_appertain_batch.gather(1, opes.unsqueeze(-1)).squeeze(-1)
    opes = torch.where(eligible.float().sum(-1)[:,opes]>0, opes, -1).view(-1)
    jobs = torch.where(opes!=-1, jobs, -1)
    return opes, jobs

    # ope_remain_time = jobs_remain_time.gather(1, state.opes_appertain_batch) * eligible.float().sum(-1)
    # # Choose the operation with the most remaining operations
    # opes = ope_remain_time.argmax(dim=1)
    # # Get the corresponding machine and job

    # jobs = state.opes_appertain_batch.gather(1, opes.unsqueeze(-1)).squeeze(-1)
    # return opes, jobs

def FIFO(state:EnvState, device):
    """
    Selects the next operation to be scheduled using the First-In-First-Out (FIFO) rule.

    Args:
        state (torch.Tensor): The current state of the system.
        device (torch.device): The device on which the computation will be performed.

    Returns:
        torch.Tensor: A tensor containing the indices of the selected operation, machine, and job.
    """
    eligible = state.next_ope_eligible_batch.to(device)
    eligible_ready_time = torch.where(eligible>0, state.ready_time, torch.ones_like(state.ready_time) * float('inf'))
    # Get the operation indices
    opes = eligible_ready_time.argmin(dim=-1)
    # Get the corresponding job indices
    jobs = state.opes_appertain_batch.gather(1, opes.unsqueeze(-1)).squeeze(-1)
    return opes, jobs

def SPT(state:EnvState, opes, device):
    proc_time = torch.where(state.ope_ma_adj_batch == 1, state.proc_times_batch, torch.inf)
    mas = proc_time[state.batch_idxes, opes].argmin(dim=1)
    return mas

def EFT(state:EnvState, opes, device, machine_end_times):
    machine_end_times = torch.tensor(machine_end_times, device=device).unsqueeze(0).unsqueeze(0).\
        expand_as(state.proc_times_batch)
    ma_mask_end_times = torch.where(state.ope_ma_adj_batch == 1, machine_end_times, torch.inf)
    mas = ma_mask_end_times[state.batch_idxes, opes].argmin(dim=1)
    return mas

def LUM(state:EnvState, opes, device, machine_load):
    machine_load = torch.tensor(machine_load, device=device).unsqueeze(0).unsqueeze(0).\
        expand_as(state.proc_times_batch)
    ma_mask_load = torch.where(state.ope_ma_adj_batch == 1, machine_load, torch.inf)
    mas = ma_mask_load[state.batch_idxes, opes].argmin(dim=1)
    return mas

def Muhammad(state:EnvState, device, p):
    '''
        A priority-based heuristic algorithm (PBHA) for optimizing integrated process planning and scheduling problem
        https://doi.org/10.1080/23311916.2015.1070494
    '''

    rand_num_id = state.combs_id_batch.squeeze(0)*torch.rand_like(state.combs_id_batch)
    choice = rand_num_id.max(dim=2)[1].view(-1)
    choice_comb = random.choices(choice, p, k=1)
    ope_eli = state.combs_batch[0, choice_comb]*state.next_ope_eligible_batch
    if ope_eli.sum() == 0:
        return torch.tensor([-1], device=device), torch.tensor([-1], device=device)
    ope = torch.tensor([random.choices(ope_eli.nonzero()[0].tolist(), k=1)[0]], device=device)
    job = torch.tensor([state.opes_appertain_batch[0, ope]], device=device)
    return ope, job
    
    
        
def greedy_rule(state:EnvState, ope_rule:str, mas_rule:str, device, other_info = None, p = None):
    """
    Greey dispatching rule for the IPPS environment.
    """
    mode_dict = {
        'SPT': SPT,
        'RandChoice': RandChoice,
        'MOR': MOR,
        'MWKR': MWKR,
        'FIFO': FIFO,
        'EFT': EFT,
        'LUM': LUM,
        'Muhammad': Muhammad
    }
    # Get the operation indices
    if p is not None:
        opes, jobs = mode_dict[ope_rule](state, device, p)
    else:
        opes, jobs = mode_dict[ope_rule](state, device)
    # Get the machine indices
    if opes[0] == -1:
        mas = torch.tensor([-1], device=device)
    else:
        if other_info is not None:
            mas = mode_dict[mas_rule](state, opes, device, other_info)
        else:
            mas = mode_dict[mas_rule](state, opes, device)
    
    com = torch.stack([opes, mas, jobs], dim=1)
    choice = torch.where(state.eligible_pairs[state.batch_idxes, opes, mas], com, torch.full_like(com, -1))
    
    return choice.t()
    